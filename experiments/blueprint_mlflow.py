from argparse import ArgumentParser
from datetime import datetime, timedelta
import os, sys

sys.path.append(os.getcwd())

from typing import get_args

import random

import numpy as np

import torch

import lightning as L
from lightning.pytorch import seed_everything

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, HGTLoader
import torch_geometric.transforms as T


import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Param
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_PARENT_RUN_ID

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import session as RaySession, CheckpointConfig

from db_transformer.nn.lightning import LightningWrapper
from db_transformer.nn.lightning.callbacks import (
    BestMetricsLoggerCallback,
    MLFlowLoggerCallback,
)
from db_transformer.data import (
    CTUDataset,
    CTUDatasetName,
)

from experiments.blueprint_instances.instances import create_blueprint_model

DEFAULT_DATASET_NAME: CTUDatasetName = "CORA"

DEFAULT_EXPERIMENT_NAME = "pelesjak-deep-db-tests"

RANDOM_SEED = 42

MAX_NEIGHBORS = 50


def prepare_run(config: tune.TuneConfig):
    session = RaySession.get_session()
    assert session != None

    mlflow_config = config.pop("mlflow_config", None)
    client: MlflowClient = mlflow_config["client"]

    experiment_name = mlflow_config.pop("experiment_name", None)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    parent_run_id = mlflow_config.pop("parent_run_id", None)

    run_name: str = mlflow_config.pop("run_name", None)

    run = client.create_run(
        experiment_id,
        run_name=run_name + f"_{session.trial_id}",
        tags={
            MLFLOW_USER: "pelesjak",
            MLFLOW_PARENT_RUN_ID: parent_run_id,
            "Dataset": config["dataset"],
            "trial_id": session.trial_id,
        },
    )
    return session, client, run


def train_model(config: tune.TuneConfig):
    print(f"Cuda available: {torch.cuda.is_available()}")
    log_dir = config.pop("log_dir", None)
    data_dir = config.pop("data_dir", None)
    session, client, run = prepare_run(config)

    run_id = run.info.run_id

    params = [Param(k, str(v)) for (k, v) in config.items()]
    client.log_batch(run_id, params=params)

    try:
        device = (
            torch.device(config.pop("device", "cpu"))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"Device: {device}")

        dataset = CTUDataset(config["dataset"], data_dir=data_dir, force_remake=False)

        target = dataset.defaults.target

        data, col_stats_dict = dataset.build_hetero_data(force_rematerilize=False)

        n_total = data[dataset.defaults.target_table].y.shape[0]
        data: HeteroData = T.RandomNodeSplit(
            split="train_rest", num_val=int(0.30 * n_total), num_test=0
        )(data)

        total_samples = data[target[0]].y.shape[0]

        min_batch_size = max(16, int(2 ** np.around(np.log2(total_samples / 100))))
        batch_size = min(min_batch_size * 2 ** config["batch_size_scale"], 16384)
        client.log_param(run_id, "batch_size", batch_size)

        train_loader = HGTLoader(
            data,
            num_samples=[MAX_NEIGHBORS] * config.get("gnn_layers", 1),
            batch_size=batch_size,
            input_nodes=(target[0], data[target[0]].train_mask),
            subgraph_type="bidirectional",
            shuffle=True,
        )

        val_loader = HGTLoader(
            data,
            num_samples=[MAX_NEIGHBORS] * config.get("gnn_layers", 1),
            batch_size=batch_size,
            input_nodes=(target[0], data[target[0]].val_mask),
            subgraph_type="bidirectional",
            shuffle=True,
        )

        edge_types = list(data.collect("edge_index", allow_empty=True).keys())

        model = create_blueprint_model(
            config["model_type"],
            dataset.defaults,
            {
                node: tf.col_names_dict
                for node, tf in data.collect("tf").items()
                if tf.num_rows > 0
            },
            edge_types,
            col_stats_dict,
            config,
        )

        lightning_model = LightningWrapper(
            model,
            dataset.defaults.target_table,
            lr=config["lr"],
            betas=config["betas"],
            task_type=dataset.defaults.task,
            verbose=False,
        )

        seed_everything(config["seed"], workers=True)

        metric = config["metric"]

        trainer = L.Trainer(
            accelerator=device.type,
            devices=1,
            deterministic=True,
            callbacks=[
                BestMetricsLoggerCallback(
                    monitor=f"val_{metric}",
                    cmp="max" if metric == "acc" else "min",
                    metrics=[
                        "train_loss",
                        "val_loss",
                        "test_loss",
                        f"train_{metric}",
                        f"val_{metric}",
                        f"test_{metric}",
                    ],
                ),
                MLFlowLoggerCallback(
                    run_id,
                    client,
                    session,
                    metrics=[
                        "train_loss",
                        "best_train_loss",
                        "val_loss",
                        "best_val_loss",
                        "test_loss",
                        "best_test_loss",
                        f"train_{metric}",
                        f"best_train_{metric}",
                        f"val_{metric}",
                        f"best_val_{metric}",
                        f"test_{metric}",
                        f"best_test_{metric}",
                    ],
                ),
            ],
            max_time=timedelta(hours=2),
            max_epochs=config["epochs"],
            min_epochs=2,
            max_steps=config["epochs"] * 2,
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(lightning_model, train_loader, val_dataloaders=val_loader)
        client.set_terminated(run_id)

    except Exception as e:
        client.set_tag(run_id, "exception", str(e))
        client.set_terminated(run_id, "FAILED")


def run_experiment(
    ray_address: str,
    tracking_uri: str,
    experiment_name: str,
    dataset: CTUDatasetName,
    num_samples: int,
    useCuda=False,
    log_dir: str = None,
    run_name: str = None,
    model_type: str = "transformer",
    epochs: int = 500,
    metric: str = None,
    random_seed: int = RANDOM_SEED,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
    run_name = f"{dataset}_{time_str}" if run_name is None else run_name

    log_dir = (
        os.path.join(os.getcwd(), "logs") if log_dir is None else os.path.abspath(log_dir)
    )

    with mlflow.start_run(run_name=run_name) as run:
        client = mlflow.tracking.MlflowClient(tracking_uri)

        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)

        analysis: tune.ExperimentAnalysis = tune.run(
            train_model,
            verbose=1,
            metric=f"best_val_{metric}",
            mode="max",
            search_alg=OptunaSearch(),
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            num_samples=num_samples,
            storage_path=log_dir,
            resources_per_trial=(
                {"gpu": 0.25, "cpu": 1, "memory": 4_000_000_000}
                if useCuda
                else {"cpu": 1, "memory": 4_000_000_000}
            ),
            log_to_file=True,
            local_dir=log_dir,
            resume=False,
            config={
                "lr": tune.loguniform(0.00005, 0.001, base=10),
                "betas": [0.9, 0.999],
                "embed_dim": tune.choice([32, 64]),
                "aggr": tune.choice(["sum"]),
                "gnn_layers": tune.randint(1, 5),
                "mlp_dims": tune.choice([[], [64], [64, 64]]),
                "batch_norm": tune.choice([True, False]),
                "batch_size_scale": tune.randint(0, 8),
                "model_type": model_type,
                "dataset": dataset,
                "epochs": epochs,
                "metric": metric,
                "device": "cuda" if useCuda else "cpu",
                "seed": random_seed,
                "data_dir": os.path.join(os.getcwd(), "datasets"),
                "log_dir": log_dir,
                "mlflow_config": {
                    "client": client,
                    "experiment_name": experiment_name,
                    "run_name": run_name,
                    "parent_run_id": run.info.run_id,
                },
            },
        )
        best_trial = analysis.best_trial
        mlflow.set_tags(
            {
                MLFLOW_USER: "pelesjak",
                "Dataset": dataset,
                "trial_id": best_trial.trial_id,
            }
        )
        mlflow.log_params(best_trial.config)
        mlflow.log_metrics(
            {
                k: v
                for (k, v) in analysis.best_result.items()
                if k is not None and v is not None and type(v) in [float, int]
            }
        )
        print(f"Best config: {analysis.best_result}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        choices=get_args(CTUDatasetName),
    )
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--model_type", type=str, default="transformer")

    args = parser.parse_args()
    print(args)

    run_experiment(
        args.ray_address,
        "http://147.32.83.171:2222",
        args.experiment,
        args.dataset,
        args.num_samples,
        args.cuda,
        args.log_dir,
        args.run_name,
        args.model_type,
        args.epochs,
        args.metric,
        args.seed,
    )
