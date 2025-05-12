from argparse import ArgumentParser
from datetime import datetime, timedelta
import os, sys


os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

sys.path.append(os.getcwd())

from typing import Literal, get_args

import random

import numpy as np

import torch

import lightning as L
from lightning.pytorch import seed_everything

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, HGTLoader
import torch_geometric.transforms as T

from torch_frame.data import StatType

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
    CTU_REPOSITORY_DEFAULTS,
    TaskType,
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

    session.report(
        {f"val_{config['metric']}": (-1e15 if config["higher_is_better"] else 1e15)}
    )

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
        seed_everything(config["seed"], workers=True)

        data: HeteroData = T.RandomNodeSplit(
            split="train_rest", num_val=int(0.30 * n_total), num_test=0
        )(data)

        total_samples = data[target[0]].y.shape[0]

        min_batch_size = max(16, int(2 ** np.around(np.log2(total_samples / 500))))
        batch_size = min(min_batch_size * 2 ** config["batch_size_scale"], 16384)
        client.log_param(run_id, "batch_size", batch_size)

        train_loader = HGTLoader(
            data,
            num_samples=[MAX_NEIGHBORS] * max(1, config.get("gnn_layers", 1)),
            batch_size=batch_size,
            input_nodes=(target[0], data[target[0]].train_mask),
            shuffle=True,
        )

        val_loader = HGTLoader(
            data,
            num_samples=[MAX_NEIGHBORS] * max(1, config.get("gnn_layers", 1)),
            batch_size=batch_size,
            input_nodes=(target[0], data[target[0]].val_mask),
            shuffle=True,
        )

        edge_types = list(data.collect("edge_index", allow_empty=True).keys())

        model = create_blueprint_model(
            config["model_type"],
            defaults=dataset.defaults,
            col_names_dict={
                node: tf.col_names_dict
                for node, tf in data.collect("tf").items()
                if tf.num_rows > 0
            },
            edge_types=edge_types,
            col_stats_dict=col_stats_dict,
            config=config,
        )
        print(
            "model_size {:.3f}M".format(
                sum(p.numel() for p in model.parameters()) / 1_000_000
            )
        )
        client.log_param(
            run_id,
            "model_size",
            "{:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1_000_000),
        )
        num_classes = (
            len(col_stats_dict[target[0]][target[1]][StatType.COUNT][0])
            if dataset.defaults.task == TaskType.CLASSIFICATION
            else 1
        )
        client.log_param(run_id, "num_classes", num_classes)

        lightning_model = LightningWrapper(
            model,
            dataset.defaults.target_table,
            lr=config["lr"],
            betas=config["betas"],
            task_type=dataset.defaults.task,
            num_classes=(
                len(col_stats_dict[target[0]][target[1]][StatType.COUNT][0])
                if dataset.defaults.task == TaskType.CLASSIFICATION
                else 1
            ),
            verbose=False,
        )

        val_metric = config["metric"]
        higher_is_better = config["higher_is_better"]

        log_metrics = []
        all_metrics = []
        for m_name in ["loss", *lightning_model.metrics.keys()]:
            log_metrics.extend([f"train_{m_name}", f"val_{m_name}"])
        for m_name in log_metrics:
            all_metrics.extend([m_name, f"best_{m_name}"])

        trainer = L.Trainer(
            accelerator=device.type,
            devices=1,
            deterministic=True,
            callbacks=[
                BestMetricsLoggerCallback(
                    monitor=f"val_{val_metric}",
                    cmp="max" if higher_is_better else "min",
                    metrics=log_metrics,
                ),
                MLFlowLoggerCallback(
                    run_id,
                    client,
                    session,
                    metrics=all_metrics,
                ),
            ],
            max_epochs=1000,
            min_epochs=2,
            max_steps=4500,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(lightning_model, train_loader, val_dataloaders=val_loader)
        client.set_terminated(run_id)

    except Exception as e:
        print(str(e))
        client.set_tag(run_id, "exception", str(e))
        client.set_terminated(run_id, "FAILED")


def get_tune_config(
    model_type: Literal[
        "excelformer",
        "honza",
        "mlp",
        "saint",
        "tabnet",
        "tabtransformer",
        "transformer",
        "trompt",
    ],
):
    if model_type == "mlp":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "none",
            "gnn_layers": 0,
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
        }
    if model_type == "honza":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "sum",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
        }
    if model_type == "transformer":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "attn",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
            "num_heads": tune.choice([1, 4, 8]),
            "dropout": tune.choice([0.0, 0.2]),
            "positional": False,
            "encoder": "all",
        }

    if model_type == "transformer-basic":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "attn",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
            "num_heads": tune.choice([1, 4, 8]),
            "dropout": tune.choice([0.0, 0.2]),
            "positional": False,
            "encoder": "basic",
        }

    if model_type == "transformer-text":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "attn",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
            "num_heads": tune.choice([1, 4, 8]),
            "dropout": tune.choice([0.0, 0.2]),
            "positional": False,
            "encoder": "with_embeddings",
        }

    if model_type == "transformer-time":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "attn",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
            "num_heads": tune.choice([1, 4, 8]),
            "dropout": tune.choice([0.0, 0.2]),
            "positional": False,
            "encoder": "with_time",
        }
    if model_type == "saint":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": tune.choice(["attn", "sum"]),
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
            "num_heads": tune.choice([4, 8]),
            "dropout": 0.1,
        }
    if model_type == "trompt":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "sum",
            "gnn_layers": tune.randint(1, 5),
            "num_trompt_layers": tune.choice([2, 4, 6, 8]),
        }
    if model_type == "tabnet":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "sum",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "num_layers": tune.choice([3, 5, 7]),
        }
    if model_type == "tabtransformer":
        return {
            "embed_dim": tune.choice([16, 32, 64]),
            "aggr": "sum",
            "gnn_layers": tune.randint(1, 5),
            "mlp_dims": tune.choice([[], [64], [64, 64]]),
            "batch_norm": tune.choice([True, False]),
            "num_heads": tune.choice([2, 4, 8]),
            "num_layers": tune.choice([1, 2, 3, 6]),
            "dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
        }

    raise ValueError(f"Unknown model type '{model_type}'")


def run_experiment(
    ray_address: str,
    tracking_uri: str,
    experiment_name: str,
    dataset: CTUDatasetName,
    num_samples: int,
    use_cuda=False,
    num_cpus: int = 1,
    num_gpus: int = 0,
    log_dir: str = None,
    run_name: str = None,
    model_type: str = "transformer",
    random_seed: int = RANDOM_SEED,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
    run_name = f"{dataset}_{time_str}" if run_name is None else run_name

    defaults = CTU_REPOSITORY_DEFAULTS[dataset]

    log_dir = (
        os.path.join(os.getcwd(), "logs") if log_dir is None else os.path.abspath(log_dir)
    )

    with mlflow.start_run(run_name=run_name) as run:
        client = mlflow.tracking.MlflowClient(tracking_uri)

        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            log_to_driver=True,
            num_cpus=num_cpus if ray_address == "local" else None,
            num_gpus=num_gpus if ray_address == "local" else None,
        )

        if defaults.task == TaskType.CLASSIFICATION:
            metric = "auroc"
            higher_is_better = True
        elif defaults.task == TaskType.REGRESSION:
            metric = "mae"
            higher_is_better = False
        else:
            raise ValueError(f"Unknown task type '{defaults.task}'")

        analysis: tune.ExperimentAnalysis = tune.run(
            train_model,
            name=run_name,
            metric=f"val_{metric}",
            mode="max" if higher_is_better else "min",
            verbose=1,
            search_alg=OptunaSearch(
                metric=f"val_{metric}",
                mode="max" if higher_is_better else "min",
            ),
            stop={"time_total_s": 3600 * 4},  #  4 hours
            max_concurrent_trials=6,
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            num_samples=num_samples,
            storage_path=log_dir,
            resources_per_trial=(
                {"gpu": 0.25, "cpu": 1} if use_cuda else {"gpu": 0, "cpu": 1}
            ),
            log_to_file=True,
            # local_dir=log_dir,
            config={
                "lr": tune.loguniform(0.00005, 0.002),
                "betas": [0.9, 0.999],
                **get_tune_config(model_type),
                "batch_size_scale": tune.randint(0, 8),
                "model_type": model_type,
                "dataset": dataset,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "device": "cuda" if use_cuda else "cpu",
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
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="honza")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    args = parser.parse_args()
    print(args)

    run_experiment(
        ray_address=args.ray_address,
        tracking_uri="http://147.32.83.171:2222",
        experiment_name=args.experiment,
        dataset=args.dataset,
        num_samples=args.num_samples,
        use_cuda=args.cuda,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        log_dir=args.log_dir,
        run_name=args.run_name,
        model_type=args.model_type,
        random_seed=args.seed,
    )
