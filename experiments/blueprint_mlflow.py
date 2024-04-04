from argparse import ArgumentParser
from datetime import datetime
import math
import os, sys

sys.path.append("/home/jakub/Documents/ŠKOLA/Ing./Diplomka/deep-db-learning")

from typing import List, Dict, Literal, Union, Optional, Any, Tuple, get_args

import random

import numpy as np

import torch

from lightning import LightningModule, Trainer, Callback as LightningCallback
from lightning.pytorch.loggers import MLFlowLogger

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import conv, Sequential, summary
import torch_geometric.transforms as T
from torch_geometric.typing import EdgeType, NodeType

import torch_frame
from torch_frame import stype, NAStrategy
from torch_frame.nn import encoder
from torch_frame.nn import TabTransformerConv
from torch_frame.data import StatType

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_PARENT_RUN_ID

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import session as RaySession

from db_transformer.nn import (
    BlueprintModel,
    EmbeddingTranscoder,
    SelfAttention,
    CrossAttentionConv,
    NodeApplied,
)
from db_transformer.data import (
    CTUDataset,
    TaskType,
    CTUDatasetName,
    CTU_REPOSITORY_DEFAULTS,
)

DEFAULT_DATASET_NAME: CTUDatasetName = "CORA"

DEFAULT_EXPERIMENT_NAME = "deep-db-tests-pelesjak"

RANDOM_SEED = 42


class BestMetricsLoggerCallback(LightningCallback):
    def __init__(
        self,
        monitor: str = "val_acc",
        cmp: Literal["min", "max"] = "max",
        metrics: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> None:
        if metrics is None:
            metrics = {
                "train_acc": "best_train_acc",
                "val_acc": "best_val_acc",
                "test_acc": "best_test_acc",
                "train_loss": "best_train_loss",
                "val_loss": "best_val_loss",
                "test_loss": "best_test_loss",
            }

        self.monitor = monitor
        self.cmp = cmp
        self.metrics = metrics
        self.best_value: Optional[float] = None
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.monitor not in trainer.callback_metrics:
            return

        mon_value = trainer.callback_metrics[self.monitor].detach().cpu().item()

        if self.best_value is not None and (
            self.cmp == "min" and mon_value >= self.best_value
        ):
            return

        if self.best_value is not None and (
            self.cmp == "max" and mon_value <= self.best_value
        ):
            return

        self.best_value = mon_value
        for metric_name, log_as in self.metrics.items():
            if metric_name not in trainer.callback_metrics:
                continue
            v = trainer.callback_metrics[metric_name].detach().cpu().item()
            pl_module.log(log_as, v, prog_bar=self.verbose)


class MLFlowLoggerCallback(LightningCallback):
    def __init__(
        self,
        run_id: str,
        mlflow_client: MlflowClient,
        ray_session: Any,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.run_id = run_id
        self.mlflow_client = mlflow_client
        self.ray_session = ray_session

        if metrics is None:
            metrics = [
                "train_acc",
                "best_train_acc",
                "val_acc",
                "best_val_acc",
                "test_acc",
                "best_test_acc",
                "train_loss",
                "best_train_loss",
                "val_loss",
                "best_val_loss",
                "test_loss",
                "best_test_loss",
            ]
        self.metrics = metrics

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metric_dict = {}
        mlflow_metrics = []
        timestamp = int(datetime.now().timestamp() * 1000)

        for metric_name in self.metrics:
            if metric_name not in trainer.callback_metrics:
                continue
            metric_dict[metric_name] = (
                trainer.callback_metrics[metric_name].detach().cpu().item()
            )
            mlflow_metrics.append(
                mlflow.entities.Metric(
                    metric_name, metric_dict[metric_name], timestamp, trainer.current_epoch
                )
            )

        self.mlflow_client.log_batch(self.run_id, metrics=mlflow_metrics, synchronous=False)

        self.ray_session.report(metric_dict)


class LightningModel(LightningModule):
    def __init__(
        self,
        model: BlueprintModel,
        target_table: str,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()
        self.model = model
        self.target_table = target_table
        self.lr = lr
        self.betas = betas
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, data: HeteroData):
        out = self.model(data.collect("tf"), data.collect("edge_index", allow_empty=True))

        target = data[self.target_table].y
        loss = self.loss_module(out, target)
        acc = (out.argmax(dim=-1) == target).type(torch.float).mean()
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def training_step(self, batch):
        loss, acc = self.forward(batch)
        batch_size = batch[self.target_table].y.shape[0]
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log("train_acc", acc, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch):
        _, acc = self.forward(batch)
        batch_size = batch[self.target_table].y.shape[0]

        self.log("val_acc", acc, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch):
        _, acc = self.forward(batch)
        batch_size = batch[self.target_table].y.shape[0]

        self.log("test_acc", acc, batch_size=batch_size, prog_bar=True)


def create_model(
    target: Tuple[str, str], embed_dim: int, data: HeteroData
) -> BlueprintModel:

    return BlueprintModel(
        target=target,
        embed_dim=embed_dim,
        col_stats_per_table=data.collect("col_stats"),
        col_names_dict_per_table={
            k: tf.col_names_dict for k, tf in data.collect("tf").items()
        },
        edge_types=list(data.collect("edge_index").keys()),
        stype_encoder_dict={
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.LinearEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
            stype.embedding: EmbeddingTranscoder(in_channels=300),
        },
        positional_encoding=True,
        num_gnn_layers=3,
        table_transform=SelfAttention(embed_dim, 16),
        table_transform_unique=True,
        # table_transform=lambda i, node, cols: ExcelFormerConv(embed_dim, len(cols), 1),
        table_combination=CrossAttentionConv(embed_dim, 4),
        table_combination_unique=True,
        # table_combination=conv.TransformerConv(embed_dim, embed_dim, heads=4, dropout=0.1, root_weight=False),
        decoder_aggregation=lambda x: torch.reshape(x, (-1, math.prod(x.shape[1:]))),
        decoder=lambda cols: torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim * len(cols),
                len(data[target[0]].col_stats[target[1]][StatType.COUNT][0]),
            ),
        ),
        output_activation=torch.nn.Softmax(dim=-1),
        positional_encoding_dropout=0.0,
        table_transform_dropout=0.1,
        table_combination_dropout=0.1,
        table_transform_residual=False,
        table_combination_residual=False,
        table_transform_norm=False,
        table_combination_norm=False,
    )


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
    session, client, run = prepare_run(config)

    run_id = run.info.run_id

    params = [mlflow.entities.Param(k, str(v)) for (k, v) in config.items()]
    client.log_batch(run_id, params=params)

    try:
        device = (
            torch.device(config.pop("device", "cpu"))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"Device: {device}")

        dataset = CTUDataset(
            config["dataset"], data_dir=config["shared_dir"], force_remake=False
        )

        target = dataset.defaults.target

        data = dataset.build_hetero_data(device, force_rematerilize=False)

        n_total = data[dataset.defaults.target_table].y.shape[0]
        data: HeteroData = T.RandomNodeSplit(
            split="train_rest", num_val=int(0.30 * n_total), num_test=0
        )(data)

        num_neighbors = {
            edge_type: [30] * 5 for edge_type in data.collect("edge_index").keys()
        }

        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=10000,
            input_nodes=(target[0], data[target[0]].train_mask),
            subgraph_type="bidirectional",
        )

        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=1000,
            input_nodes=(target[0], data[target[0]].val_mask),
            subgraph_type="bidirectional",
        )

        lightning_model = LightningModel(
            create_model(target, config["embed_dim"], data).to(device),
            dataset.defaults.target_table,
            lr=config["lr"],
            betas=config["betas"],
        ).to(device)

        trainer = Trainer(
            accelerator=device.type,
            devices=1,
            deterministic=True,
            callbacks=[
                BestMetricsLoggerCallback(),
                MLFlowLoggerCallback(run_id, client, session),
            ],
            max_epochs=config["epochs"],
            max_steps=-1,
        )

        trainer.fit(lightning_model, train_loader, val_dataloaders=val_loader)

        client.set_terminated(run_id)

    except Exception as ex:
        client.set_tag(run_id, "exception", str(ex))
        raise ex


def run_experiment(
    tracking_uri: str,
    experiment_name: str,
    dataset: CTUDatasetName,
    num_samples: int,
    useCuda=False,
    run_name: str = None,
    random_seed: int = RANDOM_SEED,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
    run_name = f"{dataset}_{time_str}" if run_name is None else run_name

    with mlflow.start_run(run_name=run_name) as run:
        client = mlflow.tracking.MlflowClient(tracking_uri)

        ray.init(num_cpus=4, num_gpus=1)

        analysis: tune.ExperimentAnalysis = tune.run(
            train_model,
            verbose=1,
            metric="best_val_acc",
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=num_samples,
            storage_path=os.getcwd() + "/ray_tune_results",
            resources_per_trial={"gpu": 1, "cpu": 2} if useCuda else None,
            config={
                "lr": 0.0001,  # tune.loguniform(0.00005, 0.001),
                "betas": [0.9, 0.999],
                "embed_dim": tune.choice([16, 32]),
                "aggr": tune.choice(["sum"]),
                "gnn": tune.choice([[], [16]]),
                "mlp": tune.choice([[8], [16], [32]]),
                "batch_norm": tune.choice([True]),
                "layer_norm": tune.choice([False]),
                "dataset": dataset,
                "epochs": 10,
                "device": "cuda" if useCuda else "cpu",
                "seed": random_seed,
                "shared_dir": os.path.join(os.getcwd(), "datasets"),
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


parser = ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default=DEFAULT_DATASET_NAME, choices=get_args(CTUDatasetName)
)
parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=RANDOM_SEED)

args = parser.parse_args()
print(args)

run_experiment(
    "http://147.32.83.171:2222",
    args.experiment,
    args.dataset,
    args.num_samples,
    args.cuda,
    args.run_name,
    args.seed,
)
