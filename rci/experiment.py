# %%
import argparse
import os
import random
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    get_args,
)

import lovely_tensors as lt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_PARENT_RUN_ID

import numpy as np

from simple_parsing import ArgumentParser
from sqlalchemy.engine import Connection

import torch
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import HeteroData
from torch_geometric.data.data import EdgeType, NodeType
from torch_geometric.loader import DataLoader

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import session as RaySession

from db_transformer.data.dataset_defaults.fit_dataset_defaults import (
    FIT_DATASET_DEFAULTS,
    FITDatasetDefaults,
    TaskType,
)
from db_transformer.data.embedder import CatEmbedder, NumEmbedder
from db_transformer.data.embedder.embedders import TableEmbedder
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.utils import HeteroDataBuilder
from db_transformer.helpers.timer import Timer
from db_transformer.schema.columns import CategoricalColumnDef, NumericColumnDef
from db_transformer.schema.schema import ColumnDef, Schema

from models import HeteroGNN
from models.layers import NodeApplied, PerFeatureNorm

# %load_ext autoreload
# %autoreload 2


# %%
@dataclass
class DataConfig:
    pass


DatasetType = Literal[
    "Accidents",
    "Airline",
    "Atherosclerosis",
    "Basketball_women",
    "Bupa",
    "Carcinogenesis",
    "Chess",
    "CiteSeer",
    "ConsumerExpenditures",
    "CORA",
    "CraftBeer",
    "Credit",
    "cs",
    "Dallas",
    "DCG",
    "Dunur",
    "Elti",
    "ErgastF1",
    "Facebook",
    "financial",
    "ftp",
    "geneea",
    "genes",
    "Hepatitis_std",
    "Hockey",
    "imdb_ijs",
    "imdb_MovieLens",
    "KRK",
    "legalActs",
    "medical",
    "Mondial",
    "Mooney_Family",
    "MuskSmall",
    "mutagenesis",
    "nations",
    "NBA",
    "NCAA",
    "Pima",
    "PremierLeague",
    "PTE",
    "PubMed_Diabetes",
    "Same_gen",
    "SAP",
    "SAT",
    "Shakespeare",
    "Student_loan",
    "Toxicology",
    "tpcc",
    "tpcd",
    "tpcds",
    "trains",
    "university",
    "UTube",
    "UW_std",
    "VisualGenome",
    "voc",
    "WebKP",
    "world",
]


DEFAULT_DATASET_NAME: DatasetType = "CORA"


AggrType = Literal["sum", "mean", "min", "max", "cat"]


@dataclass
class ModelConfig:
    dim: int = 64
    aggr: AggrType = "sum"
    gnn_layers: List[int] = field(default_factory=lambda: [])
    mlp_layers: List[int] = field(default_factory=lambda: [])
    batch_norm: bool = False
    layer_norm: bool = False


_T = TypeVar("_T")

DEFAULT_EXPERIMENT_NAME = "deep-db-tests-pelesjak"

# %%
parser = ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default=DEFAULT_DATASET_NAME, choices=get_args(DatasetType)
)
parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument("--run_name", type=str, default=None)

# %%
lt.monkey_patch()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# %%
def build_data(
    dataset=DEFAULT_DATASET_NAME,
    conn: Optional[Connection] = None,
    schema: Optional[Schema] = None,
    device=None,
) -> _T:
    has_sub_connection = conn is None

    if has_sub_connection:
        conn = FITRelationalDataset.create_remote_connection(
            dataset, dialect="postgresql"
        )

    defaults = FIT_DATASET_DEFAULTS[dataset]

    if schema is None:
        schema = FITRelationalDataset.create_schema_analyzer(
            dataset, conn, verbose=True
        ).guess_schema()

    builder = HeteroDataBuilder(
        conn,
        schema,
        target_table=defaults.target_table,
        target_column=defaults.target_column,
        separate_target=True,
        create_reverse_edges=True,
        fillna_with=0.0,
        device=device,
    )

    data_pd = builder.build_as_pandas()
    data = builder.build(with_column_names=True)

    if has_sub_connection:
        conn.close()

    return data, data_pd


# %%
def _expand_with_dummy_features(
    data: HeteroData, column_defs: Dict[NodeType, List[ColumnDef]]
):
    for node_type in data.node_types:
        x = data[node_type].x
        if x.shape[-1] == 0:
            x = torch.ones((*x.shape[:-1], 1), dtype=x.dtype)
            column_defs[node_type] = [NumericColumnDef()]
            data[node_type].x = x


# %%
def create_data(
    dataset=DEFAULT_DATASET_NAME, data_config: Optional[DataConfig] = None, device=None
):
    if data_config is None:
        data_config = DataConfig()

    usePostgres = True

    FITDatasetDefaults.case_sensitive = not usePostgres

    with FITRelationalDataset.create_remote_connection(
        dataset, dialect=("postgresql" if usePostgres else "mariadb")
    ) as conn:
        defaults = FIT_DATASET_DEFAULTS[dataset]

        print(f"Connected to db {dataset}...")

        schema_analyzer = FITRelationalDataset.create_schema_analyzer(
            dataset, conn, verbose=True
        )
        schema = schema_analyzer.guess_schema()

        print("Created a schema...")

        (data, column_defs, colnames), data_pd = build_data(
            dataset=dataset, conn=conn, schema=schema, device=device
        )

        print("Build the data...")

        _expand_with_dummy_features(data, column_defs)

        n_total = data[defaults.target_table].x.shape[0]

        data: HeteroData = RandomNodeSplit(
            "train_rest", num_val=int(0.30 * n_total), num_test=0
        )(data)

        return data, data_pd, schema, defaults, column_defs, colnames


# %%
class Model(torch.nn.Module):
    def __init__(
        self,
        schema: Schema,
        config: ModelConfig,
        edge_types: List[Tuple[str, str, str]],
        defaults: FITDatasetDefaults,
        column_defs: Dict[str, List[ColumnDef]],
        column_names: Dict[str, List[str]],
    ):
        super().__init__()
        self.defaults = defaults

        node_types = list(column_defs.keys())

        self.embedder = TableEmbedder(
            (CategoricalColumnDef, lambda: CatEmbedder(dim=config.dim)),
            (NumericColumnDef, lambda: NumEmbedder(dim=config.dim)),
            dim=config.dim,
            column_defs=column_defs,
            column_names=column_names,
        )

        node_dims = {k: len(cds) for k, cds in column_defs.items()}

        self.layer_norm = (
            NodeApplied(
                lambda nt: PerFeatureNorm(node_dims[nt], axis=-2), node_types=node_types
            )
            if config.layer_norm
            else None
        )

        assert defaults.task == TaskType.CLASSIFICATION

        out_col_def = schema[defaults.target_table].columns[defaults.target_column]

        if not isinstance(out_col_def, CategoricalColumnDef):
            raise ValueError()

        out_dim = out_col_def.card
        gnn_out_dim = config.mlp_layers[0] if config.mlp_layers else out_dim

        node_dims = {k: len(cds) * config.dim for k, cds in column_defs.items()}

        self.gnn = HeteroGNN(
            dims=config.gnn_layers,
            node_dims=node_dims,
            out_dim=gnn_out_dim,
            node_types=node_types,
            edge_types=edge_types,
            aggr=config.aggr,
            batch_norm=config.batch_norm,
        )

        if config.mlp_layers:
            mlp_layer_dims = [*config.mlp_layers, out_dim]

            mlp_layers = []

            for a, b in zip(mlp_layer_dims[:-1], mlp_layer_dims[1:]):
                mlp_layers += [torch.nn.ReLU(), torch.nn.Linear(a, b)]

                if config.batch_norm:
                    mlp_layers += [torch.nn.BatchNorm1d(b)]

            if config.batch_norm:
                del mlp_layers[-1]

            self.mlp = torch.nn.Sequential(*mlp_layers)
        else:
            self.mlp = None

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_dict: Dict[EdgeType, torch.Tensor],
    ) -> torch.Tensor:
        x_dict = self.embedder(x_dict)

        if self.layer_norm is not None:
            x_dict = self.layer_norm(x_dict)

        x_dict = {k: x.view(*x.shape[:-2], -1) for k, x in x_dict.items()}

        x_dict = self.gnn(x_dict, edge_dict)

        x = x_dict[self.defaults.target_table]

        if self.mlp is not None:
            x = self.mlp(x)

        return x


# %%
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


# %%
def train_model(config: tune.TuneConfig):
    print(f"Cuda available: {torch.cuda.is_available()}")
    session, client, run = prepare_run(config)

    run_id = run.info.run_id

    params = [mlflow.entities.Param(k, str(v)) for (k, v) in config.items()]
    client.log_batch(run_id, params=params)

    try:
        device = config.pop("device", "cpu") if torch.cuda.is_available() else "cpu"

        data, data_pd, schema, defaults, column_defs, colnames = create_data(
            config["dataset"], device=device
        )

        model = Model(
            schema=schema,
            config=ModelConfig(
                dim=config["embed_dim"],
                aggr=config["aggr"],
                gnn_layers=config["gnn"],
                mlp_layers=config["mlp"],
                batch_norm=config["batch_norm"],
                layer_norm=config["layer_norm"],
            ),
            edge_types=data.edge_types,
            defaults=defaults,
            column_defs=column_defs,
            column_names=colnames,
        )

        model = model.to(device)

        dataloader = DataLoader([data], batch_size=1)

        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(True), lr=config["lr"], betas=config["betas"]
        )

        target_tbl = data[defaults.target_table]

        best_train_acc = 0
        best_val_acc = 0

        for e in range(config["epochs"]):
            model.train()
            train_acc = 0
            train_norm = 0
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()

                out = model(batch.collect("x"), batch.collect("edge_index"))

                mask = target_tbl.train_mask

                loss = loss_fn(out[mask], target_tbl.y[mask])
                train_acc += (
                    (out[mask].argmax(dim=-1) == target_tbl.y[mask]).sum().float()
                )
                train_norm += mask.sum()

                loss.backward()
                optimizer.step()

            model.eval()
            val_acc = 0
            val_norm = 0
            for batch in dataloader:
                batch = batch.to(device)
                out = model(batch.collect("x"), batch.collect("edge_index"))
                mask = target_tbl.val_mask
                val_acc += (
                    (out[mask].argmax(dim=-1) == target_tbl.y[mask]).sum().float()
                )
                val_norm += mask.sum()

            train_acc = (train_acc / train_norm).item()
            val_acc = (val_acc / val_norm).item()

            if best_train_acc < train_acc:
                best_train_acc = train_acc

            if best_val_acc < val_acc:
                best_val_acc = val_acc

            metric_dict = {
                "best_train_acc": best_train_acc,
                "best_val_acc": best_val_acc,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }

            # print(f'val_acc: {val_acc}, best_val_acc: {best_val_acc}', end='\r')

            timestamp = int(datetime.now().timestamp() * 1000)

            metrics = [
                mlflow.entities.Metric(key, value, timestamp, e)
                for (key, value) in metric_dict.items()
            ]
            client.log_batch(run_id, metrics=metrics, synchronous=False)

            session.report(metric_dict)

        client.set_terminated(run_id)

    except Exception as ex:
        client.set_tag(run_id, "exception", str(ex))
        raise ex


# %%
def run_experiment(
    tracking_uri: str,
    experiment_name: str,
    dataset: DatasetType,
    num_samples: int,
    useCuda=False,
    run_name: str = None,
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
    run_name = f"{dataset}_{time_str}" if run_name is None else run_name

    with mlflow.start_run(run_name=run_name) as run:
        client = mlflow.tracking.MlflowClient(tracking_uri)

        ray.init(num_cpus=10, num_gpus=4)

        analysis: tune.ExperimentAnalysis = tune.run(
            train_model,
            verbose=1,
            metric="best_val_acc",
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=num_samples,
            storage_path=os.getcwd() + "/results",
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
                "epochs": 4000,
                "device": "cuda" if useCuda else "cpu",
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


# %%
# run_experiment('http://147.32.83.171:2222', DEFAULT_EXPERIMENT_NAME, "world", 1, True)

# %%
args = parser.parse_args()
print(args)

run_experiment(
    "http://147.32.83.171:2222",
    args.experiment,
    args.dataset,
    args.num_samples,
    args.cuda,
    args.run_name,
)
