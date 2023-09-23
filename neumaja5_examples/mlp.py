import argparse
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union
from typing import get_args as t_get_args

import lightning as L
import lightning.pytorch.callbacks as L_callbacks
import lovely_tensors as lt
import mlflow
import numpy as np
import torch
import torch_geometric.transforms as T
from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS, FITDatasetDefaults, TaskType
from db_transformer.data.embedder import CatEmbedder, NumEmbedder
from db_transformer.data.embedder.embedders import SingleTableEmbedder
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.utils import HeteroDataBuilder
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from db_transformer.helpers.timer import Timer
from db_transformer.schema.columns import CategoricalColumnDef, NumericColumnDef
from db_transformer.schema.schema import ColumnDef, Schema
from simple_parsing import ArgumentParser, DashVariant
from sqlalchemy.engine import Connection
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

lt.monkey_patch()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DEFAULT_DATASET_NAME = 'voc'


@dataclass
class DataConfig:
    pass


AggrType = Literal['sum', 'mean', 'min', 'max', 'cat']

DatasetType = Literal[
    'Accidents', 'Airline', 'Atherosclerosis', 'Basketball_women', 'Bupa', 'Carcinogenesis',
    'Chess', 'CiteSeer', 'ConsumerExpenditures', 'CORA', 'CraftBeer', 'Credit', 'cs', 'Dallas', 'DCG', 'Dunur',
    'Elti', 'ErgastF1', 'Facebook', 'financial', 'ftp', 'geneea', 'genes', 'Hepatitis_std', 'Hockey', 'imdb_ijs',
    'imdb_MovieLens', 'KRK', 'legalActs', 'medical', 'Mondial', 'Mooney_Family', 'MuskSmall', 'mutagenesis',
    'nations', 'NBA', 'NCAA', 'Pima', 'PremierLeague', 'PTE', 'PubMed_Diabetes', 'Same_gen', 'SAP', 'SAT',
    'Shakespeare', 'Student_loan', 'Toxicology', 'tpcc', 'tpcd', 'tpcds', 'trains', 'university', 'UTube',
    'UW_std', 'VisualGenome', 'voc', 'WebKP', 'world'
]


@dataclass
class ModelConfig:
    dim: int = 64
    layers: List[int] = field(default_factory=lambda: [])
    batch_norm: bool = False
    dropout: float = 0.0


class LinearBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: ModelConfig) -> None:
        super().__init__()

        self.block = torch.nn.ModuleList()
        self.block.append(torch.nn.Linear(in_dim, out_dim, bias=not config.batch_norm))
        if config.batch_norm:
            self.block.append(torch.nn.BatchNorm1d(out_dim))
        self.block.append(torch.nn.ReLU())
        if config.dropout > 0.0:
            self.block.append(torch.nn.Dropout1d(p=config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class Model(torch.nn.Module):
    def __init__(self,
                 schema: Schema,
                 config: ModelConfig,
                 n_features: int,
                 defaults: FITDatasetDefaults,
                 column_defs: List[ColumnDef],
                 column_names: List[str]):
        super().__init__()

        self.embedder = SingleTableEmbedder(
            (CategoricalColumnDef, lambda: CatEmbedder(dim=config.dim)),
            (NumericColumnDef, lambda: NumEmbedder(dim=config.dim)),
            dim=config.dim,
            column_defs=column_defs,
            column_names=column_names,
        )

        assert defaults.task == TaskType.CLASSIFICATION

        out_col_def = schema[defaults.target_table].columns[defaults.target_column]

        if not isinstance(out_col_def, CategoricalColumnDef):
            raise ValueError()

        in_dim = n_features * config.dim
        out_dim = out_col_def.card

        dims = [in_dim, *config.layers, out_dim]
        dims, (last_a, last_b) = dims[:-1], tuple(dims[-2:])

        self.layers = torch.nn.ModuleList([
            LinearBlock(a, b, config)
            for a, b in zip(dims[:-1], dims[1:])
        ])

        self.layers.append(torch.nn.Linear(last_a, last_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        x = x.view(*x.shape[:-2], -1)

        for layer in self.layers:
            x = layer(x)
        return x


class TheLightningModel(L.LightningModule):
    def __init__(self, model: Model, defaults: FITDatasetDefaults, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.defaults = defaults
        self.loss_module = torch.nn.CrossEntropyLoss()

        self._best_train_loss = float('inf')
        self._best_train_acc = 0.0
        self._best_val_acc = 0.0

    def forward(self, data: HeteroData, mode: Literal['train', 'test', 'val']):
        target_tbl = data[self.defaults.target_table]

        out = self.model(target_tbl.x)

        if mode == 'train':
            mask = target_tbl.train_mask
        elif mode == 'val':
            mask = target_tbl.val_mask
        elif mode == 'test':
            mask = target_tbl.test_mask
        else:
            raise ValueError()

        loss = self.loss_module(out[mask], target_tbl.y[mask])
        acc = (out[mask].argmax(dim=-1) == target_tbl.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")

        self.log("train_loss", loss, batch_size=1, prog_bar=True)
        if loss < self._best_train_loss:
            self._best_train_loss = loss
            self.log("best_train_loss", loss, batch_size=1, prog_bar=True)

        self.log("train_acc", acc, batch_size=1, prog_bar=True)
        if acc > self._best_train_acc:
            self._best_train_acc = acc
            self.log("best_train_acc", acc, batch_size=1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")

        self.log("val_acc", acc, batch_size=1, prog_bar=True)
        if acc > self._best_val_acc:
            self._best_val_acc = acc
            self.log("best_val_acc", acc, batch_size=1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")

        self.log("test_acc", acc, batch_size=1, prog_bar=True)
        if acc > self._best_test_acc:
            self._best_test_acc = acc
            self.log("best_test_acc", acc, batch_size=1, prog_bar=True)


_T = TypeVar('_T')


class SimpleDataset(Dataset[_T]):
    def __init__(self, data: Union[List[_T], Tuple[_T], _T], *other_data: _T) -> None:
        all_data = []

        if isinstance(data, list) or isinstance(data, tuple):
            all_data.extend(data)
        else:
            all_data.append(data)

        all_data.extend(other_data)

        self._data = all_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index) -> HeteroData:
        return self._data[index]


def build_data(
        c: Callable[[HeteroDataBuilder], _T],
        dataset=DEFAULT_DATASET_NAME,
        conn: Optional[Connection] = None,
        schema: Optional[Schema] = None,
        device=None) -> _T:
    has_sub_connection = conn is None

    if has_sub_connection:
        conn = FITRelationalDataset.create_remote_connection(dataset)

    if schema is None:
        schema = SchemaAnalyzer(conn, verbose=True).guess_schema()

    defaults = FIT_DATASET_DEFAULTS[dataset]

    builder = HeteroDataBuilder(conn,
                                schema,
                                target_table=defaults.target_table,
                                target_column=defaults.target_column,
                                separate_target=True,
                                create_reverse_edges=True,
                                fillna_with=0.0,
                                device=device)
    out = c(builder)

    if has_sub_connection:
        conn.close()

    return out


def create_data(dataset=DEFAULT_DATASET_NAME, data_config: Optional[DataConfig] = None, device=None):
    if data_config is None:
        data_config = DataConfig()

    with FITRelationalDataset.create_remote_connection(dataset) as conn:
        schema_analyzer = SchemaAnalyzer(conn,
                                         verbose=True)

        schema = schema_analyzer.guess_schema()

        defaults = FIT_DATASET_DEFAULTS[dataset]

        data_pd, (data, column_defs, colnames) = build_data(
            lambda builder: (builder.build_as_pandas(), builder.build(with_column_names=True)),
            dataset=dataset,
            conn=conn,
            schema=schema,
            # device=device
        )

        n_total = data[defaults.target_table].x.shape[0]
        T.RandomNodeSplit('train_rest', num_val=int(0.10 * n_total), num_test=int(0.10 * n_total))(data)

        return data, data_pd, schema, defaults, column_defs, colnames


def create_model(
        data: HeteroData,
        schema: Schema,
        column_defs: Dict[str, List[ColumnDef]],
        colnames: Dict[str, List[str]],
        dataset_name=DEFAULT_DATASET_NAME,
        model_config: Optional[ModelConfig] = None,
        device=None):
    if model_config is None:
        model_config = ModelConfig()

    defaults = FIT_DATASET_DEFAULTS[dataset_name]

    model = Model(schema=schema,
                  config=model_config,
                  n_features=len(column_defs[defaults.target_table]),
                  defaults=defaults,
                  column_defs=column_defs[defaults.target_table],
                  column_names=colnames[defaults.target_table])

    return model


class TimerOrEpochsCallback(L_callbacks.Callback):
    def __init__(self, epochs: int, min_train_time_s: float, epochs_multiplier: int = 10) -> None:
        self.timer = Timer(cuda=False, unit='s')
        self.min_train_time_s = min_train_time_s
        self.epochs = epochs
        self.epochs_multiplier = epochs_multiplier

    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.timer.start()

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if trainer.current_epoch == self.epochs:
            seconds = self.timer.end()
            if seconds >= self.min_train_time_s:
                # should stop
                trainer.should_stop = True
                trainer.strategy.barrier()
            else:
                # continue until ten-fold this many epochs
                self.epochs *= self.epochs_multiplier


def main(
        dataset_name: str = DEFAULT_DATASET_NAME,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
        epochs: int = 100,
        learning_rate: float = 3e-4,
        min_train_time_s: float = 60.,
        cuda: bool = False):
    if data_config is None:
        data_config = DataConfig()

    if model_config is None:
        model_config = ModelConfig()

    device = 'cuda' if cuda else 'cpu'
    data, data_pd, schema, defaults, column_defs, colnames = create_data(dataset_name, data_config, device)
    print(schema)
    print(data)

    model = create_model(data, schema, column_defs, colnames, dataset_name, model_config, device)
    print(model)

    lightning_model = TheLightningModel(model, defaults=defaults, lr=learning_rate)

    trainer = L.Trainer(
        accelerator='gpu' if cuda else 'cpu',
        deterministic=True,
        callbacks=[L_callbacks.Timer(),
                   TimerOrEpochsCallback(epochs=epochs, min_train_time_s=min_train_time_s)],
        min_epochs=epochs,
        max_epochs=-1,
        max_steps=-1,
    )

    dataloader = DataLoader([data], batch_size=1)

    trainer.fit(lightning_model, dataloader, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser(add_option_string_dash_variants=DashVariant.DASH)
    parser.add_argument('dataset', choices=t_get_args(DatasetType))
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr", "-r", type=float, default=0.0001)
    parser.add_argument("--min-train-time", "-t", type=float, default=60.)
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_arguments(ModelConfig, dest="model_config")
    parser.add_arguments(DataConfig, dest="data_config")
    args = parser.parse_args()
    model_config: ModelConfig = args.model_config
    data_config: DataConfig = args.data_config
    dataset: DatasetType = args.dataset
    cuda: bool = args.cuda
    epochs: int = args.epochs
    do_mlflow: bool = args.mlflow
    learning_rate: float = args.learning_rate
    min_train_time_s: float = args.min_train_time

    def _run_main():
        main(dataset, data_config, model_config, epochs, learning_rate, min_train_time_s, cuda)

    if do_mlflow:
        os.environ['MLFLOW_TRACKING_URI'] = 'http://147.32.83.171:2222'
        mlflow.set_experiment("deep_rl_learning NEW - neumaja5")
        mlflow.pytorch.autolog()

        file_name = os.path.basename(__file__)
        with mlflow.start_run(run_name=f"{file_name} - {dataset} - {uuid.uuid4()}") as run:
            mlflow.set_tag('dataset', dataset)
            mlflow.set_tag('Model Source', file_name)

            try:
                _run_main()
            except Exception as ex:
                mlflow.set_tag('exception', str(ex))
                raise ex
    else:
        _run_main()
