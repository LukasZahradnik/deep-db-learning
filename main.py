import os
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union
from typing import get_args as t_get_args

import lightning as L
import lightning.pytorch.callbacks as L_callbacks
import mlflow
import numpy as np
import torch
import torch_geometric.transforms as T
from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS, FITDatasetDefaults, TaskType
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.utils import HeteroDataBuilder
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from db_transformer.helpers.timer import Timer
from db_transformer.schema.schema import ColumnDef, Schema
from simple_parsing import ArgumentParser, DashVariant
from sqlalchemy.engine import Connection
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# lt.monkey_patch()

# device = torch.device('cuda')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

mlflow.set_tracking_uri("http://localhost:2222/")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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

DEFAULT_DATASET_NAME: DatasetType = 'voc'


@dataclass
class ModelConfig:
    dim: int = 64
    attn: Literal['encoder', 'attn'] = 'attn'
    aggr: AggrType = 'sum'
    gnn_sub_layers: int = 1
    attn_sub_layers: int = 1
    gnn_layers: List[int] = field(default_factory=lambda: [])
    mlp_layers: List[int] = field(default_factory=lambda: [])
    batch_norm: bool = False
    layer_norm: bool = False


class TheLightningModel(L.LightningModule):
    def __init__(self, model, defaults: FITDatasetDefaults, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.defaults = defaults
        self.loss_module = torch.nn.CrossEntropyLoss()

        self._best_train_loss = float('inf')
        self._best_train_acc = 0.0
        self._best_val_acc = 0.0
        self._best_test_acc = 0.0

    def forward(self, data: HeteroData, mode: Literal['train', 'test', 'val']):
        target_tbl = data[self.defaults.target_table]

        out = self.model(data.x_dict, data.edge_index_dict)

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


def get_schema_analyzer(conn: Connection, defaults: FITDatasetDefaults) -> SchemaAnalyzer:
    target_type = 'categorical' if defaults.task == TaskType.CLASSIFICATION else 'numeric'

    return SchemaAnalyzer(
        conn,
        target=(defaults.target_table, defaults.target_column),
        target_type=target_type,
        verbose=True
    )


def build_data(
        c: Callable[[HeteroDataBuilder], _T],
        dataset=DEFAULT_DATASET_NAME,
        conn: Optional[Connection] = None,
        schema: Optional[Schema] = None,
        device=None) -> _T:
    has_sub_connection = conn is None

    if has_sub_connection:
        conn = FITRelationalDataset.create_remote_connection(dataset)

    defaults = FIT_DATASET_DEFAULTS[dataset]

    if schema is None:
        schema = get_schema_analyzer(conn, defaults).guess_schema()

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
        defaults = FIT_DATASET_DEFAULTS[dataset]

        schema_analyzer = FITRelationalDataset.create_schema_analyzer(dataset, conn, verbose=True)
        schema = schema_analyzer.guess_schema()

        data_pd, (data, column_defs, colnames) = build_data(
            lambda builder: (builder.build_as_pandas(), builder.build(with_column_names=True)),
            dataset=dataset,
            conn=conn,
            schema=schema,
            # device=device
        )

        n_total = data[defaults.target_table].x.shape[0]
        T.RandomNodeSplit('train_rest', num_val=int(0.2 * n_total), num_test=0)(data)

        return data, data_pd, schema, defaults, column_defs, colnames


def create_model(
        data: HeteroData,
        schema: Schema,
        column_defs: Dict[str, List[ColumnDef]],
        colnames: Dict[str, List[str]],
        dataset_name=DEFAULT_DATASET_NAME,
        model_config: Optional[ModelConfig] = None,
        device=None):
    from experiment import DBGNN
    from db_transformer.transformer import DBTransformer

    if model_config is None:
        model_config = ModelConfig()

    defaults = FIT_DATASET_DEFAULTS[dataset_name]
    model_config.gnn_layers = [0] * 5

    out_dim = schema[defaults.target_table].columns[defaults.target_column].card

    # if False:
    # model = DBGNN(
    #     model_config.dim, out_dim, model_config.dim, len(model_config.gnn_layers), data.metadata(), schema,
    #     column_defs=column_defs,
    #     column_names=colnames,
    #     config=model_config,
    #     target_table=defaults.target_table,
    # )

    model = DBTransformer(
        model_config.dim, out_dim, model_config.dim, len(model_config.gnn_layers), data.metadata(), 1, schema,
        column_defs=column_defs,
        column_names=colnames,
        config=model_config,
        target_table=defaults.target_table,
    )

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
    model = create_model(data, schema, column_defs, colnames, dataset_name, model_config, device)

    lightning_model = TheLightningModel(model, defaults=defaults, lr=learning_rate)

    trainer = L.Trainer(
        accelerator='gpu' if cuda else 'cpu',
        devices=1,
        deterministic=True,
        callbacks=[L_callbacks.Timer(),
                   L_callbacks.ModelCheckpoint('./torch-models/',
                                               filename=dataset_name + '-{epoch}-{train_acc:.3f}-{val_acc:.3f}',
                                               mode='max', monitor='val_acc'),
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
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr", "-r", type=float, default=0.0001)
    parser.add_argument("--min-train-time", "-t", type=float, default=60.)
    parser.add_arguments(ModelConfig, dest="model_config")
    parser.add_arguments(DataConfig, dest="data_config")
    args = parser.parse_args()
    model_config: ModelConfig = args.model_config
    data_config: DataConfig = args.data_config
    dataset: DatasetType = args.dataset
    cuda: bool = False
    epochs: int = args.epochs
    do_mlflow: bool = False
    learning_rate: float = args.learning_rate
    min_train_time_s: float = args.min_train_time


    def _run_main():
        main(dataset, data_config, model_config, epochs, learning_rate, min_train_time_s, cuda)


    if do_mlflow:
        mlflow.set_experiment("rl_db - lukas - new experiments II")
        mlflow.pytorch.autolog()

        file_name = os.path.basename(__file__)
        with mlflow.start_run(run_name=f"gatconv+mlp") as run:
            mlflow.set_tag('dataset', dataset)
            mlflow.set_tag('Model Source', file_name)

            for k, v in asdict(model_config).items():
                mlflow.log_param(k, v)

            for k, v in asdict(data_config).items():
                mlflow.log_param(k, v)

            try:
                _run_main()
            except Exception as ex:
                mlflow.set_tag('exception', str(ex))
                raise ex
    else:
        _run_main()