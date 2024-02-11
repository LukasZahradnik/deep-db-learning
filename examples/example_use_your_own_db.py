
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from enum import Enum

import lightning as L
import lightning.pytorch.callbacks as L_callbacks
import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from simple_parsing import ArgumentParser, DashVariant

from db_transformer.data.dataset import DBDataset
from db_transformer.data.utils import HeteroDataBuilder
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from db_transformer.helpers.timer import Timer
from db_transformer.schema.schema import ColumnDef, Schema

# device = torch.device('cuda')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


@dataclass
class DBDatasetDefaults:
    database: str
    target_table: str
    target_column: str
    task: TaskType

    @property
    def target(self) -> Tuple[str, str]:
        return self.target_table, self.target_column


AggrType = Literal["sum", "mean", "min", "max", "cat"]


@dataclass
class ModelConfig:
    dim: int = 64
    attn: Literal["encoder", "attn"] = "attn"
    aggr: AggrType = "sum"
    gnn_sub_layers: int = 1
    attn_sub_layers: int = 1
    gnn_layers: List[int] = field(default_factory=lambda: [])
    mlp_layers: List[int] = field(default_factory=lambda: [])
    batch_norm: bool = False
    layer_norm: bool = False


class TheLightningModel(L.LightningModule):
    def __init__(self, model, target_table: str, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.target_table = target_table
        self.loss_module = torch.nn.CrossEntropyLoss()

        self._best_train_loss = float("inf")
        self._best_train_acc = 0.0
        self._best_val_acc = 0.0
        self._best_test_acc = 0.0

    def forward(self, data: HeteroData, mode: Literal["train", "test", "val"]):
        target_tbl = data[self.target_table]

        out = self.model(data.x_dict, data.edge_index_dict)

        if mode == "train":
            mask = target_tbl.train_mask
        elif mode == "val":
            mask = target_tbl.val_mask
        elif mode == "test":
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


def create_data(connection_url: str, defaults: DBDatasetDefaults, device=None):

    with DBDataset.create_connection(
        f"{connection_url}/{defaults.database}",
    ) as conn:

        schema_analyzer = SchemaAnalyzer(
            conn,
            target=defaults.target,
            target_type=(
                "categorical" if defaults.task == TaskType.CLASSIFICATION else "numeric"
            ),
            verbose=True,
        )

        schema = schema_analyzer.guess_schema()

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
        data, column_defs, colnames = builder.build(with_column_names=True)

        n_total = data[defaults.target_table].x.shape[0]
        data = T.RandomNodeSplit("train_rest", num_val=int(0.2 * n_total), num_test=0)(data)

        return data, data_pd, schema, column_defs, colnames


def create_model(
    data: HeteroData,
    schema: Schema,
    column_defs: Dict[str, List[ColumnDef]],
    colnames: Dict[str, List[str]],
    defaults: DBDatasetDefaults,
    model_config: Optional[ModelConfig] = None,
    device=None,
):
    from db_transformer.nn.db_gnn import DBGNN
    from db_transformer.nn.transformer import DBTransformer

    if model_config is None:
        model_config = ModelConfig()

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
        model_config.dim,
        out_dim,
        model_config.dim,
        len(model_config.gnn_layers),
        data.metadata(),
        1,
        schema,
        column_defs=column_defs,
        column_names=colnames,
        config=model_config,
        target_table=defaults.target_table,
    )

    return model


class TimerOrEpochsCallback(L_callbacks.Callback):
    def __init__(
        self, epochs: int, min_train_time_s: float, epochs_multiplier: int = 10
    ) -> None:
        self.timer = Timer(cuda=False, unit="s")
        self.min_train_time_s = min_train_time_s
        self.epochs = epochs
        self.epochs_multiplier = epochs_multiplier

    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.timer.start()

    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
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
    connection_url: str,
    defaults: DBDatasetDefaults,
    model_config: Optional[ModelConfig] = None,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    min_train_time_s: float = 60.0,
    cuda: bool = False,
):

    if model_config is None:
        model_config = ModelConfig()

    device = "cuda" if cuda else "cpu"

    data, data_pd, schema, column_defs, colnames = create_data(
        connection_url, defaults, device
    )
    model = create_model(
        data, schema, column_defs, colnames, defaults, model_config, device
    )

    lightning_model = TheLightningModel(model, defaults.target_table, lr=learning_rate)

    trainer = L.Trainer(
        accelerator="gpu" if cuda else "cpu",
        devices=1,
        deterministic=True,
        callbacks=[
            L_callbacks.Timer(),
            L_callbacks.ModelCheckpoint(
                "./torch-models/",
                filename=defaults.database + "-{epoch}-{train_acc:.3f}-{val_acc:.3f}",
                mode="max",
                monitor="val_acc",
            ),
            TimerOrEpochsCallback(epochs=epochs, min_train_time_s=min_train_time_s),
        ],
        min_epochs=epochs,
        max_epochs=-1,
        max_steps=-1,
    )

    dataloader = DataLoader([data], batch_size=1)

    trainer.fit(lightning_model, dataloader, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser(add_option_string_dash_variants=DashVariant.DASH)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr", "-r", type=float, default=0.0001)
    parser.add_argument("--min-train-time", "-t", type=float, default=60.0)
    parser.add_arguments(ModelConfig, dest="model_config")
    args = parser.parse_args()
    model_config: ModelConfig = args.model_config
    cuda: bool = False
    epochs: int = args.epochs
    learning_rate: float = args.learning_rate
    min_train_time_s: float = args.min_train_time

    connection_url = "mariadb+mysqlconnector://guest:ctu-relational@78.128.250.186:3306"

    defaults = DBDatasetDefaults(
        database="CORA",
        target_table="paper",
        target_column="class_label",
        task=TaskType.CLASSIFICATION,
    )

    main(
        connection_url,
        defaults,
        model_config,
        epochs,
        learning_rate,
        min_train_time_s,
        cuda,
    )
