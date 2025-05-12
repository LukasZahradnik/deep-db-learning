from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F

import lightning as L

from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from torchmetrics import (
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    NormalizedRootMeanSquaredError,
    R2Score,
    AUROC,
)

from db_transformer.data import TaskType


class LightningWrapper(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        target_table: str = None,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.999),
        loss_module: Optional[torch.nn.Module] = None,
        metrics: Optional[Dict[str, torch.nn.Module]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        num_classes: Optional[int] = None,
        data_key: str = "tf",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_table = target_table
        self.lr = lr
        self.betas = betas
        self.task_type = task_type
        self.num_classes = num_classes
        self.data_key = data_key
        self.verbose = verbose

        if loss_module is None:
            if task_type == TaskType.CLASSIFICATION:
                loss_module = torch.nn.CrossEntropyLoss(reduction="mean")
            else:
                loss_module = torch.nn.MSELoss(reduction="mean")

        if metrics is None:
            metrics = {}
            if task_type == TaskType.CLASSIFICATION:
                assert (
                    num_classes is not None
                ), "num_classes is required to initialize metrics for classification task"
                task = "binary" if num_classes == 2 else "multiclass"
                metrics["acc"] = Accuracy(
                    task=task, num_classes=num_classes, average="micro"
                )
                metrics["f1_micro"] = F1Score(
                    task=task, num_classes=num_classes, average="micro"
                )
                metrics["f1_macro"] = F1Score(
                    task=task, num_classes=num_classes, average="macro"
                )
                metrics["auroc"] = AUROC(
                    task=task, num_classes=num_classes, average="macro"
                )

            if task_type == TaskType.REGRESSION:
                metrics["mae"] = MeanAbsoluteError()
                metrics["mse"] = MeanSquaredError()
                metrics["msle"] = MeanSquaredLogError()
                metrics["nrmse"] = NormalizedRootMeanSquaredError(normalization="range")

        self.loss_module = loss_module
        self.metrics = metrics

    def forward(self, data: HeteroData, mode: str = "train"):
        if self.target_table is not None:
            out: torch.Tensor = self.model(
                data.collect(self.data_key), data.collect("edge_index", allow_empty=True)
            )
            target: torch.Tensor = data[self.target_table].y

        else:
            out: torch.Tensor = self.model(data[0])
            target: torch.Tensor = data[1]

        out = out.squeeze(dim=-1)

        loss = self.loss_module(out, target)

        batch_size = target.shape[0]
        with torch.no_grad():

            self.log(
                f"{mode}_loss",
                loss,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=self.verbose,
            )

            if self.task_type == TaskType.CLASSIFICATION and self.num_classes == 2:
                out = out.argmax(dim=1)

            metric_dict = {
                f"{mode}_{name}": metric(out, target)
                for name, metric in self.metrics.items()
            }
            self.log_dict(
                metric_dict,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=self.verbose,
            )

        return loss

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        # self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.opt,
        #     mode="min",
        #     factor=0.5,
        #     patience=2,
        #     cooldown=2,
        #     min_lr=1e-5,
        # )
        return {
            "optimizer": self.opt,
            # "lr_scheduler": {
            #     "scheduler": self.reduce_lr_on_plateau,
            #     "monitor": "val_loss",
            # },
        }

    def training_step(self, batch):
        loss = self.forward(batch, "train")
        return loss

    def validation_step(self, batch):
        self.forward(batch, "val")

    def test_step(self, batch):
        self.forward(batch, "test")
