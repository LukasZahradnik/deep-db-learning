from typing import Dict, Optional, Tuple

import torch

import lightning as L

from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from db_transformer.data import TaskType


class LightningWrapper(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        target_table: str,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.999),
        loss_module: Optional[torch.nn.Module] = None,
        metrics: Optional[Dict[str, torch.nn.Module]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        data_key: str = "tf",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_table = target_table
        self.lr = lr
        self.betas = betas
        self.task_type = task_type
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
                metrics["acc"] = (
                    lambda out, target: (out.argmax(dim=-1) == target)
                    .type(torch.float)
                    .mean()
                )

        self.loss_module = loss_module
        self.metrics = metrics

    def forward(self, data: HeteroData, mode: str = "train"):
        out = self.model(
            data.collect(self.data_key), data.collect("edge_index", allow_empty=True)
        )

        target = data[self.target_table].y

        loss = self.loss_module(out, target)

        batch_size = target.shape[0]
        self.log(
            f"{mode}_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=self.verbose,
        )

        metric_dict = {
            f"{mode}_{name}": metric(out, target) for name, metric in self.metrics.items()
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def training_step(self, batch):
        loss = self.forward(batch, "train")
        return loss

    def validation_step(self, batch):
        self.forward(batch, "val")

    def test_step(self, batch):
        self.forward(batch, "test")
