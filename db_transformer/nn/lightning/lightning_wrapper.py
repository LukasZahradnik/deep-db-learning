from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F

from torchmetrics import Accuracy

import lightning as L

from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from db_transformer.data import TaskType
from db_transformer.nn import BlueprintModel


class LightningWrapper(L.LightningModule):
    def __init__(
        self,
        model: BlueprintModel,
        target_table: str,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.999),
        loss_module: Optional[torch.nn.Module] = None,
        metrics: Optional[Dict[str, torch.nn.Module]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        num_classes: Optional[int] = None,
        data_key: str = "tf",
        verbose: bool = True,
        pretrain: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_table = target_table
        self.lr = lr
        self.betas = betas
        self.task_type = task_type
        self.data_key = data_key
        self.verbose = verbose
        self.pretrain = pretrain

        if loss_module is None:
            if task_type == TaskType.CLASSIFICATION:
                loss_module = torch.nn.CrossEntropyLoss(reduction="mean")
            else:
                loss_module = torch.nn.MSELoss(reduction="mean")

        if pretrain:
            self.pretrain_loss = torch.nn.BCELoss(reduction="mean")
            self.pretrain_metrics = {"acc": Accuracy(task="binary")}

        if metrics is None:
            metrics = {}
            if task_type == TaskType.CLASSIFICATION:
                metrics["acc"] = Accuracy(task="multiclass", num_classes=num_classes)
            if task_type == TaskType.REGRESSION:
                metrics["mae"] = torch.nn.L1Loss(reduction="mean")
                metrics["mse"] = torch.nn.MSELoss(reduction="mean")
                metrics["nrmse"] = (
                    lambda out, target: torch.sqrt(
                        F.mse_loss(out, target, reduction="mean")
                    )
                    / target.mean()
                )

        self.loss_module = loss_module
        self.metrics = metrics

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor, mode: str, pretrain: bool):
        loss_fn = self.pretrain_loss if pretrain else self.loss_module
        metrics = self.pretrain_metrics if pretrain else self.metrics

        loss = loss_fn(pred, target)

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
            f"{mode}_{name}": metric(pred, target) for name, metric in metrics.items()
        }
        self.log_dict(
            metric_dict,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=self.verbose,
        )

        return loss

    def forward_pretrain(self, data: HeteroData):
        out: Tuple[Dict[NodeType, torch.Tensor], Dict[NodeType, torch.Tensor]] = self.model(
            data.collect(self.data_key),
            data.collect("edge_index", allow_empty=True),
            pretrain=True,
        )
        pred_dict, target_dict = out

        return sum(
            [
                self.evaluate(
                    pred_dict[node].squeeze(dim=-1),
                    target_dict[node],
                    f"pretrain_{node}",
                    True,
                )
                for node in pred_dict.keys()
            ]
        )

    def forward(self, data: HeteroData, mode: str = "train"):
        out: torch.Tensor = self.model(
            data.collect(self.data_key), data.collect("edge_index", allow_empty=True), False
        )

        pred = out.squeeze(dim=-1)

        target: torch.Tensor = data[self.target_table].y

        return self.evaluate(pred, target, mode, False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def training_step(self, batch):
        if self.pretrain:
            return self.forward_pretrain(batch)
        return self.forward(batch, "train")

    def validation_step(self, batch):
        self.forward(batch, "val")

    def test_step(self, batch):
        self.forward(batch, "test")
