from datetime import datetime
from typing import List, Dict, Literal, Optional, Any

import torch

import lightning as L


class BestMetricsLoggerCallback(L.Callback):
    def __init__(
        self,
        monitor: str = "val_acc",
        cmp: Literal["min", "max"] = "max",
        metrics: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> None:
        if metrics is None:
            # fmt:off
            metrics = [
                "train_acc", "val_acc", "test_acc", "train_err", "val_err", "test_err",
                "train_loss", "val_loss", "test_loss"
            ]
            # fmt:on

        self.monitor = monitor
        self.cmp = cmp
        self.metrics = metrics
        self.best_value: Optional[float] = None
        self.verbose = verbose

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.cmp == "min":
            trainer.callback_metrics[self.monitor] = torch.Tensor([10_000_000])
            trainer.callback_metrics[f"best_{self.monitor}"] = torch.Tensor([10_000_000])

        if self.cmp == "max":
            trainer.callback_metrics[self.monitor] = torch.Tensor([0])
            trainer.callback_metrics[f"best_{self.monitor}"] = torch.Tensor([0])

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
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
        metric_dict = {}
        for metric_name in self.metrics:
            if metric_name not in trainer.callback_metrics:
                continue
            metric_dict[f"best_{metric_name}"] = (
                trainer.callback_metrics[metric_name].detach().cpu().item()
            )

        pl_module.log_dict(metric_dict, prog_bar=self.verbose)


class MLFlowLoggerCallback(L.Callback):
    def __init__(
        self,
        run_id: str,
        mlflow_client: Any,
        ray_session: Any,
        metrics: Optional[List[str]] = None,
    ) -> None:

        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        self.Metric = Metric

        self.run_id = run_id
        self.mlflow_client: MlflowClient = mlflow_client
        self.ray_session = ray_session

        if metrics is None:
            # fmt:off
            metrics = [
                "train_acc", "best_train_acc", "val_acc", "best_val_acc", "test_acc",
                "best_test_acc", "train_loss", "best_train_loss", "val_loss", 
                "best_val_loss", "test_loss", "best_test_loss", "train_err", 
                "best_train_err", "val_err", "best_val_err", "test_err", "best_test_err",
            ]
            # fmt:on
        self.metrics = metrics

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metric_dict = {}

        for metric_name in self.metrics:
            if metric_name not in trainer.callback_metrics:
                continue
            metric_dict[metric_name] = trainer.callback_metrics[metric_name]
        self.ray_session.report(metric_dict)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
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
                self.Metric(
                    metric_name, metric_dict[metric_name], timestamp, trainer.current_epoch
                )
            )

        self.mlflow_client.log_batch(self.run_id, metrics=mlflow_metrics, synchronous=False)

        self.ray_session.report(metric_dict)
