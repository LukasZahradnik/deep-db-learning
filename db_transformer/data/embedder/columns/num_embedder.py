from typing import Any
import torch

from db_transformer.schema.columns import NumericColumnDef

from .column_embedder import ColumnEmbedder


__all__ = [
    'NumEmbedder',
]


class NumEmbedder(ColumnEmbedder[NumericColumnDef]):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.weights: torch.nn.Parameter

    def create(self, column_def: Any):
        self.weights = torch.nn.Parameter(torch.randn(1, self.dim))

    def forward(self, value) -> torch.Tensor:
        return self.weights * value
