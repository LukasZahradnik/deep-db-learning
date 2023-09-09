from typing import Any, Optional
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

        self.linear: torch.nn.Linear

    def create(self, column_def: Any, device: Optional[str] = None):
        self.linear = torch.nn.Linear(1, self.dim, device=device)

    def forward(self, value) -> torch.Tensor:
        if len(value.shape) == 1:
            value = value.unsqueeze(dim=1)

        return self.linear(value)

