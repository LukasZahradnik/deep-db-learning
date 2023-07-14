from typing import Any
import torch

from db_transformer.schema.columns import NumericColumnDef

from .column_convertor import ColumnConvertor


__all__ = [
    'NumConvertor',
]


class NumConvertor(ColumnConvertor[NumericColumnDef]):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.weights: torch.nn.Parameter
        self.num_params = torch.nn.ParameterDict()

    def create(self, column_def: Any):
        self.weights = torch.nn.Parameter(torch.randn(1, self.dim))

    def forward(self, value) -> torch.Tensor:
        if value is None:
            value = 0  # TODO how to handle None?

        return self.weights * torch.tensor([float(value)])
