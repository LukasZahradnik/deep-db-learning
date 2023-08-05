from typing import Any
import torch

from db_transformer.schema.columns import NumericColumnDef

from .column_convertor import ColumnConvertor


__all__ = [
    'NumConvertor',
]


class NumConvertor(ColumnConvertor[NumericColumnDef]):
    def create(self, column_def: Any):
        pass

    def forward(self, value) -> torch.Tensor:
        if value is None:
            value = 0  # TODO how to handle None?

        return torch.tensor([float(value)])
