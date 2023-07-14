from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch

from db_transformer.schema.columns import ColumnDef

_TColumnDef = TypeVar('_TColumnDef', bound=ColumnDef)

__all__ = [
    'ColumnConvertor',
]


class ColumnConvertor(torch.nn.Module, Generic[_TColumnDef], ABC):
    @abstractmethod
    def create(self, column_def: _TColumnDef):
        pass

    @abstractmethod
    def forward(self, value) -> torch.Tensor:
        pass
