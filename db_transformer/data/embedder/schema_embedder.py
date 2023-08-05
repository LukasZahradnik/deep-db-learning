from abc import ABC, abstractmethod
import torch

from db_transformer.schema.columns import ColumnDef
from db_transformer.schema.schema import Schema


__all__ = [
    'SchemaEmbedder',
]


class SchemaEmbedder(torch.nn.Module, ABC):
    @abstractmethod
    def create(self, schema: Schema):
        pass

    @abstractmethod
    def has(self, table_name: str, column_name: str, column: ColumnDef) -> bool:
        pass

    @abstractmethod
    def forward(self, value, table_name: str, column_name: str, column: ColumnDef) -> torch.Tensor:
        pass
