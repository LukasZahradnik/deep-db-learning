from abc import ABC, abstractmethod
import torch

from db_transformer.schema.columns import ColumnDef
from db_transformer.schema.schema import Schema


__all__ = [
    "SchemaConvertor",
]


class SchemaConvertor(ABC):
    @abstractmethod
    def create(self, schema: Schema):
        pass

    @abstractmethod
    def has(self, table_name: str, column_name: str, column: ColumnDef) -> bool:
        pass

    @abstractmethod
    def forward(
        self, value, table_name: str, column_name: str, column: ColumnDef
    ) -> torch.Tensor:
        pass

    def __call__(
        self, value, table_name: str, column_name: str, column: ColumnDef
    ) -> torch.Tensor:
        return self.forward(value, table_name, column_name, column)
