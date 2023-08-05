from typing import Callable, Dict, Optional, Type
import warnings
import torch
from .columns.column_embedder import ColumnEmbedder
from db_transformer.schema.columns import ColumnDef
from db_transformer.schema.schema import Schema
from .schema_embedder import SchemaEmbedder


__all__ = [
    'PerTypeEmbedder',
]


class PerTypeEmbedder(SchemaEmbedder):
    def __init__(self, factories: Dict[Type[ColumnDef], Callable[[], Optional[ColumnEmbedder]]]):
        super().__init__()
        self.factories = factories
        self.column_embedders = torch.nn.ModuleDict()

    def create(self, schema: Schema):
        for table_name, table_schema in schema.items():
            for column_name, column_def in table_schema.columns.items():
                key = table_name + '/' + column_name

                # find the correct factory
                factory = None
                for t, f in self.factories.items():
                    if isinstance(column_def, t):
                        factory = f
                        break

                if factory is None:
                    warnings.warn(f"Columns of type {column_def} are missing a embedder.")
                else:
                    try:
                        embedder = factory()
                        if embedder is not None:
                            embedder.create(column_def)
                    except Exception as e:
                        raise Exception(
                            f"Creating embedder for column {table_name}.{column_name} failed") from e
                    if embedder is not None:
                        self.column_embedders[key] = embedder

    def has(self, table_name: str, column_name: str, column: ColumnDef) -> bool:
        key = table_name + '/' + column_name
        return key in self.column_embedders

    def forward(self, value, table_name: str, column_name: str, column: ColumnDef) -> torch.Tensor:
        key = table_name + '/' + column_name
        assert self.has(table_name, column_name, column)
        return self.column_embedders[key].forward(value)
