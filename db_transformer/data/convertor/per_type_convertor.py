from typing import Callable, Dict, Optional, Type
import warnings
import torch
from .columns.column_convertor import ColumnConvertor
from db_transformer.schema.columns import ColumnDef
from db_transformer.schema.schema import Schema
from .schema_convertor import SchemaConvertor


__all__ = [
    "PerTypeConvertor",
]


class PerTypeConvertor(SchemaConvertor):
    def __init__(
        self, factories: Dict[Type[ColumnDef], Callable[[], Optional[ColumnConvertor]]]
    ):
        super().__init__()
        self.factories = factories
        self.column_convertors = {}

    def create(self, schema: Schema):
        for table_name, table_schema in schema.items():
            for column_name, column_def in table_schema.columns.items():
                key = table_name + "/" + column_name

                # find the correct factory
                factory = None
                for t, f in self.factories.items():
                    if isinstance(column_def, t):
                        factory = f
                        break

                if factory is None:
                    warnings.warn(f"Columns of type {column_def} are missing a convertor.")
                else:
                    try:
                        convertor = factory()
                        if convertor is not None:
                            convertor.create(column_def)
                    except Exception as e:
                        raise Exception(
                            f"Creating convertor for column {table_name}.{column_name} failed"
                        ) from e
                    if convertor is not None:
                        self.column_convertors[key] = convertor

    def has(self, table_name: str, column_name: str, column: ColumnDef) -> bool:
        key = table_name + "/" + column_name
        return key in self.column_convertors

    def forward(
        self, value, table_name: str, column_name: str, column: ColumnDef
    ) -> torch.Tensor:
        key = table_name + "/" + column_name
        assert self.has(table_name, column_name, column)
        return self.column_convertors[key].forward(value)
