from collections.abc import Mapping
from typing import Any, Dict, List, Set, Type, TypeVar, Union
from attrs import define, field
import warnings
import inspect

from torch_geometric.data.dataset import Tuple

from db_transformer.helpers.objectpickle import (
    SimpleSerializer,
    TypedDeserializer,
    TypedSerializer,
    deserialize,
)
from db_transformer.helpers.collections import OrderedDotDict


__all__ = [
    "ColumnDef",
    "named_column_def",
    "column_def_to_name",
    "ColumnDefSerializer",
    "ColumnDefDeserializer",
    "ColumnDefs",
    "ForeignKeyDef",
    "TableSchema",
    "Schema",
]

_KNOWN_COLUMN_TYPES: Dict[str, Type] = {}


_T = TypeVar("_T")


class ColumnDef:
    """Column definition base class."""

    pass


def named_column_def(name: str):
    """
    A decorator that gives a known name to a ColumnDef class.
    """

    def class_wrapper(cls: _T) -> _T:
        if name in _KNOWN_COLUMN_TYPES and cls != _KNOWN_COLUMN_TYPES[name]:
            warnings.warn(
                f"Redefining the underlying class for column definition named '{name}' with a different class. "
                "This may cause problems when serializing/deserializing column names."
            )

        _KNOWN_COLUMN_TYPES[name] = cls

        @classmethod
        def get_column_type_name(cls) -> Union[str, Type]:
            return name

        # adds a class method that returns the name
        setattr(cls, "get_column_type_name", get_column_type_name)

        return cls

    return class_wrapper


def column_def_to_name(column_def: Union[object, Type[object]]) -> str:
    if callable(getattr(column_def, "get_column_type_name", None)):
        return column_def.get_column_type_name()

    if getattr(column_def, "__qualname__", None) is not None:
        cls = column_def
    else:
        cls = column_def.__class__
    return f"({cls.__module__}, {cls.__qualname__})"


class ColumnDefSerializer(TypedSerializer):
    """
    A serializer that serializes an arbitrary ColumnDef, but makes the serialization output prettier.
    """

    def __init__(self):
        super().__init__(
            # not self - delegate to default behavior for children
            delegate_serializer=SimpleSerializer(child_serializer=TypedSerializer()),
            type_key="type",
        )

    def _get_type(self, cls: Type) -> Any:
        if callable(getattr(cls, "get_column_type_name", None)):
            type = cls.get_column_type_name()

            if not isinstance(type, str):
                raise TypeError(
                    f"get_column_type_name() must return a string. (Class {cls})"
                )

            return type

        return super()._get_type(cls)


class ColumnDefDeserializer(TypedDeserializer):
    """
    A deserializer for ColumnDef that can deserialize a dictionary created using ColumnDefSerializer.
    """

    def __init__(self):
        super().__init__(
            child_deserializer=TypedDeserializer(),  # not self - delegate to default behavior for children
            type_key="type",
        )

    def _get_class(self, type: Any) -> Type:
        if isinstance(type, str):
            if type not in _KNOWN_COLUMN_TYPES:
                raise ValueError(f"Unknown ColumnDef type {type}")

            return _KNOWN_COLUMN_TYPES[type]
        return super()._get_class(type)


class ColumnDefs(OrderedDotDict[ColumnDef]):
    """
    Represents the column definitions of one table.
    It is basically a dictionary of column_name -> `ColumnDef`.
    """

    SERIALIZER = ColumnDefSerializer()
    DESERIALIZER = ColumnDefDeserializer()

    def __setitem__(self, key: str, value: ColumnDef):
        # automatically cast the value
        return super().__setitem__(
            key, self.DESERIALIZER(value) if isinstance(value, dict) else value
        )

    def __getstate__(self) -> object:
        # let's do the serialization of the type ourselves to make it a little nicer

        return {k: self.SERIALIZER(v) for k, v in self.items()}

    def __setstate__(self, state: dict):
        for k, v in state.items():
            self[k] = self.DESERIALIZER(v)

    def is_in_primary_key(self, column_name: str) -> bool:
        col = self[column_name]

        if hasattr(col, "key"):
            return bool(col.key)
        if isinstance(col, Mapping) and "key" in col:
            return bool(col["key"])

        return False

    def get_primary_key(self) -> Set[str]:
        return set((col_name for col_name in self if self.is_in_primary_key(col_name)))


@define()
class ForeignKeyDef:
    """
    Represents one foreign key.
    """

    columns: List[str] = field(converter=list)
    """
    The referencing columns (in this table)
    """

    ref_table: str
    """
    The referenced table name
    """

    ref_columns: List[str] = field(converter=list)
    """
    The referenced columns (in the referenced table)
    """


@define()
class TableSchema:
    """
    Represents the schema of a single DB table.
    Holds information about the table columns as well as foreign keys.
    """

    columns: ColumnDefs = field(converter=ColumnDefs)
    foreign_keys: List[ForeignKeyDef] = field(
        converter=lambda vs: [
            deserialize(v, ForeignKeyDef) if isinstance(v, dict) else v for v in vs
        ],
        repr=lambda fks: (
            "[\n" + ",\n".join(["    " + str(fk) for fk in fks]) + "\n]" if fks else "[]"
        ),
    )

    def get_primary_key(self) -> Set[str]:
        """
        Retrieve the primary key
        """
        return self.columns.get_primary_key()


class Schema(OrderedDotDict[TableSchema]):
    """
    Represents the schema of the whole database.
    It is basically a dictionary of table_name -> `TableSchema`.
    """

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(
            key, deserialize(value, TableSchema) if isinstance(value, dict) else value
        )

    def __getstate__(self) -> object:
        # no need to serialize the internal state with extra type info
        # because we know that all types are `TableSchema` :)
        simple_serializer = SimpleSerializer()
        return {k: simple_serializer(v) for k, v in self.items()}

    def __setstate__(self, state: dict):
        # deserialize each as TableSchema
        for k, v in state.items():
            self[k] = deserialize(v, type=TableSchema)
