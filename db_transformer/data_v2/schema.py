from enum import Enum
from typing import Any, Dict, Literal, Optional, Type, Union
from attrs import asdict, define, field
from collections.abc import Iterable, Mapping

import attrs

from db_transformer.data_v2.types import DotDict, TypeCheckedDotDict


# Note: `attrs` are used instead of Python's dataclasses because they support additional functionality,
# specifically kw_only, which is in Python's dataclasses only since Python 3.10.
# Other than that, the goal was to be user-friendly but also not too dependent on too many external pkgs.

def to_obj(value) -> Any:
    # TODO: documentation

    if isinstance(value, object) and callable(getattr(value, "to_obj", None)):
        return value.to_obj()  # type: ignore
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, Mapping):
        return {k: to_obj(value[k]) for k in value}
    elif isinstance(value, object) and attrs.has(value.__class__):
        return {k: to_obj(v) for k, v in asdict(value).items()}
    elif not isinstance(value, str) and ((isinstance(value, object) and getattr(value, "__iter__", None) is not None) or (isinstance(value, Iterable))):
        return [to_obj(v) for v in value]  # type: ignore
    else:
        return value


@define(kw_only=True)
class SchemaBase:
    """
    Base class for the schema model. Subclass for classes such as `ColumnDef`, `TableSchema` or `Schema`.
    """

    @classmethod
    def from_obj(cls, d: Union['SchemaBase', Mapping[str, Any]]):
        """
        Converts a `Mapping` (e.g. a `dict`) to an instance of self (e.g. a schema). 
        Must perform type-conversion hierarchically.
        """
        if isinstance(d, cls):
            return d

        if not isinstance(d, Mapping):
            raise ValueError(f"{cls.__name__}.from_obj() must be initialized with a mapping. Received: {d}")
        return cls(**d)


class ColumnType(str, Enum):
    """
    Database column type enum. Determines how the column will be interpreted by the machine learning pipeline.
    """

    FOREIGN_KEY = 'foreign_key'
    """Foreign key type column."""

    CATEGORICAL = 'cat'
    """Column containing data that should be interpreted as categorical (i.e. "enum-like"),
    irrespective of whether it is a string, an integer, etc."""

    NUMERIC = 'num'
    """Column containing numeric data that should be interpreted strictly as-is 
    (as opposed to representing categorical data)."""


@define(kw_only=True)
class ColumnDef(SchemaBase):
    """
    Base column definition class. 
    Denotes the type of the column as well as any other properties of the column specific to its type.

    Should not be instantiated directly. Instead, you should use subclasses of `ColumnDef`, as a separate sub-class
    is defined for each column type, containing type-specific values.
    """

    type: ColumnType
    """The type of the column."""

    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    """Whether the column is part of the table's primary key"""

    def __init__(self, **kwargs):
        """
        Do not instantiate directly! Use either `ColumnDef.from_obj()`, or type-specific subclasses of `ColumnDef`.
        """

        if self.__class__ == ColumnDef:
            raise TypeError("Do not instantiate ColumnDef directly. Instead, use the specific subclass or from_obj().")

        self.__attrs_init__(**kwargs)  # type: ignore

    @classmethod
    def from_obj(cls, d: Union['ColumnDef', Mapping[str, Any]]) -> 'ColumnDef':
        """
        Converts a column definition stored as a `Mapping` (e.g. a `dict`) to an instance of `ColumnDef`. 
        Ensures that the data is converted to an appropriate `ColumnDef` sub-class.
        """

        if isinstance(d, ColumnDef):
            return d

        if 'type' not in d:
            raise ValueError('A column definition must have a type')

        type = ColumnType(d['type'])

        # pass directly to the constructor, as each subclass is also an `attrs` dataclass.
        return _COLUMN_TYPE_DEF_CLASS[type](**d)

    def __attrs_post_init__(self):
        self.type = ColumnType(self.type)
        if _COLUMN_TYPE_DEF_CLASS[self.type] != self.__class__:
            raise ValueError(f"Trying to initialize ColumnDef of type '{self.type}' as {self.__class__.__name__}. "
                             f"ColumnDef of type {self.type} must use class {_COLUMN_TYPE_DEF_CLASS[self.type].__name__}.")


@define(kw_only=True)
class ForeignKeyColumnDef(ColumnDef):
    """Foreign key column definition - will be interpreted as a foreign key column by the machine learning pipeline."""

    type: Literal[ColumnType.FOREIGN_KEY] = field(default=ColumnType.FOREIGN_KEY, repr=False)
    table: str = field(validator=attrs.validators.instance_of(str))
    # TODO: documentation

    column: str = field(validator=attrs.validators.instance_of(str))
    # TODO: documentation


@define(kw_only=True)
class CategoricalColumnDef(ColumnDef):
    """Categorical column definition - containing data that should be interpreted as categorical (i.e. "enum-like"),
    irrespective of whether it is a string, an integer, etc."""

    type: Literal[ColumnType.CATEGORICAL] = field(default=ColumnType.CATEGORICAL, repr=False)
    card: int = field(validator=attrs.validators.instance_of(int))
    # TODO: documentation


@define(kw_only=True)
class NumericColumnDef(ColumnDef):
    """Numeric column definition - containing numeric data that should be interpreted strictly as-is 
    (as opposed to representing categorical data)."""

    type: Literal[ColumnType.NUMERIC] = field(default=ColumnType.NUMERIC, repr=False)


_COLUMN_TYPE_DEF_CLASS: Dict[ColumnType, Type[ColumnDef]] = {
    ColumnType.FOREIGN_KEY: ForeignKeyColumnDef,
    ColumnType.CATEGORICAL: CategoricalColumnDef,
    ColumnType.NUMERIC: NumericColumnDef,
}


class TableSchema(TypeCheckedDotDict[ColumnDef], SchemaBase):
    # TODO: documentation

    def __init__(self, __items: Optional[Union['DotDict[Any]', Dict[str, Any]]] = None, /, **kwargs: ColumnDef):
        super().__init__(ColumnDef.from_obj, __items, **kwargs)


class Schema(TypeCheckedDotDict[TableSchema], SchemaBase):
    # TODO: documentation

    def __init__(self, __items: Optional[Union['DotDict[Any]', Dict[str, Any]]] = None, /, **kwargs: TableSchema):
        super().__init__(TableSchema.from_obj, __items, **kwargs)
