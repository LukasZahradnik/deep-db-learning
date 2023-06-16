from attrs import define, field
import attrs

from .schema import named_column_def

# Note: `attrs` are used instead of Python's dataclasses because they support additional functionality,
# specifically kw_only, which is in Python's dataclasses only since Python 3.10.

__all__ = [
    'ForeignKeyColumnDef',
    'CategoricalColumnDef',
    'NumericColumnDef',
    'OmitColumnDef',
]


@named_column_def('foreign_key')
@define(kw_only=True)
class ForeignKeyColumnDef:
    """
    Foreign key column definition - will be interpreted as a foreign key column by the machine learning pipeline.
    """

    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    """Whether the column is part of the table's primary key"""


@named_column_def('cat')
@define(kw_only=True)
class CategoricalColumnDef:
    """
    Categorical column definition - containing data that should be interpreted as categorical (i.e. "enum-like"),
    irrespective of whether it is a string, an integer, etc.
    """

    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    """Whether the column is part of the table's primary key"""

    card: int = field(validator=attrs.validators.instance_of(int))
    """The cardinality of the categorical variable"""


@named_column_def('num')
@define(kw_only=True)
class NumericColumnDef:
    """
    Numeric column definition - containing numeric data that should be interpreted strictly as-is 
    (as opposed to representing categorical data).
    """

    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    """Whether the column is part of the table's primary key"""


@named_column_def('omit')
@define(kw_only=True)
class OmitColumnDef:
    """
    Column definition that marks the column as to be ignored by the machine learning pipeline.
    """

    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    """Whether the column is part of the table's primary key"""

