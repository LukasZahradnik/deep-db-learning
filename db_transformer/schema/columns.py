from attrs import define, field
import attrs

from .schema import ColumnDef, named_column_def

# Note: `attrs` are used instead of Python's dataclasses because they support additional functionality,
# specifically kw_only, which is in Python's dataclasses only since Python 3.10.

__all__ = [
    "CategoricalColumnDef",
    "NumericColumnDef",
    "DateColumnDef",
    "DateTimeColumnDef",
    "DurationColumnDef",
    "TimeColumnDef",
    "TextColumnDef",
    "OmitColumnDef",
]


@define(kw_only=True)
class _AttrsColumnDef(ColumnDef):
    """
    Internal base class for all default ColumnDef types - just contains fields that
    should be available in all column definitions, namely the 'key' boolean.
    """

    key: bool = field(default=False, validator=attrs.validators.instance_of(bool))
    """Whether the column is part of the table's primary key"""


@named_column_def("cat")
@define(kw_only=True)
class CategoricalColumnDef(_AttrsColumnDef):
    """
    Categorical column definition - containing data that should be interpreted as categorical (i.e. "enum-like"),
    irrespective of whether it is a string, an integer, etc.
    """

    card: int = field(validator=attrs.validators.instance_of(int))
    """The cardinality of the categorical variable"""


@named_column_def("num")
@define(kw_only=True)
class NumericColumnDef(_AttrsColumnDef):
    """
    Numeric column definition - containing numeric data that should be interpreted strictly as-is
    (as opposed to representing categorical data).
    """

    pass


@named_column_def("date")
@define(kw_only=True)
class DateColumnDef(_AttrsColumnDef):
    pass


@named_column_def("datetime")
@define(kw_only=True)
class DateTimeColumnDef(_AttrsColumnDef):
    pass


@named_column_def("duration")
@define(kw_only=True)
class DurationColumnDef(_AttrsColumnDef):
    pass


@named_column_def("time")
@define(kw_only=True)
class TimeColumnDef(_AttrsColumnDef):
    pass


@named_column_def("text")
@define(kw_only=True)
class TextColumnDef(_AttrsColumnDef):
    pass


@named_column_def("omit")
@define(kw_only=True)
class OmitColumnDef(_AttrsColumnDef):
    """
    Column definition that marks the column as to be ignored by the machine learning pipeline.
    """

    pass
