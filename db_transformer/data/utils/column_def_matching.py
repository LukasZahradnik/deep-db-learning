import inspect
from typing import Callable, Optional, Sequence, Tuple, Type, TypeVar, Union

from db_transformer.schema.schema import ColumnDef

__ALL__ = ['ColumnDefMatcher', 'ColumnDefMatcherLike', 'get_matcher']

ColumnDefMatcher = Callable[[ColumnDef], bool]
ColumnDefMatcherLike = Union[Type[ColumnDef], None, ColumnDefMatcher]


def get_matcher(m: ColumnDefMatcherLike) -> ColumnDefMatcher:
    if m is None:
        return lambda _: True

    if inspect.isclass(m):
        # assume is a specific ColumnDef class
        return lambda o: isinstance(o, m)

    if callable(m):
        return m

    raise ValueError()


_T = TypeVar('_T')


def find_value_for_matcher(matchers: Sequence[Tuple[ColumnDefMatcher, _T]], column_def: ColumnDef) -> Optional[_T]:
    for matcher, value in matchers:
        if matcher(column_def):
            return value

    return None
