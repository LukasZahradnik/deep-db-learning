from abc import ABC, abstractmethod
from typing import Generic, Protocol, Sequence, TypeVar

import pandas as pd
from db_transformer.schema.columns import ColumnDef

_TColumnDef = TypeVar('_TColumnDef', bound=ColumnDef)

__ALL__ = [
    'SeriesConverter',
    'BaseSeriesConverter',
]


class SeriesConverter(Protocol):
    @abstractmethod
    def __call__(self, column_def: ColumnDef, column: pd.Series) -> Sequence[pd.Series]:
        ...


class BaseSeriesConverter(Generic[_TColumnDef], ABC):
    @abstractmethod
    def __call__(self, column_def: _TColumnDef, column: pd.Series) -> Sequence[pd.Series]:
        pass
