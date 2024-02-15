from abc import ABC, abstractmethod
from typing import Generic, Protocol, Sequence, Tuple, TypeVar

import pandas as pd

from db_transformer.schema.columns import ColumnDef

_TColumnDef = TypeVar("_TColumnDef", bound=ColumnDef)

__ALL__ = [
    "SeriesConverter",
]


class SeriesConverter(Generic[_TColumnDef], ABC):
    @abstractmethod
    def __call__(
        self, column_def: _TColumnDef, column: pd.Series
    ) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        pass
