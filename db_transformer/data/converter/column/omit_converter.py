from typing import Sequence, Tuple

import pandas as pd

from db_transformer.schema.columns import ColumnDef

from .series_converter import SeriesConverter

__ALL__ = ["OmitConverter"]


class OmitConverter(SeriesConverter):
    def __call__(
        self, column_def: ColumnDef, column: pd.Series
    ) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        return (), ()
