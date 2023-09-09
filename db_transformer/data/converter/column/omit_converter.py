from typing import Sequence

import pandas as pd
from db_transformer.schema.columns import ColumnDef

from .series_converter import SeriesConverter

__ALL__ = ['OmitConverter']


class OmitConverter(SeriesConverter):
    def __call__(self, column_def: ColumnDef, column: pd.Series) -> Sequence[pd.Series]:
        return ()
