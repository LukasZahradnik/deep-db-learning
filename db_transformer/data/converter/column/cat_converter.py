from typing import Sequence

import pandas as pd
from db_transformer.schema.schema import ColumnDef

from .series_converter import SeriesConverter

__ALL__ = ['CategoricalConverter']


class CategoricalConverter(SeriesConverter):
    def __call__(self, column_def: ColumnDef, column: pd.Series) -> Sequence[pd.Series]:
        distinct_vals = column.unique()

        # give None index of 0
        if None in distinct_vals:
            distinct_vals = [v for v in distinct_vals if v is not None]
            distinct_vals.insert(0, None)

        value_map = {v: i for i, v in enumerate(distinct_vals)}
        return column.map(value_map),
