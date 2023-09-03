from typing import Callable, Dict, Literal, Sequence, Tuple, Union

import pandas as pd
import pandas.core.indexes.accessors
from .series_converter import SeriesConverter
from db_transformer.schema.schema import ColumnDef


__ALL__ = ['PandasConverter']


class PandasConverter(SeriesConverter):
    def __init__(self, *segments: Tuple[str, Callable[[pd.Series], pd.Series]], skip_if_allsame=True) -> None:
        self.segments = segments
        self.skip_if_allsame = skip_if_allsame

    @classmethod
    def single(cls, func: Callable[[pd.Series], pd.Series], skip_if_allsame=True) -> SeriesConverter:
        return PandasConverter(('', func), skip_if_allsame=skip_if_allsame)

    def __call__(self, column_def: ColumnDef, column: pd.Series) -> Sequence[pd.Series]:
        out = []

        for segment_suffix, segment in self.segments:
            new_series = segment(column)

            # skip if all values that weren't na in the original are all the same
            if self.skip_if_allsame and new_series[column.notna()].nunique() <= 1:
                continue

            # add suffix to name
            orig_name = new_series.name
            assert isinstance(orig_name, str)
            new_series.name = orig_name + segment_suffix

            # add to list of all
            out.append(new_series)

        return out
