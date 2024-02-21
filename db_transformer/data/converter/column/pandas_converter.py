from typing import Callable, List, Sequence, Tuple

import pandas as pd

from db_transformer.schema.schema import ColumnDef

from .series_converter import SeriesConverter

__ALL__ = ["PandasConverter"]


class PandasConverter(SeriesConverter):
    def __init__(
        self,
        *segments: Tuple[str, Callable[[pd.Series], Tuple[pd.Series, ColumnDef]]],
        skip_if_allsame=True
    ) -> None:
        self.segments = segments
        self.skip_if_allsame = skip_if_allsame

    @classmethod
    def single(
        cls, func: Callable[[pd.Series], Tuple[pd.Series, ColumnDef]], skip_if_allsame=True
    ) -> SeriesConverter:
        return PandasConverter(("", func), skip_if_allsame=skip_if_allsame)

    def __call__(
        self, column_def: ColumnDef, column: pd.Series
    ) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        out: List[pd.Series] = []
        out_column_defs: List[ColumnDef] = []

        for segment_suffix, segment in self.segments:
            new_series, column_def = segment(column)

            # skip if all values that weren't na in the original are all the same
            # TODO remove? Is kinda confusing when you have less features than you expected to have
            if self.skip_if_allsame and new_series[column.notna()].nunique() <= 1:
                continue

            # add suffix to name
            orig_name = new_series.name
            assert isinstance(orig_name, str)
            new_series.name = orig_name + segment_suffix

            # add to list of all
            out.append(new_series)
            out_column_defs.append(column_def)

        return out, out_column_defs
