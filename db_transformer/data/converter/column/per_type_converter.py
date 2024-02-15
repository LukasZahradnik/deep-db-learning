from typing import Sequence, Tuple

import pandas as pd

from db_transformer.data.converter.column.series_converter import SeriesConverter
from db_transformer.data.utils.column_def_matching import (
    ColumnDefMatcherLike,
    find_value_for_matcher,
    get_matcher,
)
from db_transformer.schema.schema import ColumnDef

__ALL__ = ["PerTypeSeriesConverter"]


class PerTypeSeriesConverter(SeriesConverter[ColumnDef]):
    def __init__(self, *converters: Tuple[ColumnDefMatcherLike, SeriesConverter]) -> None:
        self.converters = [(get_matcher(k), v) for k, v in converters]

    def __call__(
        self, column_def: ColumnDef, column: pd.Series
    ) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        converter = find_value_for_matcher(self.converters, column_def)

        if converter is None:
            return (), ()

        try:
            series, this_column_defs = converter(column_def, column)
        except Exception as e:
            raise RuntimeError(f"Failed to convert {column_def} using {converter}") from e

        if len(series) != len(this_column_defs):
            raise ValueError(
                f"{converter} returned {len(series)} pd.Series objects, "
                f"but {len(this_column_defs)} column definition objects "
                f"for column def {column_def}."
            )

        return series, this_column_defs
