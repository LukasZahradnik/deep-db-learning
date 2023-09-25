from typing import List, Sequence, Tuple

import pandas as pd
from pandas.core.frame import itertools

from db_transformer.schema.schema import ColumnDef

from .series_converter import SeriesConverter

__ALL__ = ['ConverterList']


class ConverterList(SeriesConverter):
    def __init__(self, *converters: SeriesConverter) -> None:
        self.converters = converters

    def __call__(self, column_def: ColumnDef, column: pd.Series) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        out: List[pd.Series] = []
        out_column_defs: List[ColumnDef] = []

        seen_names = set()

        for c in self.converters:
            series, column_defs = c(column_def, column)

            out_column_defs.extend(column_defs)

            for serie in series:
                the_name = str(serie.name)

                if the_name in seen_names:
                    for i in itertools.count():
                        the_name2 = the_name + str(i)
                        if the_name2 not in seen_names:
                            seen_names.add(the_name2)
                            serie.name = the_name2
                            break
                else:
                    seen_names.add(the_name)
                out.append(serie)

        return out, out_column_defs
