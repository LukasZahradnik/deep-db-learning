from typing import Sequence
from collections import OrderedDict

import pandas as pd
from pandas.core.frame import itertools
from .series_converter import SeriesConverter
from db_transformer.schema.schema import ColumnDef


__ALL__ = ['ConverterList']


class ConverterList(SeriesConverter):
    def __init__(self, *converters: SeriesConverter) -> None:
        self.converters = converters

    def __call__(self, column_def: ColumnDef, column: pd.Series) -> Sequence[pd.Series]:
        out_dict: OrderedDict[str, pd.Series] = OrderedDict()

        for c in self.converters:
            this_series = c(column_def, column)

            for new_series in this_series:
                the_name = str(new_series.name)

                if the_name in out_dict:
                    for i in itertools.count():
                        the_name2 = the_name + str(i)
                        if the_name2 not in out_dict:
                            new_series.name = the_name2
                            out_dict[the_name2] = new_series
                            break
                else:
                    out_dict[the_name] = new_series

        return list(out_dict.values())
