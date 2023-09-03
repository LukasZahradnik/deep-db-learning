from typing import Dict, Literal, Optional, Tuple, Type, overload

import pandas as pd
from db_transformer.data.converter.column.series_converter import SeriesConverter
from db_transformer.data.converter.dataframe_converter import DataFrameConverter
from db_transformer.schema.schema import ColumnDef, Schema, TableSchema


__ALL__ = ['PerTypeDataFrameConverter']


class PerTypeDataFrameConverter(DataFrameConverter):
    def __init__(self, *converters: Tuple[Optional[Type[ColumnDef]], SeriesConverter], schema: Schema) -> None:
        self.converters = converters
        self.schema = schema

    @overload
    def convert_table(self, table_name: str,
                      df: pd.DataFrame, inplace: Literal[False] = False) -> pd.DataFrame:
        ...

    @overload
    def convert_table(self, table_name: str,
                      df: pd.DataFrame, inplace: Literal[True]) -> None:
        ...

    def convert_table(self, table_name: str,
                      df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
        df_out = df if inplace else df.copy()

        table_schema = self.schema[table_name]

        for column_name in list(df.columns):
            if column_name not in table_schema.columns:
                # not in schema -> remove
                del df_out[column_name]
                continue

            column_def = table_schema.columns[column_name]
            converter = self._get_converter_for(column_def)

            if converter is None:
                # no converter -> keep as-is
                continue

            if converter is not None:
                this_series_list = converter(column_def, df[column_name])
                del df_out[column_name]

                for this_series in this_series_list:
                    df_out[this_series.name] = this_series

        return None if inplace else df_out

    def _get_converter_for(self, column_def: ColumnDef) -> Optional[SeriesConverter]:
        for column_def_type, converter in self.converters:
            if column_def_type is None or isinstance(column_def, column_def_type):
                return converter

        return None
