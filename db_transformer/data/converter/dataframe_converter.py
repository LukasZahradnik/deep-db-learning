from typing import Dict, Literal, Optional, Protocol, Tuple, Union, overload

import pandas as pd

from db_transformer.data.converter.column.series_converter import SeriesConverter
from db_transformer.schema.schema import ColumnDef, Schema

__ALL__ = ["DataFrameConverter", "SimpleDataFrameConverter"]


class DataFrameConverter(Protocol):
    @overload
    def convert_table(
        self, table_name: str, df: pd.DataFrame, inplace: Literal[False] = False
    ) -> Tuple[pd.DataFrame, Dict[str, ColumnDef]]: ...

    @overload
    def convert_table(
        self, table_name: str, df: pd.DataFrame, inplace: Literal[True]
    ) -> Dict[str, ColumnDef]: ...

    def convert_table(
        self, table_name: str, df: pd.DataFrame, inplace: bool = False
    ) -> Union[Dict[str, ColumnDef], Tuple[pd.DataFrame, Dict[str, ColumnDef]]]: ...


class SimpleDataFrameConverter(DataFrameConverter):
    def __init__(
        self,
        series_converter: SeriesConverter,
        schema: Schema,
        target_converter: Optional[SeriesConverter] = None,
        target: Optional[Tuple[str, str]] = None,
    ) -> None:
        self.series_converter = series_converter
        self.target_converter = target_converter
        self.target = target

        if target_converter is not None and target is None:
            raise ValueError(
                "When target_converter is specified, target must be specified as well."
            )

        self.schema = schema

    @overload
    def convert_table(
        self, table_name: str, df: pd.DataFrame, inplace: Literal[False] = False
    ) -> Tuple[pd.DataFrame, Dict[str, ColumnDef]]: ...

    @overload
    def convert_table(
        self, table_name: str, df: pd.DataFrame, inplace: Literal[True]
    ) -> Dict[str, ColumnDef]: ...

    def convert_table(
        self, table_name: str, df: pd.DataFrame, inplace: bool = False
    ) -> Union[Dict[str, ColumnDef], Tuple[pd.DataFrame, Dict[str, ColumnDef]]]:
        df_out = df if inplace else df.copy()
        out_column_defs: Dict[str, ColumnDef] = {}

        table_schema = self.schema[table_name]

        for column_name in list(df.columns):
            if column_name not in table_schema.columns:
                # not in schema -> remove
                del df_out[column_name]
                continue

            column_def = table_schema.columns[column_name]

            if (table_name, column_name) == self.target:
                assert self.target_converter is not None
                converter = self.target_converter
            else:
                converter = self.series_converter

            try:
                series, this_column_defs = converter(column_def, df[column_name])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert {table_name}.{column_name} using {converter}"
                ) from e

            if len(series) != len(this_column_defs):
                raise ValueError(
                    f"{converter} returned {len(series)} pd.Series objects, "
                    f"but {len(this_column_defs)} column definition objects "
                    f"for column {column_name} and table {table_name}."
                )

            del df_out[column_name]

            for serie, column_def in zip(series, this_column_defs):
                the_out_column_name = str(serie.name)
                df_out[the_out_column_name] = serie
                out_column_defs[the_out_column_name] = column_def

        if inplace:
            return out_column_defs
        else:
            return df_out, out_column_defs
