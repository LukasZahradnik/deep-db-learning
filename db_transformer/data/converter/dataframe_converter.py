from typing import Literal, Optional, Protocol, overload

import pandas as pd
from db_transformer.schema.schema import TableSchema


__ALL__ = ['DataFrameConverter']


class DataFrameConverter(Protocol):
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
        ...
