from typing import Dict, List, Literal, Protocol, Tuple, Union, overload

import pandas as pd

from db_transformer.schema.schema import ColumnDef

__ALL__ = ['DataFrameConverter']


class DataFrameConverter(Protocol):
    @overload
    def convert_table(self, table_name: str,
                      df: pd.DataFrame, inplace: Literal[False] = False) -> Tuple[pd.DataFrame, Dict[str, ColumnDef]]:
        ...

    @overload
    def convert_table(self, table_name: str,
                      df: pd.DataFrame, inplace: Literal[True]) -> Dict[str, ColumnDef]:
        ...

    def convert_table(self, table_name: str,
                      df: pd.DataFrame, inplace: bool = False) -> Union[Dict[str, ColumnDef],
                                                                        Tuple[pd.DataFrame, Dict[str, ColumnDef]]]:
        ...
