import warnings
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, overload

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sqlalchemy import Connection
from torch_geometric.data import HeteroData

from db_transformer.data.converter import (
    CategoricalConverter,
    ConverterList,
    DataFrameConverter,
    DateConverter,
    DateTimeConverter,
    IdentityConverter,
    OmitConverter,
    PerTypeDataFrameConverter,
    TimeConverter,
    TimestampConverter,
)
from db_transformer.data.converter.column.series_converter import SeriesConverter
from db_transformer.db.db_inspector import DBInspector, DBInspectorInterface
from db_transformer.schema.columns import (
    CategoricalColumnDef,
    DateColumnDef,
    DateTimeColumnDef,
    NumericColumnDef,
    TimeColumnDef,
)
from db_transformer.schema.schema import ColumnDef, ForeignKeyDef, Schema

__ALL__ = ['HeteroDataBuilder']


class HeteroDataBuilder:
    def __init__(self,
                 inspector_or_connection: Union[DBInspectorInterface, Connection],
                 schema: Schema,
                 target_table: str,
                 target_column: str,
                 df_converter: Optional[DataFrameConverter] = None,
                 create_reverse_edges: bool = True,
                 separate_target: bool = True,
                 device=None,
                 ):
        if isinstance(inspector_or_connection, Connection):
            db_inspector = DBInspector(inspector_or_connection)
        else:
            db_inspector = inspector_or_connection

        self.db_inspector = db_inspector
        self.schema = schema

        self.target_table = target_table
        self.target_column = target_column

        if df_converter is None:
            df_converter = self.extend_default_df_converter(schema=schema)

        self.df_converter = df_converter
        self.create_reverse_edges = create_reverse_edges
        self.separate_target = separate_target
        self.device = device

    @staticmethod
    def extend_default_df_converter(*converters: Tuple[Optional[Type[ColumnDef]], SeriesConverter],
                                    schema: Schema) -> DataFrameConverter:
        col_converters: OrderedDict[Optional[Type[ColumnDef]], SeriesConverter] = OrderedDict((
            (CategoricalColumnDef, CategoricalConverter()),
            (NumericColumnDef, IdentityConverter()),
            (DateColumnDef, ConverterList(DateConverter(), TimestampConverter())),
            (DateTimeColumnDef, ConverterList(DateTimeConverter(), TimestampConverter())),
            (TimeColumnDef, ConverterList(TimeConverter(), TimestampConverter())),
            (None, OmitConverter()),
        ))

        col_converters.update(converters)

        return PerTypeDataFrameConverter(
            *col_converters.items(),
            schema=schema
        )

    def _table_to_dataframe_raw(self, table_name: str) -> pd.DataFrame:
        if table_name not in self.schema.keys():
            raise ValueError(f"Invalid table name (based on schema): {table_name}")

        df = pd.read_sql(
            con=self.db_inspector.connection,
            sql=table_name
        )

        return df

    def _fk_to_index(self,
                     fk_def: ForeignKeyDef,
                     table: pd.DataFrame,
                     ref_table: pd.DataFrame) -> torch.Tensor:
        assert isinstance(table.index, pd.RangeIndex)
        assert isinstance(ref_table.index, pd.RangeIndex)

        # keep just the index columns
        table = table[fk_def.columns].copy()
        ref_table = ref_table[fk_def.ref_columns].copy()

        table.index.name = '__pandas_index'
        ref_table.index.name = '__pandas_index'
        table.reset_index(inplace=True)
        ref_table.reset_index(inplace=True)

        out = pd.merge(left=table, right=ref_table,
                       how='inner',
                       left_on=fk_def.columns, right_on=fk_def.ref_columns)

        return (torch.from_numpy(out[['__pandas_index_x', '__pandas_index_y']].to_numpy())
                .t()
                .contiguous()
                .to(self.device))

    def _get_dataframes_raw(self) -> Dict[str, pd.DataFrame]:
        return {tname: self._table_to_dataframe_raw(tname) for tname in self.schema}

    def _pop_target_column(self, table_dfs: Dict[str, pd.DataFrame]) -> pd.Series:
        target_series: pd.Series = table_dfs[self.target_table][self.target_column]
        del table_dfs[self.target_table][self.target_column]
        return target_series

    def _convert_dataframe(self, name: str, df: pd.DataFrame) -> Dict[str, ColumnDef]:
        column_defs = self.df_converter.convert_table(name, df, inplace=True)
        return column_defs

    def _convert_dataframes(self, table_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, ColumnDef]]:
        out_column_defs = {}

        for table_name, df in table_dfs.items():
            types_this = self.df_converter.convert_table(table_name, df, inplace=True)
            out_column_defs[table_name] = types_this

        return out_column_defs

    def _convert_dataframes_and_target(self,
                                       table_dfs: Dict[str, pd.DataFrame]) -> Tuple[Tuple[pd.DataFrame, Dict[str, ColumnDef]],
                                                                                    Dict[str, Dict[str, ColumnDef]]]:
        # separate target from features
        target_df = self._pop_target_column(table_dfs).to_frame(self.target_column)

        # convert target dataframe
        target_column_defs = self._convert_dataframe(self.target_table, target_df)
        if len(target_df.columns) != 1:
            warnings.warn(f"The original target {self.target_table}.{self.target_column} "
                          f"was converted to multiple features: "
                          f"{[str(c) for c in target_df.columns]}. "
                          f"You may want to double-check whether this is correct.")

        # convert feature dataframes
        column_defs = self._convert_dataframes(table_dfs)

        return (target_df, target_column_defs), column_defs

    @overload
    def build_as_pandas(self, with_target: Literal[True]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        ...

    @overload
    def build_as_pandas(self, with_target: Literal[False] = False) -> Dict[str, pd.DataFrame]:
        ...

    def build_as_pandas(self, with_target: bool = False) -> Union[
            Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], pd.DataFrame]]:
        table_dfs = self._get_dataframes_raw()

        if with_target:
            (target_df, _), _ = self._convert_dataframes_and_target(table_dfs)
            return table_dfs, target_df
        else:
            # just remove the target
            del table_dfs[self.target_table][self.target_column]
            self._convert_dataframes(table_dfs)
            return table_dfs

    def build(self) -> Tuple[HeteroData, Dict[str, List[ColumnDef]]]:
        out = HeteroData()

        # get all tables
        table_dfs = self._get_dataframes_raw()

        # convert all foreign keys
        for table_name, table_schema in self.schema.items():
            for fk_def in table_schema.foreign_keys:
                id = table_name, '-'.join(fk_def.columns), fk_def.ref_table
                out[id].edge_index = self._fk_to_index(fk_def, table_dfs[table_name], table_dfs[fk_def.ref_table])

        (target_df, _), column_defs = self._convert_dataframes_and_target(table_dfs)

        # set HeteroData target
        out[self.target_table].y = torch.from_numpy(target_df.to_numpy(dtype=np.float32)).to(self.device)

        out_column_defs: Dict[str, List[ColumnDef]] = {}

        # set HeteroData features (with target now removed)
        for table_name, df in table_dfs.items():
            out[table_name].x = torch.from_numpy(df.to_numpy(dtype=np.float32)).to(self.device)
            out_column_defs[table_name] = [column_defs[table_name][str(col)] for col in df.columns]

        if self.create_reverse_edges:
            # add reverse edges
            out: HeteroData = T.ToUndirected()(out)

        return out, out_column_defs
