from collections import OrderedDict
import torch.nn.functional as F
import torch
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
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
                 target_one_hot: bool = False,
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
        self.target_one_hot = target_one_hot

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

    def table_to_pandas(self, table_name: str) -> pd.DataFrame:
        if table_name not in self.schema.keys():
            raise ValueError(f"Invalid table name (based on schema): {table_name}")

        df = pd.read_sql(
            con=self.db_inspector.connection,
            sql=table_name
        )

        return df

    def fk_to_index(self,
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

        return torch.from_numpy(out[['__pandas_index_x', '__pandas_index_y']].to_numpy()).t().contiguous()

    def build(self) -> HeteroData:
        out = HeteroData()

        # get all tables
        table_dfs = {tname: self.table_to_pandas(tname) for tname in self.schema}

        # convert all foreign keys
        for table_name, table_schema in self.schema.items():
            for fk_def in table_schema.foreign_keys:
                id = table_name, '-'.join(fk_def.columns), fk_def.ref_table
                out[id].edge_index = self.fk_to_index(fk_def, table_dfs[table_name], table_dfs[fk_def.ref_table])

        target_table_name = '__target_' + self.target_table

        # convert all tables
        for table_name, df in table_dfs.items():
            self.df_converter.convert_table(table_name, df, inplace=True)

            table_names = [table_name]

            if table_name == self.target_table:
                table_names.append(target_table_name)

                # set target
                if self.target_one_hot:
                    out[target_table_name].y = F.one_hot(torch.from_numpy(df[self.target_column].to_numpy(dtype=int)))
                else:
                    out[target_table_name].y = torch.from_numpy(df[self.target_column].to_numpy())
                del df[self.target_column]

            for tn in table_names:
                if len(df.columns) == 0:
                    out[tn].x = torch.ones((len(df), 1))
                else:
                    out[tn].x = torch.from_numpy(df.to_numpy(dtype=float))

        # add reverse edges
        out = T.ToUndirected()(out)

        # duplicate edges for target table and non-target table of the same name
        for (x, name, y), d in out.edge_items():
            if x == self.target_table:
                id = target_table_name, name, y
                out[id].edge_index = torch.clone(d['edge_index'])

        return out
