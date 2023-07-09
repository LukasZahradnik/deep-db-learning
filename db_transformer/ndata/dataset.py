import os.path
from collections import defaultdict
from typing import Union, List, Tuple, Optional

import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.data import BaseData

from db_transformer.helpers.database import copy_database, get_table_len
from db_transformer.ndata.convertor.cat_convertor import CatConvertor
from db_transformer.ndata.convertor.num_convertor import NumConvertor
from db_transformer.ndata.strategy.strategy import BaseStrategy
from db_transformer.schema import Schema, NumericColumnDef, CategoricalColumnDef, KeyColumnDef, ForeignKeyColumnDef
from sqlalchemy import create_engine
from db_transformer.db import SchemaAnalyzer


class BaseDBDataset(Dataset):
    def __init__(
        self,
        database: str,
        target_table: str,
        sql_connection_string: str,
        root: str,
        strategy: BaseStrategy,
        schema: Optional[Schema] = None,
    ):
        self.schema = schema
        self.database = database
        self.target_table = target_table
        self.sql_connection_string = sql_connection_string

        self.processed_data = []
        self.length = 0
        self.strategy = strategy

        # TODO: Temporary
        self.convertors = {
            NumericColumnDef: NumConvertor(32),
            CategoricalColumnDef: CatConvertor(32),
        }

        # TODO: Save the schema in the processed_file_names (offline support)
        # TODO: Reconsider guessing the schema here - we have to share it somehow with the model itself
        if self.schema is None:
            engine = create_engine(self.sql_connection_string)

            with engine.connect() as connection:
                self.schema = SchemaAnalyzer(connection).guess_schema()
        super().__init__(root)

    def len(self) -> int:
        return self.length

    def download(self):
        copy_database(self.sql_connection_string, self.db_connection_string, self.schema.keys())

    def process(self):
        engine = create_engine(self.sql_connection_string)

        with engine.connect() as connection:
            self.length = get_table_len(self.target_table, connection)

    @property
    def db_connection_string(self) -> str:
        db_file = f"{self.database}.db"
        return f"sqlite:///{os.path.join(self.raw_dir, db_file)}"

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.database}.db"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.database}.{self.target_table}.meta"]

    def get(self, idx: int) -> BaseData:
        engine = create_engine(self.sql_connection_string)
        with engine.connect() as connection:
            data = self.strategy.get_db_data(idx, connection, self.target_table, self.schema)
        return self._to_hetero_data(data)

    def _to_hetero_data(self, data) -> HeteroData:
        hetero_data = HeteroData()
        primary_keys = defaultdict(dict)

        for table_name, table_data in data.items():
            table_tensor_data = []
            process_cols = []

            for i, (col_name, col) in enumerate(self.schema[table_name].columns.items()):
                if type(col) in self.convertors:
                    process_cols.append((i, col_name, col, self.convertors[type(col)]))
                    self.convertors[type(col)].create(table_name, col_name, col)

            primary_col = [
                i for i, col in enumerate(self.schema[table_name].columns.values()) if self.schema[table_name].columns.is_in_primary_key(col)
            ]
            table_primary_keys = {}

            for index, row in enumerate(table_data):
                row_tensor_data = [
                    convertor(row[i], table_name, col_name, col) for i, col_name, col, convertor in process_cols
                ]

                if not row_tensor_data:
                    break

                if primary_col:
                    table_primary_keys[row[primary_col[0]]] = index
                table_tensor_data.append(torch.cat(row_tensor_data))
            else:
                if table_primary_keys:
                    primary_keys[table_name] = table_primary_keys
                hetero_data[table_name].x = torch.stack(table_tensor_data)

        for table_name, table_data in data.items():
            foreign_cols = {
                col_name: i
                for i, (col_name, col) in enumerate(self.schema[table_name].columns.items())
                if isinstance(col, ForeignKeyColumnDef)
            }

            foreign_cols = [
                (foreign_cols[foreigns.columns[0]], foreigns) for foreigns in self.schema[table_name].foreign_keys
            ]

            if not foreign_cols:
                continue

            for index, col in foreign_cols:
                edge_index = []
                table_primary_keys = primary_keys[col.ref_table]

                for i_index, row in enumerate(table_data):
                    if row[index] is None:
                        continue

                    edge_index.append([i_index, table_primary_keys[row[index]]])
                hetero_data[table_name, col.columns[0], col.ref_table].edge_index = (
                    torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                )

        return hetero_data
