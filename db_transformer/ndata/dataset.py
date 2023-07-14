from collections import defaultdict
import os.path
from typing import List, Optional, Tuple, Union

from sqlalchemy.engine import Connection, create_engine
from sqlalchemy.engine.url import URL
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.data import BaseData

from db_transformer.db import SchemaAnalyzer
from db_transformer.db.db_inspector import DBInspector
from db_transformer.helpers.database import copy_database, get_table_len
from db_transformer.ndata.convertor.cat_convertor import CatConvertor
from db_transformer.ndata.convertor.num_convertor import NumConvertor
from db_transformer.ndata.strategy.strategy import BaseStrategy
from db_transformer.schema import (
    CategoricalColumnDef,
    ForeignKeyColumnDef,
    NumericColumnDef,
    Schema,
)


class DBDataset(Dataset):
    def __init__(
        self,
        database: str,
        target_table: str,
        connection_url: Union[str, URL],
        root: str,
        strategy: BaseStrategy,
        download: bool,
        schema: Optional[Schema] = None,
    ):
        # if the schema is None, it will be processed in the `process` method.
        self.schema = schema
        self.database = database
        self.target_table = target_table

        self.upstream_connection_url = connection_url
        self.connection: Optional[Connection] = None
        """
        The sqlalchemy `Connection` instance.
        If download=True, then it is the connection to the local database.
        If download=False, then it is the connection to the upstream database.
        """

        self._do_download = download

        self.processed_data = []
        self._length = 0
        self.strategy = strategy

        # TODO: Temporary
        self.convertors = {
            NumericColumnDef: NumConvertor(32),
            CategoricalColumnDef: CatConvertor(32),
        }

        super().__init__(root)  # initialize NOW before we guess the schema !

    @property
    def local_connection_url(self) -> Union[str, URL]:
        db_file = f"{self.database}.db"
        return f"sqlite:///{os.path.join(self.raw_dir, db_file)}"

    @property
    def connection_url(self) -> Union[str, URL]:
        if self.has_download:
            return self.local_connection_url
        else:
            return self.upstream_connection_url

    def __enter__(self):
        return self

    def close(self):
        if self.connection is not None:
            self.connection.close()

    def __exit__(self, type, value, tb):
        self.close()

    def len(self) -> int:
        return self._length

    @property
    def has_download(self):
        # this override is needed because then Dataset skips calling the `download` method
        # if this returns false
        return self._do_download

    def _create_inspector(self, connection: Connection) -> DBInspector:
        return DBInspector(connection)

    def download(self):
        if not self.has_download:
            raise RuntimeError(f"This {self.__class__.__name__} was initialized with download=False, so "
                               "the upstream database is accessed directly instead of downloading. "
                               "download() thus shouldn't be executed.")

        with Connection(create_engine(self.upstream_connection_url)) as upstream_connection:
            if self.connection is None:
                self.connection = Connection(create_engine(self.local_connection_url))

            copy_database(src_inspector=self._create_inspector(upstream_connection), dst=self.connection)

    def _create_schema_analyzer(self, connection: Connection) -> SchemaAnalyzer:
        return SchemaAnalyzer(self._create_inspector(connection))

    def process(self):
        if self.connection is None:
            self.connection = Connection(create_engine(self.connection_url))

        # TODO: Save the schema in the processed_file_names (instead of always re-running the analysis)
        # TODO: Reconsider guessing the schema here - we have to share it somehow with the model itself
        if self.schema is None:
            self.schema = self._create_schema_analyzer(self.connection).guess_schema()

        self._length = get_table_len(self.target_table, self.connection)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.database}.db"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.database}.{self.target_table}.meta"]

    def get(self, idx: int) -> BaseData:
        if self.connection is None:
            self.connection = Connection(create_engine(self.connection_url))

        assert self.schema is not None
        data = self.strategy.get_db_data(
            idx, self.connection, self.target_table, self.schema)
        return self._to_hetero_data(data)

    def _to_hetero_data(self, data) -> HeteroData:
        assert self.schema is not None
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
                i for i, col_name in enumerate(self.schema[table_name].columns.keys()) if self.schema[table_name].columns.is_in_primary_key(col_name)
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
