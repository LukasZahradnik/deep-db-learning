import os.path
import sys
from collections import defaultdict
from os import path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch_geometric.transforms as T
from sqlalchemy.engine import Connection, Engine, create_engine, make_url
from sqlalchemy.engine.url import URL
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.data import BaseData

from db_transformer.data.convertor import (
    CatConvertor,
    DateConvertor,
    DateTimeConvertor,
    DurationConvertor,
    NumConvertor,
    PerTypeConvertor,
    TimeConvertor,
)
from db_transformer.data.convertor.schema_convertor import SchemaConvertor
from db_transformer.data.strategy.strategy import BaseStrategy
from db_transformer.db import SchemaAnalyzer
from db_transformer.db.db_inspector import DBInspector
from db_transformer.helpers.database import copy_database, get_table_len
from db_transformer.schema import (
    CategoricalColumnDef,
    DateColumnDef,
    DateTimeColumnDef,
    DurationColumnDef,
    NumericColumnDef,
    OmitColumnDef,
    Schema,
    TimeColumnDef,
)


class DBDataset(Dataset):
    def __init__(
        self,
        database: str,
        target_table: str,
        target_column: str,
        connection_url: Union[str, URL],
        root: str,
        strategy: BaseStrategy,
        download: bool,
        schema: Optional[Schema] = None,
        verbose=True,
        cache_in_memory=False,
    ):
        # if the schema is None, it will be processed in the `process` method.
        self.schema = schema
        self.database = database
        self.target_table = target_table
        self.target_column = target_column

        self.upstream_connection_url = connection_url
        self.connection: Optional[Connection] = None
        """
        The sqlalchemy `Connection` instance.
        If download=True, then it is the connection to the local database.
        If download=False, then it is the connection to the upstream database.
        """

        self._do_download = download
        self._verbose = verbose

        self.processed_data = []
        self._length = 0
        self.strategy = strategy
        self.label_convertor = None
        self.ones = torch.tensor([1])  # Label placeholder

        self.cache_in_memory = cache_in_memory
        self.cache = []

        self.convertor = PerTypeConvertor(
            {
                OmitColumnDef: lambda: None,  # skip warnings
                NumericColumnDef: lambda: NumConvertor(),
                CategoricalColumnDef: lambda: CatConvertor(),
                DateColumnDef: lambda: None,
                # DateConvertor(dim, segments=["year", "month", "day"]),
                # TimeColumnDef: lambda: TimeConvertor(dim, segments=['total_seconds']),
                DateTimeColumnDef: lambda: None, 
                # DateTimeConvertor(
                    # dim, segments=["year", "month", "day", "total_seconds"]
                # ),
                # DurationColumnDef: lambda: DurationConvertor(dim),
            }
        )

        super().__init__(root)  # initialize NOW before we guess the schema !

    @classmethod
    def get_default_raw_dir(cls, root: str) -> str:
        return path.join(root, "raw")

    @classmethod
    def get_default_local_file(cls, root: str, dataset_name: str) -> str:
        return path.join(cls.get_default_raw_dir(root), f"{dataset_name}.db")

    @property
    def raw_dir(self) -> str:
        assert self.root is not None
        return self.get_default_raw_dir(self.root)

    @property
    def local_file(self) -> str:
        assert self.root is not None
        return self.get_default_local_file(root=self.root, dataset_name=self.database)

    @property
    def local_connection_url(self) -> Union[str, URL]:
        return f"sqlite:///{self.local_file}"

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

    @classmethod
    def create_connection(cls, engine_or_url: Union[str, URL, Engine], **kwargs) -> Connection:
        """Create a new SQLAlchemy Connection instance (Don't forget to close it after you are done using it!)."""
        if isinstance(engine_or_url, Engine):
            engine = engine_or_url
        else:
            engine = create_engine(engine_or_url)

        return Connection(engine)

    @classmethod
    def create_local_connection(cls, root: str, dataset_name: str) -> Connection:
        """Create a new SQLAlchemy Connection instance to the local database copy.

        Create a new SQLAlchemy Connection instance to the local database copy.
        Don't forget to close the Connection after you are done using it!
        """
        url = "sqlite:///" + cls.get_default_local_file(root, dataset_name)
        return Connection(create_engine(url))

    def download(self):
        if not self.has_download:
            raise RuntimeError(
                f"This {self.__class__.__name__} was initialized with download=False, so "
                "the upstream database is accessed directly instead of downloading. "
                "download() thus shouldn't be executed."
            )

        if self.connection is None:
            self.connection = self.create_connection(self.local_connection_url)

        try:
            with self.create_connection(self.upstream_connection_url) as upstream_connection:
                if self._verbose:
                    print("Copying database...", file=sys.stderr)

                copy_database(
                    src_inspector=self._create_inspector(upstream_connection),
                    dst=self.connection,
                    verbose=self._verbose,
                )
        except Exception as e:
            self.connection.close()

            if os.path.exists(self.local_file):
                os.remove(self.local_file)
            raise e

    def _create_schema_analyzer(self, connection: Connection) -> SchemaAnalyzer:
        return SchemaAnalyzer(self._create_inspector(connection),
                              target=(self.target_table, self.target_column),
                              verbose=self._verbose)

    def process(self):
        if self.connection is None:
            self.connection = self.create_connection(self.connection_url)

        # TODO: Save the schema in the processed_file_names (instead of always re-running the analysis)
        # TODO: Reconsider guessing the schema here - we have to share it somehow with the model itself
        if self.schema is None:
            if self._verbose:
                print("Guessing schema...", file=sys.stderr)

            self.schema = self._create_schema_analyzer(self.connection).guess_schema()

        self.convertor.create(self.schema)

        target_col = self.schema[self.target_table].columns[self.target_column]
        if isinstance(target_col, CategoricalColumnDef):
            self.label_convertor = CatConvertor()
            self.label_convertor.create(target_col)

        self._length = get_table_len(self.target_table, self.connection)

        if self.cache_in_memory:
            self.cache = [None] * self._length

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.database}.db"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.database}.{self.target_table}.meta"]

    def get_as_dict(self, idx: int) -> Tuple[Dict[str, Set[Tuple[Any, ...]]], Any]:
        if self.connection is None:
            self.connection = self.create_connection(self.connection_url)

        assert self.schema is not None
        return self.strategy.get_db_data(idx, self.connection, self.target_table, self.schema)

    def get(self, idx: int) -> BaseData:
        if self.cache_in_memory and self.cache[idx] is not None:
            return self.cache[idx]

        data, target_row = self.get_as_dict(idx)
        hetero_data = self._to_hetero_data(data, target_row)

        if self.cache_in_memory:
            self.cache[idx] = hetero_data
        return hetero_data

    def _to_hetero_data(self, data, target_row) -> HeteroData:
        assert self.schema is not None
        hetero_data = HeteroData()
        primary_keys = defaultdict(dict)

        col_to_index = {col_name: i for i, col_name in enumerate(self.schema[self.target_table].columns.keys())}
        label = target_row[col_to_index[self.target_column]]

        if self.label_convertor is not None:
            hetero_data.y = torch.argmax(self.label_convertor.to_one_hot(label))
        else:
            hetero_data.y = torch.tensor(float(label))

        for table_name, table_data in data.items():
            table_tensor_data = []
            process_cols = []

            for i, (col_name, col) in enumerate(self.schema[table_name].columns.items()):
                if self.convertor.has(table_name, col_name, col):
                    process_cols.append((i, col_name, col))

            primary_col = [
                i
                for i, col_name in enumerate(self.schema[table_name].columns.keys())
                if self.schema[table_name].columns.is_in_primary_key(col_name)
            ]
            table_primary_keys = {}

            for index, row in enumerate(table_data):
                row_tensor_data = [
                    self.ones
                    if table_name == self.target_table and col_name == self.target_column and row == target_row
                    else self.convertor(row[i], table_name, col_name, col)
                    for i, col_name, col in process_cols
                ]

                if len(row_tensor_data) == 0:
                    row_tensor_data.append(self.ones)

                if not row_tensor_data:
                    break

                if primary_col:
                    table_primary_keys[row[primary_col[0]]] = index
                table_tensor_data.append(torch.cat(row_tensor_data))
            else:
                if table_primary_keys:
                    primary_keys[table_name] = table_primary_keys
                hetero_data[table_name].x = torch.flatten(torch.stack(table_tensor_data), 1)

        for table_name, table_data in data.items():
            col_to_index = {col_name: i for i, col_name in enumerate(self.schema[table_name].columns.keys())}

            foreign_cols = [
                (col_to_index[foreigns.columns[0]], foreigns) for foreigns in self.schema[table_name].foreign_keys
            ]

            if not foreign_cols:
                continue

            for index, col in foreign_cols:
                edge_index = []
                table_primary_keys = primary_keys[col.ref_table]

                for i_index, row in enumerate(table_data):
                    if row[index] is None or row[index] not in table_primary_keys:
                        continue

                    edge_index.append([i_index, table_primary_keys[row[index]]])
                hetero_data[table_name, col.columns[0], col.ref_table].edge_index = (
                    torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                )

        hetero_data = T.ToUndirected()(hetero_data)
        hetero_data = T.AddSelfLoops()(hetero_data)

        return hetero_data
