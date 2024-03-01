from collections import OrderedDict
import json
import os
from pathlib import Path
import warnings

from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from sqlalchemy.engine import Connection, create_engine
from sqlalchemy.schema import MetaData, Table as SQLTable
from sqlalchemy.sql import select, text

import torch

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from sentence_transformers import SentenceTransformer

from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.utils import infer_df_stype

from relbench.data import Dataset as RelBenchDataset, BaseTask, Database, Table
from db_transformer.data.relbench.utils import (
    merge_schema_infered_stype,
)
from db_transformer.db.db_inspector import DBInspector
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from db_transformer.schema.schema import ColumnDef, ForeignKeyDef, Schema
from db_transformer.data.relbench.ctu_repository_defauts import (
    CTUDatasetName,
    CTU_REPOSITORY_DEFAULTS,
)
from db_transformer.helpers.objectpickle import serialize, deserialize


from db_transformer.data.relbench.ctu_task import CTUTask


class GloveTextEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return torch.from_numpy(self.model.encode(sentences))


class CTUDataset(RelBenchDataset):
    def __init__(
        self,
        name: CTUDatasetName,
        data_dir: str = "./datasets",
        tasks: List[type[BaseTask]] = None,
        save_db: bool = True,
        force_remake: bool = False,
    ):
        if name not in CTU_REPOSITORY_DEFAULTS.keys():
            raise KeyError(f"Relational CTU dataset '{name}' is unknown.")

        self.name = name
        self.data_dir = data_dir

        db = None

        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

        if not force_remake and os.path.exists(self.db_dir):
            self.schema = self._load_schema()
            db = Database.load(self.db_dir)
            if len(db.table_dict) == 0:
                db = None

        if db == None:
            db, self.schema = self.make_db(name)
            if save_db:
                db.save(self.db_dir)
                self._save_schema()

        if tasks == None:
            tasks = [CTUTask]

        self.text_embedder_cfg = TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(), batch_size=32
        )

        super().__init__(db, pd.Timestamp.today(), pd.Timestamp.today(), tasks)

    @property
    def root_dir(self):
        return os.path.join(self.data_dir, self.name)

    @property
    def db_dir(self):
        return os.path.join(self.root_dir, "db")

    @property
    def schema_path(self):
        return os.path.join(self.db_dir, "schema.json")

    def validate_and_correct_db(self):
        return

    def get_task(self) -> BaseTask:
        return CTUTask(dataset=self)

    def build_hetero_data(
        self, device=None
    ) -> Tuple[HeteroData, Dict[str, List[ColumnDef]]]:
        data = HeteroData()

        # get all tables
        table_dfs = {
            table_name: table.df for table_name, table in self.db.table_dict.items()
        }

        materialized_dir = os.path.join(self.root_dir, "materialized")
        Path(materialized_dir).mkdir(parents=True, exist_ok=True)

        out_column_defs: Dict[str, List[ColumnDef]] = {}
        for table_name, table_schema in self.schema.items():
            df = table_dfs[table_name]

            # convert all foreign keys
            for fk_def in table_schema.foreign_keys:
                ref_df = table_dfs[fk_def.ref_table]
                id = table_name, "-".join(fk_def.columns), fk_def.ref_table
                try:
                    data[id].edge_index = self._fk_to_index(fk_def, df, ref_df)
                except Exception as e:
                    warnings.warn(f"Failed to join on foreign key {id}. Reason: {e}")

            col_to_stype = infer_df_stype(df)
            col_to_stype = merge_schema_infered_stype(self.schema[table_name], col_to_stype)

            dataset = Dataset(
                df=df,
                col_to_stype=col_to_stype,
                col_to_text_embedder_cfg=self.text_embedder_cfg,
            ).materialize(path=os.path.join(materialized_dir, table_name))

            data[table_name].tf = dataset.tensor_frame
            data[table_name].col_stats = dataset.col_stats

        # add reverse edges
        data: HeteroData = T.ToUndirected()(data)

        return data, out_column_defs

    @classmethod
    def get_url(cls, dataset: CTUDatasetName) -> str:
        connector = "mariadb+mysqlconnector"
        port = 3306
        return f"{connector}://guest:ctu-relational@78.128.250.186:{port}/{dataset}"

    @classmethod
    def create_remote_connection(cls, dataset: CTUDatasetName):
        """Create a new SQLAlchemy Connection instance to the remote database.

        Create a new SQLAlchemy Connection instance to the remote database.
        Don't forget to close the Connection after you are done using it!
        """
        return Connection(create_engine(cls.get_url(dataset)))

    @classmethod
    def make_db(cls, dataset: CTUDatasetName) -> Tuple[Database, Schema]:
        remote_conn = cls.create_remote_connection(dataset)

        defaults = CTU_REPOSITORY_DEFAULTS[dataset]

        inspector = DBInspector(remote_conn)

        analyzer = SchemaAnalyzer(remote_conn)
        schema = analyzer.guess_schema()

        remote_md = MetaData()
        remote_md.reflect(bind=inspector.engine)

        tables = {}

        for table_name in inspector.get_tables():
            pk = inspector.get_primary_key(table_name)
            pkey_col = list(pk)[0] if len(pk) == 1 else None

            fk_dict = {
                list(fk)[0]: fk_const.ref_table
                for fk, fk_const in inspector.get_foreign_keys(table_name).items()
                if len(fk) == 1
            }
            src_table = SQLTable(table_name, remote_md)

            dtypes: Dict[str, str] = {}

            for c in src_table.columns:
                str_type = c.type.__str__().split("(")[0]
                dtype = MARIADB_TO_PANDAS.get(str_type, None)
                if dtype is not None:
                    dtypes[c.name] = dtype
                else:
                    warnings.warn(f"Unknown data type {c.type}")

            statement = select(src_table.columns)
            query = statement.compile(remote_conn.engine)
            df = pd.read_sql_query(sql=text(query.string), con=remote_conn, dtype=dtypes)
            tables[table_name] = Table(
                df=df, fkey_col_to_pkey_table=fk_dict, pkey_col=pkey_col
            )

        return Database(tables), schema

    def _fk_to_index(
        self,
        fk_def: ForeignKeyDef,
        table: pd.DataFrame,
        ref_table: pd.DataFrame,
        device=None,
    ) -> torch.Tensor:
        assert isinstance(table.index, pd.RangeIndex)
        assert isinstance(ref_table.index, pd.RangeIndex)

        # keep just the index columns
        table = table[fk_def.columns].copy()
        ref_table = ref_table[fk_def.ref_columns].copy()

        table.index.name = "__pandas_index"
        ref_table.index.name = "__pandas_index"
        table.reset_index(inplace=True)
        ref_table.reset_index(inplace=True)

        out = pd.merge(
            left=table,
            right=ref_table,
            how="inner",
            left_on=fk_def.columns,
            right_on=fk_def.ref_columns,
        )

        return (
            torch.from_numpy(out[["__pandas_index_x", "__pandas_index_y"]].to_numpy())
            .t()
            .contiguous()
            .to(device)
        )

    def _save_schema(self):
        with open(self.schema_path, "w+") as f:
            json.dump(serialize(self.schema), f, indent=4)

    def _load_schema(self) -> Schema:
        with open(self.schema_path, "r") as f:
            return deserialize(json.load(f))


MARIADB_TO_PANDAS = {
    "TINYINT": pd.Int8Dtype(),
    "SMALLINT": pd.Int16Dtype(),
    "MEDIUMINT": pd.Int32Dtype(),
    "INT": pd.Int32Dtype(),
    "INTEGER": pd.Int32Dtype(),
    "BIGINT": pd.Int64Dtype(),
    "TINYINT UNSIGNED": pd.UInt8Dtype(),
    "SMALLINT UNSIGNED": pd.UInt16Dtype(),
    "MEDIUMINT UNSIGNED": pd.UInt32Dtype(),
    "INT UNSIGNED": pd.UInt32Dtype(),
    "INTEGER UNSIGNED": pd.UInt32Dtype(),
    "BIGINT UNSIGNED": pd.UInt64Dtype(),
    "FLOAT": pd.Float32Dtype(),
    "DOUBLE": pd.Float64Dtype(),
    "DECIMAL": pd.Float64Dtype(),
    "DATE": "datetime64[ns]",
    "TIME": "object",
    "DATETIME": "datetime64[ns]",
    "TIMESTAMP": "datetime64[ns]",
    "CHAR": "string",
    "VARCHAR": "string",
    "TEXT": "string",
    "MEDIUMTEXT": "string",
    "LONGTEXT": "string",
    "ENUM": "categorical",
    "SET": "object",
    "BINARY": "object",
    "VARBINARY": "object",
    "BLOB": "object",
    "MEDIUMBLOB": "object",
    "LONGBLOB": "object",
}
