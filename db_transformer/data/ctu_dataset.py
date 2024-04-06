from collections.abc import Iterable
import json
import os
from pathlib import Path
import shutil
import warnings

from typing import Dict, List, Optional, Tuple

import pandas as pd

from sqlalchemy.engine import Connection, create_engine
from sqlalchemy.schema import MetaData, Table as SQLTable
from sqlalchemy.sql import select, text

from sklearn.preprocessing import MultiLabelBinarizer

import torch

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from sentence_transformers import SentenceTransformer

import torch_frame
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.utils import infer_series_stype

from relbench.data import Database, Table

from db_transformer.db.db_inspector import DBInspector
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from db_transformer.helpers.progress import wrap_progress
from db_transformer.schema import columns, ColumnDef, ForeignKeyDef, Schema, TableSchema
from db_transformer.data.dataset_defaults.ctu_repository_defauts import (
    CTUDatasetName,
    CTU_REPOSITORY_DEFAULTS,
)
from db_transformer.helpers.objectpickle import serialize, deserialize


class GloveTextEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return torch.from_numpy(self.model.encode(sentences, show_progress_bar=False))


class CTUDataset:
    def __init__(
        self,
        name: CTUDatasetName,
        data_dir: str = "./datasets",
        save_db: bool = True,
        force_remake: bool = False,
    ):
        if name not in CTU_REPOSITORY_DEFAULTS.keys():
            raise KeyError(f"Relational CTU dataset '{name}' is unknown.")

        self.name = name
        self.defaults = CTU_REPOSITORY_DEFAULTS[name]
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
                db = Database.load(self.db_dir)
                self._save_schema()

        self.text_embedder_cfg = TextEmbedderConfig(text_embedder=GloveTextEmbedding())
        self.db = db

    @property
    def root_dir(self):
        return os.path.join(self.data_dir, self.name)

    @property
    def db_dir(self):
        return os.path.join(self.root_dir, "db")

    @property
    def schema_path(self):
        return os.path.join(self.db_dir, "schema.json")

    def build_hetero_data(
        self, device: str = None, force_rematerilize: bool = False
    ) -> HeteroData:
        data = HeteroData()

        # get all tables
        table_dfs = {
            table_name: table.df for table_name, table in self.db.table_dict.items()
        }

        materialized_dir = os.path.join(self.root_dir, "materialized")
        if force_rematerilize and os.path.exists(materialized_dir):
            shutil.rmtree(materialized_dir)
        Path(materialized_dir).mkdir(parents=True, exist_ok=True)

        for table_name, table_schema in wrap_progress(
            self.schema.items(), verbose=True, desc="Building data"
        ):
            df = table_dfs[table_name]

            if df.empty:
                continue

            # convert all foreign keys
            for fk_def in table_schema.foreign_keys:
                ref_df = table_dfs[fk_def.ref_table]
                id = table_name, "-".join(fk_def.columns), fk_def.ref_table
                try:
                    data[id].edge_index = self._fk_to_index(fk_def, df, ref_df, device)
                except Exception as e:
                    warnings.warn(f"Failed to join on foreign key {id}. Reason: {e}")

            col_to_stype = self._schema_to_stype_dict(self.schema[table_name])

            target_col = (
                self.defaults.target_column
                if table_name == self.defaults.target_table
                else None
            )

            for col in df.columns:
                if col not in col_to_stype:
                    continue
                if df[col].dtype.__str__().startswith("timedelta"):
                    df[col] = df[col].dt.nanoseconds
                if (
                    df[col].dtype.__str__().startswith("object")
                    and col_to_stype[col] == torch_frame.stype.categorical
                ):
                    mlb = MultiLabelBinarizer()
                    df = df.join(
                        pd.DataFrame(mlb.fit_transform(df[col]), columns=mlb.classes_)
                    )
                    col_to_stype.pop(col)
                    df.drop(columns=[col], inplace=True)
                    for c in mlb.classes_:
                        col_to_stype[c] = torch_frame.stype.categorical

            if len(col_to_stype) == 0 or (
                len(col_to_stype) == 1 and target_col is not None
            ):
                col_to_stype["__filler"] = torch_frame.stype.categorical
                df["__filler"] = 0

            if target_col is not None and target_col not in col_to_stype:
                col_to_stype[target_col] = infer_series_stype(df[target_col])

            def __build_frame_dataset(table, col_to_stype):
                return Dataset(
                    df=table,
                    col_to_stype=col_to_stype,
                    col_to_text_embedder_cfg=self.text_embedder_cfg,
                    target_col=target_col,
                ).materialize(device, path=os.path.join(materialized_dir, table_name))

            try:
                dataset = __build_frame_dataset(df, col_to_stype)

            except pd.errors.OutOfBoundsDatetime as e:
                for col, _stype in col_to_stype.items():
                    if _stype != torch_frame.stype.timestamp:
                        continue
                    df[col].loc[MAX_TIMESTAMP.date() < df[col]] = MAX_TIMESTAMP.date()
                    df[col].loc[MIN_TIMESTAMP.date() > df[col]] = MIN_TIMESTAMP.date()

                dataset = __build_frame_dataset(df, col_to_stype)

            stype_to_col_str = "\n".join(
                [f"\t{k}: {v}" for k, v in dataset.tensor_frame.col_names_dict.items()]
            )
            print(f"Table {table_name} has stypes:\n{stype_to_col_str}")

            data[table_name].tf = dataset.tensor_frame.to(device)
            data[table_name].col_stats = dataset.col_stats
            if table_name == self.defaults.target_table:
                data[table_name].y = dataset.tensor_frame.y

        # add reverse edges
        data: HeteroData = T.ToUndirected()(data)

        return data

    @classmethod
    def get_url(cls, dataset: CTUDatasetName) -> str:
        connector = "mariadb+mysqlconnector"
        port = 3306
        return f"{connector}://guest:ctu-relational@relational.fel.cvut.cz:{port}/{dataset}"

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

        inspector = DBInspector(remote_conn)

        analyzer = SchemaAnalyzer(
            remote_conn,
            verbose=True,
            target=CTU_REPOSITORY_DEFAULTS[dataset].target,
            target_type=CTU_REPOSITORY_DEFAULTS[dataset].task.to_type(),
            post_guess_schema_hook=CTU_REPOSITORY_DEFAULTS[dataset].schema_fixer,
        )
        schema = analyzer.guess_schema()

        remote_md = MetaData()
        remote_md.reflect(bind=inspector.engine)

        tables = {}

        for table_name in wrap_progress(
            inspector.get_tables(), verbose=True, desc="Downloading tables"
        ):
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

    def _schema_to_stype_dict(
        self, table_schema: TableSchema
    ) -> Dict[str, torch_frame.stype]:
        merged: Dict[str, torch_frame.stype] = {}
        for col_name, col_def in table_schema.columns.items():
            _stype = COLUMN_DEF_STYPE[type(col_def)]
            if _stype is not None:
                merged[col_name] = _stype
        return merged

    def _save_schema(self):
        with open(self.schema_path, "w+") as f:
            json.dump(serialize(self.schema), f, indent=4)

    def _load_schema(self) -> Schema:
        with open(self.schema_path, "r") as f:
            return deserialize(json.load(f))


MAX_TIMESTAMP = pd.Timestamp("2262-04-10")
MIN_TIMESTAMP = pd.Timestamp("1677-09-23")

COLUMN_DEF_STYPE: Dict[ColumnDef, torch_frame.stype] = {
    columns.CategoricalColumnDef: torch_frame.stype.categorical,
    columns.DateColumnDef: torch_frame.stype.timestamp,
    columns.DateTimeColumnDef: torch_frame.stype.timestamp,
    columns.NumericColumnDef: torch_frame.stype.numerical,
    columns.DurationColumnDef: torch_frame.stype.numerical,
    columns.TextColumnDef: torch_frame.stype.text_embedded,
    columns.TimeColumnDef: torch_frame.stype.numerical,
    columns.OmitColumnDef: None,
}


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
    "DATE": "object",
    "TIME": "object",
    "DATETIME": "object",
    "TIMESTAMP": "object",
    "CHAR": "string",
    "VARCHAR": "string",
    "TEXT": "string",
    "MEDIUMTEXT": "string",
    "LONGTEXT": "string",
    "ENUM": pd.CategoricalDtype(),
    "SET": "object",
    "BINARY": "object",
    "VARBINARY": "object",
    "BLOB": "object",
    "MEDIUMBLOB": "object",
    "LONGBLOB": "object",
}
