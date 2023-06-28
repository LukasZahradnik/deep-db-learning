import os.path
from typing import Union, List, Tuple

from torch_geometric.data import Dataset

from db_transformer.helpers.database import copy_database, get_table_len
from db_transformer.schema import Schema
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from db_transformer.db import SchemaAnalyzer


class BaseDBDataset(Dataset):
    def __init__(self, schema: Schema, database: str, target_table: str, sql_connection_string, root=None, transform=None, pre_transform=None, pre_filter=None):
        self.schema = schema
        self.database = database
        self.target_table = target_table
        self.sql_connection_string = sql_connection_string

        self.processed_data = []
        self.length = 0

        # TODO: Save the schema in the processed_file_names (offline support)
        if self.schema is None:
            engine = create_engine(self.sql_connection_string)

            with Session(engine) as session:
                self.schema = SchemaAnalyzer(engine, session).guess_schema()
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self) -> int:
        return self.length

    def download(self):
        copy_database(self.sql_connection_string, self.db_connection_string, self.schema.keys())

    def process(self):
        engine = create_engine(self.sql_connection_string)

        with Session(engine) as session:
            self.length = get_table_len(self.target_table, session)

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
