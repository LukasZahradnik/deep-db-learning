import sys, os, warnings

sys.path.append(os.getcwd())

from typing import Dict, List

import pandas as pd

from sqlalchemy.engine import Connection, create_engine
from sqlalchemy.schema import MetaData, Table as SQLTable
from sqlalchemy.sql import select, text

from db_transformer.db.db_inspector import DBInspector

from relbench.data import Dataset, BaseTask, Database, Table
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc

from db_transformer.data.relbench.ctu_repository_defauts import (
    CTUDatasetName,
    CTU_REPOSITORY_DEFAULTS,
)

from db_transformer.data.relbench.ctu_task import CTUTask


class CTUDataset(Dataset):
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
        db_dir = os.path.join(data_dir, name, "db")

        if not force_remake and os.path.exists(db_dir):
            db = Database.load(db_dir)
            if len(db.table_dict) == 0:
                db = None

        if db == None:
            db = self.make_db(name)
            if save_db:
                db.save(db_dir)

        if tasks == None:
            tasks = [CTUTask]

        super().__init__(db, pd.Timestamp.today(), pd.Timestamp.today(), tasks)

    def validate_and_correct_db(self):
        return

    def get_task(self) -> BaseTask:
        return CTUTask(dataset=self)

    @classmethod
    def get_url(cls, dataset: str) -> str:
        connector = "mariadb+mysqlconnector"
        port = 3306
        return f"{connector}://guest:ctu-relational@78.128.250.186:{port}/{dataset}"

    @classmethod
    def create_remote_connection(cls, dataset: str):
        """Create a new SQLAlchemy Connection instance to the remote database.

        Create a new SQLAlchemy Connection instance to the remote database.
        Don't forget to close the Connection after you are done using it!
        """
        return Connection(create_engine(cls.get_url(dataset)))

    @classmethod
    def make_db(cls, dataset: str) -> Database:
        remote_conn = cls.create_remote_connection(dataset)

        inspector = DBInspector(remote_conn)

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

        return Database(tables)


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


if __name__ == "__main__":
    dataset = CTUDataset(name="classicmodels", force_remake=False)
    task = dataset.get_task()

    print(task.train_table.df)
    # print(dataset.db.table_dict)
