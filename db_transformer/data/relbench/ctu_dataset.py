from typing import List, overload
import os

import pandas as pd

from sqlalchemy.engine import Connection, create_engine
from sqlalchemy.schema import MetaData, Table
from sqlalchemy.sql import select

from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS, TaskType
from db_transformer.db.db_inspector import DBInspector

from relbench.data import Dataset, Task, Database, Table as RelBenchTable
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc


class CTUDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./datasets",
        tasks: List[type[Task]] = None,
        save_db: bool = True
    ):
        if dataset_name not in FIT_DATASET_DEFAULTS.keys():
            raise KeyError(f"Relational CTU database '{dataset_name}' is unknown. "
                            "Please specify target_table and target_column explicitly.")
            
        self.dataset_name = dataset_name
        self.data_dir = data_dir

        db = None
        db_dir = os.path.join(data_dir, dataset_name, 'db')
        
        if os.path.exists(db_dir):
            db = Database.load(db_dir)
            if len(db.table_dict) == 0:
                db = None
        
        if db == None:
            db = self.make_db(dataset_name)
            if save_db:
                db.save(db_dir)
                
        if tasks == None:
            tasks = [CTUTask]
        
        super().__init__(
            db, pd.Timestamp.today(), pd.Timestamp.today(), tasks
        )


    @classmethod
    def get_url(cls, dataset: str) -> str:
        connector = "mariadb+mariadbconnector"
        port = 3306
        return f"{connector}://guest:potato-relational@potato.felk.cvut.cz:{port}/{dataset}"

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
            
            fk_dict = {list(fk)[0]: fk_const.ref_table for fk, fk_const in inspector.get_foreign_keys(table_name).items() if len(fk) == 1}
            src_table = Table(table_name, remote_md)
            
            df = pd.read_sql_query(sql=select(src_table.columns), con=remote_conn.engine)
            tables[table_name] = RelBenchTable(df=df, fkey_col_to_pkey_table=fk_dict, pkey_col=pkey_col)
            
        return Database(tables)
    


class CTUTask(Task):
    
    name = 'ctu-task'
    
    def __init__(self, name: str, dataset: CTUDataset):
        self.defaults = FIT_DATASET_DEFAULTS[dataset.dataset_name]
        metrics = []
        
        super().__init__(dataset, pd.Timedelta(days=0), self.defaults.target_column, metrics)
        
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> RelBenchTable:
        target_table = db.table_dict[self.defaults.target_table]
        df = target_table.df[[target_table.pkey_col, self.defaults.target_column]].copy(deep=True)
        
        # TODO: make random splits
        
        return RelBenchTable(df, {target_table.pkey_col: self.defaults.target_table})



# if __name__ == "__main__":
#     print('ctu_dataset')
