from typing import Optional

from db_transformer.ndata.dataset import BaseDBDataset
from db_transformer.ndata.strategy.bfs import BFSStrategy
from db_transformer.ndata.strategy.strategy import BaseStrategy
from db_transformer.schema import Schema


class FITRelationalDataset(BaseDBDataset):
    def __init__(
        self, database: str, target_table: str, root: str, strategy: BaseStrategy, schema: Optional[Schema] = None
    ):
        sql_connection_str = f"mariadb+mariadbconnector://guest:relational@relational.fit.cvut.cz:3306/{database}"

        super().__init__(database, target_table, sql_connection_str, root, strategy, schema)
