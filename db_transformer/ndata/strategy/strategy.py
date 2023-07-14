from typing import Dict, Set

from sqlalchemy.engine import Connection

from db_transformer.schema import Schema


class BaseStrategy:
    def get_db_data(self, idx: int, connection: Connection, target_table: str, schema: Schema) -> Dict[str, Set]:
        raise NotImplemented
