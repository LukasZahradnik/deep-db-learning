from typing import Dict, Set, Tuple, Any

from sqlalchemy.engine import Connection

from db_transformer.schema import Schema


class BaseStrategy:
    def get_db_data(self, idx: int, connection: Connection, target_table: str, schema: Schema) -> Tuple[Dict[str, Set], Any]:
        raise NotImplemented
