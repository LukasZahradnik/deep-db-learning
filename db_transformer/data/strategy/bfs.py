from collections import defaultdict
from typing import Any, Dict, Set, Tuple

from sqlalchemy.engine import Connection
from sqlalchemy.sql import column, select, table

from db_transformer.data.strategy.strategy import BaseStrategy
from db_transformer.schema import Schema


class BFSStrategy(BaseStrategy):
    def __init__(self, max_depth: int):
        self.max_depth = max_depth

    def _get_keys(self, data, index: int):
        return [d[index] for d in data]

    def get_db_data(
        self, idx: int, connection: Connection, target_table: str, schema: Schema
    ) -> Tuple[Dict[str, Set[Tuple[Any, ...]]], Any]:
        queue = [(target_table, 0, None, None, None)]
        empty_set = set()
        table_data = defaultdict(lambda: set())
        target_row = None

        while len(queue) != 0:
            table_name, depth, parent, key, keys = queue.pop(0)

            if depth >= self.max_depth:
                return table_data, target_row

            col_to_index = {
                col: index for index, col in enumerate(schema[table_name].columns)
            }

            table_obj = table(table_name)
            if depth == 0:
                query = select("*", table_obj).limit(1).offset(idx)
            else:
                query = select("*", table_obj).where(column(key).in_(set(keys)))

            results = connection.execute(query).all()
            if depth == 0:
                target_row = results[0]

            for res in results:
                table_data[table_name].add(res.tuple())

            if depth + 1 == self.max_depth:
                continue

            processed_foreigns = set()
            for col in schema[table_name].foreign_keys:
                # TODO: This supports only one col per key
                fkeys = self._get_keys(
                    table_data.get(table_name, empty_set), col_to_index[col.columns[0]]
                )
                queue.append(
                    (col.ref_table, depth + 1, table_name, col.ref_columns[0], fkeys)
                )
                processed_foreigns.add(col.ref_table)

            # TODO: This assumes that other tables are referencing the first column of the current table
            pkeys = self._get_keys(table_data.get(table_name, empty_set), 0)

            for next_table, next_schema in schema.items():
                if next_table in processed_foreigns or (
                    parent is not None and next_table == parent
                ):
                    continue

                for foreign_key in next_schema.foreign_keys:
                    if foreign_key.ref_table == table_name:
                        queue.append(
                            (
                                next_table,
                                depth + 1,
                                table_name,
                                foreign_key.columns[0],
                                pkeys,
                            )
                        )

        return table_data, target_row
