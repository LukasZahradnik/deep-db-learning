from collections import defaultdict
from typing import Dict, Set

from sqlalchemy import Connection, table, select, column

from db_transformer.ndata.strategy.strategy import BaseStrategy
from db_transformer.schema import Schema


class BFSStrategy(BaseStrategy):
    def __init__(self, max_depth: int):
        self.max_depth = max_depth

    def _get_keys(self, data, index: int):
        return [d[index] for d in data]

    def get_db_data(self, idx: int, connection: Connection, target_table: str, schema: Schema) -> Dict[str, Set]:
        queue = [(target_table, 0, None, None, None)]
        table_data = defaultdict(lambda: set())

        while len(queue) != 0:
            table_name, depth, parent, key, keys = queue.pop(0)

            if depth >= self.max_depth:
                return table_data

            col_to_index = {col: index for index, col in enumerate(schema[table_name].columns)}

            table_obj = table(table_name)
            if depth == 0:
                query = select("*", table_obj).limit(1).offset(idx)
            else:
                query = select("*", table_obj).where(column(key).in_(set(keys)))

            for res in connection.execute(query).all():
                res_tuple = res.tuple()

                if depth + 1 == self.max_depth and len(schema[table_name].foreign_keys) != 0:
                    res_tuple = [*res_tuple]

                    # TODO: This supports only one col per key
                    for col in schema[table_name].foreign_keys:
                        res_tuple[col_to_index[col.columns[0]]] = None
                    res_tuple = tuple(res_tuple)
                table_data[table_name].add(res_tuple)

            if depth + 1 == self.max_depth:
                continue

            processed_foreigns = set()
            for col in schema[table_name].foreign_keys:
                # TODO: This supports only one col per key
                fkeys = self._get_keys(table_data[table_name], col_to_index[col.columns[0]])
                queue.append((col.ref_table, depth + 1, table_name, col.ref_columns[0], fkeys))
                processed_foreigns.add(col.ref_table)

            # TODO: This assumes that other tables are referencing the first column of the current table
            pkeys = self._get_keys(table_data[table_name], 0)

            for next_table, next_schema in schema.items():
                if next_table in processed_foreigns or (parent is not None and next_table == parent):
                    continue

                for foreign_key in next_schema.foreign_keys:
                    if foreign_key.ref_table == table_name:
                        queue.append((next_table, depth + 1, table_name, foreign_key.columns[0], pkeys))

        return table_data
