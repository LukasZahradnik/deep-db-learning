from collections import defaultdict

from torch_geometric.data import HeteroData
from torch_geometric.data.data import BaseData

from sqlalchemy import Connection
from sqlalchemy import create_engine, table, select, column

from db_transformer.ndata.dataset import BaseDBDataset
from db_transformer.schema import Schema


class DFSDBDataset(BaseDBDataset):
    def __init__(self, schema: Schema, database: str, target_table: str, depth: int, sql_connection_string, root=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.max_depth = depth
        super().__init__(schema, database, target_table, sql_connection_string, root, transform, pre_transform,
                         pre_filter)

    def get(self, idx: int) -> BaseData:
        engine = create_engine(self.sql_connection_string)
        with engine.connect() as connection:
            data = self._load_data(idx, connection)
        return self._to_hetero_data(data)

    def _get_keys(self, data, index: int):
        return [d[index] for d in data]

    def _to_hetero_data(self, data) -> HeteroData:
        # TODO: Convert to tensors and build graph
        pass

    def _load_data(self, idx: int, connection: Connection):
        queue = [(self.target_table, 0, None, None, None)]
        table_data = defaultdict(lambda: set())

        while len(queue) != 0:
            table_name, depth, parent, key, keys = queue.pop(0)

            if depth >= self.max_depth:
                return table_data

            col_to_index = {col: index for index, col in enumerate(self.schema[table_name].columns)}

            table_obj = table(table_name)
            if depth == 0:
                query = select("*", table_obj).limit(1).offset(idx)
            else:
                query = select("*", table_obj).where(column(key).in_(set(keys)))

            for res in connection.execute(query).all():
                res_tuple = res.tuple()

                if depth + 1 == self.max_depth and len(self.schema[table_name].foreign_keys) != 0:
                    res_tuple = [*res_tuple]

                    # TODO: This supports only one col per key
                    for col in self.schema[table_name].foreign_keys:
                        res_tuple[col_to_index[col.columns[0]]] = None
                    res_tuple = tuple(res_tuple)
                table_data[table_name].add(res_tuple)

            if depth + 1 == self.max_depth:
                continue

            processed_foreigns = set()
            for col in self.schema[table_name].foreign_keys:
                # TODO: This supports only one col per key
                fkeys = self._get_keys(table_data[table_name], col_to_index[col.columns[0]])
                queue.append((col.ref_table, depth + 1, table_name, col.ref_columns[0], fkeys))
                processed_foreigns.add(col.ref_table)

            # TODO: This assumes that other tables are referencing the first column of the current table
            pkeys = self._get_keys(table_data[table_name], 0)

            for next_table, next_schema in self.schema.items():
                if next_table in processed_foreigns or (parent is not None and next_table == parent):
                    continue

                for foreign_key in next_schema.foreign_keys:
                    if foreign_key.ref_table == table_name:
                        queue.append((next_table, depth + 1, table_name, foreign_key.columns[0], pkeys))

        return table_data
