from typing import Set, Dict
import json



class DBDataLoader:
    def __init__(self, schema_path: str, batch_size: int):
        with open(schema_path, mode="r") as fp:
            self.schema = json.load(fp)

        self.selects = []

        self.processed = {}
        self.max_depth = 3
        self.batch_size = batch_size

        self.batches = []

    def get_keys(self, data, index):
        return [d[index] for d in data]

    def clear_keys(self, data, index):
        for i, d in enumerate(data):
            dd = list(d)
            dd[index] = None

            data[i] = tuple(dd)

    def load(self, con, table: str, index: int):
        queue = [(table, 0, None, None, None)]
        root = None

        table_data = {}

        while len(queue) != 0:
            t, depth, parent, key, keys = queue.pop(0)

            if depth >= self.max_depth:
                return root

            if root is None:
                data = con.execute(f"SELECT * FROM {t} LIMIT {self.batch_size} OFFSET {index * self.batch_size};")
                data = data.fetchall()

                if not data:
                    return False

                root = True
                table_data[t] = set(data)
            else:
                k = set(keys)
                data = con.execute(f"SELECT * FROM {t} WHERE {key} IN ({ ','.join('?' * len(k))});", list(k))
                data = data.fetchall()

                if depth + 1 == self.max_depth:
                    for i, (key, info) in enumerate(self.schema[t].items()):
                        if info.get("type") == "foreign_key":
                            self.clear_keys(data, i)

                if t not in table_data:
                    table_data[t] = set(data)
                else:
                    table_data[t].update(data)

                if depth + 1 == self.max_depth:
                    continue

            foreigns = set()
            for i, (key, info) in enumerate(self.schema[t].items()):
                if info.get("type") == "foreign_key":
                    queue.append((info.get("table"), depth + 1, table, key, self.get_keys(table_data[t], i)))
                    foreigns.add(info.get("table"))

            for next_table in self.schema:
                if next_table in foreigns or (parent is not None and next_table == parent):
                    continue

                for i, (key, info) in enumerate(self.schema[next_table].items()):
                    if info.get("type") == "foreign_key" and info.get("table") == t:
                        queue.append((next_table, depth + 1, table, key, self.get_keys(table_data[t], i)))

        self.batches.append(table_data)
        return True
