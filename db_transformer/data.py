import json
import sqlite3
from typing import List, Tuple

import torch
from torch_geometric.data import HeteroData

from db_transformer.dataloader import DBDataLoader


from collections import defaultdict

encoders = defaultdict(lambda: defaultdict(lambda: CatEncoder()))

class CatEncoder:
    def __init__(self):
        self.cat_dict = {}
        self.name = ""

    def get_index(self, category: str) -> int:
        index = self.cat_dict.get(category, None)
        if index is None:
            index = len(self.cat_dict)
            self.cat_dict[category] = index
        return index



class TableData:
    def __init__(self, name, table_schema):
        super().__init__()

        self.name = name
        self.cat_encoders = []
        self.cat_index = []
        self.num_index = []
        self.key_index = []
        self.keys = []

        self.primary_key = -1

        self.categories = []
        self.num_continuous = 0

        self.x_numer = None
        self.x_cat = None

        self.x_keys = [0]
        self.schema = table_schema

        self.empty = False

        index = 0
        for i, (column_name, column) in enumerate(table_schema.items()):
            if column.get("key"):
                index += 1
                self.primary_key = i
                continue

            if column["type"] == "cat":
                index += 1
                self.cat_encoders.append(encoders[name][column_name])
                self.cat_encoders[-1].name = column_name
                self.cat_index.append(i)
                self.categories.append(column["card"])

            if column["type"] == "num" and not column.get("key"):
                index += 1
                self.num_index.append(i)
                self.num_continuous += 1

            if column["type"] == "foreign_key":
                self.x_keys.append(0)
                self.key_index.append(i)
                self.keys.append((column["table"], column["column"]))
                index += 1

    def process(self, table_raw_row):
        preprocessed_data = []
        for i, data in enumerate(table_raw_row):
            if i in self.num_index:
                preprocessed_data.append(float(data) if data != "None" else 0.)
            elif i in self.cat_index:
                # Make this faster?
                preprocessed_data.append(self.cat_encoders[self.cat_index.index(i)].get_index(data))
            elif i in self.key_index:
                preprocessed_data.append(data)
            else:
                preprocessed_data.append(data)
        return preprocessed_data

    def to_torch(self, processed_data, key_indexing):
        x_categ = []
        x_numer = []

        if processed_data:
            self.empty = False
        else:
            self.empty = True

        key_indexing[self.name] = {}

        for i, row in enumerate(processed_data):
            numeric = [row[x] for x in self.num_index]
            # numeric.extend(0.0 for _ in self.key_index)

            if self.primary_key != -1:
                key_indexing[self.name][row[self.primary_key]] = i

            x_categ.append([row[x] for x, encode in zip(self.cat_index, self.cat_encoders)])
            x_numer.append(numeric)

        self.x_numer = torch.tensor(x_numer)
        self.x_cat = torch.tensor(x_categ, dtype=torch.long)

    def add_foreign_keys(self, hetero_data: HeteroData, preprocessed_data, key_indexing) -> HeteroData:
        edge_indices = [[] for _ in self.key_index]

        for i, row in enumerate(preprocessed_data):
            for j, (key_index, (table_name, _)) in enumerate(zip(self.key_index, self.keys)):
                if not row[key_index] and row[key_index] != 0:
                    continue
                if row[key_index] not in key_indexing[table_name]:
                    continue

                edge_indices[j].append([i, key_indexing[table_name][row[key_index]]])

        for edge_index, (table_name, column) in zip(edge_indices, self.keys):
            if not edge_index:
                continue
            key = f"{self.name}_{table_name}_{column}"
            # _ = hetero_data[key]
            _ = hetero_data[self.name]
            _ = hetero_data[table_name]
            hetero_data[self.name, key, table_name].edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return hetero_data


class DataLoader:
    def __init__(self, dir_path: str, batch_size: int, max_depth: int, dtype, split: float):
        self.dir_path = dir_path
        self.batch_size = batch_size

        with open(f"{dir_path}/schema.json", mode="r") as fp:
            self.schema = json.load(fp)

        self.dl = DBDataLoader(f"{dir_path}/schema.json", batch_size, 2)
        self.len = 0

        self.metadata = [[], []]
        self.tables = []
        self.dtype = dtype
        self.split = split
        self.label_encoder = CatEncoder()

        for table, values in self.schema.items():
            self.tables.append(TableData(table, values))

            for column_name, column in values.items():
                if column["type"] == "foreign_key":
                    t, c = column["table"], column["column"]
                    key = f"{table}_{t}_{c}"
                    self.metadata[1].append((table, key, t))
                    self.metadata[0].append(t)
                    self.metadata[0].append(table)
        self.metadata = (list(set(self.metadata[0])), self.metadata[1])

    def load_data_loader(self, db, target):
        with sqlite3.connect(db) as con:
            index = 0

            while True:
                if self.split > index * self.batch_size:
                    self.dl.load(con, target, index, self.batch_size, index * self.batch_size)
                elif self.split - ((index - 1) * self.batch_size) > 0:
                    self.dl.load(con, target, index, self.split - ((index - 1) * self.batch_size), index * self.batch_size)
                    break
                else:
                    break
                index += 1

            index = 0
            while True:
                if not self.dl.load(con, target, index, self.batch_size, self.split + (index * self.batch_size), False):
                    break
                index += 1

        self.len = len(self.dl.batches)
        self.test_len = len(self.dl.test_batches)
        print(self.len, self.test_len)

    def load(self, index, out_table: str, label_index: int = -1, train=True) -> Tuple[List[TableData], HeteroData, torch.Tensor]:
        data = HeteroData()
        for k in self.metadata[1]:
            data[k].edge_index = torch.tensor([[], []], dtype=torch.long)
        for k in self.metadata[0]:
            data[k].x = torch.tensor([])
        table_data = []
        label_encoder = self.label_encoder
        labels = []
        raw_data = []
        key_indexing = {}

        batches = self.dl.batches
        if not train:
            batches = self.dl.test_batches

        for table_name, table in self.schema.items():
            table_data_instance = TableData(table_name, table)

            processed = []

            if table_name in batches[index]:
                for line_data in batches[index][table_name]:
                    if table_name == out_table:
                        labels.append(torch.tensor(label_encoder.get_index(line_data[label_index]), dtype=self.dtype))
                    processed.append(table_data_instance.process(line_data))
            raw_data.append(processed)
            table_data.append(table_data_instance)
            table_data_instance.to_torch(processed, key_indexing)

        for table, raw in zip(table_data, raw_data):
            table.add_foreign_keys(data, raw, key_indexing)

        return table_data, data, torch.stack(labels)
