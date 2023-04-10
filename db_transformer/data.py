import json
from typing import List, Tuple

import torch
from torch_geometric.data import HeteroData


class CatEncoder:
    def __init__(self):
        self.cat_dict = {}

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

        for i, (column_name, column) in enumerate(table_schema.items()):
            if column.get("key"):
                self.primary_key = i
                continue

            if column["type"] == "cat":
                self.cat_encoders.append(CatEncoder())
                self.cat_index.append(i)
                self.categories.append(column["card"])

            if column["type"] == "num" and not column.get("key"):
                self.num_index.append(i)
                self.num_continuous += 1

            if column["type"] == "foreign_key":
                self.x_keys.append(0)
                self.key_index.append(i)
                self.keys.append((column["table"], column["column"]))

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

        key_indexing[self.name] = {}

        for i, row in enumerate(processed_data):
            numeric = [row[x] for x in self.num_index]
            # numeric.extend(0.0 for _ in self.key_index)

            if self.primary_key != -1:
                key_indexing[self.name][row[self.primary_key]] = i

            x_categ.append([row[x] for x, encode in zip(self.cat_index, self.cat_encoders)])
            x_numer.append(numeric)

        self.x_numer = torch.tensor(x_numer)
        self.x_cat = torch.tensor(x_categ)

    def add_foreign_keys(self, hetero_data: HeteroData, preprocessed_data, key_indexing) -> HeteroData:
        edge_indices = [[] for _ in self.key_index]

        for i, row in enumerate(preprocessed_data):
            for j, (key_index, (table_name, _)) in enumerate(zip(self.key_index, self.keys)):
                edge_indices[j].append([i, key_indexing[table_name][row[key_index]]])

        for edge_index, (table_name, column) in zip(edge_indices, self.keys):
            key = f"{self.name}_{table_name}_{column}"
            _ = hetero_data[key]
            _ = hetero_data[table_name]
            hetero_data[key, key, table_name].edge_index = torch.tensor(edge_index).t().contiguous()
        return hetero_data


class DataLoader:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

        with open(f"{dir_path}/schema.json", mode="r") as fp:
            self.schema = json.load(fp)

    def load(self, out_table: str, label_index: int = -1) -> Tuple[List[TableData], HeteroData, torch.Tensor]:
        data = HeteroData()
        table_data = []
        label_encoder = CatEncoder()
        labels = []
        raw_data = []
        key_indexing = {}

        for table_name, table in self.schema.items():
            table_data_instance = TableData(table_name, table)

            with open(f"{self.dir_path}/{table_name}.csv", mode="r") as fp:
                processed = []

                for line in fp.readlines():
                    line_data = line.strip().split(",")

                    if table_name == out_table:
                        labels.append(torch.tensor(label_encoder.get_index(line_data[label_index])))

                    processed.append(table_data_instance.process(line_data))
                raw_data.append(processed)
            table_data.append(table_data_instance)
            table_data_instance.to_torch(processed, key_indexing)

        for table, raw in zip(table_data, raw_data):
            table.add_foreign_keys(data, raw, key_indexing)

        return table_data, data, torch.stack(labels)
