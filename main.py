import json
import torch.nn
from dbtransformer import DBTransformer

torch.manual_seed(1)


class CatEncoder:
    def __init__(self):
        self.cat_dict = {}

    def get_index(self, category: str) -> int:
        index = self.cat_dict.get(category, None)
        if index is None:
            index = len(self.cat_dict)
            self.cat_dict[category] = index
        return index


class TableInitEmbed:
    FOREIGN_KEYS = {}

    def __init__(self, name, table):
        super(TableInitEmbed, self).__init__()

        self.name = name
        self.cat_encoders = []
        self.cat_index = []
        self.num_index = []
        self.key_index = []
        self.keys = []

        self.primary_key = 0

        self.categories = []
        self.num_continuous = 0

        for i, (column_name, column) in enumerate(table.items()):
            if column["type"] == "cat":
                self.cat_encoders.append(CatEncoder())
                self.cat_index.append(i)
                self.categories.append(column["card"])
            if column["type"] == "num" and not column.get("key"):
                self.num_index.append(i)
                self.num_continuous += 1
            if column["type"] == "foreign_key":
                self.num_continuous += 1
                self.key_index.append(i)
                self.keys.append((column["table"], column["column"]))
            if column.get("key"):
                self.primary_key = i

    def process(self, line):
        res = []
        for i, data in enumerate(line):
            if i in self.num_index:
                res.append(float(data) if data != "None" else 0.)
            elif i in self.cat_index:
                for j, k in enumerate(self.cat_index):
                    if i == k:
                        res.append(self.cat_encoders[j].get_index(data))
                        break
            elif i in self.key_index:
                res.append(data)
            else:
                res.append(data)
        return res

    def to_torch(self, res):
        x_categ = []
        x_numer = []

        TableInitEmbed.FOREIGN_KEYS[self.name] = {}

        for i, row in enumerate(res):
            numeric = [row[x] for x in self.num_index]
            numeric.extend([1.0 for x in self.key_index])

            TableInitEmbed.FOREIGN_KEYS[self.name][row[self.primary_key]] = i

            x_categ.append([row[x] for x, encode in zip(self.cat_index, self.cat_encoders)])
            x_numer.append(numeric)

        return torch.tensor(x_numer, requires_grad=False), torch.tensor(x_categ, requires_grad=False)

    def add_foreign_keys(self, res):
        idxs = [[] for _ in self.key_index]

        for i, row in enumerate(res):
            for k, (j, (table_name, _)) in enumerate(zip(self.key_index, self.keys)):
                idxs[k].append(TableInitEmbed.FOREIGN_KEYS[table_name][row[j]])

        return tuple(torch.tensor(idx, dtype=torch.long, requires_grad=False) for idx in idxs)


def get_tables(dataset_name: str, out_table: str, label_index: int = -1):
    with open(f"./dataset/{dataset_name}/schema.json", mode="r") as fp:
        schema = json.load(fp)

    label_enc = CatEncoder()
    tables = []
    labels = []
    raw_data = []

    for table_name in schema.keys():
        table = TableInitEmbed(table_name, schema[table_name])
        tables.append(table)

        with open(f"./dataset/{dataset_name}/{table_name}.csv", mode="r") as fp:
            processed = []
            for line in fp.readlines():
                line_data = line.strip().split(",")
                if table_name == out_table:
                    label = torch.tensor(label_enc.get_index(line_data[label_index]))
                    labels.append(label)
                processed.append(table.process(line_data))
            raw_data.append(processed)

    input_data = [[*table.to_torch(raw)] for table, raw in zip(tables, raw_data)]
    foreign_indices = [table.add_foreign_keys(raw) for table, raw in zip(tables, raw_data)]

    return tables, input_data, foreign_indices, torch.stack(labels)


def train():
    dataset_name = "financial"
    target_table = "loan"

    print("Getting tables")
    tables, input_data, foreign_indices, labels = get_tables(dataset_name, target_table)

    epochs = 1000
    print(len(tables), len(input_data))

    dim = 64
    dim_out = 4
    heads = 4
    layers = 4
    attn_dropout = 0.1
    ff_dropout = 0.1

    transformer = DBTransformer(dim, dim_out, heads, attn_dropout, ff_dropout, tables, layers)
    optim = torch.optim.Adam(transformer.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(epochs):
        x = transformer(input_data, foreign_indices, target_table)

        loss = loss_fn(x, labels)
        print("Loss: ", loss)

        optim.zero_grad(set_to_none=True)
        loss.backward()

        optim.step()

        if i % 10 == 0:
            s = 0
            lab = torch.argmax(x, dim=1)
            for r, p in zip(labels, lab):
                if r == p:
                    s += 1
            print(s, len(labels))

train()
