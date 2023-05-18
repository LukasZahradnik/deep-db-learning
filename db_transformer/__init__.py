from typing import List

import torch
import torch_geometric
from einops import rearrange
from torch_geometric.data import HeteroData

from db_transformer.embed import ColumnEmbedder
from db_transformer.transformer import SimpleTableTransformer
from db_transformer.data import TableData


class MyMLP(torch.nn.Module):
    def __init__(self, *, dim):
        super().__init__()
        self.dim = dim

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, keys, x):
        if x is not None:
            x = torch.cat((keys, x), dim=1)
        else:
            x = keys
        shape = x.shape
        x = rearrange(x, "a b c -> a (b c)")
        x = self.seq(x)

        x = x.reshape(shape)
        x[:, 0, :] = torch.sum(x, dim=1)

        return [x[:, :keys.shape[1]], x[:, keys.shape[1]:]]


class DBTransformer(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, transformer_func, gnn_func, tables: List[TableData], layers: int):
        super().__init__()

        self.tables: List[TableData] = tables
        self.dim = dim

        self.embedder = [ColumnEmbedder(table.categories, table.num_continuous, dim) for table in tables]

        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleList([transformer_func(dim, table) for table in tables]) for _ in range(layers)
        ])

        self.message_passing = torch.nn.ModuleList([
            gnn_func(dim) for _ in range(layers)
        ])

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim_out)
        )

    def forward(self, table_data: List[TableData], hetero_data: HeteroData, output_table: str):
        table_to_int = {}
        for i, table in enumerate(self.tables):
            table_to_int[table.name] = i

        def get_init_keys(tdata):
            if tdata.x_keys:
                n = len(tdata.x_cat) if tdata.x_cat is not None else 0
                m = len(tdata.x_numer) if tdata.x_numer is not None else 0

                return torch.zeros((max(n, m), len(tdata.x_keys), self.dim))
            return None

        xs = {
            table.name: [get_init_keys(tt), embedder(tt.x_cat, tt.x_numer)]
            for tt, embedder, table in zip(table_data, self.embedder, self.tables) if not tt.empty
        }

        for layer, message_passing in zip(self.layers, self.message_passing):
            xs = {
                table.name: model(*xs[table.name])
                for model, table in zip(layer, table_data) if not table.empty
            }

            for table, x in xs.items():
                table_inst = table_data[table_to_int[table]]
                hetero_data[table].x = x[0][:, 0, :]

                # print(table, hetero_data[table].x)
                # for key_index, (table_name, column) in zip(table_inst.key_index, table_inst.keys):
                #    hetero_data[table_name].x = x[0][:, 0, :]

            for k in hetero_data.edge_index_dict.keys():
                if hetero_data[k].edge_index.shape == (2, 0):
                    del hetero_data[k]

            out = message_passing(hetero_data.x_dict, hetero_data.edge_index_dict)

            for table, x in xs.items():
                table_inst = table_data[table_to_int[table]]

                if table in out.keys():
                    newx = [out[table]]
                else:
                    newx = [x[0][:, 0]]

                for i, (table_name, column) in enumerate(table_inst.keys):
                    key = f"{table}_{table_name}_{column}"
                    if out[key]:
                        newx.append(out[key])
                    else:
                        newx.append(x[0][:, i + 1])

                newx = [rearrange(xx, "b n -> b 1 n") for xx in newx]
                x[0] = torch.cat(newx, dim=1)

        x = xs[output_table][0][:, 0, :]
        x = self.to_logits(x)

        return x
