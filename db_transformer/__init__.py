from typing import List

import torch
import torch_geometric
from einops import rearrange
from torch_geometric.data import HeteroData

from db_transformer.embed import ColumnEmbedder
from db_transformer.transformer import SimpleTableTransformer
from db_transformer.data import TableData


class DBTransformer(torch.nn.Module):
    def __init__(self, metadata, dim: int, dim_out: int, heads: int, attn_dropout: float, ff_dropout: float, tables, layers: int):
        super().__init__()

        self.tables = tables
        self.dim = dim
        self.embedder = [ColumnEmbedder(table.categories, table.num_continuous, dim) for table in tables]

        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleList([
                SimpleTableTransformer(
                    dim=dim,
                    dim_out=dim,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    table=table
                ) for table in tables
            ]) for _ in range(layers)
        ])

        self.message_passing = torch.nn.ModuleList([
            torch_geometric.nn.to_hetero(torch_geometric.nn.GraphSAGE(dim, dim, 1), metadata, aggr="mean")
            for _ in range(layers)
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
                return torch.zeros((max(len(tdata.x_cat), len(tdata.x_numer)), len(tdata.x_keys), self.dim))
            return None

        xs = {
            table.name: [get_init_keys(table), embedder(table_data.x_cat, table_data.x_numer)]
            for table_data, embedder, table in zip(table_data, self.embedder, self.tables)
        }

        for layer, message_passing in zip(self.layers, self.message_passing):
            xs = {
                table.name: model(*xs[table.name])
                for model, table in zip(layer, self.tables)
            }

            for table, x in xs.items():
                table_inst = table_data[table_to_int[table]]
                hetero_data[table].x = x[0][:, 0, :].clone()

                for key_index, (table_name, column) in zip(table_inst.key_index, table_inst.keys):
                    key = f"{table}_{table_name}_{column}"
                    hetero_data[key].x = x[0][:, 0, :] #.clone()

            out = message_passing(hetero_data.x_dict, hetero_data.edge_index_dict)

            for table, x in xs.items():
                table_inst = table_data[table_to_int[table]]

                if table in out.keys():
                    newx = [out[table]]
                else:
                    newx = [x[0][:, 0]]

                for i, (table_name, column) in enumerate(table_inst.keys):
                    key = f"{table}_{table_name}_{column}"
                    newx.append(out[key])

                newx = [rearrange(xx, "b n -> b 1 n") for xx in newx]
                x[0] = torch.cat(newx, dim=1)


        x = xs[output_table][0][:, 0, :]
        x = self.to_logits(x)

        return x
