import torch
from torch_geometric.nn import HeteroConv, Linear, MessagePassing

from db_transformer.data.embedder import PerTypeEmbedder, NumEmbedder, CatEmbedder
from db_transformer.schema import OmitColumnDef, NumericColumnDef, CategoricalColumnDef, DateColumnDef, \
    DateTimeColumnDef


class TransformerGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, aggr="mean"):
        super().__init__(aggr=aggr, node_dim=-3)

        self.in_channels = in_channels

        self.lin = Linear(in_channels, in_channels, bias=True)
        self.transformer = torch.nn.TransformerEncoderLayer(self.in_channels, num_heads, dim_feedforward=64, batch_first=True)

        self.b_proj = torch.nn.Linear(in_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.b_proj.reset_parameters()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        x_j = self.b_proj(x_j)
        x_c = torch.concat((x_i, x_j), dim=1)

        x = self.transformer(x_c)
        x = x[:, :x_i.shape[1], :]

        return x


class DBTransformerLayer(torch.nn.Module):
    def __init__(self, dim, out_channels, metadata, num_heads):
        super().__init__()

        convs = {m: TransformerGNN(dim, out_channels, num_heads) for m in metadata[1]}
        self.hetero = HeteroConv(convs, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        return self.hetero(x_dict, edge_index_dict)


class Embedder(torch.nn.Module):
    def __init__(self, dim, schema):
        super().__init__()

        self.schema = schema
        self.embedder = PerTypeEmbedder(
            {
                OmitColumnDef: lambda: None,
                NumericColumnDef: lambda: NumEmbedder(dim),
                CategoricalColumnDef: lambda: CatEmbedder(dim),
                DateColumnDef: lambda: None,
                DateTimeColumnDef: lambda: None,
            }
        )

        self.dim = dim
        self.embedder.create(schema)

    def forward(self, table_name, value):
        embedded_cols = []

        for col_name, col in self.schema[table_name].columns.items():
            if self.embedder.has(table_name, col_name, col):
                embedded_cols.append((col_name, col))

        d = [
            self.embedder(value[:, i], table_name, col_name, col)
            for i, (col_name, col) in enumerate(embedded_cols)
        ]

        if not d:
            return torch.ones((value.shape[0], 1, self.dim))

        return torch.stack(d, dim=1)


class DBTransformer(torch.nn.Module):
    def __init__(self, dim, out_channels, layers, metadata, num_heads, out_table, schema):
        super().__init__()

        self.out_table = out_table
        self.out_lin = Linear(dim, out_channels, bias=True)
        self.out_channels = out_channels
        self.dim = dim
        self.layers = layers

        self.schema = schema
        self.embedder = Embedder(dim, schema)

        self.transformer_layers = torch.nn.ModuleList([
            DBTransformerLayer(dim, out_channels, metadata, num_heads)
            for _ in range(layers)
        ])


    def forward(self, x_dict, edge_index_dict):
        new_x_dict = {}
        for table_name, value in x_dict.items():
            if table_name == "_target_table":
                new_x_dict[table_name] = torch.ones((value.shape[0], 1, self.dim))

            if table_name not in self.schema:
                continue

            new_x_dict[table_name] = self.embedder(table_name, value)

        x = new_x_dict

        for layer in self.transformer_layers:
            x = layer(x, edge_index_dict)

        x = x["_target_table"]
        x = x[:, 0, :]
        x = self.out_lin(x)

        return torch.softmax(x, dim=1)
