import torch
from torch_geometric.nn import HeteroConv, Linear, MessagePassing

from db_transformer.data.embedder import TableEmbedder, NumEmbedder, CatEmbedder
from db_transformer.schema import OmitColumnDef, NumericColumnDef, CategoricalColumnDef, DateColumnDef, \
    DateTimeColumnDef


class TransformerGNN(MessagePassing):
    def __init__(self, in_channels, ff_dim, num_heads, aggr="mean"):
        super().__init__(aggr=aggr, node_dim=-3)

        self.in_channels = in_channels

        self.lin = Linear(in_channels, in_channels, bias=True)

        self.transformer = torch.nn.MultiheadAttention(self.in_channels, num_heads, batch_first=True)
        # self.transformer = torch.nn.TransformerEncoderLayer(self.in_channels, num_heads, dim_feedforward=ff_dim, batch_first=True)

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

        x, _ = self.transformer(x_i, x_c, x_c)
        # x = x[:, :x_i.shape[1], :]

        return x


class DBTransformerLayer(torch.nn.Module):
    def __init__(self, dim, ff_dim, metadata, num_heads, aggr):
        super().__init__()

        convs = {m: TransformerGNN(dim, ff_dim, num_heads, aggr) for m in metadata[1]}
        self.hetero = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        return self.hetero(x_dict, edge_index_dict)


class Embedder(torch.nn.Module):
    def __init__(self, dim, schema, column_defs, column_names, config):
        super().__init__()

        self.schema = schema
        self.embedder = TableEmbedder(
            (CategoricalColumnDef, lambda: CatEmbedder(dim=config.dim)),
            (NumericColumnDef, lambda: NumEmbedder(dim=config.dim)),
            dim=config.dim,
            column_defs=column_defs,
            column_names=column_names,
        )

        self.dim = dim
        self.embedder.create(schema)

    def get_embed_cols(self, table_name: str):
        return [
            (col_name, col)
            for col_name, col in self.schema[table_name].columns.items()
            if self.embedder.has(table_name, col_name, col)
        ]

    def forward(self, table_name, value):
        embedded_cols = self.get_embed_cols(table_name)

        d = [
            self.embedder(value[:, i], table_name, col_name, col)
            for i, (col_name, col) in enumerate(embedded_cols)
        ]

        if not d:
            return torch.ones((value.shape[0], 1, self.dim))

        return torch.stack(d, dim=1)


class DBTransformer(torch.nn.Module):
    def __init__(self, dim, out_channels, ff_dim, layers, metadata, num_heads, schema, column_defs, column_names, aggr="mean"):
        super().__init__()

        self.out_lin = Linear(dim, out_channels, bias=True)
        self.out_channels = out_channels
        self.dim = dim
        self.layers = layers

        self.schema = schema
        self.embedder = Embedder(dim, schema)

        self.transformer_layers = torch.nn.ModuleList([
            DBTransformerLayer(dim, ff_dim, metadata, num_heads, aggr)
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
