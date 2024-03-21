import torch
from torch_geometric.nn import HeteroConv, Linear, MessagePassing

from db_transformer.nn.embedder import MultiTableEmbedder, NumEmbedder, CatEmbedder
from db_transformer.schema import NumericColumnDef, CategoricalColumnDef


class TransformerGNN(MessagePassing):
    def __init__(self, in_channels, ff_dim, num_heads, aggr="mean"):
        super().__init__(aggr=aggr, node_dim=-3)

        self.in_channels = in_channels

        # self.lin = Linear(in_channels, in_channels, bias=True)

        self.transformer = torch.nn.MultiheadAttention(
            self.in_channels, num_heads, batch_first=True
        )
        # self.transformer = torch.nn.TransformerEncoderLayer(self.in_channels, num_heads, dim_feedforward=ff_dim, batch_first=True)

        # self.b_proj = torch.nn.Linear(in_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin.reset_parameters()
        # self.b_proj.reset_parameters()
        pass

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_j = self.b_proj(x_j)
        x_c = torch.concat((x_i, x_j), dim=1)
        x, _ = self.transformer(x_i, x_c, x_c)
        # x = x[:, :x_i.shape[1], :]

        return x


class DBTransformerLayer(torch.nn.Module):
    def __init__(self, dim, ff_dim, metadata, num_heads, schema, aggr):
        super().__init__()

        if False:
            self.self_attn = torch.nn.ModuleDict(
                {
                    key: torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
                    for key in schema.keys()
                }
            )

        convs = {m: TransformerGNN(dim, ff_dim, num_heads, aggr) for m in metadata[1]}
        self.hetero = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        #  x_dict = {k: self.self_attn[k](x, x, x)[0] for k, x in x_dict.items()}

        return self.hetero(x_dict, edge_index_dict)


class DBTransformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        out_channels,
        ff_dim,
        layers,
        metadata,
        num_heads,
        schema,
        column_defs,
        column_names,
        config,
        target_table,
        aggr="mean",
    ):
        super().__init__()

        self.out_lin = Linear(dim, out_channels, bias=True)
        self.out_channels = out_channels
        self.dim = dim
        self.layers = layers

        self.target_table = target_table
        self.schema = schema
        self.embedder = MultiTableEmbedder(
            (CategoricalColumnDef, lambda: CatEmbedder(dim=config.dim)),
            (NumericColumnDef, lambda: NumEmbedder(dim=config.dim)),
            dim=config.dim,
            column_defs=column_defs,
            column_names=column_names,
        )

        self.transformer_layers = torch.nn.ModuleList(
            [
                DBTransformerLayer(dim, ff_dim, metadata, num_heads, schema, aggr)
                for _ in range(layers)
            ]
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.embedder(x_dict)

        for layer in self.transformer_layers:
            x_dict = layer(x_dict, edge_index_dict)

        x = x_dict[self.target_table]
        x = x[:, 0, :]
        x = self.out_lin(x)

        return torch.softmax(x, dim=1)
