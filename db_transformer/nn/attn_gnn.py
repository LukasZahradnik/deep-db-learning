import torch
from torch_geometric.nn import HeteroConv, Linear, SAGEConv

from db_transformer.nn.embedder import TableEmbedder, CatEmbedder, NumEmbedder
from db_transformer.schema.columns import CategoricalColumnDef, NumericColumnDef


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


class AttnDBGNNLayer(torch.nn.Module):
    def __init__(self, dim, metadata, aggr, device):
        super().__init__()

        convs = {
            m: SAGEConv(-1, dim, aggr=aggr, add_self_loops=False, device=device)
            for m in metadata[1]
        }

        self.attns = torch.nn.ModuleDict(
            {m: torch.nn.MultiheadAttention(dim, 1, device=device) for m in metadata[0]}
        )

        self.hetero = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        xs = {}
        for m, x in x_dict.items():
            tmp, _ = self.attns[m](x, x, x)
            xs[m] = tmp[:, 0]

            if len(xs[m].shape) == 1:
                xs[m] = xs[m].unsqueeze(dim=1)

        xs = self.hetero(xs, edge_index_dict)

        new_dict = {}

        for m, x in x_dict.items():
            if len(x_dict[m].shape) == 2:
                new_dict[m] = xs[m]
            else:
                new_dict[m] = torch.concat(
                    (xs[m].unsqueeze(dim=1), x_dict[m][:, 1:]), dim=1
                )
        return new_dict


class AttnDBGNN(torch.nn.Module):
    def __init__(
        self,
        dim,
        out_channels,
        proj_dim,
        layers,
        metadata,
        schema,
        aggr="mean",
        device=None,
    ):
        super().__init__()

        self.out_lin = Linear(proj_dim, out_channels, bias=True)
        self.out_channels = out_channels
        self.dim = dim
        self.layers = layers

        self.schema = schema
        self.embedder = Embedder(dim, schema, device=device)

        self.gnn_layers = torch.nn.ModuleList(
            [AttnDBGNNLayer(proj_dim, metadata, aggr, device=device) for _ in range(layers)]
        )

        # self.table_projection = torch.nn.ModuleDict({
        #     key: torch.nn.Linear(max(len(self.embedder.get_embed_cols(key)), 1) * self.dim + self.dim, proj_dim, device=device)
        #     for key in self.schema.keys()
        # })

        self.proj_dim = proj_dim

    def forward(self, x_dict, edge_index_dict):
        new_x_dict = {}
        for table_name, value in x_dict.items():
            if table_name == "_target_table":
                new_x_dict[table_name] = torch.ones((value.shape[0], self.proj_dim))

            if table_name not in self.schema:
                continue

            val = self.embedder(table_name, value)
            val = torch.concat((torch.ones(val.shape[0], 1, val.shape[2]), val), dim=1)
            # val = val.reshape((val.shape[0], val.shape[1] * val.shape[2]))

            new_x_dict[table_name] = val  # self.table_projection[table_name](val)

        x = new_x_dict

        for layer in self.gnn_layers:
            x = layer(x, edge_index_dict)

        x = x["_target_table"]
        x = x[:, :]
        x = self.out_lin(x)

        return torch.softmax(x, dim=1)
