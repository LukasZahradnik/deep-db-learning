import torch
from torch_geometric.nn import HeteroConv, Linear, GCNConv, SAGEConv

from db_transformer.transformer import Embedder


class DBGNNLayer(torch.nn.Module):
    def __init__(self, dim, metadata, aggr):
        super().__init__()

        convs = {m: SAGEConv(-1, dim, aggr=aggr, add_self_loops=False) for m in metadata[1]}
        self.hetero = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        return self.hetero(x_dict, edge_index_dict)


class DBGNN(torch.nn.Module):
    def __init__(self, dim, out_channels, proj_dim, layers, metadata, schema, aggr="mean"):
        super().__init__()

        self.out_lin = Linear(proj_dim, out_channels, bias=True)
        self.out_channels = out_channels
        self.dim = dim
        self.layers = layers

        self.schema = schema
        self.embedder = Embedder(dim, schema)

        self.gnn_layers = torch.nn.ModuleList([
            DBGNNLayer(proj_dim, metadata, aggr)
            for _ in range(layers)
        ])

        self.table_projection = torch.nn.ModuleDict({
            key: torch.nn.Linear(len(self.embedder.get_embed_cols(key)) * self.dim, proj_dim)
            for key in self.schema.keys()
        })

        self.proj_dim = proj_dim


    def forward(self, x_dict, edge_index_dict):
        new_x_dict = {}
        for table_name, value in x_dict.items():
            if table_name == "_target_table":
                new_x_dict[table_name] = torch.ones((value.shape[0], self.proj_dim))

            if table_name not in self.schema:
                continue

            val = self.embedder(table_name, value)
            val = val.reshape((val.shape[0], val.shape[1] * val.shape[2]))

            new_x_dict[table_name] = self.table_projection[table_name](val)

        x = new_x_dict

        for layer in self.gnn_layers:
            x = layer(x, edge_index_dict)

        x = x["_target_table"]
        x = x[:, :]
        x = self.out_lin(x)

        return torch.softmax(x, dim=1)
