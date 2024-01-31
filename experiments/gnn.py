import torch
from torch_geometric.nn import HeteroConv, Linear, GCNConv, SAGEConv, GATConv, GATv2Conv

from db_transformer.transformer import Embedder
from db_transformer.data.embedder.embedders import SingleTableEmbedder, TableEmbedder
from db_transformer.data.embedder import CatEmbedder, NumEmbedder
from db_transformer.schema.columns import CategoricalColumnDef, NumericColumnDef
from db_transformer.schema.schema import ColumnDef, Schema


class DBGNNLayer(torch.nn.Module):
    def __init__(self, dim, metadata, aggr, schema):
        super().__init__()

        self.table_projection = torch.nn.ModuleDict({
            key: torch.nn.Sequential(
                torch.nn.Linear(dim, dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(dim // 2, dim),
            )
            for key in schema.keys()
        })

        convs = {m: GATv2Conv(-1, dim, aggr=aggr, add_self_loops=False) for m in metadata[1]}
        self.hetero = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        torch.use_deterministic_algorithms(False)
        x_dict = {k: self.table_projection[k](x) for k, x in x_dict.items()}
        x = self.hetero(x_dict, edge_index_dict)

        torch.use_deterministic_algorithms(True)
        return x


class DBGNN(torch.nn.Module):
    def __init__(self, dim, out_channels, proj_dim, layers, metadata, schema, column_defs, column_names, config, target_table, aggr="mean"):
        super().__init__()

        self.target_table = target_table

        self.out_lin = Linear(proj_dim, out_channels, bias=True)
        self.out_channels = out_channels
        self.dim = dim
        self.layers = layers

        self.schema = schema
        self.embedder = TableEmbedder(
            (CategoricalColumnDef, lambda: CatEmbedder(dim=config.dim)),
            (NumericColumnDef, lambda: NumEmbedder(dim=config.dim)),
            dim=config.dim,
            column_defs=column_defs,
            column_names=column_names,
        )

        self.gnn_layers = torch.nn.ModuleList([
            DBGNNLayer(proj_dim, metadata, aggr, schema)
            for _ in range(layers)
        ])

        self.table_projection = torch.nn.ModuleDict({
            key: torch.nn.Linear(max(len(column_defs[key]), 1) * self.dim, proj_dim)
            for key in self.schema.keys()
        })

        self.proj_dim = proj_dim


    def forward(self, x_dict, edge_index_dict):
        x_dict = self.embedder(x_dict)
        x_dict = {k: self.table_projection[k](x.view(*x.shape[:-2], -1)) for k, x in x_dict.items()}
        
        for layer in self.gnn_layers:
            x_dict = layer(x_dict, edge_index_dict)

        x = x_dict[self.target_table]
        x = self.out_lin(x)

        return torch.softmax(x, dim=1)
