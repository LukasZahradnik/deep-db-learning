import torch
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear


class MyHeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super().__init__()

        convs = {m: SAGEConv(in_channels, out_channels) for m in metadata[1]}
        self.convs = HeteroConv(convs, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        return self.convs(x_dict, edge_index_dict)
