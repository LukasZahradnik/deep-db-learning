import torch
from torch_geometric.nn import HeteroConv, Linear, MessagePassing


class TransformerGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__(aggr="mean")

        self.in_channels = in_channels

        self.lin = Linear(in_channels, in_channels, bias=True)
        self.transformer = torch.nn.TransformerEncoderLayer(self.in_channels, num_heads, dim_feedforward=64)

        self.b_proj = torch.nn.Linear(in_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.b_proj.reset_parameters()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        i_shape = x_i.shape
        x_i = x_i.reshape((i_shape[0], int(i_shape[1] / self.in_channels), self.in_channels))

        shape = x_j.shape
        x_j = x_j.reshape((shape[0], int(shape[1] / self.in_channels), self.in_channels))

        x_j = self.b_proj(x_j)
        x_c = torch.concat((x_i, x_j), dim=1)

        x = self.transformer(x_c)

        x = x[:, :x_i.shape[1], :]

        x = x.reshape(i_shape)

        return x

class DBTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, num_heads, out_table):
        super().__init__()

        self.out_table = out_table
        self.out_lin = Linear(in_channels, out_channels, bias=True)
        self.out_channels = out_channels
        self.in_channels = in_channels

        convs = {m: TransformerGNN(in_channels, out_channels, num_heads) for m in metadata[1]}
        self.hetero = HeteroConv(convs, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x = self.hetero(x_dict, edge_index_dict)
        x = x[self.out_table]

        x = x.reshape((x.shape[0], int(x.shape[1] / self.in_channels), self.in_channels))
        x = self.out_lin(x)
        x = torch.sum(x, dim=1)

        return torch.softmax(x, dim=1)
