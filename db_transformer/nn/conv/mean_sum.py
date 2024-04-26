from typing import List, Optional, Union

import torch

from torch_geometric.nn import MessagePassing, Aggregation


class MeanSumConv(MessagePassing):
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        per_column_embedding: bool = True,
    ):
        super().__init__(aggr=aggr, node_dim=-3 if per_column_embedding else -2)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor):
        return x_i + x_j.mean(dim=1).unsqueeze(1)
