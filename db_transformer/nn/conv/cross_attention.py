from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, Aggregation

from db_transformer.nn.aggr.attention import AttentionAggregation


class CrossAttentionConv(MessagePassing):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        aggr: Optional[Union[str, List[str], Aggregation]] = None,
        per_column_embedding: bool = True,
    ):
        if aggr is None:
            aggr = AttentionAggregation(embed_dim)
        super().__init__(aggr=aggr, node_dim=-3 if per_column_embedding else -2)

        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, query=x, key=x, value=x)

    def message(self, query_i, key_j, value_j):
        # key = torch.concat((key_i, key_j), dim=-2)
        # value = torch.concat((value_i, value_j), dim=-2)
        x, _ = self.attn(query_i, key_j, value_j)
        return x
