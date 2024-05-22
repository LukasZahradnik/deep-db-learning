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
        dropout: float = 0.0,
        aggr: Optional[Union[str, List[str], Aggregation]] = "attn",
        per_column_embedding: bool = True,
    ):
        if aggr == "attn":
            aggr = AttentionAggregation(embed_dim)
        super().__init__(aggr=aggr, node_dim=-3 if per_column_embedding else -2)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, query=x, key=x, value=x)

    def message(self, query_i, key_j, value_j):
        x, _ = self.attn(query_i, key_j, value_j)
        return x
