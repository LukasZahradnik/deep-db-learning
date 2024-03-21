import math
from typing import Optional

import torch

from torch_geometric.nn import Aggregation
from torch_geometric.utils import softmax, scatter


class AttentionAggregation(Aggregation):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(
        self,
        x: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ):
        if dim == -2:
            x = x.unsqueeze(-2)

        x_mean = scatter(x, index, dim, dim_size, reduce="mean")

        key = x_mean.index_select(dim=-3, index=index)
        query = x
        value = x

        alpha = (query @ key.permute((0, 2, 1))) / math.sqrt(self.channels)
        alpha = softmax(alpha, index, ptr, dim_size)

        x_attn = alpha @ value

        if dim == -2:
            x_attn = x_attn.squeeze(-2)

        return self.reduce(x_attn, index, ptr, dim_size, dim, reduce="sum")
