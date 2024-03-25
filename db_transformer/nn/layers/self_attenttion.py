from typing import Dict, List

import torch

from torch_geometric.data.data import NodeType


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, node_types: List[NodeType]) -> None:
        super().__init__()

        self.attn = torch.nn.ModuleDict(
            {
                node_type: torch.nn.MultiheadAttention(embed_dim, 1, batch_first=True)
                for node_type in node_types
            }
        )

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        out_dict = {}

        for node_type, x in x_dict.items():
            out_dict[node_type], _ = self.attn[node_type](x, x, x)

        return out_dict
