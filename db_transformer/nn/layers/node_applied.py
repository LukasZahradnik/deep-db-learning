from typing import Callable, Dict, List

import torch
from torch_geometric.data.data import NodeType

class NodeApplied(torch.nn.Module):
    def __init__(self,
                 factory: Callable[[NodeType], torch.nn.Module],
                 node_types: List[NodeType],
                 ) -> None:
        super().__init__()

        self.node_types = node_types
        self.items = torch.nn.ModuleDict({
            k: factory(k)
            for k in node_types
        })

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        out_dict: Dict[NodeType, torch.Tensor] = {}

        for k in self.node_types:
            out_dict[k] = self.items[k](x_dict[k])

        return out_dict
