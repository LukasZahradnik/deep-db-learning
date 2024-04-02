from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch_geometric.data.data import NodeType


class NodeApplied(torch.nn.Module):
    def __init__(
        self,
        factory: Callable[[NodeType], Union[torch.nn.Module, Callable[[Any], Any]]],
        node_types: List[NodeType],
        learnable: bool = True,
        dynamic_args: bool = False,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.dynamic_args = dynamic_args
        if learnable:
            self.node_layer_dict = torch.nn.ModuleDict({k: factory(k) for k in node_types})
        else:
            self.node_layer_dict = {k: factory(k) for k in node_types}

    def forward(self, x_dict: Dict[NodeType, Any], *argv) -> Dict[NodeType, Any]:
        out_dict: Dict[NodeType, Any] = {}

        if self.dynamic_args:
            input_dict = defaultdict(list)
            for k in self.node_types:
                input_dict[k].append(x_dict[k])
                for arg in argv:
                    if not isinstance(arg, dict):
                        input_dict[k].append(arg)
                    else:
                        input_dict[k].append(arg[k])

            for k in self.node_types:
                out_dict[k] = self.node_layer_dict[k](*input_dict[k])
        else:
            for k in self.node_types:
                out_dict[k] = self.node_layer_dict[k](x_dict[k])

        return out_dict
