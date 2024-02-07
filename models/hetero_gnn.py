from typing import Dict, List, Literal

import torch
from torch_geometric.data.data import NodeType, EdgeType
from torch_geometric.nn import Sequential

from .layers import HeteroGNNLayer, NodeApplied

AggrType = Literal['sum', 'mean', 'min', 'max', 'cat']

class HeteroGNN(torch.nn.Module):
    def __init__(self, dims: List[int],
                 node_dims: Dict[NodeType, int],
                 out_dim: int,
                 node_types: List[NodeType],
                 edge_types: List[EdgeType],
                 aggr: AggrType,
                 batch_norm: bool,
                 ) -> None:
        super().__init__()

        the_dims = [*dims, out_dim]

        layers = []
        layers += [
            (HeteroGNNLayer(
                node_dims,
                the_dims[0],
                node_types=node_types,
                edge_types=edge_types,
                aggr=aggr,
                batch_norm=batch_norm), 'x, edge_index -> x'),
        ]

        for a, b in zip(the_dims[:-1], the_dims[1:]):
            layers += [
                NodeApplied(lambda nt: torch.nn.ReLU(inplace=True), node_types=node_types),
                (HeteroGNNLayer(a, b,
                                node_types=node_types,
                                edge_types=edge_types,
                                aggr=aggr,
                                batch_norm=batch_norm), 'x, edge_index -> x')
            ]

        self.layers = Sequential('x, edge_index', layers)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        x_dict = self.layers(x_dict, edge_index_dict)
        return x_dict
