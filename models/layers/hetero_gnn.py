from typing import Dict, List, Mapping, Union, Literal

import torch
from torch_geometric.data.data import NodeType, EdgeType
from torch_geometric.nn import BatchNorm, HeteroConv, SAGEConv

from .node_applied import NodeApplied

AggrType = Literal['sum', 'mean', 'min', 'max', 'cat']

class HeteroGNNLayer(torch.nn.Module):
    def __init__(self,
                 dim: Union[Dict[NodeType, int], int],
                 out_dim: int,
                 node_types: List[NodeType],
                 edge_types: List[EdgeType],
                 aggr: AggrType,
                 batch_norm: bool,
                 ):
        super().__init__()

        if isinstance(dim, Mapping):
            convs = {et: SAGEConv((dim[et[0]], dim[et[2]]), out_dim,
                                  aggr=aggr, add_self_loops=False) for et in edge_types}
        else:
            convs = {et: SAGEConv(dim, out_dim,
                                  aggr=aggr, add_self_loops=False) for et in edge_types}
        self.hetero = HeteroConv(convs, aggr=aggr)
        self.norm = NodeApplied(lambda nt: BatchNorm(out_dim), node_types) if batch_norm else None
        self._in_dim = dim
        self._out_dim = out_dim

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        out = self.hetero(x_dict, edge_index_dict)
        if self.norm is not None:
            out = self.norm(out)
        return out

    def extra_repr(self) -> str:
        in_dim_repr = "{various}" if not isinstance(self._in_dim, int) else self._in_dim

        return f"in={in_dim_repr}, out={self._out_dim}"