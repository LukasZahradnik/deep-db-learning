from typing import Dict, List, Literal, Union, Mapping

import torch
from torch_geometric.data.data import NodeType, EdgeType
from torch_geometric.nn import Sequential, BatchNorm, HeteroConv, SAGEConv

from db_transformer.nn.layers import NodeApplied

AggrType = Literal["sum", "mean", "min", "max", "cat"]


class HeteroGNNLayer(torch.nn.Module):
    def __init__(
        self,
        dim: Union[Dict[NodeType, int], int],
        out_dim: int,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        aggr: AggrType,
        batch_norm: bool,
    ):
        super().__init__()

        if isinstance(dim, Mapping):
            convs = {
                et: SAGEConv(
                    (dim[et[0]], dim[et[2]]), out_dim, aggr=aggr, add_self_loops=False
                )
                for et in edge_types
            }
        else:
            convs = {
                et: SAGEConv(dim, out_dim, aggr=aggr, add_self_loops=False)
                for et in edge_types
            }
        self.hetero = HeteroConv(convs, aggr=aggr)
        self.norm = (
            NodeApplied(lambda nt: BatchNorm(out_dim), node_types) if batch_norm else None
        )
        self._in_dim = dim
        self._out_dim = out_dim

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Dict[NodeType, torch.Tensor]:
        out = self.hetero(x_dict, edge_index_dict)
        if self.norm is not None:
            out = self.norm(out)
        return out

    def extra_repr(self) -> str:
        in_dim_repr = "{various}" if not isinstance(self._in_dim, int) else self._in_dim

        return f"in={in_dim_repr}, out={self._out_dim}"


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        dims: List[int],
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
            (
                HeteroGNNLayer(
                    node_dims,
                    the_dims[0],
                    node_types=node_types,
                    edge_types=edge_types,
                    aggr=aggr,
                    batch_norm=batch_norm,
                ),
                "x, edge_index -> x",
            ),
        ]

        for a, b in zip(the_dims[:-1], the_dims[1:]):
            layers += [
                NodeApplied(lambda nt: torch.nn.ReLU(inplace=True), node_types=node_types),
                (
                    HeteroGNNLayer(
                        a,
                        b,
                        node_types=node_types,
                        edge_types=edge_types,
                        aggr=aggr,
                        batch_norm=batch_norm,
                    ),
                    "x, edge_index -> x",
                ),
            ]

        self.layers = Sequential("x, edge_index", layers)

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Dict[NodeType, torch.Tensor]:
        x_dict = self.layers(x_dict, edge_index_dict)
        return x_dict
