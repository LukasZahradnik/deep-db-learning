import copy
from typing import Callable, List, Dict, Union, Optional, Any, Tuple

import torch

from torch_geometric.nn import Sequential, conv
from torch_geometric.nn.fx import Transformer
from torch_geometric.typing import EdgeType, NodeType

import torch_frame
from torch_frame import stype
from torch_frame.nn import StypeEncoder
from torch_frame.data import StatType

from db_transformer.nn import (
    CrossAttentionConv,
    DBEmbedder,
    NodeApplied,
    PositionalEncoding,
)


class BlueprintModel(torch.nn.Module):
    def __init__(
        self,
        target: Tuple[str, str],
        embed_dim: int,
        col_stats_per_table: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_names_dict_per_table: Dict[NodeType, Dict[stype, List[str]]],
        edge_types: Optional[List[EdgeType]] = None,
        stype_embedder_dict: Optional[Dict[torch_frame.stype, StypeEncoder]] = None,
        positional_encoding: Optional[bool] = True,
        positional_encoding_dropout: Optional[float] = 0.0,
        num_gnn_layers: Optional[int] = 1,
        table_transform: Optional[
            Union[torch.nn.Module, Callable[[int], torch.nn.Module]]
        ] = None,
        table_combination: Optional[
            Union[torch.nn.Module, Callable[[int, str], torch.nn.Module]]
        ] = None,
    ):
        super().__init__()

        self.target_table = target[0]
        self.target_col = target[1]
        self.node_types = list(col_stats_per_table.keys())
        print(self.node_types)
        self.edge_types = edge_types

        embedder_layers: List[Tuple[str, torch.nn.Module]] = []

        embedder_layers.append(
            (
                DBEmbedder(
                    embed_dim=embed_dim,
                    col_stats_per_table=col_stats_per_table,
                    col_names_dict_per_table=col_names_dict_per_table,
                    stype_embedder_dict=stype_embedder_dict,
                    return_cols=False,
                ),
                "tf_dict -> x_dict",
            )
        )

        if positional_encoding:
            embedder_layers.append(
                (
                    NodeApplied(
                        lambda _: PositionalEncoding(
                            embed_dim, positional_encoding_dropout
                        ),
                        self.node_types,
                    ),
                    "x_dict -> x_dict",
                )
            )

        self.embedder = Sequential("tf_dict", embedder_layers)

        if table_combination is None:
            table_combination = CrossAttentionConv(embed_dim, 4)
        is_combination_module = isinstance(table_combination, torch.nn.Module)

        layers: List[Tuple[str, torch.nn.Module]] = []
        for i in range(num_gnn_layers):
            if table_transform is not None:
                layers.append(
                    (
                        (
                            copy.deepcopy(table_transform)
                            if isinstance(table_transform, torch.nn.Module)
                            else table_transform(i)
                        ),
                        f"x_dict -> x_dict",
                    )
                )

            convs = {
                edge_type: (
                    copy.deepcopy(table_combination)
                    if is_combination_module
                    else table_combination(i, edge_type)
                )
                for edge_type in self.edge_types
            }

            layers.append((conv.HeteroConv(convs), f"x_dict, edge_dict -> x_dict"))

        self.hetero_gnn = Sequential("x_dict, edge_dict", layers)

        self.target_transform = torch.nn.Linear(
            embed_dim,
            len(col_stats_per_table[self.target_table][self.target_col][StatType.COUNT][0]),
        )

    def forward(
        self,
        tf_dict: Dict[str, torch_frame.TensorFrame],
        edge_dict: Dict[str, torch.Tensor],
    ):
        x_dict = self.embedder(tf_dict)

        x_dict = self.hetero_gnn(x_dict, edge_dict)

        x_target = x_dict[self.target_table]

        x_target = self.target_transform(x_target.sum(dim=-2))

        return torch.softmax(x_target, dim=-1)
