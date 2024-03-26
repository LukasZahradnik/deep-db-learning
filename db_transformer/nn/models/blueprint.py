import copy
from typing import Callable, List, Dict, Union, Optional, Any, Tuple

import torch
from torch.nn import functional as F

from torch_geometric.nn import Sequential, conv
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
        num_gnn_layers: Optional[int] = 1,
        table_transform: Optional[
            Union[torch.nn.Module, Callable[[int], torch.nn.Module]]
        ] = None,
        table_combination: Optional[
            Union[torch.nn.Module, Callable[[int, str], torch.nn.Module]]
        ] = None,
        positional_encoding_dropout: Optional[float] = 0.0,
        table_transform_dropout: Optional[float] = 0.0,
        table_combination_dropout: Optional[float] = 0.0,
        table_transform_residual: Optional[bool] = False,
        table_combination_residual: Optional[bool] = False,
        table_transform_norm: Optional[bool] = True,
        table_combination_norm: Optional[bool] = True,
    ):
        super().__init__()

        self.target_table = target[0]
        self.target_col = target[1]
        self.node_types = list(col_stats_per_table.keys())
        self.edge_types = edge_types
        self.embedded_stypes = (
            list(stype_embedder_dict.keys())
            if stype_embedder_dict is not None
            else [stype.categorical, stype.numerical]
        )
        self.embedded_cols: Dict[NodeType, List[str]] = {}
        for node, cols_dict in col_names_dict_per_table.items():
            self.embedded_cols[node] = []
            for stype, cols in cols_dict.items():
                if stype not in self.embedded_stypes:
                    continue
                self.embedded_cols[node].extend(cols)

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
                        f"x_dict_{i} -> x_dict",
                    )
                )
                if table_transform_dropout > 0:
                    layers.append(
                        (
                            NodeApplied(
                                lambda _: torch.nn.Dropout(table_transform_dropout),
                                self.node_types,
                            ),
                            "x_dict -> x_dict",
                        )
                    )
                if table_transform_residual:
                    layers.append(
                        (
                            NodeApplied(
                                lambda _: lambda x1, x2: x1 + x2,
                                self.node_types,
                                learnable=False,
                            ),
                            f"x_dict_{i}, x_dict -> x_dict",
                        )
                    )
                if table_transform_norm:
                    layers.append(
                        (
                            NodeApplied(
                                lambda node: torch.nn.BatchNorm1d(
                                    len(self.embedded_cols[node])
                                ),
                                self.node_types,
                            ),
                            "x_dict -> x_dict",
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

            layers.append((conv.HeteroConv(convs), f"x_dict, edge_dict -> x_dict_{i+1}"))

            if table_combination_dropout > 0:
                layers.append(
                    (
                        NodeApplied(
                            lambda _: torch.nn.Dropout(table_combination_dropout),
                            self.node_types,
                        ),
                        f"x_dict_{i+1} -> x_dict_{i+1}",
                    )
                )

            if table_combination_residual:
                layers.append(
                    (
                        NodeApplied(
                            lambda _: lambda x1, x2: x1 + x2,
                            self.node_types,
                            learnable=False,
                        ),
                        f"x_dict_{i+1}, x_dict -> x_dict_{i+1}",
                    )
                )

            if table_combination_norm:
                layers.append(
                    (
                        NodeApplied(
                            lambda node: torch.nn.BatchNorm1d(
                                len(self.embedded_cols[node])
                            ),
                            self.node_types,
                        ),
                        f"x_dict_{i+1} -> x_dict_{i+1}",
                    )
                )

        self.hetero_gnn = Sequential("x_dict_0, edge_dict", layers)

        self.out_transform = torch.nn.Linear(
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

        x_target = self.out_transform(x_target.sum(dim=-2))

        return torch.softmax(x_target, dim=-1)
