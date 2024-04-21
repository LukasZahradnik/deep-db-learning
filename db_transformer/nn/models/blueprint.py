import copy
from typing import Callable, List, Dict, Union, Optional, Any, Tuple

import torch
from torch.nn import functional as F

from torch_geometric.nn import HeteroConv, Sequential
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

AggrFunction = Callable[[torch.Tensor], torch.Tensor]


class BlueprintModel(torch.nn.Module):
    def __init__(
        self,
        target: Tuple[str, str],
        embed_dim: int,
        col_stats_per_table: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_names_dict_per_table: Dict[NodeType, Dict[stype, List[str]]],
        edge_types: Optional[List[EdgeType]] = None,
        stype_encoder_dict: Optional[Dict[torch_frame.stype, StypeEncoder]] = None,
        positional_encoding: bool = True,
        positional_encoding_dropout: float = 0,
        per_column_embedding: bool = True,
        num_gnn_layers: int = 1,
        table_transform: Optional[
            Union[torch.nn.Module, Callable[[int, NodeType, List[str]], torch.nn.Module]]
        ] = None,
        table_transform_unique: bool = True,
        table_transform_dropout: Optional[float] = None,
        table_transform_residual: bool = False,
        table_transform_norm: bool = True,
        table_combination: Optional[
            Union[
                torch.nn.Module,
                Callable[[int, str, Tuple[List[str], List[str]]], torch.nn.Module],
            ]
        ] = None,
        table_combination_unique: bool = True,
        table_combination_dropout: Optional[float] = None,
        table_combination_residual: bool = False,
        table_combination_norm: bool = True,
        decoder_aggregation: AggrFunction = torch.nn.Identity(),
        decoder: Optional[
            Union[torch.nn.Module, Callable[[List[str]], torch.nn.Module]]
        ] = None,
        output_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.target_table = target[0]
        self.target_col = target[1]
        self.node_types = list(col_names_dict_per_table.keys())
        self.edge_types = edge_types
        self.embedded_stypes = (
            list(stype_encoder_dict.keys())
            if stype_encoder_dict is not None
            else [stype.categorical, stype.numerical]
        )
        self.embedded_cols: Dict[NodeType, List[str]] = {}
        for node, cols_dict in col_names_dict_per_table.items():
            self.embedded_cols[node] = []
            for stype, cols in cols_dict.items():
                if stype not in self.embedded_stypes:
                    continue
                self.embedded_cols[node].extend(cols)
            if len(self.embedded_cols[node]) == 0:
                self.embedded_cols[node] = ["__filler"]

        embedder_layers: List[Tuple[str, torch.nn.Module]] = []

        embedder_layers.append(
            (
                DBEmbedder(
                    embed_dim=embed_dim,
                    col_stats_per_table=col_stats_per_table,
                    col_names_dict_per_table=col_names_dict_per_table,
                    stype_encoder_dict=stype_encoder_dict,
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
        if not per_column_embedding:
            embedder_layers.append(
                (
                    NodeApplied(
                        lambda _: lambda x: x.view(*x.shape[:-2], -1),
                        self.node_types,
                    ),
                    "x_dict -> x_dict",
                )
            )

        self.embedder = Sequential("tf_dict", embedder_layers)

        layers: List[Tuple[str, torch.nn.Module]] = []

        def create_table_transform(node):
            if isinstance(table_transform, torch.nn.Module):
                return (
                    copy.deepcopy(table_transform)
                    if table_transform_unique
                    else table_transform
                )
            else:
                return table_transform(i, node, self.embedded_cols[node])

        def create_table_combination(edge_type):
            if is_combination_module:
                return (
                    copy.deepcopy(table_combination)
                    if table_combination_unique
                    else table_combination
                )
            else:
                return table_combination(
                    i,
                    edge_type,
                    (self.embedded_cols[edge_type[0]], self.embedded_cols[edge_type[2]]),
                )

        for i in range(num_gnn_layers):
            if table_transform is not None:
                layers.append(
                    (
                        NodeApplied(
                            create_table_transform,
                            self.node_types,
                        ),
                        f"x_dict_{i} -> x_dict",
                    )
                )
                if table_transform_dropout is None:
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
                                dynamic_args=True,
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

            if edge_types is None or len(edge_types) == 0:
                layers.append((torch.nn.Identity(), f"x_dict -> x_dict_{i+1}"))
                continue

            if table_combination is None:
                table_combination = CrossAttentionConv(embed_dim, 4)
            is_combination_module = isinstance(table_combination, torch.nn.Module)
            convs = {
                edge_type: create_table_combination(edge_type)
                for edge_type in self.edge_types
            }

            layers.append((HeteroConv(convs), f"x_dict, edge_dict -> x_dict_{i+1}"))

            if table_combination_dropout is None:
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
                            dynamic_args=True,
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

        if len(layers) == 0:
            layers.append((torch.nn.Identity(), "x_dict_0 -> x_dict"))
        self.hetero_gnn = Sequential("x_dict_0, edge_dict", layers)

        self.decoder_aggregation = decoder_aggregation

        self.decoder = torch.nn.Sequential()
        if decoder is not None:

            self.decoder.append(
                decoder
                if isinstance(decoder, torch.nn.Module)
                else decoder(self.embedded_cols[self.target_table])
            )

        if output_activation is not None:
            self.decoder.append(output_activation)

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        edge_dict: Dict[EdgeType, torch.Tensor],
    ):
        tf_dict, edge_dict = self._remove_empty(tf_dict, edge_dict)

        x_dict = self.embedder(tf_dict)

        x_dict = self.hetero_gnn(x_dict, edge_dict)

        x_target = x_dict[self.target_table]

        x_target = self.decoder_aggregation(x_target)

        x_target = self.decoder(x_target)

        return x_target

    def _remove_empty(
        cls,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        edge_dict: Dict[EdgeType, torch.Tensor],
    ) -> Tuple[Dict[NodeType, torch_frame.TensorFrame], Dict[EdgeType, torch.Tensor]]:
        out_tf_dict = {}
        out_edge_dict = {}
        for node, tf in tf_dict.items():
            if tf.num_rows > 0:
                out_tf_dict[node] = tf

        for edge, edge_index in edge_dict.items():
            (src, _, dst) = edge
            if src in out_tf_dict and dst in out_tf_dict:
                out_edge_dict[edge] = edge_index

        return out_tf_dict, out_edge_dict
