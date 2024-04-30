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
    MeanAddConv,
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
        post_embedder: Optional[
            Union[torch.nn.Module, Callable[[NodeType, List[str]], torch.nn.Module]]
        ] = None,
        positional_encoding: bool = True,
        positional_encoding_dropout: float = 0,
        num_gnn_layers: int = 1,
        pre_combination: Optional[
            Union[torch.nn.Module, Callable[[int, NodeType, List[str]], torch.nn.Module]]
        ] = None,
        table_combination: Optional[
            Union[
                torch.nn.Module,
                Callable[[int, EdgeType, Tuple[List[str], List[str]]], torch.nn.Module],
            ]
        ] = None,
        post_combination: Optional[
            Union[torch.nn.Module, Callable[[int, NodeType, List[str]], torch.nn.Module]]
        ] = None,
        decoder_aggregation: AggrFunction = torch.nn.Identity(),
        pretrain_decoder: Optional[
            Union[torch.nn.Module, Callable[[NodeType, List[str]], torch.nn.Module]]
        ] = None,
        decoder: Optional[
            Union[torch.nn.Module, Callable[[List[str]], torch.nn.Module]]
        ] = None,
        output_activation: Optional[torch.nn.Module] = None,
        pretrain_swap_prob: float = 0.2,
    ):
        super().__init__()

        self.target_table = target[0]
        self.target_col = target[1]
        self.node_types = list(col_names_dict_per_table.keys())
        self.edge_types = edge_types
        self.pretrain_swap_prob = max(min(pretrain_swap_prob, 1.0), 0.0)

        self.embedder = DBEmbedder(
            embed_dim=embed_dim,
            col_stats_dict=col_stats_per_table,
            col_names_dict_per_table=col_names_dict_per_table,
            stype_encoder_dict=stype_encoder_dict,
        )

        if positional_encoding:
            self.positional_encoding = NodeApplied(
                lambda _: PositionalEncoding(embed_dim, positional_encoding_dropout),
                self.node_types,
            )
        else:
            self.positional_encoding = None

        if post_embedder is not None:
            self.post_embedder = NodeApplied(
                lambda node: (
                    post_embedder
                    if isinstance(post_embedder, torch.nn.Module)
                    else post_embedder(node, self.embedder.active_cols_dict[node])
                ),
                self.node_types,
            )
        else:
            self.post_embedder = None

        layers: List[Tuple[str, torch.nn.Module]] = []

        if table_combination is None:
            table_combination = MeanAddConv()

        def create_pre_combination_transform(node):
            if isinstance(pre_combination, torch.nn.Module):
                return pre_combination
            else:
                return pre_combination(i, node, self.embedder.active_cols_dict[node])

        def create_table_combination(edge_type):
            if isinstance(table_combination, torch.nn.Module):
                return table_combination
            else:
                return table_combination(
                    i,
                    edge_type,
                    (
                        self.embedder.active_cols_dict[edge_type[0]],
                        self.embedder.active_cols_dict[edge_type[2]],
                    ),
                )

        def create_post_combination_transform(node):
            if isinstance(post_combination, torch.nn.Module):
                return post_combination
            else:
                return post_combination(i, node, self.embedder.active_cols_dict[node])

        for i in range(num_gnn_layers):
            if pre_combination is not None:
                layers.append(
                    (
                        NodeApplied(
                            create_pre_combination_transform,
                            self.node_types,
                        ),
                        f"x_dict_{i} -> x_dict_{i}",
                    )
                )

            if edge_types is None or len(edge_types) == 0:
                layers.append((torch.nn.Identity(), f"x_dict_{i} -> x_dict_{i+1}"))
                continue

            convs = {
                edge_type: create_table_combination(edge_type)
                for edge_type in self.edge_types
            }

            layers.append((HeteroConv(convs), f"x_dict_{i}, edge_dict -> x_dict_{i+1}"))

            if post_combination is not None:
                layers.append(
                    (
                        NodeApplied(
                            create_post_combination_transform,
                            self.node_types,
                            dynamic_args=True,
                        ),
                        f"x_dict_{i}, x_dict_{i+1} -> x_dict_{i+1}",
                    )
                )

        if len(layers) == 0:
            layers.append((torch.nn.Identity(), "x_dict_0 -> x_dict"))
        self.hetero_gnn = Sequential("x_dict_0, edge_dict", layers)

        self.decoder_aggregation = decoder_aggregation

        if pretrain_decoder is not None:
            self.pretrain_decoder = NodeApplied(
                lambda node: (
                    pretrain_decoder
                    if isinstance(decoder, torch.nn.Module)
                    else pretrain_decoder(node, self.embedder.active_cols_dict[node])
                ),
                self.node_types,
            )
        else:
            self.pretrain_decoder = None

        self.decoder = torch.nn.Sequential()
        if decoder is not None:
            self.decoder.append(
                decoder
                if isinstance(decoder, torch.nn.Module)
                else decoder(self.embedder.active_cols_dict[self.target_table])
            )

        if output_activation is not None:
            self.decoder.append(output_activation)

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        edge_dict: Dict[EdgeType, torch.Tensor],
        pretrain: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[Dict[NodeType, torch.Tensor], Dict[NodeType, torch.Tensor]]
    ]:
        tf_dict, edge_dict = self._remove_empty(tf_dict, edge_dict)

        x_dict = self.embedder(tf_dict)

        if pretrain:
            assert self.pretrain_decoder is not None
            x_dict, mask_dict = self._pretrain_masked_swap(x_dict)

        if self.positional_encoding is not None:
            x_dict = self.positional_encoding(x_dict)

        if self.post_embedder is not None:
            x_dict = self.post_embedder(x_dict)

        x_dict = self.hetero_gnn(x_dict, edge_dict)

        if pretrain:
            return (
                self.pretrain_decoder(
                    {node: self.decoder_aggregation(x) for node, x in x_dict.items()}
                ),
                mask_dict,
            )

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

    def _pretrain_masked_swap(self, x_dict: Dict[NodeType, torch.Tensor]):
        mask_dict: Dict[NodeType, torch.Tensor] = {}
        x_swap_dict: Dict[NodeType, torch.Tensor] = {}
        for node, x in x_dict.items():
            b, c, d = x.shape
            # Get indicies for embeddings swap
            idx = torch.randperm(b * c)
            # Check for indicies that stayed the same
            not_same_mask = torch.logical_not(torch.eq(idx, torch.arange(0, b * c)))
            # Create mask for embeddings from Bernoulli distribution
            mask_dict[node] = torch.logical_and(
                torch.bernoulli(torch.ones(b, c) * self.pretrain_swap_prob),
                not_same_mask.view(b, c),
            ).float()
            print(mask_dict[node].shape)
            # Create swaped tensor
            x_swap = x.view(b * c, d).index_select(dim=0, index=idx).view(b, c, d)
            # Select from swap tensor only when mask is active
            x_swap_dict[node] = x.masked_scatter(
                mask_dict[node].bool().unsqueeze(-1).expand(b, c, d), x_swap
            )

        return x_swap_dict, mask_dict
