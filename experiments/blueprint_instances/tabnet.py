from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType

from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import BlueprintModel, TabNetEncoder, TabNetDecoder
from db_transformer.nn.conv.cross_attention import CrossAttentionConv

from .utils import get_decoder, get_encoder


def create_tabnet_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, List[str]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    num_layers = config.get("num_layers", 1)
    num_heads = config.get("num_heads", 1)
    residual = config.get("residual", False)
    dropout = config.get("dropout", 0.0)

    is_classification = defaults.task == TaskType.CLASSIFICATION

    output_dim = (
        len(col_stats_dict[target[0]][target[1]][StatType.COUNT][0])
        if is_classification
        else 1
    )

    return BlueprintModel(
        target=target,
        embed_dim=embed_dim,
        col_stats_per_table=col_stats_dict,
        col_names_dict_per_table=col_names_dict,
        edge_types=edge_types,
        stype_encoder_dict=get_encoder("tabnet"),
        positional_encoding=False,
        num_gnn_layers=gnn_layers,
        pre_combination=lambda i, node, cols: TabNetEncoder(
            channels=embed_dim,
            out_channels=embed_dim,
            num_cols=len(cols),
            num_layers=num_layers,
        ),
        table_combination=lambda i, edge, cols: CrossAttentionConv(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        ),
        # residual connection
        post_combination=(
            None if not residual else lambda i, node, cols: lambda x, x_next: x + x_next
        ),
        decoder_aggregation=lambda x: x.view(*x.shape[:-2], -1),
        decoder=lambda cols: get_decoder(embed_dim * len(cols), output_dim, [], False),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
    )
