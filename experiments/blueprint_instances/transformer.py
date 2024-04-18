from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType

from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import (
    BlueprintModel,
    CrossAttentionConv,
    SelfAttention,
)

from .utils import get_decoder, get_encoder


def create_transformer_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, List[str]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    batch_norm = config.get("batch_norm", False)
    mlp_dims = config.get("mlp_dims", [])
    num_heads = config.get("num_heads", 1)
    residual = config.get("residual", False)
    dropout = config.get("dropout", 0)

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
        stype_encoder_dict=get_encoder("basic"),
        positional_encoding=False,
        per_column_embedding=True,
        num_gnn_layers=gnn_layers,
        table_transform=lambda i, node, cols: torch.nn.Sequential(
            SelfAttention(embed_dim // 2**i, num_heads=num_heads),
            torch.nn.Linear(embed_dim // 2**i, embed_dim // 2 ** (i + 1)),
        ),
        table_transform_unique=True,
        table_combination=lambda i, edge, cols: CrossAttentionConv(
            embed_dim // 2 ** (i + 1), num_heads=num_heads
        ),
        table_combination_unique=True,
        decoder_aggregation=lambda x: x.view(*x.shape[:-2], -1),
        decoder=lambda cols: get_decoder(
            len(cols) * embed_dim // 2 ** max(gnn_layers, 0),
            output_dim,
            mlp_dims,
            batch_norm,
        ),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
        table_transform_dropout=dropout,
        table_combination_dropout=dropout,
        table_transform_residual=False,
        table_combination_residual=residual,
        table_transform_norm=batch_norm,
        table_combination_norm=batch_norm,
    )
