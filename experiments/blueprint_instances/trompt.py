from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType

from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import BlueprintModel, CrossAttentionConv, Trompt

from .utils import get_decoder, get_encoder


def create_trompt_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, List[str]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    encoder = config.get("encoder", "basic")
    gnn_layers = config.get("gnn_layers", 1)
    batch_norm = config.get("batch_norm", False)
    mlp_dims = config.get("mlp_dims", [])
    num_heads = config.get("num_heads", 1)

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
        stype_encoder_dict=get_encoder(encoder),
        positional_encoding=False,
        num_gnn_layers=gnn_layers,
        post_embedder=lambda node, cols: (
            Trompt(
                embed_dim,
                embed_dim,
                num_cols=len(cols),
                num_prompts=embed_dim,
            )
        ),
        table_combination=lambda i, edge, cols: CrossAttentionConv(
            embed_dim, num_heads=num_heads, per_column_embedding=False
        ),
        decoder=lambda cols: get_decoder(
            embed_dim,
            output_dim,
            mlp_dims,
            batch_norm,
        ),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
    )
