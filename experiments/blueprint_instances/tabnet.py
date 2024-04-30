from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType

from torch_frame import stype
from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import (
    BlueprintModel,
    TabNetEncoder,
    MeanAddConv,
)

from .utils import get_decoder, get_encoder


def create_tabnet_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    init_embed_dim = 16

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    mlp_dims = config.get("mlp_dims", [])

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
        embed_dim=init_embed_dim,
        col_stats_per_table=col_stats_dict,
        col_names_dict_per_table=col_names_dict,
        edge_types=edge_types,
        stype_encoder_dict=get_encoder("tabnet"),
        positional_encoding=False,
        num_gnn_layers=gnn_layers,
        pre_combination=lambda i, node, cols: TabNetEncoder(
            channels=init_embed_dim,
            out_channels=len(cols) * init_embed_dim,
            num_cols=len(cols),
            num_layers=num_layers,
            split_feat_channels=embed_dim,
            split_attn_channels=embed_dim,
        ),
        table_combination=lambda i, edge, cols: MeanAddConv(per_column_embedding=False),
        decoder=lambda cols: get_decoder(
            init_embed_dim * len(cols),
            output_dim,
            mlp_dims,
            False,
            out_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        ),
        positional_encoding_dropout=0.0,
    )
