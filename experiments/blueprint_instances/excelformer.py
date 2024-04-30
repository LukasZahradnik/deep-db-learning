from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType

from torch_frame import stype
from torch_frame.data import StatType
from torch_frame.nn import ExcelFormerConv, ExcelFormerDecoder

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import BlueprintModel, CrossAttentionConv

from .utils import get_encoder


def create_excelformer_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    num_layers = config.get("num_layers", 6)
    num_heads = config.get("num_heads", 1)
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
        stype_encoder_dict=get_encoder("excelformer"),
        positional_encoding=False,
        num_gnn_layers=gnn_layers,
        pre_combination=lambda i, node, cols: torch.nn.Sequential(
            *[
                ExcelFormerConv(
                    embed_dim,
                    len(cols),
                    num_heads,
                    dropout,
                    dropout,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        ),
        table_combination=lambda i, edge, cols: CrossAttentionConv(
            embed_dim, num_heads=num_heads
        ),
        decoder=lambda cols: ExcelFormerDecoder(embed_dim, output_dim, len(cols)),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
    )
