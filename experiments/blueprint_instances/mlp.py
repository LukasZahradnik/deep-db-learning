from typing import List, Dict, Any, Tuple

import torch

from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.nn import conv
from torch_geometric.data import HeteroData

from torch_frame import stype, NAStrategy
from torch_frame.nn import encoder
from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import (
    EmbeddingTranscoder,
    BlueprintModel,
    CrossAttentionConv,
    SelfAttention,
)

from .utils import get_decoder


def create_mlp_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, List[str]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    batch_norm = config.get("batch_norm", False)
    mlp_dims = config.get("mlp_dims", [])

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
        stype_encoder_dict={
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.LinearEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        },
        positional_encoding=False,
        per_column_embedding=True,
        num_gnn_layers=0,
        decoder_aggregation=lambda x: x.view(*x.shape[:-2], -1),
        decoder=lambda cols: get_decoder(
            len(cols) * embed_dim,
            output_dim,
            mlp_dims,
            batch_norm,
        ),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
    )
