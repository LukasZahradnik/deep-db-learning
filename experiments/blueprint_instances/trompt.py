from typing import List, Dict, Any

import torch

from torch_geometric.typing import NodeType, EdgeType

from torch_frame import NAStrategy, stype
from torch_frame.data import StatType
from torch_frame.nn import encoder

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import BlueprintModel, MeanAddConv, TromptEncoder, TromptDecoder


def create_trompt_model(
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    target = defaults.target

    embed_dim = config.get("embed_dim", 64)
    gnn_layers = config.get("gnn_layers", 1)
    num_trompt_layers = 6

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
                post_module=torch.nn.LayerNorm([embed_dim]),
            ),
            stype.numerical: encoder.LinearEncoder(
                na_strategy=NAStrategy.MEAN,
                post_module=torch.nn.Sequential(
                    torch.nn.ReLU(), torch.nn.LayerNorm([embed_dim])
                ),
            ),
        },
        positional_encoding=False,
        post_embedder=lambda node, cols: (
            TromptEncoder(
                channels=embed_dim,
                num_cols=len(cols),
                num_prompts=embed_dim,
                num_layers=num_trompt_layers,
            )
        ),
        num_gnn_layers=gnn_layers,
        table_combination=lambda i, edge, cols: MeanAddConv(),
        decoder=lambda cols: TromptDecoder(
            embed_dim,
            output_dim,
            num_prompts=embed_dim,
            num_encoder_layers=num_trompt_layers,
        ),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
    )
