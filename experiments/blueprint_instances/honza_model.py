from typing import List, Dict, Any, Tuple

import torch

from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.nn import conv
from torch_geometric.data import HeteroData

from torch_frame import stype, NAStrategy
from torch_frame.nn import encoder
from torch_frame.data import StatType

from db_transformer.data import CTUDatasetDefault, TaskType
from db_transformer.nn import EmbeddingTranscoder, BlueprintModel


def create_honza_model(
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

    is_classification = defaults.task == TaskType.CLASSIFICATION

    output_dim = (
        len(col_stats_dict[target[0]][target[1]][StatType.COUNT][0])
        if is_classification
        else 1
    )

    def get_decoder(out_gnn):
        mlp_dims = config.get("mlp_dims", [])

        mlp_dims = [out_gnn, *mlp_dims, output_dim]

        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            if i > 0:
                if batch_norm:
                    mlp_layers.append(torch.nn.BatchNorm1d(mlp_dims[i]))
                mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(torch.nn.Linear(mlp_dims[i], mlp_dims[i + 1]))

        return torch.nn.Sequential(*mlp_layers)

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
            # stype.embedding: EmbeddingTranscoder(),
        },
        positional_encoding=False,
        per_column_embedding=False,
        num_gnn_layers=gnn_layers,
        table_transform=lambda i, node, cols: (
            torch.nn.Identity()
            if i == 0
            else torch.nn.Sequential(
                (
                    torch.nn.BatchNorm1d(
                        len(cols) * int(embed_dim / 2**i),
                    )
                    if batch_norm
                    else torch.nn.Identity()
                ),
                torch.nn.ReLU(),
            )
        ),
        table_transform_unique=True,
        table_combination=lambda i, edge, cols: conv.SAGEConv(
            (
                len(cols[0]) * int(embed_dim / 2**i),
                len(cols[1]) * int(embed_dim / 2**i),
            ),
            len(cols[1]) * int(embed_dim / 2 ** (i + 1)),
            aggr="sum",
        ),
        table_combination_unique=True,
        decoder_aggregation=torch.nn.Identity(),
        decoder=lambda cols: get_decoder(len(cols) * int(embed_dim / 2**gnn_layers)),
        output_activation=torch.nn.Softmax(dim=-1) if is_classification else None,
        positional_encoding_dropout=0.0,
        table_transform_dropout=0.0,
        table_combination_dropout=0.0,
        table_transform_residual=False,
        table_combination_residual=False,
        table_transform_norm=False,
        table_combination_norm=False,
    )
