from typing import Dict, List, Literal, Optional

import torch

from torch_frame import stype, NAStrategy
from torch_frame.nn import encoder

from db_transformer.nn import EmbeddingTranscoder


def get_encoder(
    type: Optional[
        Literal[
            "basic", "with_time", "with_embeddings", "all", "excelformer", "saint", "tabnet"
        ]
    ] = None
) -> Dict[stype, encoder.StypeEncoder]:

    if type == "basic" or type is None:
        return {
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.LinearEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        }
    if type == "excelformer" or type is None:
        return {
            stype.numerical: encoder.ExcelFormerEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        }
    if type == "saint" or type is None:
        return {
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.LinearEncoder(
                na_strategy=NAStrategy.MEAN, post_module=torch.nn.ReLU()
            ),
        }
    if type == "tabnet" or type is None:
        return {
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.StackEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        }
    if type == "with_time" or type is None:
        return {
            **get_encoder("basic"),
            stype.timestamp: encoder.TimestampEncoder(),
        }
    if type == "with_embeddings" or type is None:
        return {
            **get_encoder("basic"),
            stype.embedding: EmbeddingTranscoder(),
        }
    if type == "all" or type is None:
        return {
            **get_encoder("with_embeddings"),
            stype.timestamp: encoder.TimestampEncoder(),
        }
    raise ValueError(f"Unknown encoder type '{type}'")


def get_decoder(out_gnn: int, output_dim: int, mlp_dims: List[int] = [], batch_norm=False):

    mlp_dims = [out_gnn, *mlp_dims, output_dim]

    mlp_layers = []
    for i in range(len(mlp_dims) - 1):
        if i > 0:
            if batch_norm:
                mlp_layers.append(torch.nn.BatchNorm1d(mlp_dims[i]))
            mlp_layers.append(torch.nn.ReLU())
        mlp_layers.append(torch.nn.Linear(mlp_dims[i], mlp_dims[i + 1]))

    return torch.nn.Sequential(*mlp_layers)
