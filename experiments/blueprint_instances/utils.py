from typing import Dict, List, Literal, Optional

import torch

from torch_frame import stype, NAStrategy
from torch_frame.nn import encoder


def get_encoder(
    type: Optional[
        Literal[
            "basic",
            "with_time",
            "with_embeddings",
            "all",
            "excelformer",
            "saint",
            "tabnet",
            "tabtransformer",
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
    if type == "excelformer":
        return {
            stype.numerical: encoder.ExcelFormerEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        }
    if type == "saint":
        return {
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.LinearEncoder(
                na_strategy=NAStrategy.MEAN, post_module=torch.nn.ReLU()
            ),
        }
    if type == "tabnet":
        return {
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.StackEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        }
    if type == "tabtransformer":
        return {
            stype.categorical: encoder.EmbeddingEncoder(
                na_strategy=NAStrategy.MOST_FREQUENT,
            ),
            stype.numerical: encoder.StackEncoder(
                na_strategy=NAStrategy.MEAN,
            ),
        }
    if type == "with_time":
        return {
            **get_encoder("basic"),
            stype.timestamp: encoder.TimestampEncoder(),
        }
    if type == "with_embeddings":
        return {
            **get_encoder("basic"),
            stype.embedding: encoder.LinearEmbeddingEncoder(),
        }
    if type == "all":
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
