from typing import List

import torch


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
