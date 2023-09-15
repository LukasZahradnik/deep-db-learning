import torch

from .column_embedder import ColumnEmbedder

__all__ = [
    'NumEmbedder',
]


class NumEmbedder(ColumnEmbedder):
    """Passes features (or a feature) through a linear layer.

    Input tensor: [..., num_features] -> output: [..., num_features, dim]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.embedding = torch.nn.Linear(1, self.dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        squeezed = False

        if value.shape[-1] != 1:
            squeezed = True
            value = value.unsqueeze(dim=-1)

        out = self.embedding(value)

        if not squeezed:
            out = out.unsqueeze(-2)

        return out
