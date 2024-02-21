import torch

from .column_embedder import ColumnEmbedder


__ALL__ = [
    "IdentityEmbedder",
]


class IdentityEmbedder(ColumnEmbedder):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value
