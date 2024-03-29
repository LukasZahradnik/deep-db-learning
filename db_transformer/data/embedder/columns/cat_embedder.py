import torch

from db_transformer.schema.columns import CategoricalColumnDef

from .column_embedder import ColumnEmbedder

__all__ = [
    "CatEmbedder",
]


class CatEmbedder(ColumnEmbedder[CategoricalColumnDef]):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.embedding: torch.nn.Embedding

    def create(self, column_def: CategoricalColumnDef):
        self.embedding = torch.nn.Embedding(column_def.card, self.dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.embedding(value.long())
