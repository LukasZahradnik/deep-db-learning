import torch

from db_transformer.data.embedder import TableEmbedder, NumEmbedder, CatEmbedder
from db_transformer.schema import NumericColumnDef, CategoricalColumnDef


class Embedder(torch.nn.Module):
    def __init__(self, dim, schema, column_defs, column_names, config):
        super().__init__()

        self.schema = schema
        self.embedder = TableEmbedder(
            (CategoricalColumnDef, lambda: CatEmbedder(dim=config.dim)),
            (NumericColumnDef, lambda: NumEmbedder(dim=config.dim)),
            dim=config.dim,
            column_defs=column_defs,
            column_names=column_names,
        )

        self.dim = dim
        self.embedder.create(schema)

    def get_embed_cols(self, table_name: str):
        return [
            (col_name, col)
            for col_name, col in self.schema[table_name].columns.items()
            if self.embedder.has(table_name, col_name, col)
        ]

    def forward(self, table_name, value):
        embedded_cols = self.get_embed_cols(table_name)

        d = [
            self.embedder(value[:, i], table_name, col_name, col)
            for i, (col_name, col) in enumerate(embedded_cols)
        ]

        if not d:
            return torch.ones((value.shape[0], 1, self.dim))

        return torch.stack(d, dim=1)
