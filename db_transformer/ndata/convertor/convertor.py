import torch

from db_transformer.schema.columns import _AttrsColumnDef


class BaseConvertor(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def create(self, table_name: str, column_name: str, column: _AttrsColumnDef):
        pass

    def forward(self, value, table_name: str, column_name: str, column: _AttrsColumnDef) -> torch.Tensor:
        pass
