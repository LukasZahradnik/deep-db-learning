import torch

from db_transformer.schema.columns import ColumnDef


class BaseConvertor(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def create(self, table_name: str, column_name: str, column: ColumnDef):
        raise NotImplemented

    def forward(self, value, table_name: str, column_name: str, column: ColumnDef) -> torch.Tensor:
        raise NotImplemented
