import torch

from db_transformer.ndata.convertor.convertor import BaseConvertor
from db_transformer.schema.columns import NumericColumnDef


class NumConvertor(BaseConvertor):
    def __init__(self, dim: int):
        super().__init__(dim)

        self.num_params = torch.nn.ParameterDict()

    def create(self, table_name: str, column_name: str, column: NumericColumnDef):
        name = f"{table_name}/{column_name}"
        if name not in self.num_params:
            self.num_params[name] = torch.nn.Parameter(torch.randn(1, self.dim))

    def forward(self, value, table_name: str, column_name: str, column: NumericColumnDef) -> torch.Tensor:
        name = f"{table_name}/{column_name}"

        return self.num_params[name] * torch.tensor([float(value)])
