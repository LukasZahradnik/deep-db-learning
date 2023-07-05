from collections import defaultdict

import torch

from db_transformer.ndata.convertor.convertor import BaseConvertor
from db_transformer.schema.columns import CategoricalColumnDef


class CatConvertor(BaseConvertor):
    def __init__(self, dim: int):
        super().__init__(dim)

        self.cat_modules = torch.nn.ModuleDict()
        self.cat_to_int = defaultdict(dict)

    def create(self, table_name: str, column_name: str, column: CategoricalColumnDef):
        name = f"{table_name}/{column_name}"
        if name not in self.cat_modules:
            self.cat_modules[name] = torch.nn.Embedding(column.card, self.dim)

    def forward(self, value, table_name: str, column_name: str, column: CategoricalColumnDef) -> torch.Tensor:
        name = f"{table_name}/{column_name}"

        if value not in self.cat_to_int[name]:
            self.cat_to_int[name][value] = len(self.cat_to_int[name])
        index = self.cat_to_int[name][value]

        return self.cat_modules[name](torch.tensor(index)).unsqueeze(dim=0)
