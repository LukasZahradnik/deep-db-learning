import torch

from db_transformer.schema.columns import CategoricalColumnDef

from .column_convertor import ColumnConvertor


__all__ = [
    'CatConvertor',
]


class CatConvertor(ColumnConvertor[CategoricalColumnDef]):
    def __init__(self) -> None:
        super().__init__()

        self.classes = 0
        self.value_to_idx = {None: 0}  # index zero reserved for None values

    def create(self, column_def: CategoricalColumnDef):
        self.classes = column_def.card

    def forward(self, value) -> torch.Tensor:
        if value not in self.value_to_idx:
            self.value_to_idx[value] = len(self.value_to_idx)
        index = self.value_to_idx[value]

        if index >= self.classes:
            raise ValueError(
                f"Found at least {len(self.value_to_idx)} unique values "
                f"(expected cardinality: {self.classes}). "
                f"The values are: {list(self.value_to_idx.keys())}")

        return torch.tensor([index])

    def to_one_hot(self, value) -> torch.Tensor:
        if value not in self.value_to_idx:
            self.value_to_idx[value] = len(self.value_to_idx)
        index = self.value_to_idx[value]

        one_hot = torch.zeros((self.classes,))
        one_hot[index] = 1.0

        return one_hot

