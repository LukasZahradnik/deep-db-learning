from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data.data import NodeType

from db_transformer.data.utils.column_def_matching import ColumnDefMatcherLike, find_value_for_matcher, get_matcher
from db_transformer.schema.schema import ColumnDef

from .columns.column_embedder import ColumnEmbedder
from .columns.identity_embedder import IdentityEmbedder

__ALL__ = ['SingleTableEmbedder', 'TableEmbedder']


class SingleTableEmbedder(torch.nn.Module):
    def __init__(self,
                 *column_embedders: Tuple[ColumnDefMatcherLike, Callable[[], ColumnEmbedder]],
                 dim: int,
                 column_defs: Sequence[ColumnDef],
                 column_names: Optional[List[str]] = None) -> None:
        super().__init__()

        self.column_defs = column_defs
        self.column_names = column_names
        self.dim = dim

        column_embedders = tuple((get_matcher(k), v) for k, v in column_embedders)

        self.embedders = torch.nn.ModuleList()

        for column_def in column_defs:
            provider = find_value_for_matcher(column_embedders, column_def)
            embedder = provider() if provider is not None else IdentityEmbedder()
            embedder.create(column_def)
            self.embedders.append(embedder)

    def _fix_dimensionality(self, vals: List[torch.Tensor]) -> List[torch.Tensor]:
        # ensure all values are of shape [..., 1, dim]
        vals = [
            v if v.shape[-2] == 1 else v.unsqueeze(-2)
            for v in vals
        ]

        return vals

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != len(self.embedders):
            raise ValueError(f"There are {len(self.embedders)} features expected for this embedder, "
                             f"but the input tensor has shape {list(input.shape)}, i.e. {input.shape[-1]} features.")

        if input.shape[-1] == 0:
            return input.unsqueeze(-1).repeat([*([1] * input.dim()), self.dim])

        vals = []

        for i in range(len(self.embedders)):
            val = torch.select(input, dim=-1, index=i).unsqueeze(-1)
            try:
                val = self.embedders[i](val)
            except Exception as e:
                if self.column_names is not None:
                    colname = '\'' + self.column_names[i] + '\' '
                else:
                    colname = ''

                raise RuntimeError(f"Failed to embed feature {colname}at index {i}: {self.column_defs[i]}") from e
            vals.append(val)

        vals = self._fix_dimensionality(vals)

        out = torch.concat(vals, dim=-2)
        return out


class TableEmbedder(torch.nn.Module):
    def __init__(self,
                 *column_embedders: Tuple[ColumnDefMatcherLike, Callable[[], ColumnEmbedder]],
                 dim: int,
                 column_defs: Dict[str, List[ColumnDef]],
                 column_names: Dict[str, List[str]]) -> None:
        super().__init__()

        self.table_embedders = torch.nn.ModuleDict({
            t: SingleTableEmbedder(*column_embedders, dim=dim, column_defs=column_defs_this, column_names=column_names[t])
            for t, column_defs_this in column_defs.items()
        })

    def forward(self, node_data: Dict[NodeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        out = {}

        for k in node_data:
            try:
                out[k] = self.table_embedders[k](node_data[k])
                if out[k].shape[1] == 0:
                    out[k] = torch.ones((out[k].shape[0], 1, out[k].shape[2]))
            except Exception as e:
                raise RuntimeError(f"Failed to embed table {k}") from e

        return out
