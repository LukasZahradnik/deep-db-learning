from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union
import warnings

import torch

from torch_geometric.typing import NodeType

from torch_frame import stype, TensorFrame, NAStrategy
from torch_frame.data import StatType
from torch_frame.nn import (
    StypeWiseFeatureEncoder,
    StypeEncoder,
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)


class TableEmbedder(StypeWiseFeatureEncoder):
    def __init__(
        self,
        embed_dim: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[stype, list[str]],
        stype_encoder_dict: dict[stype, StypeEncoder],
    ) -> None:
        super().__init__(embed_dim, col_stats, col_names_dict, stype_encoder_dict)
        for stype in col_names_dict:
            if stype not in stype_encoder_dict:
                warnings.warn(
                    f"Encoder for stype.{stype} is not present. Columns of this stype will be ignored."
                )
        self.active_stypes = list(stype_encoder_dict.keys())

    def forward(self, tf: TensorFrame) -> Tuple[torch.Tensor, List[str]]:
        r"""Encode :class:`TensorFrame` object into a tuple
        :obj:`(x, col_names)`.

        Args:
            tf (:class:`torch_frame.TensorFrame`): Input :class:`TensorFrame`
                object.

        Returns:
            (torch.Tensor, List[str]): A tuple of an output column-wise
                :class:`torch.Tensor` of shape
                :obj:`[batch_size, num_cols, hidden_channels]` and a list of
                column names of  :obj:`x`. The length needs to be
                :obj:`num_cols`.
        """
        all_col_names = []
        xs = []
        for _stype in tf.stypes:
            # Skip if the stype encoder is not defined.
            if _stype not in self.active_stypes:
                continue
            feat = tf.feat_dict[_stype]
            col_names = self.col_names_dict[_stype]
            x = self.encoder_dict[_stype.value](feat, col_names)
            xs.append(x)
            all_col_names.extend(col_names)
        x = torch.cat(xs, dim=1)
        return x, all_col_names


class DBEmbedder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: Union[int, Dict[NodeType, int]],
        col_stats_per_table: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_names_dict_per_table: Dict[NodeType, Dict[stype, List[str]]],
        stype_embedder_dict: Dict[stype, StypeEncoder] = None,
        return_cols: bool = True,
    ) -> None:
        super().__init__()

        self.return_cols = return_cols

        self.table_embedders = torch.nn.ModuleDict()
        for table_name in col_stats_per_table:
            table_embed_dim = (
                embed_dim[table_name] if isinstance(embed_dim, dict) else embed_dim
            )

            table_stype_embedder_dict = deepcopy(stype_embedder_dict)
            if table_stype_embedder_dict is None:
                table_stype_embedder_dict = {
                    stype.categorical: EmbeddingEncoder(
                        na_strategy=NAStrategy.MOST_FREQUENT,
                    ),
                    stype.numerical: LinearEncoder(
                        na_strategy=NAStrategy.MEAN,
                    ),
                    stype.timestamp: TimestampEncoder(),
                }

            self.table_embedders[table_name] = TableEmbedder(
                embed_dim=table_embed_dim,
                col_stats=col_stats_per_table[table_name],
                col_names_dict=col_names_dict_per_table[table_name],
                stype_encoder_dict=table_stype_embedder_dict,
            )

    def forward(self, tf_dict: Dict[NodeType, TensorFrame]) -> Dict[NodeType, torch.Tensor]:
        x_dict = {}
        cols_dict = {}
        for table_name, tf in tf_dict.items():
            x, cols = self.table_embedders[table_name](tf)
            x_dict[table_name] = x
            cols_dict[table_name] = cols

        if not self.return_cols:
            return x_dict

        return x_dict, cols_dict
