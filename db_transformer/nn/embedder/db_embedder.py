from copy import deepcopy
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Literal, Optional
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
)


class TableEmbedder(StypeWiseFeatureEncoder):
    def __init__(
        self,
        embed_dim: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[stype, List[str]],
        stype_encoder_dict: dict[stype, StypeEncoder],
    ) -> None:
        self.embed_dim = embed_dim
        super().__init__(embed_dim, col_stats, col_names_dict, stype_encoder_dict)
        for stype in col_names_dict:
            if stype not in stype_encoder_dict:
                warnings.warn(
                    f"Encoder for stype.{stype} is not present. Columns of this stype will be ignored."
                )
        self._active_stypes = {
            _stype: col_names_dict.get(_stype, [])
            for _stype in stype_encoder_dict
            if _stype in col_names_dict
        }
        self._active_cols = [
            col for _stype in self._active_stypes for col in col_names_dict.get(_stype, [])
        ]

    @property
    def active_stypes(self) -> Dict[stype, List[str]]:
        return self._active_stypes

    @property
    def active_cols(self) -> List[str]:
        return self._active_cols

    def forward(self, tf: TensorFrame) -> Tuple[torch.Tensor, List[str]]:
        r"""Encode :class:`TensorFrame` object into a tuple
        :obj:`(x, col_names)`.

        Args:
            tf (:class:`torch_frame.TensorFrame`): Input :class:`TensorFrame`
                object.

        Returns:
            torch.Tensor: An output column-wise :class:`torch.Tensor` of shape
                :obj:`[batch_size, num_cols, hidden_channels]`
        """
        xs = []
        for _stype in tf.stypes:
            # Skip if the stype encoder is not defined.
            if _stype not in self.active_stypes:
                continue
            feat = tf.feat_dict[_stype]
            col_names = self.col_names_dict[_stype]
            x = self.encoder_dict[_stype.value](feat, col_names)
            xs.append(x)
        if len(xs) == 0:
            x = torch.zeros((tf.num_rows, 1, self.embed_dim), dtype=torch.float32).to(
                tf.device
            )
        else:
            x = torch.cat(xs, dim=1)
        return x


class DBEmbedder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_names_dict_per_table: Dict[NodeType, Dict[stype, List[str]]],
        stype_encoder_dict: Optional[Dict[stype, StypeEncoder]] = None,
    ) -> None:
        super().__init__()

        self.active_cols_dict: Dict[NodeType, List[str]] = {}
        self.active_stypes_dict: Dict[stype, List[str]] = {}

        self.embedders = torch.nn.ModuleDict()
        for table_name in col_names_dict_per_table:

            table_stype_embedder_dict = deepcopy(stype_encoder_dict)
            if table_stype_embedder_dict is None:
                table_stype_embedder_dict = {
                    stype.categorical: EmbeddingEncoder(
                        na_strategy=NAStrategy.MOST_FREQUENT,
                    ),
                    stype.numerical: LinearEncoder(
                        na_strategy=NAStrategy.MEAN,
                    ),
                }

            embedder = TableEmbedder(
                embed_dim=embed_dim,
                col_stats=col_stats_dict[table_name],
                col_names_dict=col_names_dict_per_table[table_name],
                stype_encoder_dict=table_stype_embedder_dict,
            )
            self.embedders[table_name] = embedder
            self.active_cols_dict[table_name] = embedder.active_cols
            if len(embedder.active_cols) == 0:
                self.active_cols_dict[table_name] = ["__filler"]
            self.active_stypes_dict[table_name] = embedder.active_stypes

    def forward(self, tf_dict: Dict[NodeType, TensorFrame]) -> Dict[NodeType, torch.Tensor]:
        x_dict = {}
        for table_name, tf in tf_dict.items():
            x_dict[table_name] = self.embedders[table_name](tf)

        return x_dict
