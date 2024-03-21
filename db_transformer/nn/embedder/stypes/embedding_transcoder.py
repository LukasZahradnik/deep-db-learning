from typing import Any

import torch

from torch_frame import stype, NAStrategy
from torch_frame.data import StatType
from torch_frame.nn import StypeEncoder
from torch_frame.typing import TensorData, MultiEmbeddingTensor


class EmbeddingTranscoder(StypeEncoder):
    supported_stypes = {stype.embedding}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
        in_channels: int | None = None,
    ):
        self.in_channels = in_channels
        self.na_strategy = None
        super().__init__(out_channels, stats_list, stype, post_module, na_strategy)

    def init_modules(self):
        super().init_modules()
        self.lin = (
            torch.nn.Linear(in_features=self.in_channels, out_features=self.out_channels)
            if self.in_channels is not None and self.out_channels is not None
            else None
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def encode_forward(
        self,
        feat: TensorData,
        col_names: list[str] | None = None,
    ) -> torch.Tensor:
        if isinstance(feat, MultiEmbeddingTensor):
            # TODO: Fix this as it is not general
            feat = feat.values.reshape((feat.num_rows, feat.num_cols, -1))

        if self.lin is not None:
            return self.lin(feat)
        return feat
