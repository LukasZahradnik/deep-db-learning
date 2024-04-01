from typing import Any, Dict, List

import torch

from torch_geometric.typing import NodeType

from torch_frame import stype, TensorFrame, NAStrategy
from torch_frame.data import StatType
from torch_frame.nn import EmbeddingEncoder, ExcelFormer, ExcelFormerEncoder

__ALL__ = ["ExcelFormerEmbedder", "ExcelFormerTableEmbedder"]


class ExcelFormerTableEmbedder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[stype, List[str]],
        diam_dropout: float,
        aium_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()

        self.valid_stypes = [stype.categorical, stype.numerical]

        self.out_channels = out_channels

        self.embedder = ExcelFormer(
            in_channels=channels,
            out_channels=out_channels,
            num_cols=len(col_stats.keys()),
            num_layers=num_layers,
            num_heads=num_heads,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: ExcelFormerEncoder(
                    out_channels, na_strategy=NAStrategy.MEAN
                ),
                stype.categorical: EmbeddingEncoder(
                    out_channels,
                    na_strategy=NAStrategy.MOST_FREQUENT,
                ),
            },
            diam_dropout=diam_dropout,
            aium_dropout=aium_dropout,
            residual_dropout=residual_dropout,
        )
        self.no_valid_col = True
        for _stype in self.valid_stypes:
            if _stype in col_names_dict and len(col_names_dict[_stype]) > 0:
                self.no_valid_col = False

    def forward(self, tf: TensorFrame) -> torch.Tensor:
        if self.no_valid_col:
            return torch.ones((tf.num_rows, self.out_channels)).to(tf.device)
        return self.embedder(tf)


class ExcelFormerEmbedder(torch.nn.Module):
    def __init__(
        self,
        table_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        table_col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        diam_dropout: float,
    ) -> None:
        super().__init__()

        self.embedder = torch.nn.ModuleDict(
            {
                table_name: ExcelFormerTableEmbedder(
                    channels=embed_dim,
                    out_channels=embed_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    col_stats=table_col_stats[table_name],
                    col_names_dict=table_col_names_dict[table_name],
                    diam_dropout=diam_dropout,
                    aium_dropout=0,
                    residual_dropout=0,
                )
                for table_name in table_col_stats.keys()
            }
        )

    def forward(self, tf_dict: Dict[NodeType, TensorFrame]) -> Dict[str, torch.Tensor]:
        x_dict = {
            table_name: embedder(tf_dict[table_name])
            for table_name, embedder in self.embedder.items()
        }

        return x_dict
