from typing import Any, Dict, List

import torch

from torch_geometric.typing import NodeType

from torch_frame import stype, TensorFrame
from torch_frame.data import StatType
from torch_frame.nn import Trompt

__ALL__ = ["TromptEmbedder", "TromptTableEmbedder"]


class TromptTableEmbedder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_prompts: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[stype, List[str]],
    ) -> None:
        super().__init__()
        self.valid_stypes = [stype.categorical, stype.numerical]

        self.out_channels = out_channels

        self.col_names_dict = col_names_dict

        self.embedder = Trompt(
            channels=channels,
            out_channels=out_channels,
            num_prompts=num_prompts,
            num_layers=num_layers,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
        )
        self.no_valid_col = True
        for _stype in self.valid_stypes:
            if _stype in col_names_dict and len(col_names_dict[_stype]) > 0:
                self.no_valid_col = False

    def forward(self, tf: TensorFrame) -> torch.Tensor:
        if self.no_valid_col:
            return torch.ones((tf.num_rows, self.out_channels)).to(tf.device)
        return self.embedder(tf)


class TromptEmbedder(torch.nn.Module):
    def __init__(
        self,
        table_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        table_col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
        embed_dim: int,
        num_layers: int,
        num_prompts: int,
    ) -> None:
        super().__init__()

        self.embedder = torch.nn.ModuleDict(
            {
                table_name: TromptTableEmbedder(
                    channels=embed_dim,
                    out_channels=embed_dim,
                    num_prompts=num_prompts,
                    num_layers=num_layers,
                    col_stats=table_col_stats[table_name],
                    col_names_dict=table_col_names_dict[table_name],
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