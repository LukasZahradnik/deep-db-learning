from typing import Any, Dict, List

import torch

from torch_geometric.nn import conv
from torch_geometric.typing import EdgeType, NodeType

import torch_frame
from torch_frame.nn import TabTransformer


class TabTransformerWrapper(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        attn_dropout: float,
        ffn_dropout: float,
        col_stats: Dict[str, Dict[torch_frame.data.StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ) -> None:
        super().__init__()

        self.tabtransformer = TabTransformer(
            channels=channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            encoder_pad_size=2,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
        )

    def forward(self, tf: torch_frame.TensorFrame) -> torch.Tensor:
        return self.tabtransformer(tf)


class TabTransformerGNN(torch.nn.Module):
    def __init__(
        self,
        target_table: str,
        table_col_stats: Dict[NodeType, Dict[str, Dict[torch_frame.data.StatType, Any]]],
        table_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        edge_types: List[EdgeType],
        embed_dim: int,
        num_transformer_layers: int,
        num_transformer_heads: int,
        attn_dropout: float,
        out_dim: int,
    ) -> None:
        super().__init__()

        self.target_table = target_table

        self.out_dim = out_dim

        self.embedder = torch.nn.ModuleDict(
            {
                table_name: TabTransformerWrapper(
                    channels=embed_dim,
                    out_channels=embed_dim,
                    num_layers=num_transformer_layers,
                    num_heads=num_transformer_heads,
                    attn_dropout=attn_dropout,
                    ffn_dropout=0,
                    col_stats=table_col_stats[table_name],
                    col_names_dict=table_col_names_dict[table_name],
                )
                for table_name in table_col_stats.keys()
            }
        )

        # TODO: This can be also a cross-attention layer
        convs = {
            edge_type: conv.SAGEConv(in_channels=embed_dim, out_channels=embed_dim)
            for edge_type in edge_types
        }
        self.conv = conv.HeteroConv(convs)

        self.out_lin = torch.nn.Linear(embed_dim, out_dim)

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        edge_dict: Dict[EdgeType, torch.Tensor],
    ) -> torch.Tensor:
        x_dict = {
            table_name: embedder(tf_dict[table_name])
            for table_name, embedder in self.embedder.items()
        }

        x_dict = self.conv(x_dict, edge_dict)

        x_target = x_dict[self.target_table]

        x_target = self.out_lin(x_target)
        return torch.softmax(x_target, dim=-1)
