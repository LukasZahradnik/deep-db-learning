from typing import Any, Dict, List

import torch

from torch_geometric.nn import conv
from torch_geometric.typing import EdgeType, NodeType

import torch_frame
from torch_frame.nn import TabTransformer


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

        self.transformers = torch.nn.ModuleDict(
            {
                table_name: TabTransformer(
                    channels=embed_dim,
                    out_channels=embed_dim,
                    num_layers=num_transformer_layers,
                    num_heads=num_transformer_heads,
                    encoder_pad_size=2,
                    attn_dropout=attn_dropout,
                    ffn_dropout=0,
                    col_stats=table_col_stats[table_name],
                    col_names_dict=table_col_names_dict[table_name],
                )
                for table_name in table_col_stats.keys()
            }
        )

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
            table_name: transformer(tf_dict[table_name])
            for table_name, transformer in self.transformers.items()
        }

        x_dict = self.conv(x_dict, edge_dict)

        x_target = x_dict[self.target_table]

        x_target = self.out_lin(x_target)
        return torch.softmax(x_target, dim=-1)
