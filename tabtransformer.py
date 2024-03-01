from typing import Any, Dict, List

import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, HeteroConv, MLP
from torch_geometric.typing import EdgeType, NodeType

from sentence_transformers import SentenceTransformer

import torch_frame
from torch_frame.nn import TabTransformer
from torch_frame.config import TextEmbedderConfig
from torch_frame.utils import infer_df_stype
from torch_frame.data import TensorFrame

from relbench.external.nn import HeteroEncoder
from relbench.external.graph import make_pkey_fkey_graph, get_stype_proposal

from db_transformer.schema import Schema
from db_transformer.data.relbench.ctu_dataset import CTUDataset


# class TabTransformerGNN(MessagePassing):
#     def __init__(self, in_channels, ff_dim, num_heads, aggr="mean"):
#         super().__init__(aggr=aggr, node_dim=-3)

#         self.in_channels = in_channels

#         self.transformer = TabTransformer(in_channels, )

#         self.reset_parameters()


# class TabTransformerLayer(torch.nn.Module):
#     def __init__(self, dim, ff_dim, metadata, num_heads, schema, aggr):
#         super().__init__()

#         convs = {m: TabTransformerGNN(dim, ff_dim, num_heads, aggr) for m in metadata[1]}
#         self.hetero = HeteroConv(convs, aggr=aggr)

#     def forward(self, x_dict, edge_index_dict):
#         #  x_dict = {k: self.self_attn[k](x, x, x)[0] for k, x in x_dict.items()}

#         return self.hetero(x_dict, edge_index_dict)


class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=data.col_stats_dict,
        )

        self.head = MLP(
            channels,
            out_channels=out_channels,
            num_layers=1,
        )

    def forward(
        self,
        batch: HeteroData,
    ) -> torch.Tensor:
        x_dict = self.encoder(batch.tf_dict)

        return self.head(x_dict[entity_table])


if __name__ == "__main__":

    dataset = CTUDataset("Chess", force_remake=True)
    data = dataset.build_hetero_data()

    col_to_stype_dict = get_stype_proposal(dataset.db)

    # text_embedder_cfg = TextEmbedderConfig(
    #     text_embedder=GloveTextEmbedding(), batch_size=32
    # )

    # data = make_pkey_fkey_graph(dataset.db, col_to_stype_dict, text_embedder_cfg)

    # encoder = HeteroEncoder()

    TabTransformer()
