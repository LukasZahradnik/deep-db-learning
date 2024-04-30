from typing import Any, Dict

import torch

from torch_geometric.typing import NodeType
from torch_geometric.loader import DataLoader

import torch_frame


class CorruptLoader:
    def __init__(
        self,
        loader: DataLoader,
    ):
        self.loader = loader

    def __iter__(self) -> Any:
        for batch in self.loader:
            tf_dict: Dict[NodeType, torch_frame.TensorFrame] = batch.collect("tf")
            for node, tf in tf_dict.items():
                pass

            yield batch

    def _pretrain_masked_swap(self, tf_dict: Dict[NodeType, torch_frame.TensorFrame]):
        mask_dict: Dict[NodeType, torch.Tensor] = {}
        x_swap_dict: Dict[NodeType, torch.Tensor] = {}
        for node, tf in tf_dict.items():
            # b, c, d = x.shape
            # # Get indicies for embeddings swap
            # idx = torch.randperm(b * c)
            # # Check for indicies that stayed the same
            # not_same_mask = torch.logical_not(torch.eq(idx, torch.arange(0, b * c)))
            # # Create mask for embeddings from Bernoulli distribution
            # mask_dict[node] = torch.logical_and(
            #     torch.bernoulli(torch.ones(b, c) * self.pretrain_swap_prob),
            #     not_same_mask.view(b, c),
            # ).float()
            # print(mask_dict[node].shape)
            # # Create swaped tensor
            # x_swap = x.view(b * c, d).index_select(dim=0, index=idx).view(b, c, d)
            # # Select from swap tensor only when mask is active
            # x_swap_dict[node] = x.masked_scatter(
            #     mask_dict[node].bool().unsqueeze(-1).expand(b, c, d), x_swap
            # )

        return x_swap_dict, mask_dict

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.loader})"
