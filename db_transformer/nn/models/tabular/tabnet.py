from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class TabNetEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_cols: int,
        num_layers: int,
        gamma: float = 1.2,
        num_shared_glu_layers: int = 2,
        num_dependent_glu_layers: int = 2,
        split_feat_channels: Optional[int] = None,
        split_attn_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer (got {num_layers})")
        self.channels = channels
        self.split_feat_channels = (
            split_feat_channels if split_feat_channels is not None else channels
        )
        self.split_attn_channels = (
            split_attn_channels if split_attn_channels is not None else channels
        )
        self.num_layers = num_layers
        self.gamma = gamma

        # Batch norm applied to input feature.
        self.bn = torch.nn.BatchNorm1d(num_cols)

        self.attn_transformers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_transformers.append(
                AttentiveTransformer(
                    in_channels=self.split_attn_channels,
                    out_channels=channels,
                    num_cols=num_cols,
                )
            )

        self.feat_transformers = torch.nn.ModuleList()
        for _ in range(self.num_layers + 1):
            self.feat_transformers.append(
                FeatureTransformer(
                    in_channels=channels,
                    out_channels=self.split_feat_channels + self.split_attn_channels,
                    num_shared_glu_layers=num_shared_glu_layers,
                    num_dependent_glu_layers=num_dependent_glu_layers,
                )
            )

        self.lin = torch.nn.Linear(self.split_feat_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bn.reset_parameters()
        for feat_transformer in self.feat_transformers:
            feat_transformer.reset_parameters()
        for attn_transformer in self.attn_transformers:
            attn_transformer.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self, x: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # [batch_size, num_cols, channels]
        x = self.bn(x)

        if prior is None:
            # [batch_size, num_cols, channels]
            prior = torch.ones_like(x)

        # [batch_size, num_cols, split_attn_channels]
        attention_x = self.feat_transformers[0](x)
        attention_x = attention_x[:, :, self.split_feat_channels :]

        outs = []
        for i in range(self.num_layers):
            # [batch_size, num_cols, channels]
            attention_mask = self.attn_transformers[i](attention_x, prior)
            # [batch_size, num_cols, channels]
            masked_x = attention_mask * x
            # [batch_size, num_cols, split_feat_channels + split_attn_channel]
            out = self.feat_transformers[i + 1](masked_x)

            # Get the split feature
            # [batch_size, num_cols, split_feat_channels]
            feature_x = F.relu(out[:, :, : self.split_feat_channels])
            outs.append(feature_x)
            # Get the split attention
            # [batch_size, num_cols, split_attn_channels]
            attention_x = out[:, :, self.split_feat_channels :]

            # Update prior
            prior = (self.gamma - attention_mask) * prior

        out = sum(outs)
        out = self.lin(out)

        return out


class TabNetDecoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_shared_glu_layers: int = 2,
        num_dependent_glu_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer (got {num_layers})")
        self.channels = channels
        self.num_layers = num_layers

        self.feat_transformers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.feat_transformers.append(
                FeatureTransformer(
                    in_channels=channels,
                    out_channels=out_channels,
                    num_shared_glu_layers=num_shared_glu_layers,
                    num_dependent_glu_layers=num_dependent_glu_layers,
                )
            )

        self.lin = torch.nn.Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for feat_transformer in self.feat_transformers:
            feat_transformer.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(self.num_layers):
            # [batch_size, split_feat_channels + split_attn_channel]
            outs.append(self.feat_transformers[i](x))

        out = sum(outs)
        out = self.lin(out)

        return out


class FeatureTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_shared_glu_layers: int,
        num_dependent_glu_layers: int,
    ) -> None:
        super().__init__()

        self.shared_glu_block = (
            GLUBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                no_first_residual=True,
                num_glu_layers=num_shared_glu_layers,
            )
            if num_shared_glu_layers > 0
            else torch.nn.Identity()
        )

        self.dependent: torch.nn.Module
        if num_dependent_glu_layers == 0:
            self.dependent = torch.nn.Identity()
        else:
            self.dependent = GLUBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                no_first_residual=False,
                num_glu_layers=num_dependent_glu_layers,
            )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared_glu_block(x)
        x = self.dependent(x)
        return x

    def reset_parameters(self) -> None:
        if not isinstance(self.shared_glu_block, torch.nn.Identity):
            self.shared_glu_block.reset_parameters()
        if not isinstance(self.dependent, torch.nn.Identity):
            self.dependent.reset_parameters()


class GLUBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_glu_layers: int = 2,
        no_first_residual: bool = False,
    ) -> None:
        super().__init__()
        self.no_first_residual = no_first_residual
        self.glu_layers = torch.nn.ModuleList()

        for i in range(num_glu_layers):
            if i == 0:
                glu_layer = GLULayer(in_channels, out_channels)
            else:
                glu_layer = GLULayer(out_channels, out_channels)
            self.glu_layers.append(glu_layer)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            if self.no_first_residual and i == 0:
                x = glu_layer(x)
            else:
                x = x * math.sqrt(0.5) + glu_layer(x)
        return x

    def reset_parameters(self) -> None:
        for glu_layer in self.glu_layers:
            glu_layer.reset_parameters()


class GLULayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels * 2, bias=False)
        self.glu = torch.nn.GLU()
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        return self.glu(x)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()


class AttentiveTransformer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_cols: int) -> None:
        super().__init__()
        self.num_cols = num_cols
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bn = GhostBatchNorm1d(num_cols)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = prior * x
        # Using softmax instead of sparsemax since softmax performs better.
        x = F.softmax(x.view(x.shape[0], -1), dim=-1)
        return x.view(x.shape[0], self.num_cols, -1)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.bn.reset_parameters()


class GhostBatchNorm1d(torch.nn.Module):
    r"""Ghost Batch Normalization https://arxiv.org/abs/1705.08741."""

    def __init__(
        self,
        input_dim: int,
        virtual_batch_size: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = torch.nn.BatchNorm1d(self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > 0:
            num_chunks = math.ceil(len(x) / self.virtual_batch_size)
            chunks = torch.chunk(x, num_chunks, dim=0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

    def reset_parameters(self) -> None:
        self.bn.reset_parameters()
