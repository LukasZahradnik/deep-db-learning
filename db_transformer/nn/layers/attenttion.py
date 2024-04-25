import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


class IntersampleAttention(torch.nn.Module):
    def __init__(
        self, embed_dim: int, num_cols: int, num_heads: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            embed_dim * num_cols,
            num_heads,
            dropout=dropout,
            # kdim=embed_dim // num_heads,
            # vdim=embed_dim // num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape  # as mentioned above
        x = x.view(1, b, n * d)
        out, _ = self.attn(x, x, x)
        return out.view(b, n, d)
