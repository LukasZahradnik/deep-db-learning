import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()

        self.transformer_modules = nn.ModuleList([
            Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim, dropout = ff_dropout),
        ])

    def forward(self, x):
        attn, ff = self.transformer_modules
        x = attn(x) + x
        x = ff(x) + x
        return x


class SimpleTableTransformer(torch.nn.Module):
    def __init__(self, *, dim, heads, dim_head = 16, attn_dropout = 0., ff_dropout = 0., table = None):
        super().__init__()
        self.dim = dim

        # transformer
        self.transformer = Transformer(dim, heads, dim_head, attn_dropout, ff_dropout)
        self.table = table

        # self.linear_cls = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        # self.to_logits = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))

    def forward(self, keys, x):
        if x is not None:
            x = torch.cat((keys, x), dim=1)
        else:
            x = keys
        x = self.transformer(x)

        return [x[:, :keys.shape[1]], x[:, keys.shape[1]:]]
