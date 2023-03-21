from einops import rearrange, repeat

import torch
import torch.nn.functional as F

from ft_transformer import FTTransformer, Transformer, NumericalEmbedder


class ColumnEmbedder(torch.nn.Module):
    def __init__(self, categories, num_continuous, dim, num_special_tokens=2):
        super().__init__()

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding
            self.categorical_embeds = torch.nn.Embedding(total_tokens + 10, dim)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

    def forward(self, x_categ, x_numer):
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        return torch.cat(xs, dim=1)


class SimpleTableTransformer(torch.nn.Module):
    def __init__(self, *, dim, heads, dim_head = 16, dim_out = 1, attn_dropout = 0., ff_dropout = 0.):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.dim = dim

        # transformer
        self.transformer = Transformer(dim, heads, dim_head, attn_dropout, ff_dropout)

        self.linear_cls = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(dim, dim_out))
        self.to_logits = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))

    def forward(self, x):
        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        x = torch.cat((cls_tokens, x), dim = 1)
        x = self.transformer(x)

        # get cls token
        xs = x[:, 0]
        return self.linear_cls(xs), self.to_logits(x)


class DBTransformer(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, heads: int, attn_dropout: float, ff_dropout: float, tables, layers: int):
        super().__init__()

        self.tables = tables
        self.embedder = [ColumnEmbedder(table.categories, table.num_continuous, dim) for table in tables]

        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleList([
                SimpleTableTransformer(
                    dim=dim,
                    dim_out=dim,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout
                ) for _ in tables
            ]) for _ in range(layers)
        ])

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim_out)
        )

    def forward(self, inputs, foreign_indices, output_table: str):
        xs = {
            table.name: (0, embedder(cat, num))
            for (num, cat), embedder, table in zip(inputs, self.embedder, self.tables)
        }

        for layer in self.layers:
            xs = {
                table.name: model(xs[table.name][1])
                for model, table in zip(layer, self.tables)
            }

            # """Message Passing"""
            for table, column_foreign_indices in zip(self.tables, foreign_indices):
                for index, (foreign, (table_name, _)) in enumerate(zip(column_foreign_indices, table.keys)):
                    w = xs[table.name][1]
                    w[:, -(index + 1), :] = xs[table_name][0][foreign] + w[:, -(index + 1), :]

        x = xs[output_table][0]
        x = self.to_logits(x)

        return x
