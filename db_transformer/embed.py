import torch
from einops import rearrange


class NumericalEmbedder(torch.nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = torch.nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


class ColumnEmbedder(torch.nn.Module):
    def __init__(self, categories, num_continuous, dim, num_special_tokens=2):
        super().__init__()

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.dim = dim

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            # categorical embedding
            self.categorical_embeds = torch.nn.Embedding(self.total_tokens + 10, dim)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

    def forward(self, x_categ, x_numer):
        xs = []

        if self.num_unique_categories > 0:
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        if xs:
            return torch.cat(xs, dim=1)
        return None
