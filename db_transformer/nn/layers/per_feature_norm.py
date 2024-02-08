import torch

class PerFeatureNorm(torch.nn.Module):
    def __init__(self, n_features: int, axis: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm([n_features])
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, -1, self.axis)
        x = self.norm(x)
        x = torch.transpose(x, -1, self.axis)
        return x
  