import torch


class ResidualNorm(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(shape)

    def forward(self, x: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return self.norm(x + x_next)
