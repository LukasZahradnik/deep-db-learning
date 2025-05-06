import torch


class SafeBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) < 2:
            return x
        return self.bn(x)

    def reset_parameters(self):
        self.bn.reset_parameters()
