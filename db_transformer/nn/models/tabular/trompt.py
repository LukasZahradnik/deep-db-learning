import torch


from torch_frame.nn import Trompt
from torch_frame.nn.conv import TromptConv
from torch_frame.nn.decoder import TromptDecoder as _TromptDecoder


class TromptEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_cols: int,
        num_prompts: int,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer (got {num_layers})")

        self.channels = channels
        self.num_layers = num_layers
        self.num_prompts = num_prompts

        self.x_prompt = torch.nn.Parameter(torch.empty(num_prompts, channels))

        self.trompt_convs = torch.nn.ModuleList(
            [TromptConv(channels, num_cols, num_prompts) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.x_prompt, std=0.01)
        for trompt_conv in self.trompt_convs:
            trompt_conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = len(x)
        # [batch_size, num_prompts * num_layers, channels]
        outs = torch.empty(
            batch_size, self.num_prompts * self.num_layers, self.channels
        ).to(x.device)

        # [batch_size, num_prompts, channels]
        x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
        for i in range(self.num_layers):
            # [batch_size, num_prompts, channels]
            x_prompt = self.trompt_convs[i](x, x_prompt)
            outs[:, i * self.num_prompts : (i + 1) * self.num_prompts] = x_prompt

        # [batch_size, num_prompts * num_layers, channels]
        return outs


class TromptDecoder(torch.nn.Module):
    def __init__(
        self, channels: int, out_channels: int, num_prompts: int, num_encoder_layers: int
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_prompts = num_prompts
        self.num_encoder_layers = num_encoder_layers

        self.decoder = _TromptDecoder(channels, out_channels, num_prompts)

    def forward(self, x_prompts: torch.Tensor) -> torch.Tensor:
        batch_size = len(x_prompts)
        # [batch_size, num_encoder_layers, out_channels]
        outs = torch.empty(batch_size, self.num_encoder_layers, self.out_channels).to(
            x_prompts.device
        )

        # [batch_size, num_prompts, channels]
        for i, x_prompt in enumerate(x_prompts.split(self.num_prompts, dim=1)):
            # [batch_size, out_channels]
            out = self.decoder(x_prompt)
            # [batch_size, 1, out_channels]
            outs[:, i : i + 1] = out.view(-1, 1, self.out_channels)

        # [batch_size, out_channels]
        return outs.mean(dim=1)
