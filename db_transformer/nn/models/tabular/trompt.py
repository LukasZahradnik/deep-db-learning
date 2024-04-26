import torch

from torch_frame.nn.conv import TromptConv
from torch_frame.nn.decoder import TromptDecoder


class Trompt(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_cols: int,
        num_prompts: int,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer (got {num_layers})")

        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.x_prompt = torch.nn.Parameter(torch.empty(num_prompts, channels))

        self.trompt_convs = torch.nn.ModuleList(
            [TromptConv(channels, num_cols, num_prompts) for i in range(num_layers)]
        )

        # Decoder is shared across layers.
        self.trompt_decoder = TromptDecoder(channels, out_channels, num_prompts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.x_prompt, std=0.01)
        for trompt_conv in self.trompt_convs:
            trompt_conv.reset_parameters()
        self.trompt_decoder.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = len(x)
        outs = []
        # [batch_size, num_prompts, channels]
        x_prompt = self.x_prompt.repeat(batch_size, 1, 1)
        for i in range(self.num_layers):
            # [batch_size, num_prompts, channels]
            x_prompt = self.trompt_convs[i](x, x_prompt)
            # [batch_size, out_channels]
            out = self.trompt_decoder(x_prompt)
            # [batch_size, 1, out_channels]
            out = out.view(batch_size, 1, self.out_channels)
            outs.append(out)
        # [batch_size, num_layers, out_channels]
        stacked_out = torch.cat(outs, dim=1)
        return stacked_out.mean(dim=1)
