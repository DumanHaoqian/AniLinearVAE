import torch
import torch.nn as nn


def get_group_count(channels: int, max_groups: int = 8) -> int:
    """
    Make sure GroupNorm groups can divide channels.
    """
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """
    Basic block:
        Conv2d -> GroupNorm -> SiLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        groups = get_group_count(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        groups = get_group_count(out_channels)

        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block.

    Use Upsample + Conv instead of ConvTranspose2d
    to reduce checkerboard artifacts.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, 2H, 2W]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)