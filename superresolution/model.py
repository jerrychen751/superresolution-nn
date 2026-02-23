"""
3D CNN for turbulence super-resolution.

u_corrected = coarse_upsampled + model(coarse_upsampled)


"""

import torch
import torch.nn as nn


class CircularConv3d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2, # Preserves spatial dimensions as odd kernel size slides across
            padding_mode="circular", # Borders are padded with value from opposite side of volume
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock3d(nn.Module):
    """
    Conv -> Normalize -> ReLu -> Conv -> Normalize
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = CircularConv3d(channels, channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = CircularConv3d(channels, channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = output + original
        return self.relu(output)


class SuperResolutionCNN(nn.Module):
    """
    Input (batch, 3, depth, height, width): 3 channels of (u, v, w) velocity components.
        - depth, height, width corresponds to x, y, z
    Outputs: 3 channels of (du, dv, dw) correction terms for superresolution.
    """

    def __init__(self, hidden_channels: int = 32, num_blocks: int = 4) -> None:
        super().__init__()

        # nn.Sequential() forwards the output of one as input to the next module
        self.input_proj = nn.Sequential(
            CircularConv3d(3, hidden_channels),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Creates a sequence of num_blocks amount of residual blocks
        self.blocks = nn.Sequential(
            *[ResBlock3d(hidden_channels) for _ in range(num_blocks)]
        )

        # No activation: corrections can be positive or negative
        self.output_proj = CircularConv3d(hidden_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x
