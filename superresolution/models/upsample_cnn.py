"""
Super-resolution CNN with spatial upsampling: coarse (e.g. 16^3) -> fine (e.g. 128^3).

Uses interpolate + conv (resize-conv) to avoid checkerboard artifacts from transposed convolutions.
"""

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --- Building Blocks ---

class CircularConv3d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
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


class UpsampleStage(nn.Module):
    """
    Trilinear interpolation (2x) followed by convolution. Doubles spatial resolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = CircularConv3d(channels, channels)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.relu(self.bn(self.conv(x)))
        return x


# --- Model ---

class SuperResolutionUpsampleCNN(nn.Module):
    """
    Res blocks at coarse resolution, then num_upsample_stages interpolate+conv stages (each 2x) to reach fine resolution.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        num_upsample_stages: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            CircularConv3d(3, hidden_channels),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResBlock3d(hidden_channels) for _ in range(num_blocks)]
        )
        self.upsample = nn.Sequential(
            *[UpsampleStage(hidden_channels) for _ in range(num_upsample_stages)]
        )
        self.output_proj = CircularConv3d(hidden_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.upsample(x)
        x = self.output_proj(x)
        return x


# --- Preprocessing ---

def make_training_pair(
    dns_velocity: np.ndarray,
    sigma: float,
    ds_step: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (coarse, dns_velocity). Gaussian blur then stride-downsample; target is original DNS.
    """
    from ..preprocess import apply_gaussian_filter

    blurred = apply_gaussian_filter(dns_velocity, sigma=sigma)
    coarse = blurred[::ds_step, ::ds_step, ::ds_step, :]
    return coarse, dns_velocity


# --- Dataset ---

class UpsampleCNNDataset(Dataset):
    """
    Loads (nz, ny, nx, 3) numpy pairs from disk and transposes to channels-first
    (3, nz, ny, nx) layout. Input and target may have different spatial shapes
    since upsample_cnn predicts fine resolution from coarse input.
    """

    def __init__(self, inputs: list[Path], targets: list[Path]) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = np.load(self.inputs[i]).astype(np.float32)
        target_data = np.load(self.targets[i]).astype(np.float32)

        input_data = np.transpose(input_data, (3, 0, 1, 2))
        target_data = np.transpose(target_data, (3, 0, 1, 2))

        return torch.from_numpy(input_data), torch.from_numpy(target_data)
