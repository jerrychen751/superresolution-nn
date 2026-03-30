"""
3D CNN for turbulence closure modeling.

Takes coarse velocity field (e.g., 16^3) and predicts a velocity correction
at the same resolution. The correction makes the coarse field more accurate
without increasing grid resolution.

u_corrected = coarse + model(coarse)
"""

import numpy as np

import torch
import torch.nn as nn


# ── Building blocks ──────────────────────────────────────────────

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


# ── Model ────────────────────────────────────────────────────────

class ClosureCNN(nn.Module):
    """
    Input (batch, 3, D, H, W): coarse velocity field (e.g., 16^3).
    Output (batch, 3, D, H, W): velocity correction at same resolution.

    Learns the difference between the volume-averaged DNS (truth at coarse
    resolution) and the filtered+downsampled field (coarse simulation proxy).
    """

    def __init__(self, hidden_channels: int = 32, num_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            CircularConv3d(3, hidden_channels),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResBlock3d(hidden_channels) for _ in range(num_blocks)]
        )
        self.output_proj = CircularConv3d(hidden_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x


# ── Preprocessing ────────────────────────────────────────────────

def make_training_pair(
    dns_velocity: np.ndarray,
    sigma: float,
    ds_step: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates input (filtered+downsampled) and target (velocity correction)
    for closure training.

    The correction = volume_averaged_DNS - filtered_downsampled, representing
    what the coarse field is missing due to filtering.
    """
    from ..preprocess import apply_gaussian_filter, volume_average

    blurred = apply_gaussian_filter(dns_velocity, sigma=sigma)
    filtered_downsampled = blurred[::ds_step, ::ds_step, ::ds_step, :]
    truth_coarse = volume_average(dns_velocity, ds_step)
    correction = truth_coarse - filtered_downsampled
    return filtered_downsampled, correction
