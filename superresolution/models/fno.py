"""
Fourier Neural Operator models for CFD prediction.

FNO2d: original 2D model for Navier-Stokes vorticity prediction.
FNO3d: 3D variant for JHTDB velocity super-resolution. Predicts a
       residual correction at full DNS resolution, following the same
       u_corrected = u_coarse_upsampled + model(u_coarse_upsampled) paradigm
       as the CNN models.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset


# ============================================================
# 2D FNO (Navier-Stokes vorticity)
# ============================================================

# --- Building Blocks (2D) ---

class LiftLayer2d(nn.Module):
    """
    Pointwise linear map that projects raw input channels into the
    hidden dimension d_v. Uses a 1x1 convolution (equivalent to a
    per-pixel linear layer shared across all spatial locations).
    """

    def __init__(self, in_channels: int, d_v: int) -> None:
        super().__init__()
        self.linear = nn.Conv2d(in_channels, d_v, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SpectralConv2d(nn.Module):

    def __init__(self, d_v: int, k_max: int) -> None:
        """
        Learnable linear transform applied in Fourier space to the
        lowest k_max modes along each spatial axis.

        Two weight tensors handle the two quadrants of the real FFT
        output that contain independent information: low-low and
        high-low wavenumber pairs.

        d_v: hidden channel dimension (same for input and output)
        k_max: number of Fourier modes retained per axis
        """
        super().__init__()
        self.d_v = d_v
        self.k_max = k_max

        scale = 1 / (d_v * d_v)
        self.W1 = nn.Parameter(scale * torch.randn(d_v, d_v, k_max, k_max, 2))
        self.W2 = nn.Parameter(scale * torch.randn(d_v, d_v, k_max, k_max, 2))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        batch, d_v, nx, ny = v.shape
        v_ft = torch.fft.rfft2(v)  # (batch, d_v, nx, ny//2 + 1)

        out_ft = torch.zeros_like(v_ft)
        out_ft[:, :, :self.k_max, :self.k_max] = torch.einsum(
            "bixy, ioxy -> boxy",
            v_ft[:, :, :self.k_max, :self.k_max],
            torch.view_as_complex(self.W1)
        )
        out_ft[:, :, -self.k_max:, :self.k_max] = torch.einsum(
            "bixy, ioxy -> boxy",
            v_ft[:, :, -self.k_max:, :self.k_max],
            torch.view_as_complex(self.W2)
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class FourierLayer2d(nn.Module):
    """
    One FNO layer: spectral path (global) + local linear path, summed
    and passed through ReLU. The local path (1x1 conv) lets the network
    learn purely local features that the truncated spectral path misses.
    """

    def __init__(self, d_v: int, k_max: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(d_v, k_max)
        self.linear = nn.Conv2d(d_v, d_v, kernel_size=1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.spectral(v) + self.linear(v))


class ProjectionLayer2d(nn.Module):

    def __init__(self, d_v: int, d_hidden: int, d_out: int) -> None:
        """
        Two-layer pointwise MLP that maps from hidden dimension back
        to the target output channels.

        d_v: input channel dimension
        d_hidden: hidden layer dimension
        d_out: output dimension (1 for scalar vorticity)
        """
        super().__init__()
        self.fc1 = nn.Conv2d(d_v, d_hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(d_hidden, d_out, kernel_size=1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(v))
        return self.fc2(h)


# --- Model (2D) ---

class FNO2d(nn.Module):

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        d_v: int = 32,
        k_max: int = 12,
        n_fourier_layers: int = 4,
        proj_hidden: int = 128
    ) -> None:
        super().__init__()
        self.lift = LiftLayer2d(in_channels, d_v)
        self.fourier = nn.Sequential(*[FourierLayer2d(d_v, k_max) for _ in range(n_fourier_layers)])
        self.proj = ProjectionLayer2d(d_v, proj_hidden, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.lift(x)
        return self.proj(self.fourier(v))


# ============================================================
# 3D FNO (JHTDB velocity super-resolution)
# ============================================================

# --- Building Blocks (3D) ---

class LiftLayer3d(nn.Module):
    """
    Same idea as 2D lift but with Conv3d.
    Input: (batch, in_channels, nz, ny, nx)
    """

    def __init__(self, in_channels: int, d_v: int) -> None:
        super().__init__()
        self.linear = nn.Conv3d(in_channels, d_v, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SpectralConv3d(nn.Module):

    def __init__(self, d_v: int, k_max: int) -> None:
        """
        3D extension of SpectralConv2d. Multiplies learned weights with
        the lowest k_max Fourier modes along each of three spatial axes.

        rfftn along the last axis produces n//2+1 complex values. The
        other two axes keep all n values, but modes above n//2 are the
        negative-frequency mirrors. So for a cube of size n, independent
        information lives in four octants (combinations of low/high
        along the first two axes, always low along the last rfft axis).

        d_v: hidden channel dimension
        k_max: number of Fourier modes retained per axis
        """
        super().__init__()
        self.d_v = d_v
        self.k_max = k_max

        scale = 1 / (d_v * d_v)
        # Four weight tensors for the four independent octants
        self.W1 = nn.Parameter(scale * torch.randn(d_v, d_v, k_max, k_max, k_max, 2))
        self.W2 = nn.Parameter(scale * torch.randn(d_v, d_v, k_max, k_max, k_max, 2))
        self.W3 = nn.Parameter(scale * torch.randn(d_v, d_v, k_max, k_max, k_max, 2))
        self.W4 = nn.Parameter(scale * torch.randn(d_v, d_v, k_max, k_max, k_max, 2))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        batch, d_v, nz, ny, nx = v.shape
        v_ft = torch.fft.rfftn(v, dim=(-3, -2, -1))  # (batch, d_v, nz, ny, nx//2+1)
        k = self.k_max

        out_ft = torch.zeros_like(v_ft)

        # Octant 1: low-z, low-y, low-x
        out_ft[:, :, :k, :k, :k] = torch.einsum(
            "bixyz, ioxyz -> boxyz",
            v_ft[:, :, :k, :k, :k],
            torch.view_as_complex(self.W1)
        )
        # Octant 2: high-z (negative freq), low-y, low-x
        out_ft[:, :, -k:, :k, :k] = torch.einsum(
            "bixyz, ioxyz -> boxyz",
            v_ft[:, :, -k:, :k, :k],
            torch.view_as_complex(self.W2)
        )
        # Octant 3: low-z, high-y (negative freq), low-x
        out_ft[:, :, :k, -k:, :k] = torch.einsum(
            "bixyz, ioxyz -> boxyz",
            v_ft[:, :, :k, -k:, :k],
            torch.view_as_complex(self.W3)
        )
        # Octant 4: high-z, high-y, low-x
        out_ft[:, :, -k:, -k:, :k] = torch.einsum(
            "bixyz, ioxyz -> boxyz",
            v_ft[:, :, -k:, -k:, :k],
            torch.view_as_complex(self.W4)
        )

        return torch.fft.irfftn(out_ft, s=(nz, ny, nx))


class FourierLayer3d(nn.Module):
    """
    3D Fourier layer: spectral path (global) + local 1x1 conv path,
    summed and passed through GELU.

    Uses GELU instead of ReLU — smoother activation that tends to
    train better on regression tasks where small gradients near zero
    matter. ReLU kills all negative gradients; GELU allows small
    negative signals through.
    """

    def __init__(self, d_v: int, k_max: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(d_v, k_max)
        self.linear = nn.Conv3d(d_v, d_v, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(v) + self.linear(v))


class ProjectionLayer3d(nn.Module):

    def __init__(self, d_v: int, d_hidden: int, d_out: int) -> None:
        """
        d_v: input channel dimension
        d_hidden: hidden layer dimension
        d_out: output dimension (3 for velocity components u, v, w)
        """
        super().__init__()
        self.fc1 = nn.Conv3d(d_v, d_hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(d_hidden, d_out, kernel_size=1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(v))
        return self.fc2(h)


# --- Model (3D) ---

class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for velocity super-resolution.

    Predicts the residual correction (du, dv, dw) at full DNS
    resolution. Input is the coarse-upsampled velocity field (3 channels)
    plus 3D spatial grid coordinates (3 channels) = 6 input channels.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        d_v: int = 32,
        k_max: int = 12,
        n_fourier_layers: int = 4,
        proj_hidden: int = 128
    ) -> None:
        super().__init__()
        self.lift = LiftLayer3d(in_channels, d_v)
        self.fourier = nn.Sequential(*[FourierLayer3d(d_v, k_max) for _ in range(n_fourier_layers)])
        self.proj = ProjectionLayer3d(d_v, proj_hidden, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 6, nz, ny, nx) — first 3 channels are velocity,
           last 3 are grid coordinates
        Returns: (batch, 3, nz, ny, nx) — predicted correction
        """
        v = self.lift(x)
        return self.proj(self.fourier(v))


# --- Preprocessing (3D) ---

def make_training_pair(
    dns_velocity: np.ndarray,
    sigma: float,
    ds_step: int,
    spline_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same preprocessing as cnn: blur, downsample,
    spline upsample, compute correction. The FNO3d model receives
    the same (coarse_upsampled, correction) pair — the grid
    coordinates are appended at dataset load time, not here.
    """
    from ..preprocess import apply_gaussian_filter

    blurred = apply_gaussian_filter(dns_velocity, sigma=sigma)
    coarse = blurred[::ds_step, ::ds_step, ::ds_step, :]
    coarse_upsampled = zoom(
        coarse, (ds_step, ds_step, ds_step, 1), order=spline_order, mode="wrap",
    )
    correction = dns_velocity - coarse_upsampled
    return coarse_upsampled, correction


# --- Dataset ---

class FNODataset(Dataset):
    """
    Loads (nz, ny, nx, 3) numpy pairs, transposes to (3, nz, ny, nx), and
    appends 3 normalized [0,1] grid-coordinate channels to the input so the
    spectral convolution has a way to break translational symmetry. Result
    is (6, nz, ny, nx) input and (3, nz, ny, nx) target.
    """

    def __init__(self, inputs: list[Path], targets: list[Path]) -> None:
        self.inputs = inputs
        self.targets = targets
        self._grid = None

    def _get_grid(self, nz: int, ny: int, nx: int) -> np.ndarray:
        if self._grid is None:
            gz = np.linspace(0, 1, nz, dtype=np.float32)
            gy = np.linspace(0, 1, ny, dtype=np.float32)
            gx = np.linspace(0, 1, nx, dtype=np.float32)
            grid_z, grid_y, grid_x = np.meshgrid(gz, gy, gx, indexing="ij")
            self._grid = np.stack([grid_z, grid_y, grid_x], axis=0)
        return self._grid

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = np.load(self.inputs[i]).astype(np.float32)
        target_data = np.load(self.targets[i]).astype(np.float32)

        input_data = np.transpose(input_data, (3, 0, 1, 2))
        target_data = np.transpose(target_data, (3, 0, 1, 2))

        grid = self._get_grid(*input_data.shape[1:])
        input_data = np.concatenate([input_data, grid], axis=0)

        return torch.from_numpy(input_data), torch.from_numpy(target_data)
