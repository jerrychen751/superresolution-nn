from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset

class GNN(nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 3)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        prev_x = x

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + prev_x
        x = F.relu(x)

        # Output layer (no BN, no ReLU)
        x = self.conv3(x, edge_index)
        return x

def make_training_pair(
    dns_velocity: np.ndarray, # (nz, ny, nx, 3)
    sigma: float,
    ds_step: int,
    spline_order: int
) -> tuple[np.ndarray, np.ndarray]:
    from ..preprocess import apply_gaussian_filter
    from scipy.ndimage import zoom

    blurred = apply_gaussian_filter(dns_velocity, sigma)
    coarse = blurred[::ds_step, ::ds_step, ::ds_step]
    # zoom tuple determines how much to scale each axis
    coarse_upsampled = zoom(coarse, (ds_step, ds_step, ds_step, 1), order=spline_order, mode="wrap")

    correction = dns_velocity - coarse_upsampled
    return coarse_upsampled, correction # (nz, ny, nx, 3) each

def _build_grid_edges(nz: int, ny: int, nx: int) -> torch.Tensor:
    """
    Build a (2, E) edge_index for a 3D grid with periodic 6-connectivity.
    np.roll wraps around by default, so this directly encodes the torus
    topology that CircularConv3d gets via padding_mode='circular'.
    """
    N = nz * ny * nx
    idx = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    offsets = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    srcs, dsts = [], []
    for (dz, dy, dx) in offsets:
        dst = np.roll(idx, shift=(dz, dy, dx), axis=(0, 1, 2))
        srcs.append(idx.ravel())
        dsts.append(dst.ravel())

    edge_index = np.stack(
        [np.concatenate(srcs), np.concatenate(dsts)],
        axis=0,
    )
    return torch.from_numpy(edge_index)


class GNNDataset(Dataset):
    """
    Reshapes (nz, ny, nx, 3) preprocessed grids into (N, 3) node features and
    wraps them in torch_geometric Data objects sharing one precomputed
    edge_index. Graph topology is identical across every sample, so we build
    it once in __init__ and reuse it for every __getitem__ call.
    """

    def __init__(self, inputs: list[Path], targets: list[Path]) -> None:
        self.inputs = inputs
        self.targets = targets

        sample = np.load(self.inputs[0])
        nz, ny, nx, _ = sample.shape
        self.edge_index = _build_grid_edges(nz, ny, nx)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> Data:
        input_grid = np.load(self.inputs[i]).astype(np.float32)
        target_grid = np.load(self.targets[i]).astype(np.float32)

        x = torch.from_numpy(input_grid.reshape(-1, 3))
        y = torch.from_numpy(target_grid.reshape(-1, 3))

        return Data(x=x, y=y, edge_index=self.edge_index)
