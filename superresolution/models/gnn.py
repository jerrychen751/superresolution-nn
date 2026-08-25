from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset

class GNNResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = GCNConv(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = GCNConv(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x, edge_index):
        identity = x

        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out, edge_index)
        out = self.bn2(out)

        out = out + identity
        return F.relu(out)

class GNN(nn.Module):
    def __init__(self, hidden_channels=64, num_blocks=4):
        super().__init__()

        # Input projection
        self.input_proj = GCNConv(3, hidden_channels)
        self.bn_in = nn.BatchNorm1d(hidden_channels)

        # Residual blocks
        self.blocks = nn.ModuleList([
            GNNResBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = GCNConv(hidden_channels, 3)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # Input
        x = self.input_proj(x, edge_index)
        x = self.bn_in(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x, edge_index)

        # Output
        x = self.output_proj(x, edge_index)

        # Residual learning
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


# --- Inference ---

def build_inference_fn(model: nn.Module, sample_shape: tuple, device: torch.device):
    """
    Return a callable that maps one on-disk input array to a prediction in the
    same on-disk layout. Builds edge_index once in the closure since the graph
    topology is identical across every sample.
    """
    nz, ny, nx, _ = sample_shape
    edge_index = _build_grid_edges(nz, ny, nx).to(device)

    def predict(input_array: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(input_array.reshape(-1, 3)).to(device)
        data = Data(x=x, edge_index=edge_index)
        pred = model(data).cpu().numpy()
        return pred.reshape(input_array.shape)
    return predict
