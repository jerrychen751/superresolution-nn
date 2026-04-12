"""
We train the model using DNS (high-res 3D simulation of fluid flow).

Training data shape is (nx, ny, nz, 3) where 3 represents velocity in x/y/z directions, with these 3 data at all points.

The goal is to allow the model to predict a correction term so that a low-res simulation can use the correction term to obtain similar accuracy to high-res.
"""


from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import hydra
from hydra.core.config_store import ConfigStore
from .config import SuperResolutionConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=SuperResolutionConfig)


# Individual transforms

def apply_gaussian_filter(velocity: np.ndarray, sigma: float):
    """
    Artificially blur the DNS data to generate model input. Uses a weighted sum of a kernel at each point where weights are generated from Gaussian PDF.
    
    Args:
        velocity (nx, ny, nz, 3): DNS velocity flow field
        sigma (int): Standard deviation of normal distribution
    """
    filtered = np.empty_like(velocity)
    
    # Apply 3 1D convolutions
    for c in range(velocity.shape[-1]): # 0, 1, or 2
        filtered[..., c] = gaussian_filter(
            velocity[..., c], sigma=sigma, mode="wrap",
        )
    return filtered


def volume_average(velocity: np.ndarray, ds_step: int) -> np.ndarray:
    """
    Coarsen a velocity field by averaging each (ds_step)^3 block into one value.
    """
    nz, ny, nx, c = velocity.shape
    reshaped = velocity.reshape(
        nz // ds_step, ds_step,
        ny // ds_step, ds_step,
        nx // ds_step, ds_step,
        c,
    )
    return reshaped.mean(axis=(1, 3, 5))


# Normalization

class NormalizationStats:

    def __init__(self):
        self.mean: np.ndarray | None = None  # shape: (3,)
        self.std: np.ndarray | None = None   # shape: (3,)

    def fit(self, data_list):
        # Flatten each cube to (N^3, 3), then stack along axis 0
        all_data = np.concatenate(
            [d.reshape(-1, 3) for d in data_list], axis=0,
        )
        self.mean = all_data.mean(axis=0).astype(np.float32)
        # Guard against division by zero for constant channels
        std = all_data.std(axis=0).astype(np.float32)
        self.std = np.maximum(std, 1e-8)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

    def save(self, path):
        assert self.mean is not None and self.std is not None
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        stats = cls()
        data = np.load(path)
        stats.mean = data["mean"]
        stats.std = data["std"]
        return stats


# Full preprocessing pipeline
@hydra.main(version_base=None, config_path="configs", config_name="cnn")
def prepare_dataset(cfg: SuperResolutionConfig):
    # Resolve data directories
    if cfg.raw_data_dir:
        raw_dir = Path(cfg.raw_data_dir)
    else:
        raw_dir = Path(__file__).resolve().parent / "data" / "raw"

    if cfg.processed_data_dir:
        processed_dir = Path(cfg.processed_data_dir)
    else:
        processed_dir = Path(__file__).resolve().parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("velocity_t*.npy"))
    if not raw_files:
        print(f"No raw files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw cubes")

    # Check if all processed outputs already exist
    expected = []
    for fp in raw_files:
        t_str = fp.stem.split("_t")[1]
        expected.append(processed_dir / f"input_t{t_str}.npy")
        expected.append(processed_dir / f"target_t{t_str}.npy")
    expected.append(processed_dir / "input_stats.npz")
    expected.append(processed_dir / "target_stats.npz")

    if all(p.exists() for p in expected):
        print("All processed files already exist, skipping preprocessing.")
        return

    # Select the make_training_pair function based on the active model
    mode = cfg.model.name
    if mode == "cnn":
        from .models.cnn import make_training_pair
    elif mode == "upsample_cnn":
        from .models.upsample_cnn import make_training_pair
    elif mode == "closure_cnn":
        from .models.closure_cnn import make_training_pair
    elif mode == "fno":
        from .models.fno import make_training_pair
    elif mode == "gnn":
        from .models.gnn import make_training_pair
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    # Pass 1: build pairs
    inputs = []
    targets = []
    time_labels = []

    for fp in raw_files:
        t_str = fp.stem.split("_t")[1]
        time_labels.append(t_str)

        print(f"[preprocess] {fp.name}...", end=" ", flush=True)
        fine = np.load(fp).astype(np.float32)
        coarse, target = make_training_pair(
            fine,
            sigma=cfg.preprocess.sigma,
            ds_step=cfg.preprocess.downsample_step,
            spline_order=cfg.preprocess.spline_interpolation_order,
        )
        inputs.append(coarse)
        targets.append(target)
        print(
            f"target range: [{target.min():.4f}, {target.max():.4f}]"
        )

    # Pass 2: normalize and save
    input_stats = NormalizationStats()
    input_stats.fit(inputs)
    target_stats = NormalizationStats()
    target_stats.fit(targets)

    print(f"Input stats — mean: {input_stats.mean}, std: {input_stats.std}")
    print(f"Target stats — mean: {target_stats.mean}, std: {target_stats.std}")

    for i, t_str in enumerate(time_labels):
        inp_norm = input_stats.normalize(inputs[i])
        tgt_norm = target_stats.normalize(targets[i])
        for name, arr in [("input", inp_norm), ("target", tgt_norm)]:
            out = processed_dir / f"{name}_t{t_str}.npy"
            tmp = processed_dir / f"{name}_t{t_str}.tmp.npy"
            np.save(tmp, arr)
            tmp.rename(out)

    for name, stats in [("input_stats", input_stats), ("target_stats", target_stats)]:
        out = processed_dir / f"{name}.npz"
        tmp = processed_dir / f"{name}.tmp.npz"
        stats.save(tmp)
        tmp.rename(out)
    print(f"Saved {len(time_labels)} pairs + stats to {processed_dir}")


if __name__ == "__main__":
    prepare_dataset()
