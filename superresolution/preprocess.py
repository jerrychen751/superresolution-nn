"""
We train the model using DNS (high-res 3D simulation of fluid flow).

Training data shape is (nx, ny, nz, 3) where 3 represents velocity in x/y/z directions, with these 3 data at all points.

The goal is to allow the model to predict a correction term so that a low-res simulation can use the correction term to obtain similar accuracy to high-res.
"""


from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# Individual transforms

def apply_gaussian_filter(velocity, sigma=1.0):
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


def make_training_pair(dns_velocity, ds_step: int = 2):
    """
    Creates a pair of (input_features, expected_output) for the training of the model.

    Args:
        dns_velocity (nx, ny, nz, 3): Fine resolution simulation velocity flow field.
        ds_step (int): Step size to use when downsampling after Gaussian blur.
    """
    # The blur alone is not enough (just smears the data; no loss)
    # That's why downsampling must occur afterward
    blurred = apply_gaussian_filter(dns_velocity, sigma=1.0)
    coarse_blurred = blurred[::ds_step, ::ds_step, ::ds_step, :]

    # Upsample to interpolate the 
    coarse_upsampled = zoom(coarse_blurred, (ds_step, ds_step, ds_step, 1), order=3, mode="wrap")
    correction = dns_velocity - coarse_upsampled
    return coarse_upsampled, correction


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

def prepare_dataset(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(Path(raw_dir).glob("velocity_t*.npy"))
    if not raw_files:
        print(f"No raw files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw cubes")

    # Pass 1: build pairs
    inputs = []
    targets = []
    time_labels = []

    for fp in raw_files:
        t_str = fp.stem.split("_t")[1]
        time_labels.append(t_str)

        print(f"[preprocess] {fp.name}...", end=" ", flush=True)
        fine = np.load(fp).astype(np.float32)
        coarse_up, correction = make_training_pair(fine)
        inputs.append(coarse_up)
        targets.append(correction)
        print(
            f"correction range: [{correction.min():.4f}, {correction.max():.4f}]"
        )

    # Pass 2: normalize and save
    input_stats = NormalizationStats()
    input_stats.fit(inputs)
    target_stats = NormalizationStats()
    target_stats.fit(targets)

    print(f"Input  stats — mean: {input_stats.mean}, std: {input_stats.std}")
    print(f"Target stats — mean: {target_stats.mean}, std: {target_stats.std}")

    for i, t_str in enumerate(time_labels):
        inp_norm = input_stats.normalize(inputs[i])
        tgt_norm = target_stats.normalize(targets[i])
        np.save(processed_dir / f"input_t{t_str}.npy", inp_norm)
        np.save(processed_dir / f"target_t{t_str}.npy", tgt_norm)

    input_stats.save(processed_dir / "input_stats.npz")
    target_stats.save(processed_dir / "target_stats.npz")
    print(f"Saved {len(time_labels)} pairs + stats to {processed_dir}")


if __name__ == "__main__":
    prepare_dataset()
