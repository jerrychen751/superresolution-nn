"""
We train the model using DNS (high-res 3D simulation of fluid flow).

Training data shape is (nz, ny, nx, 3) where 3 represents velocity in x/y/z directions, with these 3 data at all points.

The goal is to allow the model to predict a correction term so that a low-res simulation can use the correction term to obtain similar accuracy to high-res.
"""


from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import hydra


# Individual transforms

def apply_gaussian_filter(velocity: np.ndarray, sigma: float):
    """
    Artificially blur the DNS data to generate model input. Uses a weighted sum of a kernel at each point where weights are generated from Gaussian PDF.

    Args:
        velocity (nz, ny, nx, 3): DNS velocity flow field
        sigma (float): Standard deviation of normal distribution
    """
    filtered = np.empty_like(velocity)

    # Apply 3 1D convolutions
    for c in range(velocity.shape[-1]): # 0, 1, or 2
        filtered[..., c] = gaussian_filter(  # velocity[..., c]: (nz, ny, nx, 3) -> (nz, ny, nx)
            velocity[..., c], sigma=sigma, mode="wrap",
        )
    return filtered


def average_volumes(velocity: np.ndarray, ds_step: int) -> np.ndarray:
    """
    Coarsen a velocity field by averaging each (ds_step)^3 block into one value.
    """
    nz, ny, nx, c = velocity.shape
    reshaped = velocity.reshape(  # (nz, ny, nx, c) -> (nz_blocks, dz, ny_blocks, dy, nx_blocks, dx, c), where dz/dy/dx are the within-block offsets of size ds_step
        nz // ds_step, ds_step,
        ny // ds_step, ds_step,
        nx // ds_step, ds_step,
        c,
    )
    return reshaped.mean(axis=(1, 3, 5))  # (nz_blocks, dz, ny_blocks, dy, nx_blocks, dx, c) -> (nz_blocks, ny_blocks, nx_blocks, c)


# Full preprocessing pipeline
@hydra.main(version_base=None, config_path="configs", config_name="cnn")
def prepare_dataset(cfg):
    raw_dir = Path(cfg.raw_data_dir)
    processed_dir = Path(cfg.processed_data_dir)

    raw_files = sorted(raw_dir.glob("velocity_t*.npy"))
    if not raw_files:
        print(f"No raw files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw cubes")

    # Chronological train/val/test split. Test gets the remainder so rounding never loses a file.
    ratio_sum = cfg.preprocess.train_ratio + cfg.preprocess.val_ratio + cfg.preprocess.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"preprocess split ratios must sum to 1.0, got {ratio_sum}")
    n = len(raw_files)
    n_train = int(cfg.preprocess.train_ratio * n)
    n_val = int(cfg.preprocess.val_ratio * n)
    splits = {
        "train": raw_files[:n_train],
        "val": raw_files[n_train:n_train + n_val],
        "test": raw_files[n_train + n_val:],
    }
    for split_name, split_files in splits.items():
        (processed_dir / split_name).mkdir(parents=True, exist_ok=True)
        print(f"  {split_name}: {len(split_files)} files")

    # Check if all processed outputs already exist
    expected = []
    for split_name, split_files in splits.items():
        for fp in split_files:
            t_str = fp.stem.split("_t")[1]
            expected.append(processed_dir / split_name / f"input_t{t_str}.npy")
            expected.append(processed_dir / split_name / f"target_t{t_str}.npy")

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

    for split_name, split_files in splits.items():
        split_dir = processed_dir / split_name
        for fp in split_files:
            t_str = fp.stem.split("_t")[1]

            print(f"[preprocess] {split_name}/{fp.name}...", end=" ", flush=True)
            fine = np.load(fp).astype(np.float32)
            coarse, target = make_training_pair(
                fine,
                sigma=cfg.preprocess.sigma,
                ds_step=cfg.preprocess.downsample_step,
                spline_order=cfg.preprocess.spline_interpolation_order,
            )
            for name, arr in [("input", coarse), ("target", target)]:
                out = split_dir / f"{name}_t{t_str}.npy"
                tmp = split_dir / f"{name}_t{t_str}.tmp.npy"
                np.save(tmp, arr)
                tmp.rename(out)
            print(f"target range: [{target.min():.4f}, {target.max():.4f}]")

    print(f"Saved {n} pairs to {processed_dir} (train/val/test)")


if __name__ == "__main__":
    prepare_dataset()
