"""
Download 3D velocity cubes from the JHU Turbulence Database.

Usage:
    python -m superresolution.download_data
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

# Configuration
DATASET = "isotropic1024coarse"  # 1024^3 forced isotropic turbulence
TOKEN = "edu.gatech.jerrychen-6b7455a7"
OUTPUT_PATH = "./jhtdb_output"  # metadata cache for givernylocal
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
NUM_CUBES = 20
MAX_TIME_STEP = 5024


def get_jhtdb_conn() -> turb_dataset:
    return turb_dataset(
        dataset_title=DATASET,
        output_path=OUTPUT_PATH,
        auth_token=TOKEN,
    )


def get_velocity_cube(
    conn: turb_dataset,
    time_step: int,
    origin: tuple[int, int, int] = (1, 1, 1),
    cube_size: int = 32,
) -> np.ndarray:
    ox, oy, oz = origin

    # All indices are 1-based and bounds are inclusive
    axes_ranges = np.array([
        [ox, ox + cube_size - 1], # x
        [oy, oy + cube_size - 1], # y
        [oz, oz + cube_size - 1], # z
        [time_step, time_step], # single snapshot
    ])
    strides = np.array([1, 1, 1, 1])

    result = getCutout(
        conn,
        var="velocity",
        xyzt_axes_ranges_original=axes_ranges,
        xyzt_strides=strides,
    )

    # xarray Dataset has one data variable per time step (e.g. "velocity_0001")
    var_name = list(result.data_vars)[0]
    velocity = result[var_name].to_numpy()  # shape: (z, y, x, 3)

    return velocity.astype(np.float32)


def download_cubes(time_steps: Iterable[int], cube_size: int = 32) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_jhtdb_conn()

    for t in time_steps:
        out_path = RAW_DIR / f"velocity_t{t:04d}.npy"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[download] time step {t}...", end=" ", flush=True)
        velocity = get_velocity_cube(conn, t, cube_size=cube_size)
        np.save(out_path, velocity)
        print(f"saved {out_path}  shape={velocity.shape}")


if __name__ == "__main__":
    step = MAX_TIME_STEP // NUM_CUBES
    time_steps = list(range(1, MAX_TIME_STEP, step))
    print(f"Downloading {len(time_steps)} cubes at time steps: {time_steps}")
    download_cubes(time_steps)
    print("Done.")
