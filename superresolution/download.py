"""
Download 3D velocity cubes from the JHU Turbulence Database.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

import hydra


def get_velocity_cube(
    conn: turb_dataset,
    time_step: int,
    origin: tuple[int, int, int] = (1, 1, 1),
    nx: int = 128,
    ny: int = 128,
    nz: int = 128,
) -> np.ndarray:
    ox, oy, oz = origin

    # All indices are 1-based and bounds are inclusive
    axes_ranges = np.array([
        [ox, ox + nx - 1], # x
        [oy, oy + ny - 1], # y
        [oz, oz + nz - 1], # z
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


def download_cubes(
    conn: turb_dataset,
    raw_dir: Path,
    time_steps: Iterable[int],
    nx: int,
    ny: int,
    nz: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    for t in time_steps:
        out_path = raw_dir / f"velocity_t{t:04d}.npy"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[download] time step {t}...", end=" ", flush=True)
        velocity = get_velocity_cube(conn, t, nx=nx, ny=ny, nz=nz) # (nz, ny, nx, 3)
        tmp_path = out_path.parent / f"velocity_t{t:04d}.npy.{os.getpid()}.tmp"
        with tmp_path.open("wb") as fh:
            np.save(fh, velocity)
        tmp_path.replace(out_path)  # atomic on same filesystem
        print(f"saved {out_path}  shape={velocity.shape}")


@hydra.main(version_base=None, config_path="configs", config_name="cnn")
def main(cfg):
    raw_dir = Path(cfg.raw_data_dir)

    conn = turb_dataset(
        dataset_title=cfg.download.jhtdb_dataset,
        output_path=cfg.download.jhtdb_cache_dir,
        auth_token=cfg.download.jhtdb_token,
    )

    step = cfg.download.max_timesteps // cfg.download.num_cubes
    time_steps = list(range(1, cfg.download.max_timesteps, step))
    print(f"Downloading {len(time_steps)} cubes at time steps: {time_steps}")

    download_cubes(
        conn=conn,
        raw_dir=raw_dir,
        time_steps=time_steps,
        nx=cfg.download.nx,
        ny=cfg.download.ny,
        nz=cfg.download.nz,
    )
    print("Done.")


if __name__ == "__main__":
    main()
