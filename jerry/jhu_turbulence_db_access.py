"""
JHTDB Demo: querying the Johns Hopkins Turbulence Database with givernylocal.

givernylocal gives you two ways to pull data from JHTDB:

  getData   - returns interpolated (or differentiated) values at arbitrary
              (x, y, z) points you specify. Useful for sampling on custom grids,
              scattered points, or computing derivatives.
  getCutout - returns raw grid-aligned data in a 3D rectangular slab. No
              interpolation, just the native simulation grid. This is what you
              want for bulk downloads or ML training data.

Install:  pip install givernylocal
Docs:     https://turbulence.idies.jhu.edu/database/local/python
Datasets: https://turbulence.idies.jhu.edu/database

The public testing token is limited to 4096 points per call.
Request a personal token from turbulence@lists.johnshopkins.edu for heavier use.
"""

import numpy as np
import matplotlib.pyplot as plt
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, getCutout


# Setup
# Every query goes through a dataset handle. The first call downloads JSON
# metadata (grid spacing, time range, etc.) and caches it in OUT_DIR.
DATASET = "isotropic1024coarse"  # 1024^3 forced isotropic turbulence
DEMO_TOKEN = "edu.jhu.pha.turbulence.testing-201406"
TOKEN = "edu.gatech.jerrychen-6b7455a7"
OUT_DIR = "./jhtdb_output"

cube = turb_dataset(
    dataset_title=DATASET,
    output_path=OUT_DIR,
    auth_token=TOKEN,
)


# Example 1: velocity field on a 2D slice
# Sample a 32x32 grid in the x-y plane at z=pi, t=0.
# This shows the basic getData workflow: build a point array, call getData,
# reshape the output back to your grid.

N = 32  # N*N = 1024 points, well under the 4096 testing limit
xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
ys = np.linspace(0, 2 * np.pi, N, endpoint=False)
xg, yg = np.meshgrid(xs, ys, indexing="ij")  # (N,), (N,) -> 2x (nx, ny)

# getData expects an (M, 3) array where each row is one [x, y, z] query point.
# The domain is [0, 2pi)^3 with periodic boundaries.
points = np.column_stack([  # 3x (N*N,) -> (N*N, 3)
    xg.ravel(),
    yg.ravel(),
    np.full(N * N, np.pi),
])

results = getData(
    cube,
    var="velocity",  # also: 'pressure', 'force'
    timepoint_original=0.0,  # simulation time; this dataset covers t in [0, 10.056]
    temporal_method="none",  # snap to nearest stored timestep
    spatial_method_original="lag4",  # Lagrange 4th-order interpolation to the query points
    spatial_operator="field",  # return the field itself (vs 'gradient', 'hessian', 'laplacian')
    points=points,
)

# getData returns a list of DataFrames, one per queried timepoint.
# For velocity+field, columns are ['ux', 'uy', 'uz'].
df = results[0]
print(df.head())

ux = df["ux"].values.reshape(N, N)  # (N*N,) -> (nx, ny), which is why pcolormesh below needs speed.T
uy = df["uy"].values.reshape(N, N)  # (N*N,) -> (nx, ny)
speed = np.sqrt(ux**2 + uy**2)

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.pcolormesh(xs, ys, speed.T, shading="auto", cmap="inferno")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Velocity magnitude at z=pi, t=0 ({DATASET})")
fig.colorbar(c, ax=ax, label="|u|")
fig.savefig(f"{OUT_DIR}/velocity_plane.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_DIR}/velocity_plane.png")


# Example 2: pressure at scattered points
# You don't need a regular grid. getData works with any set of (x,y,z) coords.

rng = np.random.default_rng(42)
rand_points = rng.uniform(0, 2 * np.pi, size=(100, 3))

pressure_results = getData(
    cube,
    var="pressure",
    timepoint_original=0.002,
    temporal_method="none",
    spatial_method_original="m1q4",  # spline interpolation
    spatial_operator="field",
    points=rand_points,
)

p = pressure_results[0]["p"].values
print(f"\nPressure stats: mean={p.mean():.5f}, std={p.std():.5f}")


# Example 3: velocity gradient tensor
# Change spatial_operator to 'gradient' to get the full 3x3 velocity gradient
# tensor dui/dxj at each point. Returns 9 columns: duxdx, duxdy, duxdz, etc.

grad_results = getData(
    cube,
    var="velocity",
    timepoint_original=0.0,
    temporal_method="none",
    spatial_method_original="fd4noint",  # 4th-order finite differences, no interpolation
    spatial_operator="gradient",
    points=rand_points[:50],
)

grad_df = grad_results[0]
print(f"\nVelocity gradient columns: {list(grad_df.columns)}")
print(grad_df.head())


# Example 4: time series at a single point
# To get data at multiple times, use the `option` parameter with temporal
# interpolation enabled. option=[t_end, delta_t] samples from
# timepoint_original to t_end at the given interval.

single_point = np.array([[np.pi, np.pi, np.pi]])

ts_results, times = getData(
    cube,
    var="velocity",
    timepoint_original=0.0,
    temporal_method="pchip",  # need temporal interpolation for time series
    spatial_method_original="lag4",
    spatial_operator="field",
    points=single_point,
    option=[0.1, 0.002],  # sample t in [0, 0.1) every 0.002
    return_times=True,
)

# ts_results is now a list of DataFrames, one per sampled timepoint
ux_ts = np.array([r["ux"].values[0] for r in ts_results])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(times, ux_ts, lw=0.8)
ax.set_xlabel("t")
ax.set_ylabel("u_x")
ax.set_title("u_x time series at (pi, pi, pi)")
fig.savefig(f"{OUT_DIR}/timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {OUT_DIR}/timeseries.png")


# Example 5: getCutout for bulk grid data
# getCutout grabs a rectangular chunk of the native simulation grid.
# Unlike getData, there's no interpolation; you get the raw stored values.
# This is the fastest way to download large blocks of data for training.

# Axes ranges use 1-based grid indices (not physical coordinates):
# [[x_start, x_end], [y_start, y_end], [z_start, z_end], [t_start, t_end]]
axes_ranges = np.array([
    [1, 16],  # 16 grid points in x
    [1, 16],  # 16 grid points in y
    [1, 16],  # 16 grid points in z
    [1, 1],  # single snapshot
])
strides = np.array([1, 1, 1, 1])  # stride > 1 would skip grid points (coarsen)

cutout_result = getCutout(
    cube,
    var="velocity",
    xyzt_axes_ranges_original=axes_ranges,
    xyzt_strides=strides,
)

# getCutout returns an xarray Dataset with the velocity components on the grid
print(f"\nCutout type: {type(cutout_result)}")
print(cutout_result)
