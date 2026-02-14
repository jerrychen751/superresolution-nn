"""
JHTDB Demo — querying the Johns Hopkins Turbulence Database with givernylocal.

Two main entry points:
  getData   — interpolated/differentiated values at arbitrary (x, y, z) points
  getCutout — raw grid-aligned data in a 3-D rectangular prism (no interpolation)

Install:  pip install givernylocal
Docs:     https://turbulence.idies.jhu.edu/database/local/python
Datasets: https://turbulence.idies.jhu.edu/database

The public testing token (edu.jhu.pha.turbulence.testing-201406) is limited
to ≤ 4 096 points per call.
For heavier use, request a personal token from turbulence@lists.johnshopkins.edu.
"""

import numpy as np
import matplotlib.pyplot as plt
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, getCutout

# ── 0. Configuration ────────────────────────────────────────────────
DATASET = "isotropic1024coarse"          # 1024^3 forced isotropic turbulence
TOKEN   = "edu.jhu.pha.turbulence.testing-201406"   # public testing token
OUT_DIR = "./jhtdb_output"               # where output files are saved

# Create the dataset handle (downloads JSON metadata on first call).
cube = turb_dataset(
    dataset_title=DATASET,
    output_path=OUT_DIR,
    auth_token=TOKEN,
)


# ── 1. getData: sample velocity on a 2-D plane ─────────────────────
# Build a 32×32 grid of points in the x-y plane at z = π, t = 0.
N = 32  # keep N*N ≤ 4096 for the testing token
xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
ys = np.linspace(0, 2 * np.pi, N, endpoint=False)
xg, yg = np.meshgrid(xs, ys, indexing="ij")

# points shape: (N*N, 3) — each row is [x, y, z]
points = np.column_stack([
    xg.ravel(),
    yg.ravel(),
    np.full(N * N, np.pi),  # z = π
])

results = getData(
    cube,
    var="velocity",               # 'velocity' | 'pressure' | 'force'
    timepoint_original=0.0,       # simulation time (0 .. 10.056 for this dataset)
    temporal_method="none",       # 'none' (snap to nearest) | 'pchip' (interpolate)
    spatial_method_original="lag4",  # Lagrange 4th-order interpolation
    spatial_operator="field",     # 'field' | 'gradient' | 'hessian' | 'laplacian'
    points=points,
)

# results is a list of DataFrames, one per queried timepoint.
df = results[0]
print(df.head())
# Columns for velocity+field: ['ux', 'uy', 'uz']

# Reshape into a 2-D field and plot the velocity magnitude.
ux = df["ux"].values.reshape(N, N)
uy = df["uy"].values.reshape(N, N)
speed = np.sqrt(ux**2 + uy**2)

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.pcolormesh(xs, ys, speed.T, shading="auto", cmap="inferno")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"Velocity magnitude at z = π, t = 0\n({DATASET})")
fig.colorbar(c, ax=ax, label="|u|")
fig.savefig(f"{OUT_DIR}/velocity_plane.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_DIR}/velocity_plane.png")


# ── 2. getData: query pressure at random points ────────────────────
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
print(f"\nPressure stats — mean: {p.mean():.5f}, std: {p.std():.5f}")


# ── 3. getData: velocity gradient tensor ────────────────────────────
# Returns 9 components: duxdx, duxdy, duxdz, duydx, ...
grad_results = getData(
    cube,
    var="velocity",
    timepoint_original=0.0,
    temporal_method="none",
    spatial_method_original="fd4noint",  # 4th-order finite-diff, no interpolation
    spatial_operator="gradient",
    points=rand_points[:50],  # fewer points to stay under 4096 limit
)

grad_df = grad_results[0]
print(f"\nVelocity gradient columns: {list(grad_df.columns)}")
print(grad_df.head())


# ── 4. getData: time series at a single point ──────────────────────
single_point = np.array([[np.pi, np.pi, np.pi]])

ts_results, times = getData(
    cube,
    var="velocity",
    timepoint_original=0.0,
    temporal_method="pchip",        # temporal interpolation required for time series
    spatial_method_original="lag4",
    spatial_operator="field",
    points=single_point,
    option=[0.1, 0.002],            # [t_end, delta_t] → sample t ∈ [0, 0.1) every 0.002
    return_times=True,
)

# ts_results is a list of DataFrames, one per timepoint.
ux_ts = np.array([r["ux"].values[0] for r in ts_results])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(times, ux_ts, lw=0.8)
ax.set_xlabel("t"); ax.set_ylabel("u_x")
ax.set_title("u_x time series at (π, π, π)")
fig.savefig(f"{OUT_DIR}/timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {OUT_DIR}/timeseries.png")


# ── 5. getCutout: raw gridded data (no interpolation) ──────────────
# Axes ranges: [[x_start, x_end], [y_start, y_end], [z_start, z_end], [t_start, t_end]]
# These are 1-based grid indices, not physical coordinates.
axes_ranges = np.array([
    [1, 16],   # x: 16 grid points
    [1, 16],   # y: 16 grid points
    [1, 16],   # z: 16 grid points
    [1, 1],    # t: single snapshot (1-based)
])
strides = np.array([1, 1, 1, 1])  # no downsampling

cutout_result = getCutout(
    cube,
    var="velocity",
    xyzt_axes_ranges_original=axes_ranges,
    xyzt_strides=strides,
)

# cutout_result is an xarray Dataset.
print(f"\nCutout type: {type(cutout_result)}")
print(cutout_result)