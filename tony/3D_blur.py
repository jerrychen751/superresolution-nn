import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter, uniform_filter

# =====================================================
# 1. CREATE CHAOTIC 3D FIELD (TURBULENCE-LIKE DATA)
# =====================================================

print("Creating chaotic 3D scalar field...")

nx = ny = nz = 80

x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
z = np.linspace(-2, 2, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

np.random.seed(1)

# -----------------------------------------------------
# Multi-scale turbulent structure
# (large + medium + small eddies)
# -----------------------------------------------------

large_scale = (
    np.sin(2*np.pi*X) +
    np.cos(2*np.pi*Y) +
    np.sin(2*np.pi*Z)
)

medium_scale = (
    0.6*np.sin(6*np.pi*X + Y) +
    0.6*np.cos(6*np.pi*Y + Z) +
    0.6*np.sin(6*np.pi*Z + X)
)

small_scale = (
    0.3*np.sin(14*np.pi*(X+Y)) +
    0.3*np.cos(14*np.pi*(Y+Z))
)

# random turbulent noise
noise = 0.8 * np.random.randn(nx, ny, nz)

# combine scales
vol = large_scale + medium_scale + small_scale + noise

# normalize → full color usage
vol = (vol - vol.min()) / (vol.max() - vol.min())
vol = 2 * vol - 1   # range [-1, 1]

print("Volume shape:", vol.shape)
print("Value range:", vol.min(), vol.max())

# =====================================================
# 2. 3D FILTERS
# =====================================================

print("Applying box filter...")
box_filtered = uniform_filter(vol, size=5)

print("Applying gaussian filter...")
gaussian_filtered = gaussian_filter(vol, sigma=2)

print("Applying spectral filter (FFT low-pass)...")

F = np.fft.fftn(vol)
Fshift = np.fft.fftshift(F)

kx, ky, kz = np.meshgrid(
    np.linspace(-1, 1, nx),
    np.linspace(-1, 1, ny),
    np.linspace(-1, 1, nz),
    indexing="ij"
)

radius = np.sqrt(kx**2 + ky**2 + kz**2)
cutoff = 0.25  # lower = stronger smoothing

mask = radius < cutoff
F_filtered = Fshift * mask

spectral_filtered = np.real(
    np.fft.ifftn(np.fft.ifftshift(F_filtered))
)

# =====================================================
# 3. HELPER: CONVERT NUMPY → PYVISTA GRID
# =====================================================

def make_grid(vol_data):
    grid = pv.ImageData()

    # +1 because cell data
    grid.dimensions = np.array(vol_data.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    grid.cell_data["values"] = vol_data.flatten(order="F")
    return grid


# =====================================================
# 4. VISUALIZATION (4 PANELS IN ONE WINDOW)
# =====================================================

print("Launching visualization...")

plotter = pv.Plotter(shape=(2, 2), window_size=(1400, 900))

datasets = [
    (vol, "Original Volume"),
    (box_filtered, "Box Filter"),
    (gaussian_filtered, "Gaussian Filter"),
    (spectral_filtered, "Spectral Filter"),
]

for i, (data, title) in enumerate(datasets):

    row = i // 2
    col = i % 2

    plotter.subplot(row, col)

    grid = make_grid(data)

    plotter.add_volume(
        grid,
        cmap="viridis",
        opacity="sigmoid",
        shade=True,
        clim=(-1, 1)
    )

    plotter.add_text(title, font_size=12)

# Keep window alive
plotter.show(auto_close=False)

print("Viewer closed. Done.")