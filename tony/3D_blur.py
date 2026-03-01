import numpy as np
import pyvista as pv

# =====================================================
# 1. Create synthetic 3D image (volume)
# =====================================================

size = 64

x = np.linspace(-3, 3, size)
y = np.linspace(-3, 3, size)
z = np.linspace(-3, 3, size)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Create smooth 3D structure (looks like blobs/clouds)
volume = np.exp(-(X**2 + Y**2 + Z**2))

# add small random texture
volume += 0.1 * np.random.rand(size, size, size)

volume = volume.astype(np.float32)

# =====================================================
# 2. Convert to PyVista grid
# =====================================================

grid = pv.wrap(volume)

# =====================================================
# 3. Interactive visualization
# =====================================================

plotter = pv.Plotter()

plotter.add_volume(
    grid,
    opacity="sigmoid",   # nice transparency curve
)

plotter.add_axes()
plotter.show()