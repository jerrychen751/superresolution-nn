import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. Load image
# -------------------------
name = "tech_tower.jpg"  # Change this to the image you want to test

# Use cv2.IMREAD_COLOR_RGB for color images, cv2.IMREAD_GRAYSCALE for grayscale images
img = cv2.imread("tony/images/" + name, cv2.IMREAD_COLOR_RGB)

img_size = 1024    # Change this to desired size (128, 256, 512, 1024, etc.)
img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

# Convert to float32 and normalize to [0, 1]
img = img.astype(np.float32) / 255.0

# -------------------------
# 2. Apply Gaussian blur
# -------------------------
gaussian_kernel_size = 31
sigma = 5
gaussian = cv2.GaussianBlur(img, (gaussian_kernel_size, gaussian_kernel_size), sigmaX=sigma)

# -------------------------
# 3. Apply Box blur
# -------------------------
box_kernel_size = 31
box = cv2.blur(img, (box_kernel_size, box_kernel_size))

# -------------------------
# 4. Apply Spectral (Fourier) Low-Pass Filter
# -------------------------
def spectral_lowpass(image, cutoff_ratio=0.1):
    """
    cutoff_ratio: fraction of frequency radius to keep (0 < cutoff_ratio < 0.5)
    smaller = stronger filtering
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, c = image.shape
    output = np.zeros_like(image)

    # Frequency grid
    ky = np.fft.fftfreq(h)
    kx = np.fft.fftfreq(w)
    KX, KY = np.meshgrid(kx, ky)
    radius = np.sqrt(KX**2 + KY**2)

    mask = radius < cutoff_ratio

    for ch in range(c):
        F = np.fft.fft2(image[:, :, ch])
        F_filtered = F * mask
        filtered = np.fft.ifft2(F_filtered).real
        output[:, :, ch] = filtered

    if output.shape[2] == 1:
        return output[:, :, 0]

    return output

cutoff_ratio = 0.05   # Smaller = stronger filtering
spectral = spectral_lowpass(img, cutoff_ratio=cutoff_ratio)

# -------------------------
# 5. Compute residuals (optional)
# -------------------------
res_gaussian = img - gaussian
res_box = img - box
res_spectral = img - spectral

# -------------------------
# 6. Visualization
# -------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.colorbar()

plt.subplot(2,2,2)
plt.title(f"Gaussian (k={gaussian_kernel_size}, sigma={sigma})")
plt.imshow(gaussian, cmap='gray')
plt.colorbar()

plt.subplot(2,2,3)
plt.title(f"Box (k={box_kernel_size})")
plt.imshow(box, cmap='gray')
plt.colorbar()

plt.subplot(2,2,4)
plt.title(f"Spectral Low-Pass (cutoff={cutoff_ratio})")
plt.imshow(spectral, cmap='gray')
plt.colorbar()

plt.tight_layout()
plt.show()


# -------------------------------------------------
# Comparison residual plots (COMMENTED OUT)
# -------------------------------------------------
"""
plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.title("Residual (Original - Gaussian)")
plt.imshow(res_gaussian, cmap='seismic')
plt.colorbar()

plt.subplot(2,3,2)
plt.title("Residual (Original - Box)")
plt.imshow(res_box, cmap='seismic')
plt.colorbar()

plt.subplot(2,3,3)
plt.title("Residual (Original - Spectral)")
plt.imshow(res_spectral, cmap='seismic')
plt.colorbar()

plt.tight_layout()
plt.show()
"""