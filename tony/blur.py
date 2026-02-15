import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. Load image
# -------------------------
name = "tech_tower.jpg"  # Change this to the image you want to test

# Use cv2.IMREAD_COLOR_RGB for color images, cv2.IMREAD_GRAYSCALE for grayscale images
img = cv2.imread("tony/images/" + name, cv2.IMREAD_GRAYSCALE)


img_size = 1024    # Change this to the desired size (e.g., 128, 256, 512)
img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

# Convert to float32 and normalize to [0, 1]
img = img.astype(np.float32) / 255.0

# if color, img now is a 3-channel color 2-D image (H, W, 3) with values in [0, 1].
# if grayscale, img is a 2-D image (H, W) with values in [0, 1].

# -------------------------
# 2. Apply Gaussian blur
# -------------------------
gaussian_kernel_size = 31  # Change this to larger number for a stronger blur effect (should be odd, e.g., 3, 5, 7, ..., 31)
sigma = 5  # Standard deviation for Gaussian kernel
gaussian = cv2.GaussianBlur(img, (gaussian_kernel_size, gaussian_kernel_size), sigmaX=sigma)

# -------------------------
# 3. Apply Box blur
# -------------------------
# Change the kernel size to larger number for a stronger blur effect
box_kernel_size = 31
box = cv2.blur(img, (box_kernel_size, box_kernel_size))

# -------------------------
# 4. Compute residuals
# -------------------------
res_gaussian = img - gaussian
res_box = img - box
box_gaussian_diff = box - gaussian

# -------------------------
# 5. Visualization
# -------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.colorbar()

plt.subplot(2,3,2)
plt.title("Gaussian: kernel size = {}, sigma = {}".format(gaussian_kernel_size, sigma), fontsize=10)
plt.imshow(gaussian, cmap='gray')
plt.colorbar()

plt.subplot(2,3,3)
plt.title("Box: kernel size = {}".format(box_kernel_size), fontsize=10)
plt.imshow(box, cmap='gray')
plt.colorbar()

plt.subplot(2,3,4)
plt.title("Residual (Original - Gaussian)")
plt.imshow(res_gaussian, cmap='seismic')
plt.colorbar()

plt.subplot(2,3,5)
plt.title("Residual (Original - Box)")
plt.imshow(res_box, cmap='seismic')
plt.colorbar()

plt.subplot(2,3,6)
plt.title("Difference (Box - Gaussian)")
plt.imshow(box_gaussian_diff, cmap='seismic')
plt.colorbar()

plt.tight_layout()
plt.show()