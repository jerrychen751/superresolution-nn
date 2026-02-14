import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. Load image (grayscale)
# -------------------------
name = "test.jpg"  # Change this to the image you want to test

img = cv2.imread("tony/" + name, 0)

# Resize to 128x128
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

# Convert to float for better math
img = img.astype(np.float32) / 255.0

# -------------------------
# 2. Apply Gaussian blur
# -------------------------
gaussian = cv2.GaussianBlur(img, (11, 11), sigmaX=3)

# -------------------------
# 3. Apply Box blur
# -------------------------
box = cv2.blur(img, (11, 11))

# -------------------------
# 4. Compute residuals
# -------------------------
res_gaussian = img - gaussian
res_box = img - box

# -------------------------
# 5. Visualization
# -------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.colorbar()

plt.subplot(2,3,2)
plt.title("Gaussian Blur")
plt.imshow(gaussian, cmap='gray')
plt.colorbar()

plt.subplot(2,3,3)
plt.title("Box Blur")
plt.imshow(box, cmap='gray')
plt.colorbar()

plt.subplot(2,3,4)
plt.title("Residual (Gaussian)")
plt.imshow(res_gaussian, cmap='seismic')
plt.colorbar()

plt.subplot(2,3,5)
plt.title("Residual (Box)")
plt.imshow(res_box, cmap='seismic')
plt.colorbar()

plt.tight_layout()
plt.show()

