import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load Test image
# ----------------------------
name = "tech_tower.jpg"  # Change this to the image you want to test

# Use cv2.IMREAD_COLOR_RGB for color images, cv2.IMREAD_GRAYSCALE for grayscale images
img = cv2.imread("superresolution_experiments/images/" + name, cv2.IMREAD_GRAYSCALE)

n = 64    # Change this to the desired size (e.g., 49, 64, 81, 100)
img = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)

# Flattened size
N = n * n
kernel_size = 7
pad = kernel_size // 2
box_size = kernel_size ** 2

# ----------------------------
# Build blur matrix A
# ----------------------------
A = np.zeros((N, N))

def index(i, j):
    return i * n + j

for i in range(n):
    for j in range(n):
        row = index(i, j)
        
        # neighborhood
        for di in range(-pad, pad + 1):
            for dj in range(-pad, pad + 1):
                ni = i + di
                nj = j + dj
                
                if 0 <= ni < n and 0 <= nj < n:
                    col = index(ni, nj)
                    A[row, col] += 1.0 / box_size

# ----------------------------
# Forward blur
# ----------------------------
f = img.flatten()
g = A @ f
blurred = g.reshape(n, n)

# ----------------------------
# Try inversion
# ----------------------------
# Condition number
cond = np.linalg.cond(A)
print("Condition number of A:", cond)

# Solve (may be unstable)
try:
    f_recovered = np.linalg.solve(A, g)
    recovered = f_recovered.reshape(n, n)
    
    print("Reconstruction error:",
          np.linalg.norm(recovered - img))
    
except np.linalg.LinAlgError:
    print("Matrix is singular!")

# ----------------------------
# Visual comparison
# ----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1,3,2)
plt.title("Blurred")
plt.imshow(blurred, cmap='gray')

plt.subplot(1,3,3)
plt.title("Recovered")
plt.imshow(recovered, cmap='gray')

plt.tight_layout()
plt.show()
