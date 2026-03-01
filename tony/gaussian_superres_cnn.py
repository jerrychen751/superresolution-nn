import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# ----------------------------
# Parameters
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

high_res = 128
low_res = 64
sigma = 1.2
epochs = 5
batch_size = 32

# ----------------------------
# Dataset (MNIST)
# ----------------------------
transform = T.Compose([
    T.Resize((high_res, high_res)),
    T.ToTensor()
])

dataset = datasets.MNIST(root="./data",
                         train=True,
                         download=True,
                         transform=transform)

# Use only 5000 samples for speed
subset = Subset(dataset, range(5000))

loader = DataLoader(subset,
                    batch_size=batch_size,
                    shuffle=True)

# ----------------------------
# Gaussian Kernel (PyTorch)
# ----------------------------
def get_gaussian_kernel(size=9, sigma=1.2):
    ax = torch.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

kernel_size = 9
kernel = get_gaussian_kernel(kernel_size, sigma).to(device)
kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,k,k)

# ----------------------------
# Degradation: Gaussian + Downsample
# ----------------------------
def degrade(x):
    # Blur
    x = F.conv2d(x, kernel, padding=kernel_size//2)

    # Downsample by factor 2
    x = F.interpolate(x, scale_factor=0.5, mode='bilinear')

    return x

# ----------------------------
# Simple CNN Super-Resolution
# ----------------------------
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        # Upsample back to high resolution
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return self.net(x)

model = SRCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):
    total_loss = 0

    for imgs, _ in loader:
        imgs = imgs.to(device)

        low = degrade(imgs)
        preds = model(low)

        loss = criterion(preds, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# ----------------------------
# Visualization
# ----------------------------
model.eval()

imgs, _ = next(iter(loader))
imgs = imgs.to(device)
low = degrade(imgs)

with torch.no_grad():
    preds = model(low)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(imgs[0][0].cpu(), cmap='gray')

plt.subplot(1,3,2)
plt.title("Low-res (Blur+Down)")
plt.imshow(low[0][0].cpu(), cmap='gray')

plt.subplot(1,3,3)
plt.title("Reconstructed")
plt.imshow(preds[0][0].cpu(), cmap='gray')

plt.tight_layout()
plt.show()

# ----------------------------
# Test on custom image
# ----------------------------

image_path = "tony/images/buzz_real.jpg"  # <-- change to your file

# Load image
img = Image.open(image_path).convert("L")  # convert to grayscale
img = img.resize((high_res, high_res))

transform = T.ToTensor()
img_tensor = transform(img).unsqueeze(0).to(device)

# Degrade
low_img = degrade(img_tensor)

# Reconstruct
model.eval()
with torch.no_grad():
    recon_img = model(low_img)

# Plot
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_tensor[0][0].cpu(), cmap='gray')

plt.subplot(1,3,2)
plt.title("Low-res")
plt.imshow(low_img[0][0].cpu(), cmap='gray')

plt.subplot(1,3,3)
plt.title("Reconstructed")
plt.imshow(recon_img[0][0].cpu(), cmap='gray')

plt.tight_layout()
plt.show()