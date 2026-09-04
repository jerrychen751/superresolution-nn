import torch
import matplotlib.pyplot as plt
import numpy as np
from models import CAE, Decoder
from data_utils import generate_laminar_channel
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# 1. Setup and Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64 # Match the latent_dim used in your training scripts

# Load Truth Data (1 sample)
test_data = generate_laminar_channel(n_samples=1, size=64).to(device)
truth = test_data.cpu().squeeze().numpy()[0] # Horizontal velocity 'u'

# Load Trained AE
cae = CAE(latent_dim=latent_dim).to(device)
cae.load_state_dict(torch.load("cae_laminar.pth", map_location=device))
cae.eval()

# Load Trained GAN (Generator)
gan_gen = Decoder(latent_dim=latent_dim).to(device)
gan_gen.load_state_dict(torch.load("gan_generator_laminar.pth", map_location=device))
gan_gen.eval()

# 2. Inference
with torch.no_grad():
    # AE reconstruction
    ae_out = cae(test_data).cpu().squeeze().numpy()[0]
    # GAN synthesis from random noise
    noise = torch.randn(1, latent_dim, device=device)
    gan_out = gan_gen(noise).cpu().squeeze().numpy()[0]

# 3. Visualization: Efficiency and Accuracy
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Velocity Fields
titles = ['Truth (Laminar)', 'AE Reconstruction', 'GAN Synthesis']
images = [truth, ae_out, gan_out]

for i in range(3):
    im = axs[0, i].imshow(images[i], extent=[0, 1, -1, 1], cmap='jet')
    axs[0, i].set_title(titles[i])
    fig.colorbar(im, ax=axs[0, i])

# Row 2: Error Maps (Efficiency Metric)
# This shows exactly where the models fail to capture the physics
ae_error = np.abs(truth - ae_out)
gan_error = np.abs(truth - gan_out)

axs[1, 0].axis('off') # Keep layout clean
im1 = axs[1, 1].imshow(ae_error, extent=[0, 1, -1, 1], cmap='inferno')
axs[1, 1].set_title('AE Absolute Error')
fig.colorbar(im1, ax=axs[1, 1])

im2 = axs[1, 2].imshow(gan_error, extent=[0, 1, -1, 1], cmap='inferno')
axs[1, 2].set_title('GAN Absolute Error')
fig.colorbar(im2, ax=axs[1, 2])

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'velocity_fields_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: velocity_fields_comparison.png")

# 4. Profile Comparison (The "Physics" Check)
plt.figure(figsize=(8, 6))
y_coords = np.linspace(-1, 1, 64)
plt.plot(truth[:, 32], y_coords, 'k-', label='Analytical Truth', linewidth=3)
plt.plot(ae_out[:, 32], y_coords, 'r--', label='AE Profile', linewidth=2)
plt.plot(gan_out[:, 32], y_coords, 'g:', label='GAN Profile', linewidth=2)
plt.xlabel('Horizontal Velocity (u)')
plt.ylabel('Channel Height (y)')
plt.title('Centerline Velocity Profile Comparison')
plt.legend()
plt.grid(True)
plt.savefig(SCRIPT_DIR / 'velocity_profile_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: velocity_profile_comparison.png")