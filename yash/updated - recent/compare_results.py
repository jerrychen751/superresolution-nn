import torch
import matplotlib.pyplot as plt
import numpy as np
from models import CAE
from data_utils import fetch_jhtdb_channel

# --- SETTINGS ---
AUTH_TOKEN = "edu.gatech.jerrychen-6b7455a7"
MODEL_PATH = "cae_turbulent_model.pth"
LATENT_DIM = 128
SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Model
model = CAE(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set to evaluation mode

# 2. Fetch Fresh Test Data (5 samples)
print("Fetching fresh test samples from JHTDB...")
test_data = fetch_jhtdb_channel(token=AUTH_TOKEN, n_samples=400, size=SIZE).to(device)

# 3. Run Inference
with torch.no_grad():
    reconstructed = model(test_data)

# 4. Plotting Results (Comparing U-velocity component)
# We will look at the first sample [0] and its 3 channels [u, v, w]
sample_idx = 0
components = ['U (Streamwise)', 'V (Wall-normal)', 'W (Spanwise)']

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i in range(3):
    # Original Data
    orig = test_data[sample_idx, i].cpu().numpy()
    im1 = axes[i, 0].imshow(orig, cmap='viridis')
    axes[i, 0].set_title(f"Original {components[i]}")
    plt.colorbar(im1, ax=axes[i, 0])

    # Reconstructed Data
    recon = reconstructed[sample_idx, i].cpu().numpy()
    im2 = axes[i, 1].imshow(recon, cmap='viridis')
    axes[i, 1].set_title(f"Reconstructed {components[i]}")
    plt.colorbar(im2, ax=axes[i, 1])

plt.suptitle(f"JHTDB Channel 5200: Reconstruction Quality", fontsize=16)
plt.tight_layout()
plt.savefig("comparison_results.png")
plt.show()

print("Comparison plot saved as comparison_results.png")