# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from pathlib import Path
# from models import CAE
# from data_utils import fetch_jhtdb_channel # Updated import

# # --- CONFIGURATION ---
# AUTH_TOKEN = "edu.gatech.jerrychen-6b7455a7" # Replace with your actual token
# latent_dim = 128               # Increased for turbulence complexity
# lr = 0.0005                    # Lowered learning rate for stability
# epochs = 100
# batch_size = 16

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize Model
# model = CAE(latent_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.MSELoss()

# # Fetch Turbulent Data from JHTDB
# # We fetch a batch. For production, consider a PyTorch DataLoader
# data = fetch_jhtdb_channel(token=AUTH_TOKEN, n_samples=200, size=64).to(device)

# loss_history = []

# print(f"Beginning training on JHTDB channel5200 (Device: {device})")
# for epoch in range(epochs):
#     # To prevent overfitting on a small fetched set, 
#     # we shuffle the data indices each epoch
#     indices = torch.randperm(data.size(0))
#     epoch_loss = 0
    
#     # Basic mini-batching
#     for i in range(0, len(data), batch_size):
#         batch_idx = indices[i:i+batch_size]
#         batch_data = data[batch_idx]

#         optimizer.zero_grad()
#         output = model(batch_data)
#         loss = criterion(output, batch_data)
        
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / (len(data)/batch_size)
#     loss_history.append(avg_loss)

#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# torch.save(model.state_dict(), "cae_turbulent_channel.pth")
# print("Training Complete. Model saved.")

# # Reuse your existing plot_loss_curve function here...
# # plot_loss_curve(loss_history, save_path=Path(__file__).parent/'turbulent_loss_curve.png')

# def plot_loss_curve(loss_history, save_path='ae_loss_curve.png'):
#     """
#     Plots the autoencoder reconstruction loss as a function of epochs
#     and saves it as a PNG file.

#     Args:
#         loss_history (list of float): loss value recorded at every epoch
#         save_path (str): filename to save the plot to
#     """
#     epochs_axis = list(range(1, len(loss_history) + 1))
#     # range(1, 501) gives [1, 2, 3, ..., 500] — human-friendly epoch numbers
#     # starting at 1 rather than 0.

#     fig, ax = plt.subplots(figsize=(9, 5))

#     ax.plot(epochs_axis, loss_history,
#             color='steelblue', linewidth=1.5, label='Reconstruction loss (MSE)')
#     # steelblue is a clean, readable colour for a single line.

#     # Mark the final loss value at the end of the curve so you can
#     # read the converged value without hovering over the plot.
#     ax.annotate(
#         f"Final: {loss_history[-1]:.6f}",
#         xy=(epochs_axis[-1], loss_history[-1]),
#         xytext=(-80, 15),
#         textcoords='offset points',
#         fontsize=9,
#         color='steelblue',
#         arrowprops=dict(arrowstyle='->', color='steelblue', lw=1)
#     )
#     # loss_history[-1] is the last item in the list — the final epoch's loss.
#     # xytext=(-80, 15) shifts the label 80 pixels left and 15 pixels up
#     # from the annotated point so it doesn't overlap the line.

#     ax.set_xlabel('Epoch', fontsize=11)
#     ax.set_ylabel('MSE Loss  (lower = better)', fontsize=11)
#     ax.set_title('Autoencoder Training Loss Curve\n'
#                  'Reconstruction error vs. number of training epochs',
#                  fontsize=12)

#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)

#     # Use a log scale on the y axis if the loss drops by more than
#     # two orders of magnitude — makes the early rapid drop visible
#     # at the same time as the fine detail of late-stage convergence.
#     if max(loss_history) / (min(loss_history) + 1e-12) > 100:
#         ax.set_yscale('log')
#         ax.set_ylabel('MSE Loss  (log scale, lower = better)', fontsize=11)

#     plt.tight_layout()
#     plt.savefig(Path(save_path), dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Loss curve saved to: {save_path}")

# plot_loss_curve(loss_history, save_path=Path(__file__).parent/'turbulent_loss_curve.png')




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from models import CAE 
# from data_utils import fetch_jhtdb_channel

# # --- CONFIG ---
# AUTH_TOKEN = "your_token_here" 
# LATENT_DIM = 128
# LR = 0.0001
# EPOCHS = 1000
# BATCH_SIZE = 16

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. Initialize the Model
# model = CAE(LATENT_DIM).to(device)
# optimizer = optim.Adam(model.parameters(), lr=LR)
# criterion = nn.MSELoss()

# # 2. Get the Data
# print("Attempting to load data from JHTDB...")
# raw_data = fetch_jhtdb_channel(token=AUTH_TOKEN, n_samples=100, size=64)

# if raw_data is None:
#     print("Stopping script due to data fetch failure.")
#     exit()

# train_data = raw_data.to(device)

# # 3. Training Loop
# loss_history = []
# print(f"Training started on {device}...")

# for epoch in range(EPOCHS):
#     model.train()
    
#     # Shuffle for every epoch
#     indices = torch.randperm(train_data.size(0))
#     epoch_loss = 0
    
#     for i in range(0, len(train_data), BATCH_SIZE):
#         batch_idx = indices[i:i + BATCH_SIZE]
#         batch = train_data[batch_idx]

#         optimizer.zero_grad()
#         output = model(batch)
#         loss = criterion(output, batch)
        
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / (len(train_data) / BATCH_SIZE)
#     loss_history.append(avg_loss)

#     if (epoch + 1) % 50 == 0:
#         print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

# torch.save(model.state_dict(), "cae_jhtdb_model.pth")
# print("Training Complete. Model saved as cae_jhtdb_model.pth")










# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from models import CAE 
# from data_utils import fetch_jhtdb_channel

# # --- CONFIG ---
# AUTH_TOKEN = "edu.gatech.jerrychen-6b7455a7" # Use your real SciServer token
# LATENT_DIM = 128               # Turbulence needs higher compression capacity
# LR = 1e-4
# EPOCHS = 1000
# BATCH_SIZE = 16

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. Load Data
# print("Initializing JHTDB Data Acquisition...")
# try:
#     # We fetch 100 samples to start. Note: this may take a few minutes.
#     train_data = fetch_jhtdb_channel(token=AUTH_TOKEN, n_samples=100, size=64).to(device)
# except Exception as e:
#     print(f"Data loading failed: {e}")
#     exit()

# # 2. Model Setup
# model = CAE(LATENT_DIM).to(device)
# optimizer = optim.Adam(model.parameters(), lr=LR)
# criterion = nn.MSELoss()

# loss_history = []

# # 3. Training Loop
# print(f"Training on {device}...")
# for epoch in range(EPOCHS):
#     model.train()
    
#     # Shuffle indices
#     indices = torch.randperm(train_data.size(0))
#     epoch_loss = 0
    
#     for i in range(0, len(train_data), BATCH_SIZE):
#         batch_idx = indices[i:i + BATCH_SIZE]
#         batch = train_data[batch_idx]

#         optimizer.zero_grad()
#         output = model(batch)
#         loss = criterion(output, batch)
        
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / (len(train_data) / BATCH_SIZE)
#     loss_history.append(avg_loss)

#     if (epoch + 1) % 50 == 0:
#         print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

# torch.save(model.state_dict(), "cae_turbulent_model.pth")
# print("Training finished. Weights saved to cae_turbulent_model.pth")









import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Save to file without needing a display
import matplotlib.pyplot as plt
from pathlib import Path

# Importing your custom modules
from models import CAE
from data_utils import fetch_jhtdb_channel

# --- CONFIGURATION ---
AUTH_TOKEN = "edu.gatech.jerrychen-6b7455a7" 
LATENT_DIM = 128
LR = 1e-3
EPOCHS = 800
BATCH_SIZE = 16
SAVE_PATH = "cae_turbulent_model.pth"
PLOT_PATH = "ae_loss_log_curve.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Model, Optimizer, and Loss
model = CAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 2. Fetch Turbulent Data
print("Fetching turbulent data from JHTDB...")
try:
    # Adjust n_samples based on your needs
    train_data = fetch_jhtdb_channel(token=AUTH_TOKEN, n_samples=400, size=64).to(device)
except Exception as e:
    print(f"Data loading failed: {e}")
    exit()

# 3. Training Loop
loss_history = []
print(f"Training started on {device}...")

for epoch in range(EPOCHS):
    model.train()
    indices = torch.randperm(train_data.size(0))
    epoch_loss = 0
    
    for i in range(0, len(train_data), BATCH_SIZE):
        batch_idx = indices[i:i + BATCH_SIZE]
        batch = train_data[batch_idx]

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / (len(train_data) / BATCH_SIZE)
    loss_history.append(avg_epoch_loss)

    if (epoch + 1) % (EPOCHS / 10) == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_epoch_loss:.6f}")

# 4. Save Model Weights
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved as {SAVE_PATH}")

# 5. Plot Logarithmic Loss Curve
def plot_log_loss(history, save_path):
    epochs_axis = list(range(1, len(history) + 1))
    
    # Use subplots instead of plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs_axis, history, color='darkcyan', linewidth=2, label='MSE Reconstruction Loss')
    
    # Set logarithmic scale for the Y-axis
    ax.set_yscale('log')
    
    # Labels and Title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax.set_title('Autoencoder Training: Logarithmic Loss Curve', fontsize=14)
    
    # Add grid for readability
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    # Annotate final loss
    final_loss = history[-1]
    ax.annotate(f'Final Loss: {final_loss:.6f}', 
                xy=(epochs_axis[-1], final_loss), 
                xytext=(epochs_axis[-1]*0.7, final_loss*1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Logarithmic loss plot saved to {save_path}")

plot_log_loss(loss_history, PLOT_PATH)