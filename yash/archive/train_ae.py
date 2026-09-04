import torch
import torch.nn as nn
import torch.optim as optim
from models import CAE
from data_utils import generate_laminar_channel

latent_dim = 64
lr = 0.001
epochs = 500
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
data = generate_laminar_channel(n_samples = 500, size = 64).to(device)

print("Beginning of training for laminar flow with Autoencoder")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "cae_laminar.pth")
print("End of training. Model saved as cae_laminar.pth")