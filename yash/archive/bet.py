import torch
import numpy as np
from models import CAE, Decoder
from data_utils import generate_laminar_channel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64

# load one sample
test_data = generate_laminar_channel(n_samples=1, size=64).to(device)
print("Input shape:", test_data.shape)

# load and run autoencoder
cae = CAE(latent_dim=latent_dim).to(device)
cae.load_state_dict(torch.load("cae_laminar.pth", map_location=device))
cae.eval()

with torch.no_grad():
    ae_out = cae(test_data)
    print("AE output shape:", ae_out.shape)

# load and run generator
gan_gen = Decoder(latent_dim=latent_dim).to(device)
gan_gen.load_state_dict(torch.load("gan_generator_laminar.pth", map_location=device))
gan_gen.eval()

with torch.no_grad():
    noise = torch.randn(1, latent_dim, device=device)
    gan_out = gan_gen(noise)
    print("GAN output shape:", gan_out.shape)