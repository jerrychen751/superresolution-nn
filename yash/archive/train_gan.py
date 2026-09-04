import torch
import torch.nn as nn
import torch.optim as optim
from models import Decoder, Discriminator
from data_utils import generate_laminar_channel

latent_dim = 64
lr = 0.0002
epochs = 800
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Decoder(latent_dim).to(device)
netD = Discriminator().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()
real_data = generate_laminar_channel(n_samples = 500, size = 64).to(device)

print("Beginning of training for laminar flow with GAN")
for epoch in range(epochs):
    netD.zero_grad()
    
    label_real = torch.full((batch_size,), 1.0, device=device)
    idx = torch.randint(0, len(real_data), (batch_size,))
    real_batch = real_data[idx]
    
    output = netD(real_batch).view(-1)
    errD_real = criterion(output, label_real)
    errD_real.backward()

    noise = torch.randn(batch_size, latent_dim, device=device)
    fake_batch = netG(noise)
    label_fake = torch.full((batch_size,), 0.0, device=device)
    
    output = netD(fake_batch.detach()).view(-1)
    errD_fake = criterion(output, label_fake)
    errD_fake.backward()
    optimizerD.step()

    netG.zero_grad()
    label_g = torch.full((batch_size,), 1.0, device=device)
    output = netD(fake_batch).view(-1)
    errG = criterion(output, label_g)
    errG.backward()
    optimizerG.step()

    if (epoch+1) % 80 == 0:
        print(f"[{epoch+1}/{epochs}] Loss_D: {errD_real+errD_fake:.4f} Loss_G: {errG:.4f}")

torch.save(netG.state_dict(), "gan_generator_laminar.pth")
print("End of training. Model saved as gan_generator_laminar.pth")