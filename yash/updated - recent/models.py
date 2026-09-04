# import torch
# import torch.nn as nn

# class Decoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Linear(latent_dim, 64 * 16 * 16),
#             nn.Unflatten(1, (64, 16, 16)),
#             nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 3, 4, stride = 2, padding = 1),
#             nn.Tanh()
#         )
    
#     def forward(self, x): return self.main(x)

# class CAE(nn.Module):
#     def __init__(self, latent_dim = 128):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 4, stride = 2, padding = 1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride = 2, padding = 1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 16, latent_dim)
#         )
#         self.decoder = Decoder(latent_dim)

#     def forward(self, x):
#         return self.decoder(self.encoder(x))
    
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 32, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, 4, stride=2, padding=1), # Output is 64 channels
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 16, 1), # FIXED: Changed 128 to 64
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.main(x)









import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder: Compresses (3, 64, 64) -> (latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), # Output: (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # Output: (64, 16, 16)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, latent_dim)
        )
        
        # Decoder: Decompresses (latent_dim) -> (3, 64, 64)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # Output: (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # Output: (3, 64, 64)
            nn.Sigmoid() # Use Sigmoid for [0, 1] data, Tanh for [-1, 1] data
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction