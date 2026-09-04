import numpy as np
import torch

def generate_laminar_channel(n_samples = 100, size = 64):
    data = []
    y = np.linspace(-1, 1, size)
    u_profile = 1 - y**2

    for _ in range(n_samples):
        u_max = np.random.uniform(0.8, 1.2)
        slice_u = np.tile(u_profile * u_max, (size, 1)).T
        slice_v = np.zeros((size, size))
        slice_w = np.zeros((size, size))

        sample = np.stack([slice_u, slice_v, slice_w], axis = 0)
        data.append(sample)
    
    return torch.tensor(np.array(data), dtype = torch.float32)