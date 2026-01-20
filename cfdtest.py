# # import torch
# # import matplotlib.pyplot as plt

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # nx, ny = 100, 100
# # Lx, Ly = 1.0, 1.0
# # dx, dy = Lx/(nx-1), Ly/(ny-1)

# # # Time
# # dt = 1e-4
# # nt = 1000
# # nu = 0.1

# # # Create grid
# # x = torch.linspace(0, Lx, nx, device=device)
# # y = torch.linspace(0, Ly, ny, device=device)
# # X, Y = torch.meshgrid(x, y, indexing="ij")

# # # Initial condition: hot Gaussian blob
# # u = torch.exp(-50*((X-0.5)**2 + (Y-0.5)**2))

# # # Time stepping
# # for n in range(nt):
# #     u_xx = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2
# #     u_yy = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2

# #     u[1:-1,1:-1] += dt * nu * (u_xx + u_yy)

# #     # Dirichlet BCs
# #     u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 0.0

# # # Plot
# # plt.imshow(u.cpu(), origin="lower", extent=[0,1,0,1])
# # plt.colorbar()
# # plt.title("2D Heat Equation Solution")
# # plt.show()

# import torch
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Grid
# nx, ny = 101, 101
# Lx, Ly = 1.0, 1.0
# dx, dy = Lx/(nx-1), Ly/(ny-1)

# # Time
# dt = 5e-4
# nt = 500
# nu = 0.01

# # Grid tensors
# x = torch.linspace(0, Lx, nx, device=device)
# y = torch.linspace(0, Ly, ny, device=device)
# X, Y = torch.meshgrid(x, y, indexing="ij")

# # Initial condition: vortex-like pattern
# u = -torch.sin(torch.pi * X) * torch.cos(torch.pi * Y)
# v =  torch.cos(torch.pi * X) * torch.sin(torch.pi * Y)

# for n in range(nt):
#     un = u.clone()
#     vn = v.clone()

#     # First derivatives (central difference)
#     u_x = (un[2:,1:-1] - un[:-2,1:-1]) / (2*dx)
#     u_y = (un[1:-1,2:] - un[1:-1,:-2]) / (2*dy)
#     v_x = (vn[2:,1:-1] - vn[:-2,1:-1]) / (2*dx)
#     v_y = (vn[1:-1,2:] - vn[1:-1,:-2]) / (2*dy)

#     # Second derivatives
#     u_xx = (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]) / dx**2
#     u_yy = (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) / dy**2
#     v_xx = (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]) / dx**2
#     v_yy = (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]) / dy**2

#     # Update interior
#     u[1:-1,1:-1] = un[1:-1,1:-1] - dt * (
#         un[1:-1,1:-1] * u_x + vn[1:-1,1:-1] * u_y
#     ) + dt * nu * (u_xx + u_yy)

#     v[1:-1,1:-1] = vn[1:-1,1:-1] - dt * (
#         un[1:-1,1:-1] * v_x + vn[1:-1,1:-1] * v_y
#     ) + dt * nu * (v_xx + v_yy)

#     # Boundary conditions (Dirichlet)
#     u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 0.0
#     v[0,:] = v[-1,:] = v[:,0] = v[:,-1] = 0.0

# # Plot final velocity magnitude
# speed = torch.sqrt(u**2 + v**2).cpu()
# plt.imshow(speed, origin="lower", extent=[0,1,0,1])
# plt.colorbar()
# plt.title("2D Burgers' Equation | Velocity Magnitude")
# plt.show()




# import torch
# import matplotlib.pyplot as plt

# # Device (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Grid parameters
# nx, ny = 50, 50
# Lx, Ly = 1.0, 1.0
# dx, dy = Lx/(nx-1), Ly/(ny-1)

# # Time parameters
# dt = 0.01
# nt = 50  # number of time steps
# nu = 0.1  # diffusion coefficient

# # Create grid
# x = torch.linspace(0, Lx, nx, device=device)
# y = torch.linspace(0, Ly, ny, device=device)
# X, Y = torch.meshgrid(x, y, indexing="ij")

# # Initial condition: hot spot in the center
# u = torch.zeros((nx, ny), device=device)
# u[nx//2, ny//2] = 1.0

# # Time-stepping loop (explicit finite difference)
# for n in range(nt):
#     u_xx = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2
#     u_yy = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
#     u[1:-1,1:-1] += dt * nu * (u_xx + u_yy)

#     # Dirichlet boundary conditions
#     u[0,:] = 0; u[-1,:] = 0
#     u[:,0] = 0; u[:,-1] = 0

# # Plot the final temperature field
# plt.imshow(u.cpu(), origin="lower", extent=[0,1,0,1], cmap="hot")
# plt.colorbar(label="Temperature")
# plt.title("2D Heat Diffusion (CFD-style)")
# plt.show()





import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Generate training data (2D function)
# ------------------------
# Example: u(x,y) = sin(pi*x) * sin(pi*y)
nx, ny = 50, 50
x = torch.linspace(0, 1, nx)
y = torch.linspace(0, 1, ny)
X, Y = torch.meshgrid(x, y, indexing="ij")
u_true = torch.sin(torch.pi*X) * torch.sin(torch.pi*Y)

# Flatten for training (N, 2)
inputs = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
targets = u_true.flatten().unsqueeze(1).to(device)  # (N,1)

# ------------------------
# Define a small neural network
# ------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(device)

# ------------------------
# Loss and optimizer
# ------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ------------------------
# Training loop
# ------------------------
epochs = 400
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ------------------------
# Make predictions
# ------------------------
u_pred = model(inputs).detach().cpu().reshape(nx, ny)

# ------------------------
# Plot results
# ------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("True Function")
plt.imshow(u_true, origin="lower", extent=[0,1,0,1], cmap="viridis")
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Neural Network Prediction")
plt.imshow(u_pred, origin="lower", extent=[0,1,0,1], cmap="viridis")
plt.colorbar()
plt.show()









# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Grid (keep small for laptop)
# nx, ny, nz = 20, 20, 20
# x = torch.linspace(0, 1, nx)
# y = torch.linspace(0, 1, ny)
# z = torch.linspace(0, 1, nz)

# X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
# u_true = torch.sin(torch.pi*X) * torch.sin(torch.pi*Y) * torch.sin(torch.pi*Z)

# # Flatten for training
# inputs = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).to(device)
# targets = u_true.flatten().unsqueeze(1).to(device)

# # Simple network
# class Net3D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
#     def forward(self, x):
#         return self.net(x)

# model = Net3D().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.MSELoss()

# # Training
# epochs = 200
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     output = model(inputs)
#     loss = criterion(output, targets)
#     loss.backward()
#     optimizer.step()
#     if epoch % 50 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# # Predictions
# u_pred = model(inputs).detach().cpu().reshape(nx, ny, nz)

# # Visualize a central slice
# slice_idx = nz // 2
# plt.imshow(u_pred[:, :, slice_idx], origin="lower", extent=[0,1,0,1], cmap="viridis")
# plt.colorbar()
# plt.title("3D function (central slice)")
# plt.show()
