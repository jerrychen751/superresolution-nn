import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N = 64 # grid size
x = torch.linspace(0, 1, N)
y = torch.linspace(0, 1, N)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Naming convention is _true for source
# (batch size (number of samples), channels (features per sample), height, width)
f_true = torch.zeros(1, 1, N, N) # 4D tensor of zeros

# Create the two spots
f_true[:, :, N//4, N//4] = 100
f_true[:, :, 3*N//4, 3*N//4] = 100


class LaplacianOperator(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.tensor([[[[0,  1, 0],
                                  [1, -4, 1],
                                  [0,  1, 0]]]], dtype=torch.float32)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        # overwrites random starting weights with Laplacian
        # also freezes it by not using gradients during backpropagation
        self.conv.weight = nn.Parameter(weights, requires_grad=False) 

    # Define forward propagation operation
    def forward(self, x):
        return self.conv(x)

class SolverCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # for each spatial location computes weighted sum of 3x3 neighborhood
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
    
solver = SolverCNN()
physics_op = LaplacianOperator()
optimizer = torch.optim.Adam(solver.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    u_pred = solver(f_true) # CNN predicts based on ground truth
    laplacian_pred = physics_op(u_pred) # strip 1 pixel border
    f_cropped = f_true[:, :, 1:-1, 1:-1] # take out borders
    loss = torch.mean((laplacian_pred - f_cropped)**2)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Physics Error: {loss.item():.6f}")    

