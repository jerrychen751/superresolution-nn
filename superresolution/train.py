"""
Trains the model. Assumes that superresolution/data/processed already contains training data.


"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import SuperResolutionCNN

# Configuration
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
EPOCHS = 1000
TRAIN_RATIO = 0.8

class SuperResolutionDataset(Dataset):
    """
    Helps fetch the i-th training example when data is loaded from disk.
    """

    def __init__(
        self,
        inputs: list[Path],
        targets: list[Path]
    ) -> None:
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape from disk: (nz, ny, nx, 3) for axis 0, 1, 2, 3
        input_data = np.load(self.inputs[i]).astype(np.float32)
        target_data = np.load(self.targets[i]).astype(np.float32)

        # Move channels to front: (nz, ny, nx, 3) → (3, nz, ny, nx)
        # Conv3d expects (batch_size, C, D, H, W)
        input_data = np.transpose(input_data, (3, 0, 1, 2))
        target_data = np.transpose(target_data, (3, 0, 1, 2))
        return torch.from_numpy(input_data), torch.from_numpy(target_data)


def train_eval():
    # Construct Datasets
    input_fps = sorted(PROCESSED_DIR.glob('input_t*.npy'))
    target_fps = sorted(PROCESSED_DIR.glob('target_t*.npy'))
    n_train = int(TRAIN_RATIO * len(input_fps))

    train_ds = SuperResolutionDataset(input_fps[:n_train], target_fps[:n_train])
    test_ds = SuperResolutionDataset(input_fps[n_train:], target_fps[n_train:])

    # Wrap Datasets in DataLoader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True) # don't always pair up same samples
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False) # deterministic pairing

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Model instantiation
    model = SuperResolutionCNN().to(device)

    # Select optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    # Learning rate adjustments (cosine annealing)
    # T_max defines number of steps before reaching minimum in first quarter of cosine wave (1 -> 0)
    # eta_max is original learning rate of optimizer, eta_min is the lowest it can go to
    # CosineAnnealingLR starts back up at last quarter of period (0 -> 1) after reaching T_max but this behavior is unhelpful so set T_max as the number of training iterations
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Define loss function
    criterion = torch.nn.MSELoss() # (prediction - target)^2

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0 # average loss per epoch
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad() # clear old gradients
            loss.backward() # compute and attach gradients to params
            optimizer.step() # update weights

            # Increment total loss
            # loss.item() returns average MSE over the batch
            # inputs.size(0) returns batch size
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_ds) # average over all training samples

        # Track validation/test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                loss = criterion(model(inputs), targets)
                test_loss += loss.item() * inputs.size(0)
        
        test_loss /= len(test_ds)

        # Track loss statistics for both training and test
        # These should get written to a file later on, when moving to ICE
        if epoch % 50 == 0:
            print(f"Training loss: {train_loss}")
            print(f"Testing loss: {test_loss}")
            
        scheduler.step() # adjust LR before the next epoch


if __name__ == '__main__':
    train_eval()