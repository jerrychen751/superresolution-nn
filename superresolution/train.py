"""
Containing training and evaluation code for the model. Parameters are stored at checkpoint intervals.
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import SuperResolutionCNN

import hydra
from hydra.core.config_store import ConfigStore
from .config import SuperResolutionConfig


cs = ConfigStore.instance()
cs.store(name="config", node=SuperResolutionConfig)


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

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_eval(cfg: SuperResolutionConfig):
    using_ddp = int(os.getenv("WORLD_SIZE", 1)) > 1
    if using_ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # Resolve processed data directory
    if cfg.processed_data_dir:
        processed_dir = Path(cfg.processed_data_dir)
    else:
        processed_dir = Path(__file__).resolve().parent / "data" / "processed"

    # Construct Datasets
    input_fps = sorted(processed_dir.glob('input_t*.npy'))
    target_fps = sorted(processed_dir.glob('target_t*.npy'))
    n_train = int(cfg.train.train_ratio * len(input_fps))

    train_ds = SuperResolutionDataset(input_fps[:n_train], target_fps[:n_train])
    test_ds = SuperResolutionDataset(input_fps[n_train:], target_fps[n_train:])

    # Wrap Datasets in DataLoader
    if using_ddp:
        # Sampler helps partition dataset into disjoint subsets for each process
        train_sampler = DistributedSampler(
            dataset=train_ds,
            shuffle=True
        )
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=cfg.train.batch_size,
            pin_memory=True,
            num_workers=cfg.train.num_workers,
            sampler=train_sampler # stores a reference
        )
        test_sampler = DistributedSampler(
            dataset=test_ds,
            shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=cfg.train.batch_size,
            pin_memory=True,
            num_workers=cfg.train.num_workers,
            sampler=test_sampler
        )
    else:
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True, # don't always pair up same samples
            num_workers=cfg.train.num_workers,
        )
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False, # deterministic pairing
            num_workers=cfg.train.num_workers,
        )

    # Model instantiation
    model = SuperResolutionCNN().to(device)
    if using_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Select optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.train.learning_rate
    )

    # Learning rate adjustments (cosine annealing)
    # T_max defines number of steps before reaching minimum in first quarter of cosine wave (1 -> 0)
    # eta_max is original learning rate of optimizer, eta_min is the lowest it can go to
    # CosineAnnealingLR starts back up at last quarter of period (0 -> 1) after reaching T_max but this behavior is unhelpful so set T_max as the number of training iterations
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=cfg.train.eta_min
    )

    # Define loss function
    criterion = torch.nn.MSELoss() # (prediction - target)^2

    # Training loop
    for epoch in range(cfg.train.epochs):
        # Sampler uses current epoch as a seed for generating partitions
        if using_ddp:
            train_sampler.set_epoch(epoch)

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

        if using_ddp:
            local_train_loss = torch.tensor(train_loss, device=device)
            dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
            train_loss = local_train_loss.item() / len(train_ds)
        else:
            train_loss /= len(train_ds) # average over all training samples

        # Track validation/test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = criterion(model(inputs), targets)
                test_loss += loss.item() * inputs.size(0)
        
        if using_ddp:
            local_test_loss = torch.tensor(test_loss, device=device)
            dist.all_reduce(local_test_loss, dist.ReduceOp.SUM)
            test_loss = local_test_loss.item() / len(test_ds)
        else:
            test_loss /= len(test_ds)

        # Save the model at checkpoints, as well as loss stats
        if epoch % 50 == 0:
            if not using_ddp or dist.get_rank() == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict() if using_ddp else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss
                }
                torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")

            if using_ddp:
                # Block other processes until all reach this point; all ranks should hit this
                dist.barrier()

            
        scheduler.step() # adjust LR before the next epoch

    if using_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    train_eval()