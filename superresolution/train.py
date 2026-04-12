"""
Containing training and evaluation code for the model. Parameters are stored at checkpoint intervals.
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra.utils
import hydra
from hydra.core.config_store import ConfigStore
from .config import SuperResolutionConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=SuperResolutionConfig)


def _get_data_classes(model_name: str):
    """
    Return (Dataset class, DataLoader class) for the given model. GNN uses
    torch_geometric's DataLoader because its batching collates Data objects
    by concatenating nodes, not stacking tensors.
    """
    if model_name == "cnn":
        from .models.cnn import CNNDataset
        from torch.utils.data import DataLoader
        return CNNDataset, DataLoader
    if model_name == "upsample_cnn":
        from .models.upsample_cnn import UpsampleCNNDataset
        from torch.utils.data import DataLoader
        return UpsampleCNNDataset, DataLoader
    if model_name == "closure_cnn":
        from .models.closure_cnn import ClosureCNNDataset
        from torch.utils.data import DataLoader
        return ClosureCNNDataset, DataLoader
    if model_name == "fno":
        from .models.fno import FNODataset
        from torch.utils.data import DataLoader
        return FNODataset, DataLoader
    if model_name == "gnn":
        from .models.gnn import GNNDataset
        from torch_geometric.loader import DataLoader
        return GNNDataset, DataLoader
    raise ValueError(f"Unknown model name: {model_name}")


def _unpack_batch(batch, device, is_graph: bool):
    """
    Normalize a DataLoader batch into (model_input, target, batch_size). For
    graph models the Data object is itself the model input; for dense models
    it's the input tensor.
    """
    if is_graph:
        batch = batch.to(device)
        return batch, batch.y, batch.num_graphs
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets, inputs.size(0)

@hydra.main(version_base=None, config_path="configs", config_name="cnn")
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

    # Resolve data and output directories
    base_dir = Path(__file__).resolve().parent

    if cfg.processed_data_dir:
        processed_dir = Path(cfg.processed_data_dir)
    else:
        processed_dir = base_dir / "data" / "processed"

    if cfg.checkpoints_dir:
        checkpoints_dir = Path(cfg.checkpoints_dir)
    else:
        checkpoints_dir = base_dir / "checkpoints" / cfg.model.name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Construct Datasets
    input_fps = sorted(processed_dir.glob('input_t*.npy'))
    target_fps = sorted(processed_dir.glob('target_t*.npy'))
    n_train = int(cfg.train.train_ratio * len(input_fps))

    DatasetCls, LoaderCls = _get_data_classes(cfg.model.name)
    is_graph = cfg.model.name == "gnn"

    train_ds = DatasetCls(input_fps[:n_train], target_fps[:n_train])
    test_ds = DatasetCls(input_fps[n_train:], target_fps[n_train:])

    # Wrap Datasets in DataLoader
    if using_ddp:
        # Sampler helps partition dataset into disjoint subsets for each process
        train_sampler = DistributedSampler(
            dataset=train_ds,
            shuffle=True
        )
        train_loader = LoaderCls(
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
        test_loader = LoaderCls(
            dataset=test_ds,
            batch_size=cfg.train.batch_size,
            pin_memory=True,
            num_workers=cfg.train.num_workers,
            sampler=test_sampler
        )
    else:
        train_loader = LoaderCls(
            dataset=train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True, # don't always pair up same samples
            num_workers=cfg.train.num_workers,
        )
        test_loader = LoaderCls(
            dataset=test_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False, # deterministic pairing
            num_workers=cfg.train.num_workers,
        )

    # Hydra resolves _target_ to the model class and passes remaining keys as constructor kwargs (from configs/model/<name>.yaml).
    model = hydra.utils.instantiate(cfg.model.params).to(device)
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
    for epoch in range(1, cfg.train.epochs + 1):
        # Sampler uses current epoch as a seed for generating partitions
        if using_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0 # average loss per epoch
        for batch in train_loader:
            model_input, targets, bs = _unpack_batch(batch, device, is_graph)

            # Forward pass
            predictions = model(model_input)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad() # clear old gradients
            loss.backward() # compute and attach gradients to params
            optimizer.step() # update weights

            # loss.item() returns average MSE over the batch; bs scales it per-sample.
            train_loss += loss.item() * bs

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
            for batch in test_loader:
                model_input, targets, bs = _unpack_batch(batch, device, is_graph)
                loss = criterion(model(model_input), targets)
                test_loss += loss.item() * bs
        
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
                torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch}.pt")

            if using_ddp:
                # Block other processes until all reach this point; all ranks should hit this
                dist.barrier()

            
        scheduler.step() # adjust LR before the next epoch

    if using_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    train_eval()