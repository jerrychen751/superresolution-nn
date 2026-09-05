### Overview

PyTorch DDP (Distributed Data Parallel) is a method of partitioning the dataset so that training occurs on multiple GPUs, offering a near-linear speedup in training time.

The same model weights are loaded on multiple GPUs. The convention is one process per GPU, where each process has access to the full training dataset. A `DistributedSampler` object partitions the indices so that each GPU gets an even number of samples.

Each GPU runs forward and backward independently, producing its own gradients. The gradients are then averaged across each GPU (so they all become identical) and then each calls `optimizer.step()` to update model parameters using learning rate / gradient.

### Code Adjustments

##### Starting Script

When there are `N` independent GPU processes, each process needs to have some context:

- What is `world_size`, or the total number of processes?
  - When partitioning data, PyTorch needs to know how many processes there are total.
- What is `rank`, which is this particular process's ID?
  - Some things should only be done once, like saving checkpointed model weights during the training process (rank == 0).
- What is `local_rank`, which is this particular process's GPU index on this node? (A node/machine may have multiple GPUs.)

The `torchrun` command handles all of that for you by setting environment variables that most PyTorch objects, when initialized, automatically read.

```bash
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m superresolution.train \
    processed_data_dir=$PROCESSED_DIR \
    train=hpc
```

##### Process Initialization

There are 3 main new imports:

```python
from torch.utils.data import Dataset, DataLoader, DistributedSampler # Partitions dataset
import torch.distributed as dist # Main API for distributed computing
from torch.nn.parallel import DistributedDataParallel as DDP # model wrapper that syncs gradients
```

```python
using_ddp = int(os.getenv("WORLD_SIZE", 1)) > 1
if using_ddp:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
```

##### Use DistributedSampler

Along with `DataLoader` objects, we initialize `DistributedSampler` objects for train/val datasets, with shuffling for training.

```python
train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)
train_loader = DataLoader(
    ...,
    sampler=train_sampler  # replaces shuffle=True
)
```

##### Wrap Model in DDP

This step ensures that the all-reduce algorithm can work across all processes.

```python
model = ModelClass().to(device)   
if using_ddp:
    model = DDP(model, device_ids=[local_rank])
```

##### Compute Loss

The first step is to properly seed the sampler (specifically the one for training) since it shuffles. We can use the epoch number as the random seed.

```python
if using_ddp:
    train_sampler.set_epoch(epoch)
```

Then, we need to adjust how we aggregate the training loss. Since local loss on a particular process is only the loss for a partition of the data, we need to sum across all processes and then divide by the total training dataset size.

```python
if using_ddp:
    local_train_loss = torch.tensor(train_loss, device=device)
    dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
    train_loss = local_train_loss.item() / len(train_ds)
```

##### Checkpointing Model Weights + Teardown

```python
if epoch % 50 == 0:
    if not using_ddp or dist.get_rank() == 0:
        checkpoint = {
            "epoch": epoch,
            "model": model.module.state_dict() if using_ddp else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
```

Both ranks evaluate `epoch % 50 == 0` on the same integer, so no barrier is needed here. One barrier before teardown stops a rank from destroying the communicator while rank 0 is still writing.

```python
if using_ddp:
    dist.barrier()
    dist.destroy_process_group()
```