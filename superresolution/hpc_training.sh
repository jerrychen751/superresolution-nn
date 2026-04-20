#!/bin/bash

#SBATCH --job-name=superres
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2              # H200, 141GB HBM3e per GPU
#SBATCH --ntasks-per-node=1            # Stays 1; torchrun spawns child processes equal to number of GPUs (tasks = GPU ct)
#SBATCH --cpus-per-task=5               # nproc_per_node * (num_workers + 1)
#SBATCH --mem=32G
#SBATCH --time=08:00:00                # 2 gpus * 480 min = 960 gpu-min, coe-ice qos max per job
#SBATCH --output=logs/%j.out           # stdout -> logs/<jobid>.out
#SBATCH --error=logs/%j.err            # stderr -> logs/<jobid>.err

# Usage:
#   sbatch --export=MODEL=cnn hpc_training.sh
#   sbatch --export=MODEL=closure_cnn hpc_training.sh
#   sbatch --export=MODEL=upsample_cnn hpc_training.sh
#
# MODEL selects which model variant to preprocess and train.
# preprocess.py reads cfg.model.name directly to pick the matching make_training_pair.
# Each variant gets its own processed data directory on scratch.
# Set SKIP_DOWNLOAD=1 to skip the download step (use existing raw data).

set -euo pipefail

# Validate MODEL is set
if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL environment variable not set. Use: sbatch --export=MODEL=<variant> hpc_training.sh"
    exit 1
fi
echo "Model variant: $MODEL"

# Code lives on NFS home; data/checkpoints/weights/logs live on scratch (configured via env=hpc).
# Hardcoded PACE scratch path — $SCRATCH isn't exported to Slurm jobs.
PROJECT_DIR=$HOME/projects/pi-cnn
CONDA_ENV=/storage/ice1/3/9/jchen3421/conda/envs/ai

# Environment
export PATH=$CONDA_ENV/bin:$PATH

# Diagnostics
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $PROJECT_DIR

# Step 1: Download velocity cubes from JHTDB (shared across variants)
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "=== Step 1: Download (SKIPPED) ==="
else
    echo "=== Step 1: Download ==="
    python -m superresolution.download env=hpc
fi

# Step 2: Preprocess
echo "=== Step 2: Preprocess (model=$MODEL) ==="
python -m superresolution.preprocess --config-name=$MODEL env=hpc

# Multi-node rendezvous: pick first allocated node as master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

# Step 3: Train model
# Resolve torchrun's full path before srun, since srun spawns a new process
# that may not inherit the conda env's PATH modifications.
TORCHRUN=$(which torchrun)
echo "=== Step 3: Train (model=$MODEL) ==="
srun $TORCHRUN \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m superresolution.train \
    --config-name=$MODEL \
    env=hpc \
    train.batch_size=1 \
    train.num_workers=2

echo "Job $SLURM_JOB_ID finished at $(date)"
