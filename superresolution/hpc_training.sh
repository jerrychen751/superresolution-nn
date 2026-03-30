#!/bin/bash

#SBATCH --job-name=superres
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2              # GPU type and count per node
#SBATCH --ntasks-per-node=1            # Stays 1; torchrun spawns child processes equal to number of GPUs (tasks = GPU ct)
#SBATCH --cpus-per-task=5               # nproc_per_node * (num_workers + 1)
#SBATCH --mem=32G
#SBATCH --time=08:00:00                # wall-time limit
#SBATCH --output=logs/%j.out           # stdout -> logs/<jobid>.out
#SBATCH --error=logs/%j.err            # stderr -> logs/<jobid>.err

# Exit immediately if any command fails
set -euo pipefail

# Environment
source activate ai
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Diagnostics
# $SLURM_JOB_ID is environment variable automatically set by SLURM
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Code lives on NFS home; data lives on scratch (faster I/O)
PROJECT_DIR=$HOME/projects/pi-cnn
SCRATCH=/storage/ice1/3/9/jchen3421
RAW_DIR=$SCRATCH/pi-cnn/data/raw
PROCESSED_DIR=$SCRATCH/pi-cnn/data/processed

# Navigate to project root (code)
cd $PROJECT_DIR

# Create data directories if they don't exist
mkdir -p $RAW_DIR $PROCESSED_DIR

# Step 1: Download velocity cubes from JHTDB
echo "=== Step 1: Download ==="
python -m superresolution.download \
    raw_data_dir=$RAW_DIR

# Step 2: Preprocess (blur, downsample, normalize)
echo "=== Step 2: Preprocess ==="
python -m superresolution.preprocess \
    raw_data_dir=$RAW_DIR \
    processed_data_dir=$PROCESSED_DIR

# Multi-node rendezvous: pick first allocated node as master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

# Step 3: Train model
echo "=== Step 3: Train ==="
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m superresolution.train \
    processed_data_dir=$PROCESSED_DIR \
    train=hpc

echo "Job $SLURM_JOB_ID finished at $(date)"
