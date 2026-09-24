#!/bin/bash

#SBATCH --job-name=superres
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2 # H200, 141GB HBM3e per GPU
#SBATCH --ntasks-per-node=1 # torchrun spawns one process per GPU
#SBATCH --cpus-per-task=5 # nproc_per_node * (num_workers + 1)
#SBATCH --mem=32G
#SBATCH --time=08:00:00 # coe-ice QoS maximum per job
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage:
#   sbatch --export=ALL,MODEL=cnn,SKIP_DOWNLOAD=1 scripts/hpc_training.sh
#   sbatch --export=ALL,MODEL=closure_cnn,SKIP_DOWNLOAD=1 scripts/hpc_training.sh
#   sbatch --export=ALL,MODEL=upsample_cnn,SKIP_DOWNLOAD=1 scripts/hpc_training.sh
#
# MODEL selects which model variant to preprocess and train.
# TRAIN_EPOCHS retains the original 1000-epoch default; override it at submission.
# Set SKIP_DOWNLOAD=1 when data has already been prepared by hpc_preprocess.sh.

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL environment variable not set. Use: sbatch --export=ALL,MODEL=<variant>,SKIP_DOWNLOAD=1 scripts/hpc_training.sh"
    exit 1
fi
echo "Model variant: $MODEL"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
echo "Epochs: $TRAIN_EPOCHS"

# Locate the checkout from this script unless the caller supplies an override.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${SUPERRES_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
if [ -z "${SUPERRES_STORAGE_ROOT:-}" ] && [[ "$PROJECT_DIR" == /storage/ice1/* ]]; then
    export SUPERRES_STORAGE_ROOT="$(dirname "$PROJECT_DIR")/superresolution"
fi
if [ -z "${SUPERRES_VENV:-}" ]; then
    for CANDIDATE in \
        "$PROJECT_DIR/\~scratch/venvs/superresolution-nn" \
        "$PROJECT_DIR/~scratch/venvs/superresolution-nn"; do
        if [ -x "$CANDIDATE/bin/python" ]; then
            SUPERRES_VENV="$CANDIDATE"
            break
        fi
    done
fi
: "${SUPERRES_VENV:?Export SUPERRES_VENV if no project environment is auto-detected}"

export PATH="$SUPERRES_VENV/bin:$PATH"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$PROJECT_DIR"

if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "=== Step 1: Download (SKIPPED) ==="
else
    echo "=== Step 1: Download ==="
    python -m superresolution.download env=hpc
fi

echo "=== Step 2: Preprocess (model=$MODEL) ==="
python -m superresolution.preprocess --config-name=$MODEL env=hpc

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

TORCHRUN=$(which torchrun)
echo "=== Step 3: Train (model=$MODEL) ==="
srun "$TORCHRUN" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m superresolution.train \
    --config-name="$MODEL" \
    env=hpc \
    train.epochs="$TRAIN_EPOCHS" \
    train.batch_size=1 \
    train.num_workers=2

echo "Job $SLURM_JOB_ID finished at $(date)"
