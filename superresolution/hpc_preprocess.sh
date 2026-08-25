#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# One-shot preprocessing job for all 5 models.
# preprocess.py's existence check skips any train/val/test pair already on disk, so re-running is idempotent and cheap.

set -euo pipefail

PROJECT_DIR=$HOME/projects/pi-cnn
VENV=/storage/ice1/3/9/jchen3421/venvs/pi-cnn

export PATH=$VENV/bin:$PATH

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on node: $(hostname)"

cd $PROJECT_DIR

FAILED=""
for MODEL in cnn upsample_cnn closure_cnn fno gnn; do
    echo "=== Preprocess $MODEL start: $(date +%H:%M:%S) ==="
    if python -m superresolution.preprocess --config-name=$MODEL env=hpc; then
        echo "=== Preprocess $MODEL end:   $(date +%H:%M:%S) ==="
    else
        FAILED="$FAILED $MODEL"
        echo "=== Preprocess $MODEL FAILED: $(date +%H:%M:%S) ==="
    fi
done

echo "Job $SLURM_JOB_ID finished at $(date)"
if [ -n "$FAILED" ]; then
    echo "Failed models:$FAILED"
    exit 1
fi
