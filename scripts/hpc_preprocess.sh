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

# PACE data job. It retains the original five-model default. Set DOWNLOAD=1 to
# download JHTDB first, or set MODELS="cnn" for a CNN-only run.
# Existing raw cubes and processed pairs are skipped on rerun.

set -euo pipefail

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

cd "$PROJECT_DIR"

if [ "${DOWNLOAD:-0}" = "1" ]; then
    echo "=== Download JHTDB start: $(date +%H:%M:%S) ==="
    python -m superresolution.download env=hpc
    echo "=== Download JHTDB end:   $(date +%H:%M:%S) ==="
fi

FAILED=""
for MODEL in ${MODELS:-cnn upsample_cnn closure_cnn fno gnn}; do
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
