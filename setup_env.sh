#!/usr/bin/env bash
# Setup script for the pi-cnn project.
# Creates a Python venv and installs all dependencies.
#
# Usage:
#   On PACE ICE (or any system with module):
#     module load anaconda3
#     bash setup_env.sh
#
#   On a personal machine (Python 3.10+ required):
#     bash setup_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/ai}"

echo "Creating venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Core: required by superresolution/ pipeline
pip install \
    numpy scipy matplotlib \
    hydra-core==1.3.2 omegaconf==2.3.0 \
    givernylocal

# PyTorch with CUDA 12.6 (drop the --extra-index-url line for CPU-only)
pip install \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Useful for exploratory work and notebooks
pip install \
    pandas h5py netCDF4 xarray \
    scikit-learn scikit-image \
    jupyterlab tensorboard tqdm pyyaml seaborn \
    opencv-python-headless

# ML ecosystem extras (wandb for experiment tracking, einops for tensor reshaping, etc.)
pip install \
    pytorch-lightning accelerate wandb torchmetrics einops

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
