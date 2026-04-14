#!/usr/bin/env bash
# Setup script for the pi-cnn project.
# Creates a conda env and installs all dependencies via pip.
#
# Usage:
#   On PACE ICE:
#     module load anaconda3
#     CONDA_ENV=/storage/ice1/3/9/jchen3421/conda/envs/ai bash setup_env.sh
#
#   On a personal machine (needs conda/miniconda):
#     bash setup_env.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/ai}"

echo "Creating conda env at $CONDA_ENV ..."
conda create --prefix "$CONDA_ENV" python=3.11 -y

PIP="$CONDA_ENV/bin/pip"
$PIP install --upgrade pip setuptools wheel

# Core: required by superresolution/ pipeline
$PIP install \
    numpy scipy matplotlib \
    hydra-core==1.3.2 omegaconf==2.3.0 \
    givernylocal

# PyTorch with CUDA 12.6 (drop the --extra-index-url line for CPU-only)
$PIP install \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu126

# PyG for the gnn model variant; installed after torch since its setup.py inspects the torch version.
# We only use GCNConv and Data, which are pure-Python in modern PyG, so no torch-scatter wheels needed.
$PIP install torch-geometric

# Useful for exploratory work and notebooks
$PIP install \
    pandas h5py netCDF4 xarray \
    scikit-learn scikit-image \
    jupyterlab tensorboard tqdm pyyaml seaborn \
    opencv-python-headless

# ML ecosystem extras (wandb for experiment tracking, einops for tensor reshaping, etc.)
$PIP install \
    pytorch-lightning accelerate wandb torchmetrics einops

echo ""
echo "Done. Activate with:"
echo "  source activate $CONDA_ENV"
