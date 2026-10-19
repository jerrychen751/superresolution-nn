#!/usr/bin/env bash
# Prepare the harder 128^3 -> 16^3 -> 128^3 CNN dataset on PACE.
# Reuses the same 700 downloaded raw JHTDB cubes as the standard CNN dataset.
# Safe to rerun: completed input/target pairs are skipped.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [ -n "${SUPERRES_SCRATCH_ROOT:-}" ]; then
    SCRATCH_ROOT=$(readlink -f "$SUPERRES_SCRATCH_ROOT")
elif [ -e "$HOME/scratch" ]; then
    SCRATCH_ROOT=$(readlink -f "$HOME/scratch")
else
    echo "ERROR: PACE scratch was not found at $HOME/scratch."
    echo "Set SUPERRES_SCRATCH_ROOT to your PACE scratch directory and rerun."
    exit 1
fi

export SUPERRES_PROJECT_DIR="$PROJECT_DIR"
export SUPERRES_STORAGE_ROOT="${SUPERRES_STORAGE_ROOT:-$SCRATCH_ROOT/superresolution}"
export SUPERRES_VENV="${SUPERRES_VENV:-$SCRATCH_ROOT/venvs/superresolution-nn}"

PYTHON="$SUPERRES_VENV/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Project environment not found at $SUPERRES_VENV"
    echo "Creating it now..."
    bash "$PROJECT_DIR/scripts/hpc_env_setup.sh"
fi
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python was not created at $PYTHON"
    exit 1
fi

RAW_DIR="$SUPERRES_STORAGE_ROOT/data/raw"
if [ -d "$RAW_DIR" ]; then
    RAW_COUNT=$(find "$RAW_DIR" -maxdepth 1 -name 'velocity_t*.npy' | wc -l)
else
    RAW_COUNT=0
fi
if [ "$RAW_COUNT" -ne 700 ]; then
    echo "ERROR: Expected 700 existing raw JHTDB cubes in $RAW_DIR; found $RAW_COUNT."
    echo "Run bash scripts/pace_prepare_notebook.sh first to finish the download."
    exit 1
fi

echo "Raw data:       $RAW_DIR"
echo "New pair data:  $SUPERRES_STORAGE_ROOT/data/processed/cnn_16"
echo "Downsampling:   128^3 -> 16^3 -> 128^3"
echo ""

"$PYTHON" -m superresolution.preprocess --config-name=cnn_16 env=hpc

for ENTRY in train:490 val:105 test:105; do
    SPLIT=${ENTRY%%:*}
    EXPECTED=${ENTRY##*:}
    SPLIT_DIR="$SUPERRES_STORAGE_ROOT/data/processed/cnn_16/$SPLIT"
    INPUT_COUNT=$(find "$SPLIT_DIR" -maxdepth 1 -name 'input_t*.npy' | wc -l)
    TARGET_COUNT=$(find "$SPLIT_DIR" -maxdepth 1 -name 'target_t*.npy' | wc -l)
    echo "$SPLIT: $INPUT_COUNT inputs, $TARGET_COUNT targets"
    if [ "$INPUT_COUNT" -ne "$EXPECTED" ] || [ "$TARGET_COUNT" -ne "$EXPECTED" ]; then
        echo "ERROR: Expected $EXPECTED complete pairs in $SPLIT_DIR."
        exit 1
    fi
done

cat <<EOF

128 -> 16 preprocessing is complete.

Data: $SUPERRES_STORAGE_ROOT/data/processed/cnn_16

Open one of these notebooks with the Python (superresolution-nn) kernel:
  superresolution_experiments/cnn_training_16_template.ipynb
  superresolution_experiments/cnn_training_16_tony.ipynb
EOF
