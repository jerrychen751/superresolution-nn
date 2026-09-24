#!/usr/bin/env bash
# One-time PACE preparation for the CNN training notebooks.
# Safe to rerun: uv sync is idempotent, downloads skip saved cubes, and
# preprocessing skips completed input/target pairs.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [ -n "${SUPERRES_SCRATCH_ROOT:-}" ]; then
    SCRATCH_ROOT=$(readlink -f "$SUPERRES_SCRATCH_ROOT")
elif [ -e "$HOME/scratch" ]; then
    SCRATCH_ROOT=$(readlink -f "$HOME/scratch")
else
    echo "ERROR: PACE scratch was not found at $HOME/scratch."
    echo "Set it explicitly, then rerun:"
    echo "  export SUPERRES_SCRATCH_ROOT=/storage/ice1/.../<your-username>"
    exit 1
fi

export SUPERRES_PROJECT_DIR="$PROJECT_DIR"
export SUPERRES_STORAGE_ROOT="${SUPERRES_STORAGE_ROOT:-$SCRATCH_ROOT/superresolution}"
export SUPERRES_VENV="${SUPERRES_VENV:-$SCRATCH_ROOT/venvs/superresolution-nn}"
DOWNLOAD_RETRY_SECONDS="${DOWNLOAD_RETRY_SECONDS:-15}"

mkdir -p "$SUPERRES_STORAGE_ROOT"

echo "Project:     $SUPERRES_PROJECT_DIR"
echo "Environment: $SUPERRES_VENV"
echo "Data/results: $SUPERRES_STORAGE_ROOT"
echo ""

echo "=== 1/4 Create or update the project environment and Jupyter kernel ==="
bash "$PROJECT_DIR/scripts/hpc_env_setup.sh"
PYTHON="$SUPERRES_VENV/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python was not created at $PYTHON"
    exit 1
fi

echo "=== 2/4 Download 700 JHTDB cubes ==="
while true; do
    if "$PYTHON" -m superresolution.download env=hpc; then
        break
    fi
    COUNT=$(find "$SUPERRES_STORAGE_ROOT/data/raw" -maxdepth 1 \
        -name 'velocity_t*.npy' 2>/dev/null | wc -l)
    echo "JHTDB interrupted. $COUNT/700 cubes saved. Trying again in ${DOWNLOAD_RETRY_SECONDS}s."
    sleep "$DOWNLOAD_RETRY_SECONDS"
done

RAW_COUNT=$(find "$SUPERRES_STORAGE_ROOT/data/raw" -maxdepth 1 \
    -name 'velocity_t*.npy' | wc -l)
if [ "$RAW_COUNT" -ne 700 ]; then
    echo "ERROR: Expected 700 raw cubes, found $RAW_COUNT."
    exit 1
fi

echo "=== 3/4 Preprocess the CNN train/validation/test data ==="
"$PYTHON" -m superresolution.preprocess --config-name=cnn env=hpc

echo "=== 4/4 Verify the fixed team split ==="
for ENTRY in train:490 val:105 test:105; do
    SPLIT=${ENTRY%%:*}
    EXPECTED=${ENTRY##*:}
    COUNT=$(find "$SUPERRES_STORAGE_ROOT/data/processed/cnn/$SPLIT" \
        -maxdepth 1 -name 'input_t*.npy' | wc -l)
    echo "$SPLIT: $COUNT input/target pairs"
    if [ "$COUNT" -ne "$EXPECTED" ]; then
        echo "ERROR: Expected $EXPECTED $SPLIT pairs, found $COUNT."
        exit 1
    fi
done

cat <<EOF

PACE preparation is complete.

In JupyterLab:
  1. Refresh the page if the kernel is not visible.
  2. Open superresolution_experiments/cnn_training_template.ipynb.
  3. Save a copy with your name.
  4. Edit the YOUR CODE HERE sections and choose a unique EXPERIMENT_NAME.
  5. Select the kernel: Python (superresolution-nn)
  6. Restart the kernel and click Run All.

Data: $SUPERRES_STORAGE_ROOT/data/processed/cnn
Results: $SUPERRES_STORAGE_ROOT/notebook_runs/<EXPERIMENT_NAME>
EOF
