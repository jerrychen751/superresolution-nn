#!/usr/bin/env bash
# Setup script for the superresolution-nn project.
# Creates a uv-managed virtual environment and installs every dependency from pyproject.toml.
#
# Usage:
#   On PACE ICE:
#     SUPERRES_VENV=/storage/ice1/.../<user>/venvs/superresolution-nn bash scripts/hpc_env_setup.sh
#     # Legacy VENV=... usage remains supported.
#
#   On a personal machine:
#     bash scripts/hpc_env_setup.sh

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

SUPERRES_VENV="${SUPERRES_VENV:-${VENV:-$PROJECT_DIR/.venv}}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export UV_PROJECT_ENVIRONMENT="$SUPERRES_VENV"

# The wheel cache and the managed interpreter are each multi-GB, so on PACE neither fits the home quota.
# A cache on a different filesystem from the venv is copied rather than hardlinked, so all three stay together.
case "$SUPERRES_VENV" in
    "$HOME"/*) ;;
    *)
        export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$SUPERRES_VENV")/uv-cache}"
        export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "$SUPERRES_VENV")/uv-python}"
        echo "Wheel cache: $UV_CACHE_DIR"
        echo "Interpreter: $UV_PYTHON_INSTALL_DIR"
        ;;
esac

echo "Creating environment at $SUPERRES_VENV ..."
uv python install 3.11
uv sync --python 3.11 --group extras

uv run python -c "import superresolution, pathlib; print('superresolution resolves from', pathlib.Path(superresolution.__file__).parent)"
uv run python -m ipykernel install --user \
    --name superresolution-nn \
    --display-name "Python (superresolution-nn)"

echo ""
echo "Done. Activate with:"
echo "  source $SUPERRES_VENV/bin/activate"
echo "Jupyter kernel: Python (superresolution-nn)"
