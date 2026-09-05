#!/usr/bin/env bash
# Setup script for the superresolution-nn project.
# Creates a uv-managed virtual environment and installs every dependency from pyproject.toml.
#
# Usage:
#   On PACE ICE:
#     VENV=/storage/ice1/3/9/jchen3421/venvs/superresolution-nn bash scripts/hpc_env_setup.sh
#
#   On a personal machine:
#     bash scripts/hpc_env_setup.sh

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

VENV="${VENV:-$PROJECT_DIR/.venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export UV_PROJECT_ENVIRONMENT="$VENV"

# The wheel cache and the managed interpreter are each multi-GB, so on PACE neither fits the home quota.
# A cache on a different filesystem from the venv is copied rather than hardlinked, so all three stay together.
case "$VENV" in
    "$HOME"/*) ;;
    *)
        export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$VENV")/uv-cache}"
        export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "$VENV")/uv-python}"
        echo "Wheel cache: $UV_CACHE_DIR"
        echo "Interpreter: $UV_PYTHON_INSTALL_DIR"
        ;;
esac

echo "Creating environment at $VENV ..."
uv python install 3.11
uv sync --python 3.11 --group extras

uv run python -c "import superresolution, pathlib; print('superresolution resolves from', pathlib.Path(superresolution.__file__).parent)"

echo ""
echo "Done. Activate with:"
echo "  source $VENV/bin/activate"
