#!/usr/bin/env bash
# Usage: ./install.sh [--dev] [--nnuzoo] [--cpu] [--cuda 121]
# Default installs CUDA 11.8 (cu118) torch -- GPU-ready out of the box.
# --dev            also install dev dependencies
# --nnuzoo         install nnUZoo from GitHub
# --cpu            install CPU-only torch instead (overrides default GPU build)
# --cuda 121       switch to CUDA 12.1 build instead of the default cu118
set -e

DEV_DEPS=false
NNUZOO=false
CPU=false
CUDA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)    DEV_DEPS=true; shift ;;
        --nnuzoo) NNUZOO=true;   shift ;;
        --cpu)    CPU=true;      shift ;;
        --cuda)   CUDA="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--dev] [--nnuzoo] [--cpu] [--cuda 121]"; exit 1 ;;
    esac
done

# -- 1. Ensure uv is available ------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "uv not found -- installing via pip..."
    pip install uv
fi

# -- 2. Install project dependencies (uv auto-creates .venv) ------------------
# pyqt6-qt6 is ~900 MB; increase timeout for slow connections
export UV_HTTP_TIMEOUT=300

SYNC_ARGS=("sync")
if $DEV_DEPS; then SYNC_ARGS+=("--group" "dev"); fi
uv "${SYNC_ARGS[@]}"

PYTHON="./.venv/bin/python"

# -- 3. Optional: nnUZoo ------------------------------------------------------
if $NNUZOO; then
    echo "Installing nnUZoo..."
    uv pip install --python "$PYTHON" git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main
fi

# -- 4. Override torch build if requested -------------------------------------
if $CPU; then
    echo "Switching to CPU-only torch build..."
    uv pip install --python "$PYTHON" --reinstall \
        "torch==2.4.0" "torchvision==0.19.0" \
        --index-url https://download.pytorch.org/whl/cpu
elif [[ "$CUDA" == "121" ]]; then
    echo "Switching to CUDA 12.1 torch build..."
    uv pip install --python "$PYTHON" --reinstall \
        "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" \
        --index-url https://download.pytorch.org/whl/cu121
elif [[ -n "$CUDA" ]]; then
    echo "Unknown CUDA version '$CUDA'. Use 121 (cu118 is the default)."; exit 1
fi
# default (cu118) already installed by uv sync

echo ""
echo "Done. Run the app with:"
echo "  source .venv/bin/activate && python3 src/main.py"
