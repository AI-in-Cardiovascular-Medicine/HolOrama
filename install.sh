#!/usr/bin/env bash
# Usage: ./install.sh [--dev] [--nnuzoo] [--cpu] [--cuda 121]
# Default installs CUDA 11.8 (cu118) torch — GPU-ready out of the box.
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

python3 -m venv env
source env/bin/activate
pip install --quiet poetry

if $DEV_DEPS; then
    poetry install --with dev
else
    poetry install
fi

if $NNUZOO; then
    echo "Installing nnUZoo..."
    poetry run pip install git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main
fi

# Override torch build if requested; default (cu118) comes from pyproject.toml
if $CPU; then
    echo "Switching to CPU-only torch build..."
    pip install "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cpu
elif [[ "$CUDA" == "121" ]]; then
    echo "Switching to CUDA 12.1 torch build..."
    pip install "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" --index-url https://download.pytorch.org/whl/cu121
elif [[ -n "$CUDA" ]]; then
    echo "Unknown CUDA version '$CUDA'. Use 121 (cu118 is the default)."; exit 1
fi

echo ""
echo "Done. Run the app with:"
echo "  source env/bin/activate && python3 src/main.py"
