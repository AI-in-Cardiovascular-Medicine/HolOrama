#!/usr/bin/env bash
# Usage: ./install.sh [--dev] [--gpu] [--no-segmentation] [--centerlines]
#
# Installs HolOrama from source into .venv with uv. Everything comes from pyproject.toml / uv.lock;
# there are no post-install fixes any more (torch, tensorflow and nnUZoo are gone: automatic
# segmentation runs the bundled ONNX model with onnxruntime).
#
# --dev              also install dev + docs dependencies (tests, linters, Nuitka, Sphinx) and the
#                    pre-commit git hooks
# --gpu              run Automatic Segmentation on an NVIDIA GPU (onnxruntime-gpu; CUDA 12 + cuDNN
#                    come as pip wheels, only an NVIDIA driver >= 525.60 is needed). ~2.5 GB more.
# --no-segmentation  skip the optional 'segmentation' extra (onnxruntime); Automatic Segmentation
#                    then reports itself unavailable
# --centerlines      also set up the vmtk environment for Calculate Centerlines right away (pixi +
#                    conda-forge, ~1.5 GB; Linux x64 and Intel macOS only). Without it, the app
#                    offers to do this on first use.
set -euo pipefail

DEV_DEPS=false
SEGMENTATION=true
GPU=false
CENTERLINES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)             DEV_DEPS=true;      shift ;;
        --gpu)             GPU=true;           shift ;;
        --no-segmentation) SEGMENTATION=false; shift ;;
        --centerlines)     CENTERLINES=true;   shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--dev] [--gpu] [--no-segmentation] [--centerlines]"; exit 1 ;;
    esac
done

# -- 1. Ensure uv is available ------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "uv not found -- installing it (https://docs.astral.sh/uv/)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# -- 2. Install the locked environment (uv creates .venv and fetches Python if needed) --
# pyqt6-qt6 is large; increase the timeout for slow connections
export UV_HTTP_TIMEOUT=300

SYNC_ARGS=("sync" "--locked")
if $GPU; then SYNC_ARGS+=("--extra" "segmentation-gpu")
elif $SEGMENTATION; then SYNC_ARGS+=("--extra" "segmentation"); fi
if $DEV_DEPS; then SYNC_ARGS+=("--group" "dev" "--group" "docs"); fi
uv "${SYNC_ARGS[@]}"

PYTHON="./.venv/bin/python"

# -- 3. Dev: pre-commit git hooks ---------------------------------------------
if $DEV_DEPS; then
    echo "Installing pre-commit git hooks..."
    "$PYTHON" -m pre_commit install
fi

# -- 4. Optional: vmtk environment for Calculate Centerlines ------------------
if $CENTERLINES; then
    echo "Setting up the vmtk environment (pixi + conda-forge)..."
    "$PYTHON" -c "import sys; sys.path.insert(0, 'src'); from pages.ccta.left_half.cut_geometry import vmtk; vmtk.setup_env(vmtk.VmtkEnv.from_config(None))"
fi

echo ""
echo "Done. Run the app with:"
echo "  uv run src/main.py"
echo "(or: source .venv/bin/activate && python src/main.py)"
