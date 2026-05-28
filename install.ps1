# Usage: .\install.ps1 [-Dev] [-NnUZoo] [-Cpu] [-Cuda 121]
# Default installs CUDA 11.8 (cu118) torch — GPU-ready out of the box.
# -Dev             also install dev dependencies
# -NnUZoo          install nnUZoo from GitHub
# -Cpu             install CPU-only torch instead (overrides default GPU build)
# -Cuda 121        switch to CUDA 12.1 build instead of the default cu118
param(
    [switch]$Dev,
    [switch]$NnUZoo,
    [switch]$Cpu,
    [ValidateSet('121')]
    [string]$Cuda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── 1. Create and activate virtual environment ──────────────────────────────
python -m venv env
.\env\Scripts\Activate.ps1

# ── 2. Install poetry and project dependencies ──────────────────────────────
pip install --quiet poetry

if ($Dev) {
    poetry install --with dev
} else {
    poetry install
}

# ── 3. Optional: nnUZoo ─────────────────────────────────────────────────────
if ($NnUZoo) {
    Write-Host "Installing nnUZoo..."
    poetry run pip install git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main
}

# ── 4. Override torch build if requested ────────────────────────────────────
if ($Cpu) {
    Write-Host "Switching to CPU-only torch build..."
    pip install "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cpu
} elseif ($Cuda -eq '121') {
    Write-Host "Switching to CUDA 12.1 torch build..."
    pip install "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" --index-url https://download.pytorch.org/whl/cu121
}
# default (cu118) is already installed via pyproject.toml

# ── 5. Windows fix: missing libomp140.x86_64.dll (required by PyTorch) ──────
Write-Host "Applying Windows fix: downloading libomp140.x86_64.dll..."
$libompScript = @"
import urllib.request, tarfile, io, os, sys
url = 'https://conda.anaconda.org/conda-forge/win-64/llvm-openmp-14.0.0-h2d74725_0.tar.bz2'
data = urllib.request.urlopen(url).read()
dest = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib', 'libomp140.x86_64.dll')
with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as t:
    f = t.extractfile('Library/bin/libomp.dll')
    with open(dest, 'wb') as out:
        out.write(f.read())
print('libomp140 installed to:', dest)
"@
$libompScript | python

# ── 6. Windows fix: downgrade optree (>=0.14 crashes with torch 2.4.0) ──────
Write-Host "Applying Windows fix: pinning optree to 0.13.1..."
pip install "optree==0.13.1"

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "  .\env\Scripts\Activate.ps1"
Write-Host "Then run the app with:"
Write-Host "  python src\main.py"
