# Usage: .\install.ps1 [-Dev] [-NnUZoo] [-Cpu] [-Cuda 121]
# Default installs CUDA 11.8 (cu118) torch -- GPU-ready out of the box.
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

# -- 1. Ensure uv is available ------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found -- installing via pip..."
    pip install uv
    if ($LASTEXITCODE -ne 0) { throw "Failed to install uv" }
}

# -- 2. Install project dependencies (uv auto-creates .venv) ------------------
# pyqt6-qt6 is ~900 MB; increase timeout for slow connections
$env:UV_HTTP_TIMEOUT = '300'

$syncArgs = @('sync')
if ($Dev) { $syncArgs += '--group', 'dev' }
uv @syncArgs
if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }

$python = '.\.venv\Scripts\python.exe'

# -- 3. Optional: nnUZoo ------------------------------------------------------
if ($NnUZoo) {
    Write-Host "Installing nnUZoo..."
    uv pip install --python $python git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main
    if ($LASTEXITCODE -ne 0) { throw "nnUZoo install failed" }
}

# -- 4. Override torch build if requested -------------------------------------
# uv sync installed the default cu118 build via pyproject.toml sources.
# For alternate builds we reinstall torch/torchvision directly.
if ($Cpu) {
    Write-Host "Switching to CPU-only torch build..."
    uv pip install --python $python --reinstall "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "torch CPU install failed" }
} elseif ($Cuda -eq '121') {
    Write-Host "Switching to CUDA 12.1 torch build..."
    uv pip install --python $python --reinstall "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw "torch CUDA 12.1 install failed" }
}
# default (cu118) already installed by uv sync

# -- 5. Windows fix: missing libomp140.x86_64.dll (required by PyTorch) -------
Write-Host "Applying Windows fix: downloading libomp140.x86_64.dll..."
$libompScript = @'
import urllib.request, tarfile, io, os, sys
url = 'https://conda.anaconda.org/conda-forge/win-64/llvm-openmp-14.0.0-h2d74725_0.tar.bz2'
try:
    data = urllib.request.urlopen(url, timeout=60).read()
    dest = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib', 'libomp140.x86_64.dll')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as t:
        f = t.extractfile('Library/bin/libomp.dll')
        with open(dest, 'wb') as out:
            out.write(f.read())
    print('libomp140 installed to:', dest)
except Exception as e:
    print(f'WARNING: libomp140 fix skipped ({e}). Install manually if torch fails to import.')
    sys.exit(0)
'@
$libompScript | & $python

# -- 6. Windows fix: downgrade optree (>=0.14 crashes with torch 2.4.0) -------
Write-Host "Applying Windows fix: pinning optree to 0.13.1..."
uv pip install --python $python "optree==0.13.1"
if ($LASTEXITCODE -ne 0) { throw "optree pin failed" }

# -- 7. Re-pin numpy (nnunetv2 and torch cuda installs upgrade it to 2.x) ------
# torch 2.4.0 was built against numpy 1.x C API; numpy 2.x breaks torch.from_numpy
Write-Host "Re-pinning numpy to 1.26.4 (torch 2.4.0 / numpy 1.x compatibility)..."
uv pip install --python $python "numpy==1.26.4"
if ($LASTEXITCODE -ne 0) { throw "numpy re-pin failed" }

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run the app with:"
Write-Host "  python src\main.py"
