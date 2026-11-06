# Usage: .\install.ps1 [-Dev] [-Gpu] [-NoSegmentation] [-Centerlines]
#
# Installs HolOrama from source into .venv with uv. Everything comes from pyproject.toml / uv.lock;
# there are no post-install fixes any more (torch, tensorflow and nnUZoo are gone: automatic
# segmentation runs the bundled ONNX model with onnxruntime).
#
# -Dev             also install dev + docs dependencies (tests, linters, Nuitka, Sphinx) and the
#                  pre-commit git hooks
# -Gpu             run Automatic Segmentation on an NVIDIA GPU (onnxruntime-gpu; CUDA 12 + cuDNN
#                  come as pip wheels, only an NVIDIA driver >= 527.41 is needed). ~2.5 GB more.
# -NoSegmentation  skip the optional 'segmentation' extra (onnxruntime); Automatic Segmentation
#                  then reports itself unavailable
# -Centerlines     also set up the vmtk environment for Calculate Centerlines right away (pixi +
#                  conda-forge, ~1.5 GB). Without it, the app offers to do this on first use.
param(
    [switch]$Dev,
    [switch]$Gpu,
    [switch]$NoSegmentation,
    [switch]$Centerlines
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- 1. Ensure uv is available ------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found -- installing it (https://docs.astral.sh/uv/)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install uv" }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# -- 2. Install the locked environment (uv creates .venv and fetches Python if needed) --
# pyqt6-qt6 is large; increase the timeout for slow connections
$env:UV_HTTP_TIMEOUT = '300'

$syncArgs = @('sync', '--locked')
if ($Gpu) { $syncArgs += '--extra', 'segmentation-gpu' }
elseif (-not $NoSegmentation) { $syncArgs += '--extra', 'segmentation' }
if ($Dev) { $syncArgs += '--group', 'dev', '--group', 'docs' }
uv @syncArgs
if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }

$python = '.\.venv\Scripts\python.exe'

# -- 3. Dev: pre-commit git hooks ---------------------------------------------
if ($Dev) {
    Write-Host "Installing pre-commit git hooks..."
    & $python -m pre_commit install
    if ($LASTEXITCODE -ne 0) { throw "pre-commit install failed" }
}

# -- 4. Optional: vmtk environment for Calculate Centerlines ------------------
if ($Centerlines) {
    Write-Host "Setting up the vmtk environment (pixi + conda-forge)..."
    & $python -c "import sys; sys.path.insert(0, 'src'); from pages.ccta.left_half.cut_geometry import vmtk; vmtk.setup_env(vmtk.VmtkEnv.from_config(None))"
    if ($LASTEXITCODE -ne 0) { throw "vmtk environment setup failed" }
}

Write-Host ""
Write-Host "Done. Run the app with:"
Write-Host "  uv run src\main.py"
Write-Host "(or activate the environment with .\.venv\Scripts\Activate.ps1 and run: python src\main.py)"
