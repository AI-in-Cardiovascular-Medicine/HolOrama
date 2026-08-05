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

# -- 2b. Install pre-commit git hooks (only available when -Dev was passed) ---
if ($Dev) {
    Write-Host "Installing pre-commit git hooks..."
    & $python -m pre_commit install
    if ($LASTEXITCODE -ne 0) { throw "pre-commit install failed" }
}

# -- 3. Optional: nnUZoo ------------------------------------------------------
# The 0.1.0 release has four upstream bugs that break installing/importing it:
#   1. pyproject.toml has no [tool.setuptools.packages.find], so setuptools'
#      flat-layout auto-discovery refuses to build (papers/, mamba_assets/, etc.
#      all look like top-level packages alongside nnunetv2/).
#   2. Trainer classes (e.g. nnUNetTrainerU2NetP) were moved to nnunetv2/archive/
#      nnUNetTrainer/, but nnUNetPredictor only ever looks in
#      nnunetv2/training/nnUNetTrainer/ -- so checkpoints referencing them fail
#      with "Unable to locate trainer class ... in nnunetv2.training.nnUNetTrainer".
#   3. nnunetv2/nets/segmentation/*.py (u2net.py included) import
#      init_last_bn_before_add_to_0 from the nonexistent local module
#      nnunetv2.nets.dynamic_network_architectures.network_blocks, instead of the
#      real pip package dynamic_network_architectures.initialization.weight_init.
#   4. nnUNetTrainerU2NetP.build_network_architecture still uses the old nnU-Net
#      calling convention (plans_manager, dataset_json, configuration_manager, ...)
#      while the current nnUNetPredictor calls it with the new one
#      (architecture_class_name, arch_init_kwargs, ..., num_output_channels,
#      enable_deep_supervision=..., configuration_manager=...) -- the positional
#      num_output_channels lands in the old signature's enable_deep_supervision
#      slot, which then collides with the keyword of the same name ("got multiple
#      values for argument 'enable_deep_supervision'").
# All four are patched on a local clone since we don't control the upstream repo.
if ($NnUZoo) {
    Write-Host "Installing nnUZoo..."
    $nnuzooDir = Join-Path $env:TEMP "nnUZoo-0.1.0"
    if (Test-Path $nnuzooDir) { Remove-Item -Recurse -Force $nnuzooDir }
    git clone --depth 1 --branch 0.1.0 https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo $nnuzooDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone nnUZoo" }

    Add-Content -Path (Join-Path $nnuzooDir "pyproject.toml") `
        -Value "`n[tool.setuptools.packages.find]`ninclude = [`"nnunetv2*`"]`n"

    # Bug 2 + 4: replace the archived, stale trainer with a corrected copy that
    # matches the current calling convention (bug 4) instead of copying it as-is.
    $trainerPath = Join-Path $nnuzooDir "nnunetv2\training\nnUNetTrainer\nnUNetTrainerU2NetP.py"
    @'
from os.path import join

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.nets.segmentation.u2net import get_u2netp_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary


class nnUNetTrainerU2NetP(nnUNetTrainer):
    """ Swin-UMamba """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = -1  # Training from scratch
        self.early_stop_epoch = 10

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision: bool = True,
            *,
            up_sample_type: str = "convtranspose",
            configuration_manager: ConfigurationManager = None,
    ) -> nn.Module:

        model = get_u2netp_from_plans(
            num_segmentation_heads=num_output_channels,
            num_input_channels=num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=True,
        )
        if configuration_manager is not None:
            summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0, 1.0]] * 7
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def on_epoch_end(self):
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch}.pth'))
        super().on_epoch_end()

    def on_train_epoch_start(self):
        # freeze the encoder if the epoch is less than 10
        if self.current_epoch < self.freeze_encoder_epochs:
            self.print_to_log_file("Freezing the encoder")
            if self.is_ddp:
                self.network.module.freeze_encoder()
            else:
                self.network.freeze_encoder()
        else:
            self.print_to_log_file("Unfreezing the encoder")
            if self.is_ddp:
                self.network.module.unfreeze_encoder()
            else:
                self.network.unfreeze_encoder()
        super().on_train_epoch_start()

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled
'@ | Set-Content -Path $trainerPath

    $u2netPath = Join-Path $nnuzooDir "nnunetv2\nets\segmentation\u2net.py"
    (Get-Content $u2netPath) `
        -replace 'from nnunetv2\.nets\.dynamic_network_architectures\.network_blocks import init_last_bn_before_add_to_0', `
                 'from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0' |
        Set-Content $u2netPath

    uv pip install --python $python $nnuzooDir
    $nnuzooExitCode = $LASTEXITCODE
    Remove-Item -Recurse -Force $nnuzooDir
    if ($nnuzooExitCode -ne 0) { throw "nnUZoo install failed" }
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
if ($Dev) {
    Write-Host "Pre-commit git hooks installed (black / ruff / mypy will run on 'git commit')."
} else {
    Write-Host "Pre-commit git hooks NOT installed (re-run with -Dev to enable)."
}
Write-Host "Done. Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run the app with:"
Write-Host "  python src\main.py"
