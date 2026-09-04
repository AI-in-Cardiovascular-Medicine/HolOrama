# Build a standalone Windows executable of HolOrama with Nuitka.
#
# The compiled build is inference-free: torch / torchvision / tensorflow / nnUNetv2
# and the whole dev toolchain are excluded (see --nofollow-import-to below). The app
# runs fully except for "Automatic segmentation", which imports those lazily at call
# time and will surface an ImportError only if the user triggers it.
#
# Usage: .\build_nuitka.ps1
# Output: build\nuitka\main.dist\HolOrama.exe

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$python = '.\.venv\Scripts\python.exe'
# Read the version from src\version.py so the exe's file/product metadata can't drift
# from the real release (build_installer.ps1 reads the same source).
$versionLine = Select-String -Path 'src\version.py' -Pattern "__version__\s*=\s*'([^']+)'" | Select-Object -First 1
if (-not $versionLine) { throw "Could not read __version__ from src\version.py" }
$version = $versionLine.Matches[0].Groups[1].Value

# Heavy runtime deps that are only imported lazily (inside Predict.inference) and are
# NOT needed for anything except automatic segmentation. Also drop the dev toolchain.
$exclude = @(
    'torch', 'torchvision', 'torchgen', 'functorch',
    'tensorflow', 'tensorboard', 'keras', 'jax', 'jaxlib',
    'nnunetv2', 'nnunet', 'batchgenerators', 'dynamic_network_architectures',
    'pytest', 'pytest_qt', 'pytestqt', 'black', 'mypy', 'ruff', 'pre_commit', 'pylint',
    'IPython', 'notebook'
) -join ','

$nuitkaArgs = @(
    '-m', 'nuitka',
    '--standalone',
    '--assume-yes-for-downloads',
    # Use the pure-Python pefile scanner for DLL dependency detection. The default
    # legacy depends.exe (Dependency Walker) crashes on VTK's ~630 DLLs with a
    # missing-.dwp FileNotFoundError.
    '--experimental=force-dependencies-pefile',
    '--enable-plugin=pyqt6',
    '--output-dir=build\nuitka',
    '--output-filename=HolOrama.exe',
    "--nofollow-import-to=$exclude",
    # config.yaml is loaded via Path(__file__).parent / 'config.yaml' in main.py,
    # so it must sit next to the executable.
    '--include-data-files=src\config.yaml=config.yaml',
    # media/ holds the window icon (desktop_img.ico), the About video and the
    # bundled JetBrains Mono cuts in media/fonts (recursive, so fonts come along).
    '--include-data-dir=media=media',
    '--windows-icon-from-ico=media\desktop_img.ico',
    # pydicom registers codec plugins at import time via importlib.import_module
    # (e.g. pydicom.encoders.gdcm), which Nuitka's static analysis misses. Force
    # the whole package in so those stub modules exist at runtime. pylibjpeg /
    # its libjpeg plugin are discovered the same dynamic way for JPEG DICOMs.
    '--include-package=pydicom',
    '--include-package=pylibjpeg',
    '--include-package=libjpeg',
    # pylibjpeg discovers its decoders via importlib.metadata entry points, which
    # live in the packages' .dist-info metadata. Nuitka drops that metadata by
    # default, so without this the libjpeg decoder is invisible at runtime and
    # JPEG-compressed DICOMs fail with "'pylibjpeg-libjpeg' plugin is not installed".
    '--include-distribution-metadata=pylibjpeg',
    '--include-distribution-metadata=pylibjpeg-libjpeg',
    # Windows executable metadata.
    '--company-name=HolOrama',
    '--product-name=HolOrama',
    "--file-version=$version",
    "--product-version=$version",
    '--file-description=HolOrama cardiac image analysis',
    'src\main.py'
)

Write-Host "Building HolOrama.exe with Nuitka $version ..." -ForegroundColor Cyan
& $python @nuitkaArgs
if ($LASTEXITCODE -ne 0) { throw "Nuitka build failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Done. Executable at: build\nuitka\main.dist\HolOrama.exe" -ForegroundColor Green
