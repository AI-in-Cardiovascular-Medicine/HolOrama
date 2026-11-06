# Build a standalone Windows executable of HolOrama with Nuitka.
#
# The build contains the whole app, including Automatic Segmentation (onnxruntime + the
# bundled ONNX model). The one thing not compiled in is vmtk for Calculate Centerlines: it
# exists only on conda-forge, so the app sets it up itself on first use (pixi), from the
# manifest/lock bundled below. Segmentation in the exe runs on the CPU (plain onnxruntime).
# Only uv is required: the script creates its own build environment in build\venv.
#
# Usage: .\build_nuitka.ps1
# Output: build\nuitka\main.dist\HolOrama.exe

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Build from a dedicated environment, never from the developer's .venv: that one may carry the
# GPU runtime (segmentation-gpu, ~2 GB of CUDA DLLs) or stray packages Nuitka would follow into
# the exe. build\venv is the locked core + CPU 'segmentation' extra + Nuitka, nothing else.
$env:UV_PROJECT_ENVIRONMENT = 'build\venv'
$env:UV_HTTP_TIMEOUT = '300'
Write-Host "Syncing the build environment (build\venv) ..." -ForegroundColor Cyan
uv sync --locked --extra segmentation --group dev
if ($LASTEXITCODE -ne 0) { throw "uv sync of the build environment failed" }
Remove-Item Env:UV_PROJECT_ENVIRONMENT
$python = '.\build\venv\Scripts\python.exe'
# Read the version from src\version.py so the exe's file/product metadata can't drift
# from the real release (build_installer.ps1 reads the same source).
$versionLine = Select-String -Path 'src\version.py' -Pattern "__version__\s*=\s*'([^']+)'" | Select-Object -First 1
if (-not $versionLine) { throw "Could not read __version__ from src\version.py" }
$version = $versionLine.Matches[0].Groups[1].Value

# The installer is CPU-only by design; make sure the GPU runtime never slips in.
& $python -c "import onnxruntime as o, sys; sys.exit('CUDAExecutionProvider' in o.get_available_providers())"
if ($LASTEXITCODE -ne 0) { throw "build\venv has onnxruntime-gpu; delete build\venv and rerun." }

# The dev toolchain may be importable from the build venv but must never be followed into the exe.
$exclude = @(
    'pytest', 'pytest_qt', 'pytestqt', 'black', 'mypy', 'ruff', 'pre_commit', 'nuitka',
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
    # The built-in contour presets are read from presets/ next to the executable
    # (domain/intravascular/contour_presets.py and domain/ccta/presets.py).
    '--include-data-dir=src\presets=presets',
    # The automatic segmentation model, found next to segmentation/onnx_inference.py.
    '--include-data-files=src\segmentation\weights\ivus_lumen_u2netp.onnx=segmentation\weights\ivus_lumen_u2netp.onnx',
    # pixi manifest + lock of the vmtk environment, found next to cut_geometry/vmtk.py.
    '--include-data-files=src\pages\ccta\left_half\cut_geometry\vmtk_env\pixi.toml=pages\ccta\left_half\cut_geometry\vmtk_env\pixi.toml',
    '--include-data-files=src\pages\ccta\left_half\cut_geometry\vmtk_env\pixi.lock=pages\ccta\left_half\cut_geometry\vmtk_env\pixi.lock',
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

Write-Host "Building HolOrama.exe $version with Nuitka ..." -ForegroundColor Cyan
& $python @nuitkaArgs
if ($LASTEXITCODE -ne 0) { throw "Nuitka build failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Done. Executable at: build\nuitka\main.dist\HolOrama.exe" -ForegroundColor Green
