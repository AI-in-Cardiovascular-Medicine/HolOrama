# Intravascular module
## Urgent
- [] Adjust gating module to new data
- [] Export solid line as it's own contour

## Requested features
- [] clicking in longitudinal view let's user set region of interest
- [] mousewheel in longitudinal view changes cut line
- [] Tag every x mm

## Lower priority
- [] Reduce memory by not storing image_rgb next to image
- [] Reduce memory by not storing the whole pixel_array in memory but read in badges when needed.
- [] Config callable from GUI

## Segmentation tools
- [] Ensure that only tagged frames can be exported
- [] Aim for a first test data set including lumen, side branch, catheter for around 300 frames
- [] Allow brush mode in mask mode
- [] Seperation of Concerns: logic on mousemovements coupled to segmentation tool

## Bugs

# CCTA module

## Bugs

# Build & platforms (next PR: compiled builds for Linux and macOS)
Builds on fix/cleanup-build-system: the Windows installer (Nuitka + DirectML) and the source installs
on all three platforms work there. This PR adds compiled builds for Linux and macOS.

## Linux compiled build
- [] `build_nuitka.sh`, the counterpart of build_nuitka.ps1: `uv sync --locked` into build/venv, Nuitka standalone with gcc, bundle the ONNX model and vmtk_env/pixi.toml + pixi.lock as data, post-build check that they and libonnxruntime*.so are present
- [] Decide GPU or CPU-only build: onnxruntime-gpu + CUDA 12/cuDNN pip libs is ~1-1.5 GB and needs NVIDIA driver >= 525, but falls back to CPU on machines without NVIDIA (device 'auto' already handles that); CPU-only stays small
- [] Build on the oldest glibc to support (Ubuntu 22.04 runner or container), test the result on a newer distro
- [] Qt: make sure the xcb (and wayland) platform plugins are bundled; test on a machine without system Qt
- [] No `-march=native` or AVX-512 in the binary (the 0xc000001d crash on Windows): clear CFLAGS, keep a baseline target
- [] Package as tar.gz (optional AppImage) with .desktop file and icon
- [] vmtk: pixi env downloaded on first use (linux-64 is already in pixi.lock); test centerlines end to end in the compiled build

## macOS
- [] CoreML device: add 'coreml' to DEVICES in src/segmentation/onnx_inference.py (CoreMLExecutionProvider ships with the macOS onnxruntime wheel; options MLComputeUnits=ALL, ModelFormat=MLProgram), with fallback disabled and a test inference at the fixed batch shape like CUDA/DirectML
- [] Check CoreML on a real Mac: masks vs CPU (expect float16-level differences, a few px/frame as with nnUZoo) and speed vs CPU (ops CoreML can't run fall back to the CPU and can make it slower)
- [] Apple Silicon has no vmtk on conda-forge (only osx-64): try the osx-64 pixi env under Rosetta 2; if that fails, block centerlines on osx-arm64 with a clear message (VmtkEnv.platform_supported)
- [] Intel Macs: the locked onnxruntime only has macOS arm64 wheels, so segmentation is unavailable there; decide whether to drop Intel Macs or pin an older onnxruntime
- [] Nuitka app bundle (`--mode=app`, .icns icon), built on a macOS arm64 runner
- [] Distribution: unsigned apps need right-click > Open on first start; signing + notarisation needs an Apple Developer ID ($99/year); ship as .dmg

## CI / releases
- [] Release workflow: on a version tag build the Windows installer, Linux tar.gz and macOS app and attach them to the GitHub release
- [] Add macos-latest to the test matrix (core + segmentation on CPU)
- [] Smoke test each compiled build: start it headless and segment a few frames (as scripts/benchmark_segmentation.py does)

## Open decisions
- [] Linux build with GPU (large) or CPU-only
- [] Who tests on a Mac (CoreML and the app bundle can't be checked without one)
- [] Apple Silicon centerlines: Rosetta or blocked
- [] macOS signing: Apple Developer account or unsigned

## Related follow-ups (separate PRs)
- [] multimodars 0.10.0 upgrade (reverted from fix/cleanup-build-system, start from commit 4a7c839)
- [] Optional float16 model for the GPU backends (~1.5x faster on DirectML, ~3 px/frame different from float32)