# Segmentation model export

HolOrama runs its automatic IVUS segmentation model with **onnxruntime** only
(`src/segmentation/onnx_inference.py`). The two scripts here are the bridge from a trained
nnU-Net / nnUZoo checkpoint to the bundled `.onnx`, and the only place in the project that
still needs PyTorch and nnUZoo. Nothing here is needed to run, test or build the app.

| Script | Purpose |
|---|---|
| `export_onnx.py` | Rebuilds the network exactly as `nnUNetPredictor` does for inference, exports it to ONNX and stores what the runtime needs (patch size, mirror axes, normalisation, labels) as ONNX metadata. |
| `verify_onnx.py` | Runs `nnUNetPredictor` and HolOrama's ONNX pipeline on the same frames and compares the masks (Dice, differing pixels, run time). |

## Throwaway export environment (Windows)

nnUZoo 0.1.0 pins `torch==2.4.0` and needs the same patches the old `install.ps1` applied
(see git history of `install.ps1` before the ONNX switch for the corrected
`nnUNetTrainerU2NetP.py`). Keep this environment outside the repository:

```powershell
git clone --depth 1 --branch 0.1.0 https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo $env:TEMP\nnuzoo
# apply the three patches: packages.find in pyproject.toml, the corrected trainer,
# and the dynamic_network_architectures import in nnunetv2/nets/segmentation/u2net.py
uv venv -p 3.11 $env:TEMP\export-env
uv pip install --python $env:TEMP\export-env\Scripts\python.exe "torch==2.4.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $env:TEMP\export-env\Scripts\python.exe $env:TEMP\nnuzoo "numpy==1.26.4" "optree==0.13.1" onnx onnxruntime
# torch 2.4.0 on Windows also needs libomp140.x86_64.dll in torch\lib (conda-forge llvm-openmp 14)
```

## Export and verify

```powershell
$env:PYTHONIOENCODING = 'utf-8'   # torchinfo prints box-drawing characters
$py = "$env:TEMP\export-env\Scripts\python.exe"
& $py scripts\model_export\export_onnx.py <model folder> src\segmentation\weights\ivus_lumen_u2netp.onnx
& $py scripts\model_export\verify_onnx.py <model folder> src\segmentation\weights\ivus_lumen_u2netp.onnx <model folder>\fold_0\imagesTs
```

Result for the bundled model (`nnUNetTrainerU2NetP__nnUNetResEncUNetLPlans__2d`, fold 0,
`checkpoint_final.pth`) on 24 test frames, CPU, mirroring on:

- mean Dice 0.99999 (min 0.99976), 7 differing pixels in total: nnU-Net accumulates the
  tile logits in float16, the ONNX pipeline in float32
- nnUNetPredictor (torch 2.4 CPU) 12.3 s/frame, onnxruntime 3.1 s/frame (16 cores)

Only 2D models with z-score normalisation, target spacing 1 and no transpose are supported;
`export_onnx.py` refuses anything else.
