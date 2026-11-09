"""Check that HolOrama's onnxruntime pipeline (src/segmentation/onnx_inference.py) reproduces
nnUNetPredictor on real frames. Needs the same throwaway export environment as export_onnx.py
(torch + nnUZoo) plus onnxruntime.

Usage:
    python verify_onnx.py <nnUNet model folder> <model .onnx> <folder of .nii.gz frames> [--max-frames 64]

All frames of the folder (sorted by name, one 2D frame or a (frames, y, x) stack per file) are
stacked into one volume and passed to both pipelines exactly as the app passes a pullback.
Prints per-frame Dice, the number of differing pixels and the run times; exits non-zero if
the mean Dice of foreground is below --min-dice.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from domain.intravascular.segmentation_types import SegmentationSettings  # noqa: E402
from segmentation.onnx_inference import OnnxSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_folder', type=Path)
    parser.add_argument('onnx', type=Path)
    parser.add_argument('frames_dir', type=Path)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=64)
    parser.add_argument('--min-dice', type=float, default=0.995)
    parser.add_argument('--no-mirroring', action='store_true')
    args = parser.parse_args()

    files = sorted(args.frames_dir.glob('*.nii.gz'), key=natural_key)
    arrays = [sitk.GetArrayFromImage(sitk.ReadImage(str(f))) for f in files]
    frames = [a[None] if a.ndim == 2 else a for a in arrays]
    volume = np.concatenate(frames)[: args.max_frames]
    print(f'Volume {volume.shape} {volume.dtype} from {len(files)} files')

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.no_mirroring,
        perform_everything_on_device=False,
        device=torch.device('cpu'),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_folder), use_folds=(args.fold,), checkpoint_name='checkpoint_final.pth'
    )
    t = time.perf_counter()
    reference = predictor.predict_single_npy_array(volume[None].astype(np.float32), {'spacing': [1, 1, 1]})
    t_ref = time.perf_counter() - t

    settings = SegmentationSettings(model_file=str(args.onnx), device='cpu', use_mirroring=not args.no_mirroring)
    segmenter = OnnxSegmenter(settings)
    segmenter.load()
    t = time.perf_counter()
    ours = segmenter.predict(volume)
    t_onnx = time.perf_counter() - t
    assert ours is not None

    # Sub-range must give the same masks as the corresponding frames of a full run.
    sub = segmenter.predict(volume, frames=range(3, 7))
    assert sub is not None and np.array_equal(sub[3:7], ours[3:7]) and not sub[:3].any()

    dices = []
    for z in range(volume.shape[0]):
        a, b = reference[z] > 0, ours[z] > 0
        denom = a.sum() + b.sum()
        dice = 1.0 if denom == 0 else 2 * (a & b).sum() / denom
        dices.append(dice)
        print(f'frame {z:3d}: dice {dice:.5f}, differing pixels {(a != b).sum():5d}, fg {a.sum():6d}')
    n = volume.shape[0]
    print(f'\nmean dice {np.mean(dices):.5f}, min {np.min(dices):.5f}')
    print(f'nnUNetPredictor (torch cpu): {t_ref:.1f}s ({t_ref / n:.2f}s/frame)')
    print(f'onnxruntime cpu:             {t_onnx:.1f}s ({t_onnx / n:.2f}s/frame)')
    if np.mean(dices) < args.min_dice:
        sys.exit(f'FAILED: mean dice {np.mean(dices):.5f} < {args.min_dice}')
    print('OK')


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path.name)]


if __name__ == '__main__':
    main()
