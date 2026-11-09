"""Time the automatic segmentation on a pullback from the console, frame by frame.

    uv run python scripts/benchmark_segmentation.py <pullback.dcm|.nii|.nii.gz> [options]

Examples:
    uv run python scripts/benchmark_segmentation.py run.dcm                      # all frames, device from 'auto'
    uv run python scripts/benchmark_segmentation.py run.dcm --device cpu --frames 0:50
    uv run python scripts/benchmark_segmentation.py run.dcm --save masks.npy     # keep the masks for comparing

Prints the time of each stage (reading, model load, whole-volume preprocessing) and of each
frame, split into network time (onnxruntime) and the numpy work around it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from loguru import logger  # noqa: E402

from segmentation.onnx_inference import DEVICES, PRECISIONS, OnnxSegmenter, nonzero_bbox, zscore_stats  # noqa: E402


def read_pullback(path: Path) -> np.ndarray:
    """(frames, y, x) gray volume, read the way the app reads it."""
    from input_output.input.image import _parse_pixel_array, _read_dicom, _read_nifti

    if path.name.lower().endswith(('.nii', '.nii.gz')):
        pixel_array, _ = _read_nifti(str(path))
    else:
        pixel_array, _ = _read_dicom(str(path))
    images, _ = _parse_pixel_array(pixel_array)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pullback', type=Path)
    parser.add_argument('--device', choices=DEVICES, default='auto')
    parser.add_argument(
        '--precision', choices=PRECISIONS, default='auto', help='GPU precision; the CPU always runs fp32'
    )
    parser.add_argument('--frames', default=None, help='start:stop (0-based, stop excluded); default all')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--no-mirroring', action='store_true', help='skip test-time mirroring (4x fewer network calls)')
    parser.add_argument('--quiet', action='store_true', help='summary only, no line per frame')
    parser.add_argument('--save', type=Path, default=None, help='write the masks to this .npy file')
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level='WARNING')

    t = time.perf_counter()
    volume = read_pullback(args.pullback)
    print(f'read        {time.perf_counter() - t:7.2f} s   {volume.shape} {volume.dtype}')

    start, stop = 0, volume.shape[0]
    if args.frames:
        a, b = args.frames.split(':')
        start, stop = int(a or 0), int(b or volume.shape[0])
    frames = range(start, stop)

    seg = OnnxSegmenter(
        device=args.device, precision=args.precision, batch_size=args.batch_size, use_mirroring=not args.no_mirroring
    )
    t = time.perf_counter()
    seg.load()
    print(
        f'model load  {time.perf_counter() - t:7.2f} s   device: {seg.active_device} ({seg.active_precision})'
        + (f'  (GPU not used: {seg.fallback_reason})' if seg.fallback_reason else '')
    )

    # The same whole-volume steps predict() starts with, timed on their own.
    t = time.perf_counter()
    crop = nonzero_bbox(volume)
    t_bbox = time.perf_counter() - t
    t = time.perf_counter()
    zscore_stats(volume[crop])
    t_stats = time.perf_counter() - t
    print(f'preprocess  {t_bbox + t_stats:7.2f} s   (bounding box {t_bbox:.2f} s, mean/std {t_stats:.2f} s)')

    session_run = seg._session.run
    net = {'s': 0.0, 'calls': 0}

    def timed_run(*a, **k):
        t0 = time.perf_counter()
        out = session_run(*a, **k)
        net['s'] += time.perf_counter() - t0
        net['calls'] += 1
        return out

    seg._session.run = timed_run  # type: ignore[method-assign]

    state = {'done': 0, 't': 0.0, 'net': 0.0}
    per_frame: list[float] = []

    def on_progress(done: int, total: int) -> bool:
        if done > state['done']:
            now = time.perf_counter()
            n = done - state['done']
            dt, dnet = (now - state['t']) / n, (net['s'] - state['net']) / n
            per_frame.extend([dt] * n)
            if not args.quiet:
                print(
                    f'frame {start + done - 1:5d}  {dt:6.3f} s   network {dnet:6.3f} s   other {dt - dnet:6.3f} s'
                    f'   [{done}/{total}]',
                    flush=True,
                )
            state.update(done=done, t=now, net=net['s'])
        return True

    t_total = time.perf_counter()
    state['t'] = t_total
    masks = seg.predict(volume, frames, on_progress)
    total = time.perf_counter() - t_total
    assert masks is not None

    n = len(per_frame)
    if n:
        pf = np.array(per_frame)
        print(f'\n{n} frames in {total:.1f} s on {seg.active_device}  (includes preprocessing again)')
        print(
            f'per frame: mean {pf.mean():.3f} s, median {np.median(pf):.3f} s, max {pf.max():.3f} s'
            f'  (first frame {pf[0]:.3f} s, includes preprocessing)'
        )
        print(
            f'network: {net["s"]:.1f} s in {net["calls"]} calls ({net["s"] / net["calls"] * 1000:.0f} ms each),'
            f' {net["s"] / total:.0%} of the total'
        )
    if args.save:
        np.save(args.save, masks)
        print(f'masks written to {args.save}')


if __name__ == '__main__':
    main()
