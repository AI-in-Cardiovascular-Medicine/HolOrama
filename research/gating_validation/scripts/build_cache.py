"""Build the per-case cache: image signal, labels and lumen measurements (optionally frames and blur).

Usage:
    python build_cache.py                 all cases in cases.csv that are not cached yet
    python build_cache.py 348 351         only these cases
    python build_cache.py --rebuild       rebuild even if cached (e.g. after re-labelling)
    python build_cache.py --frames --blur also store 4x downsampled frames and the blur feature

Each case covers the labelled frames +-20 (the app uses the frame range chosen in the dialog).
The newest *_contours*.json in the Run1/Rest folder is used unless cases.csv names a json.
A labels/<case_id>.csv (frame,phase) overrides the phases stored in the json.
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np

import common  # noqa: F401  (sets sys.path and silences loguru)
from input_output.input.image import _parse_pixel_array, _read_dicom
from input_output.input.metadata import parse_metadata_dcm
from signal_processing.gating_pipeline import compute_correlation_signal, normalize_data

KEYS = ('area', 'minor_axis', 'major_axis', 'elliptic_ratio', 'circumference')
MARGIN = 20
DOWNSAMPLE = 4


def find_json(row: dict) -> str:
    if row['json'].strip():
        p = row['json'].strip()
        return p if os.path.isabs(p) else os.path.join(common.REPO, p)
    case_dir = os.path.join(common.DATA_ROOT, f"NARCO_{row['case_id']}")
    folders = [r for r, _, _ in os.walk(case_dir) if os.path.basename(r) == row['folder']]
    if not folders:
        raise FileNotFoundError(f"no {row['folder']} folder in {case_dir}")
    return max(glob.glob(os.path.join(folders[0], '*_contours*.json')), key=os.path.getmtime)


def find_dicom(json_path: str) -> str:
    stem = os.path.basename(json_path).split('_contours')[0]
    found = [
        p
        for p in glob.glob(os.path.join(os.path.dirname(json_path), stem + '*'))
        if not p.endswith(('.json', '.csv', '.txt'))
    ]
    return sorted(found, key=len)[0]


def labels(row: dict, data: dict) -> tuple[list[int], list[int]]:
    override = os.path.join(common.VALIDATION_DIR, 'labels', f"{row['case_id']}.csv")
    if os.path.exists(override):
        with open(override, newline='') as f:
            rows = list(csv.DictReader(f))
        return [int(r['frame']) for r in rows if r['phase'] == 'D'], [
            int(r['frame']) for r in rows if r['phase'] == 'S'
        ]

    def phase(p: str) -> list[int]:
        return sorted(int(k) for k in data if k.isdigit() and isinstance(data[k], dict) and data[k].get('phase') == p)

    return phase('D'), phase('S')


def crop_norm(frames: np.ndarray) -> np.ndarray:
    """Same crop and per-frame [0, 1] scaling as gating_pipeline.prepare_data."""
    fc = frames[:, 50:450, 50:450].astype(np.float32)
    f_lo = fc.min(axis=(1, 2), keepdims=True)
    f_hi = fc.max(axis=(1, 2), keepdims=True)
    return (fc - f_lo) / np.where(f_hi > f_lo, f_hi - f_lo, 1.0)


def blur_signal(frames: np.ndarray) -> np.ndarray:
    """Talou et al. s1(n) = -sum |grad u_n|, high when edges are blurred by transducer motion."""
    out = np.empty(len(frames))
    for n, f in enumerate(frames):
        gy, gx = np.gradient(f)
        out[n] = -np.sqrt(gx**2 + gy**2).sum()
    return out


def build(row: dict, with_frames: bool, with_blur: bool) -> None:
    t0 = time.time()
    json_path = find_json(row)
    data = json.load(open(json_path))
    tD, tS = labels(row, data)
    if not tD and not tS:
        print(f"{row['case_id']}: no labels, skipped")
        return
    pix, md_df = _read_dicom(find_dicom(json_path))
    frames, _ = _parse_pixel_array(pix)
    fs = parse_metadata_dcm(md_df, frames.shape[0]).frame_rate or common.FS_DEFAULT
    lo, hi = max(0, min(tD + tS) - MARGIN), min(frames.shape[0], max(tD + tS) + MARGIN + 1)
    fc = crop_norm(frames[lo:hi])
    out = dict(
        image_raw=normalize_data(compute_correlation_signal(fc), common.NORMALIZE_STEP),
        D=np.array(tD) - lo,
        S=np.array(tS) - lo,
        lo=lo,
        fs=fs,
        json=json_path,
    )
    for k in KEYS:
        values = np.full(hi - lo, np.nan)
        for i in range(lo, hi):
            fd = data.get(str(i))
            m = (
                fd.get('lumen', {}).get('measurements')
                if isinstance(fd, dict) and isinstance(fd.get('lumen'), dict)
                else None
            )
            if m and m.get(k) is not None:
                values[i - lo] = m[k]
        out[k] = values
    if with_blur:
        out['blur'] = blur_signal(fc)
    np.savez(common.cache_path(row['case_id']), **out)
    if with_frames:
        n, h, w = fc.shape
        small = fc.reshape(n, h // DOWNSAMPLE, DOWNSAMPLE, w // DOWNSAMPLE, DOWNSAMPLE).mean(axis=(2, 4))
        np.save(common.cache_path(row['case_id'], '_frames.npy'), small.astype(np.float16))
    print(
        f"{row['case_id']}: frames {lo}-{hi}, {len(tD)} D / {len(tS)} S labels, fs {fs}, {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--frames', action='store_true')
    ap.add_argument('--blur', action='store_true')
    args = ap.parse_args()
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    for row in common.load_cases():
        if args.cases and row['case_id'] not in args.cases:
            continue
        if os.path.exists(common.cache_path(row['case_id'])) and not args.rebuild:
            continue
        build(row, args.frames, args.blur)
