"""Shared paths, case list and scoring for the gating validation scripts."""

import csv
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(VALIDATION_DIR))
sys.path.insert(0, os.path.join(REPO, 'src'))

from loguru import logger  # noqa: E402

logger.remove()

DATA_ROOT = os.environ.get('NARCO_ROOT', r'E:\PostDoc_Anselm-Stark\06_ivus_data\01_narco_ivus_raw')
CACHE_DIR = os.environ.get('GATING_CACHE', os.path.join(VALIDATION_DIR, 'cache'))
CASES_CSV = os.path.join(VALIDATION_DIR, 'cases.csv')
RESULTS_DIR = os.path.join(VALIDATION_DIR, 'results')
FS_DEFAULT = 30.0
NORMALIZE_STEP = 100  # config.yaml gating.normalize_step
BANDPASS = (0.7, 2.2)  # config.yaml gating.bandpass_lo_frac / bandpass_hi_frac
HR_RANGE = (0.75, 3.3333)  # config.yaml gating.f_cardiac_min / f_cardiac_max


def load_cases(group: str | None = None) -> list[dict]:
    """Rows of cases.csv that are not excluded, optionally only one group."""
    with open(CASES_CSV, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if not r['exclude'].strip()]
    if group:
        rows = [r for r in rows if r['group'] == group]
    for r in rows:
        r['perfect'] = r['perfect'].strip() == '1'
        r['swap_labels'] = r['swap_labels'].strip() == '1'
        r['weight'] = 2.0 if r['perfect'] else 1.0
    return rows


def cache_path(case_id: str, suffix: str = '.npz') -> str:
    return os.path.join(CACHE_DIR, f'{case_id}{suffix}')


def load_case(row: dict) -> dict | None:
    """Cached signals plus labels with the corrections from cases.csv applied."""
    p = cache_path(row['case_id'])
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    tD, tS = np.sort(z['D']), np.sort(z['S'])
    if row['swap_labels']:
        tD, tS = tS, tD
    lo_frame, hi_frame = row['label_frame_min'].strip(), row['label_frame_max'].strip()
    if lo_frame or hi_frame:
        a = (int(lo_frame) if lo_frame else -(10**9)) - int(z['lo'])
        b = (int(hi_frame) if hi_frame else 10**9) - int(z['lo'])
        tD, tS = tD[(tD >= a) & (tD <= b)], tS[(tS >= a) & (tS <= b)]
    minor = z['minor_axis'] if 'minor_axis' in z.files else None
    if minor is not None:
        ok = ~np.isnan(minor)
        minor = np.interp(np.arange(len(minor)), np.where(ok)[0], minor[ok]) if ok.sum() > 1 else None
    return dict(
        row=row,
        case_id=row['case_id'],
        raw=z['image_raw'],
        fs=float(z['fs']),
        offset=int(z['lo']),
        tD=tD,
        tS=tS,
        lo=int(min(np.r_[tD, tS])),
        hi=int(max(np.r_[tD, tS])),
        minor=minor,
        contour={k: z[k] for k in ('area', 'minor_axis', 'major_axis', 'elliptic_ratio') if k in z.files},
        blur=z['blur'] if 'blur' in z.files else None,
        frames_path=cache_path(row['case_id'], '_frames.npy'),
    )


def fake_main_window(fs: float):
    """Minimal stand-in for MainWindow, enough for AutomaticGating."""
    rt = SimpleNamespace(
        frame_data_dct={}, metadata={'frame_rate': fs}, gating_signal={}, gated_frames_dia=[], gated_frames_sys=[]
    )
    return SimpleNamespace(
        runtime_data=rt,
        config=SimpleNamespace(gating=SimpleNamespace(normalize_step=NORMALIZE_STEP)),
        diastolic_frame_box=Mock(),
        systolic_frame_box=Mock(),
        save_contours_soon=Mock(),
    )


def score(D, S, tD, tS, lo, hi, tol):
    """Per phase [correct, swapped, n labels] inside the labelled section.

    correct: a prediction of the same phase within tol frames.
    swapped: none of the same phase but one of the other phase within tol.
    """
    D = np.array([x for x in D if lo - tol <= x <= hi + tol])
    S = np.array([x for x in S if lo - tol <= x <= hi + tol])

    def near(arr, t):
        return len(arr) > 0 and np.min(np.abs(arr - t)) <= tol

    out = []
    for truth, same, other in ((tD, D, S), (tS, S, D)):
        ok = sum(near(same, t) for t in truth)
        sw = sum((not near(same, t)) and near(other, t) for t in truth)
        out.append(np.array([ok, sw, len(truth)], dtype=float))
    return out


def violations(D, S, lo, hi, beat):
    """Non-physiological events in the section: D-D under 0.7 beat, D next to S closer than 0.25 beat."""
    D = [x for x in D if lo <= x <= hi]
    S = [x for x in S if lo <= x <= hi]
    events = sorted([(x, 'D') for x in D] + [(x, 'S') for x in S])
    short = sum((b - a) < 0.7 * beat for a, b in zip(D[:-1], D[1:]))
    close = sum((b[0] - a[0]) < 0.25 * beat for a, b in zip(events[:-1], events[1:]) if a[1] != b[1])
    return short + close
