"""Label-based analyses behind the design choices (for the methods section).

Usage:
    python analyses.py phase          phase locking of each feature's heart-rate component to D and S
    python analyses.py discriminate   which per-valley feature separates labelled D from S

phase: circular mean phase and resultant length R at the labelled D frames
       (R 1 = locked, 0 = random). An across-case R near 1 means D sits at the same
       phase in every patient, so the feature can place diastole on its own.
discriminate: for each labelled D and the S of the same beat, how often the feature is
       lower at D. A usable discriminator is far from 50 % on the same side in every case.
Optional features need build_cache.py --frames / --blur.
"""

import sys

import numpy as np
from scipy.signal import hilbert, peak_widths

import common
from signal_processing.gating_pipeline import bandpass_filter, detect_heart_rate, normalize_data


def interp(x):
    ok = ~np.isnan(x)
    return np.interp(np.arange(len(x)), np.where(ok)[0], x[ok]) if ok.sum() > 1 else None


def circ(ph):
    m = np.mean(np.exp(1j * ph))
    return np.degrees(np.angle(m)), np.abs(m)


def cases():
    for row in common.load_cases():
        c = common.load_case(row)
        if c is not None and len(c['tS']):
            c['f'] = detect_heart_rate(c['raw'], c['fs'], *common.HR_RANGE)
            yield c


def phase():
    rows = {}
    for c in cases():
        signals = {'image': c['raw']}
        signals.update({k: interp(v) for k, v in c['contour'].items()})
        if c['blur'] is not None:
            signals['blur'] = c['blur']
        line = f"{c['case_id']:>5} |"
        for k, x in signals.items():
            if x is None:
                continue
            ph = np.angle(hilbert(bandpass_filter(x, c['f'], c['fs'], 0.7, 1.3)))  # 0 deg = maximum of the HR component
            mD, rD = circ(ph[c['tD']])
            mS, _ = circ(ph[c['tS']])
            rows.setdefault(k, []).append((mD, rD, (mS - mD + 180) % 360 - 180))
            line += f' {k[:6]} D {mD:5.0f} deg R {rD:.2f} |'
        print(line)
    print('\nfeature          within-case R at D (median) | across-case R of the D phase | D to S separation (deg)')
    for k, r in rows.items():
        _, across = circ(np.radians([x[0] for x in r]))
        print(
            f'{k:16} {np.median([x[1] for x in r]):.2f} | {across:.2f} | {np.round([x[2] for x in r]).astype(int).tolist()}'
        )


def discriminate():
    summary = {}
    for c in cases():
        T = c['fs'] / c['f']
        raw = c['raw']
        g = normalize_data(bandpass_filter(raw, c['f'], c['fs'], *common.BANDPASS), common.NORMALIZE_STEP)
        w = max(2, int(round(0.15 * T)))
        pairs = [(d, s) for d in c['tD'] for s in c['tS'] if 0 < s - d < 0.8 * T and d - w >= 0 and s + w < len(g)]
        feats = {
            'valley depth': lambda i: g[i],
            'raw stillness': lambda i: raw[i - w : i + w + 1].mean(),
            'valley width': lambda i: -peak_widths(-g, [i], rel_height=0.5)[0][0],
        }
        try:
            U = np.load(c['frames_path']).astype(np.float32)
            U = U.reshape(len(U), -1)
            U -= U.mean(axis=1, keepdims=True)
            U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
            ks = range(int(round(0.85 * T)), int(round(1.15 * T)) + 1)
            feats['beat similarity'] = lambda i: -max(
                [U[i] @ U[i + k] for k in ks if i + k < len(U)] + [U[i] @ U[i - k] for k in ks if i - k >= 0]
            )
        except FileNotFoundError:
            pass
        if c['blur'] is not None:
            b = normalize_data(c['blur'], common.NORMALIZE_STEP)
            feats['blur'] = lambda i: b[i - 1 : i + 2].mean()
        res = {k: np.mean([fn(d) < fn(s) for d, s in pairs]) for k, fn in feats.items()}
        for k, v in res.items():
            summary.setdefault(k, []).append(v)
        print(f"{c['case_id']:>5} n={len(pairs):3d} | " + ' | '.join(f'{k} {v:4.0%}' for k, v in res.items()))
    print(
        '\nfeature          fraction of beats with the feature lower at D than at S: min / median / max, cases > 60 %'
    )
    for k, v in summary.items():
        print(f'{k:16} {min(v):4.0%} / {np.median(v):4.0%} / {max(v):4.0%}, {sum(x > 0.6 for x in v)}/{len(v)}')


if __name__ == '__main__':
    {'phase': phase, 'discriminate': discriminate}[sys.argv[1]]()
