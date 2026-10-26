"""Parameter sweeps of the app's beat tracker (module constants of automatic_gating).

Usage:
    python tune.py                      one-at-a-time sweep around the shipped values
    python tune.py --group tuning       restrict to one group of cases.csv

The constants are set on the module before each run, so the app code itself is scored.
Report the setting chosen here only together with a check on cases it was not tuned on.
"""

import argparse

import numpy as np

import common
import signal_processing.automatic_gating as ag
from signal_processing.gating_pipeline import bandpass_filter, detect_heart_rate, normalize_data

SWEEP = {
    'SYSTOLE_MARGIN': [0.25, 0.3, 0.35, 0.4],
    'DELAY_WEIGHT': [4, 16, 64, 256],
    'GAP_WEIGHT': [2, 5, 20],
    'TEMPO_WEIGHT': [3, 10, 30],
    'SKIP_PENALTY': [3, 6, 12],
    'CLOCK_WEIGHT': [0, 0.5, 1, 2],
}


def run(cases: list[dict]) -> tuple[float, float, float, float]:
    t = np.zeros(6)
    for c in cases:
        mw = common.fake_main_window(c['fs'])
        ag.AutomaticGating(mw).automatic_gating(c['g'], c['f'], minor_axis=c['minor'])
        D, S = mw.runtime_data.gated_frames_dia, mw.runtime_data.gated_frames_sys
        sc1 = common.score(D, S, c['tD'], c['tS'], c['lo'], c['hi'], 1)
        sc2 = common.score(D, S, c['tD'], c['tS'], c['lo'], c['hi'], 2)
        t += c['row']['weight'] * np.array([sc1[0][0], sc1[1][0], sc2[0][0], sc2[1][0], sc1[0][2], sc1[1][2]])
    return 100 * t[0] / t[4], 100 * t[1] / t[5], 100 * t[2] / t[4], 100 * t[3] / t[5]


def fmt(r):
    return f'D {r[0]:5.1f} S {r[1]:5.1f} (tol 2: D {r[2]:5.1f} S {r[3]:5.1f}) mean {np.mean(r[:2]):5.1f}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group')
    args = ap.parse_args()
    cases = []
    for row in common.load_cases(args.group):
        c = common.load_case(row)
        if c is None:
            continue
        c['f'] = detect_heart_rate(c['raw'], c['fs'], *common.HR_RANGE)
        c['g'] = normalize_data(bandpass_filter(c['raw'], c['f'], c['fs'], *common.BANDPASS), common.NORMALIZE_STEP)
        cases.append(c)
    print(f"{len(cases)} cases, perfect cases weigh double")
    shipped = {k: getattr(ag, k) for k in SWEEP}
    print('shipped', shipped, fmt(run(cases)))
    for name, values in SWEEP.items():
        results = []
        for v in values:
            setattr(ag, name, v)
            results.append(f'{v}: {fmt(run(cases))}')
        setattr(ag, name, shipped[name])
        print(f'{name:15}', ' | '.join(results), flush=True)
