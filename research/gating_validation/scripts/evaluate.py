"""Score the app's automatic gating against the labels.

Usage:
    python evaluate.py                    all cached cases, writes results/evaluation_<date>.txt and .csv
    python evaluate.py --group tuning     one group of cases.csv

Gating runs exactly as in the app: harmonic-sum heart rate, bandpass 0.7-2.2 x HR,
AutomaticGating on the image signal, with and without the lumen minor-axis clock.
Scores count labels within 1 and 2 frames inside the labelled section. Perfect cases
weigh double in the overall totals.
"""

import argparse
import csv
import datetime
import os

import numpy as np

import common
from signal_processing.automatic_gating import AutomaticGating
from signal_processing.gating_pipeline import bandpass_filter, detect_heart_rate, normalize_data

METHODS = ('image only', 'image + minor axis')


def pct(a: float, n: float) -> float:
    return 100 * a / n if n else float('nan')


def gate(case: dict, f_heart: float, g: np.ndarray, use_minor: bool):
    mw = common.fake_main_window(case['fs'])
    minor = case['minor'] if use_minor else None
    AutomaticGating(mw).automatic_gating(g, f_heart, minor_axis=minor)
    rt = mw.runtime_data
    return rt.gated_frames_dia, rt.gated_frames_sys, rt.gating_signal.get('gating_skipped', [])


def main(group: str | None) -> None:
    lines, rows = [], []
    totals: dict[tuple[str, str], np.ndarray] = {}
    for row in common.load_cases(group):
        c = common.load_case(row)
        if c is None:
            continue
        f = detect_heart_rate(c['raw'], c['fs'], *common.HR_RANGE)
        beat = c['fs'] / f
        g = normalize_data(bandpass_filter(c['raw'], f, c['fs'], *common.BANDPASS), common.NORMALIZE_STEP)
        label_bpm = 60 * c['fs'] / np.median(np.diff(c['tD'])) if len(c['tD']) > 1 else float('nan')
        text = f"{c['case_id']:>5}{'*' if row['perfect'] else ' '} {row['group']:8} HR {60 * f:4.0f} (labels {label_bpm:4.0f}) |"
        for m in METHODS:
            D, S, skipped = gate(c, f, g, m.endswith('axis'))
            sc1 = common.score(D, S, c['tD'], c['tS'], c['lo'], c['hi'], 1)
            sc2 = common.score(D, S, c['tD'], c['tS'], c['lo'], c['hi'], 2)
            v = common.violations(D, S, c['lo'], c['hi'], beat)
            n_skip = sum(c['lo'] <= a <= c['hi'] for a, _ in skipped)
            vec = np.r_[sc1[0], sc1[1], sc2[0][0], sc2[1][0], v, n_skip]
            for key in ((row['group'], m), ('all', m)):
                w = row['weight'] if key[0] == 'all' else 1.0
                totals[key] = totals.get(key, np.zeros(10)) + w * vec
            text += (
                f" {m}: D {pct(sc1[0][0], sc1[0][2]):3.0f} S {pct(sc1[1][0], sc1[1][2]):3.0f} viol {v} skip {n_skip} |"
            )
            rows.append(
                dict(
                    case_id=c['case_id'],
                    group=row['group'],
                    perfect=int(row['perfect']),
                    method=m,
                    hr_bpm=round(60 * f, 1),
                    label_bpm=round(label_bpm, 1),
                    d_labels=int(sc1[0][2]),
                    s_labels=int(sc1[1][2]),
                    d_correct_tol1=int(sc1[0][0]),
                    d_swapped_tol1=int(sc1[0][1]),
                    s_correct_tol1=int(sc1[1][0]),
                    s_swapped_tol1=int(sc1[1][1]),
                    d_correct_tol2=int(sc2[0][0]),
                    s_correct_tol2=int(sc2[1][0]),
                    violations=int(v),
                    skipped=int(n_skip),
                )
            )
        lines.append(text)

    lines.append('\nTotals: correct/swapped % within 1 frame | correct % within 2 frames | violations | skipped beats')
    lines.append("('all' weighs perfect cases double)")
    for key in sorted(totals, key=lambda k: (k[0] != 'all', k)):
        t = totals[key]
        lines.append(
            f'{key[0]:8} {key[1]:20} D {100*t[0]/t[2]:5.1f}/{100*t[1]/t[2]:4.1f}  S {100*t[3]/t[5]:5.1f}/{100*t[4]/t[5]:4.1f}'
            f' | D {100*t[6]/t[2]:5.1f} S {100*t[7]/t[5]:5.1f} | {int(t[8])} | {int(t[9])}'
        )
    print('\n'.join(lines))
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    base = os.path.join(common.RESULTS_DIR, f'evaluation_{stamp}' + (f'_{group}' if group else ''))
    with open(base + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwritten {base}.txt / .csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group')
    main(ap.parse_args().group)
