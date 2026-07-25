import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import matplotlib.pyplot as plt
import numpy as np

from gating.breathing_pipeline import assign_breathing_bins, register_phase

with open(r"C:\Users\ansel\Desktop\Neuer Ordner\IVUS_NARCO_122\Run1\PDWN7A4I_contours_ho_0_3_0.json", "r") as file:
    test = json.load(file)
print(test['gating_signal'].keys())
print(test['0']['phase'])
breath = test['gating_signal']['breathing_display_signal']
frame_keys = sorted((k for k in test if k != 'gating_signal'), key=int)
areas = [test[k]['lumen']['measurements']['area'] for k in frame_keys]
phases = [test[k]['phase'] for k in frame_keys]

frame_indices = [int(k) for k in frame_keys]
dia_idx = [i for i, p in zip(frame_indices, phases) if p == 'D']
dia_areas = [a for a, p in zip(areas, phases) if p == 'D']
sys_idx = [i for i, p in zip(frame_indices, phases) if p == 'S']
sys_areas = [a for a, p in zip(areas, phases) if p == 'S']

breathing_peaks = [219, 433, 628, 889, 1182, 1443]
breathing_valley = [289, 522, 725, 978, 1320, 1611]

n_bins = 5
frame_indices_arr = np.array(frame_indices)
areas_arr = np.array(areas)
bin_cmap = plt.get_cmap('viridis')


def assign_breathing_bins_by_amplitude(frames, breath_signal, peaks, valleys, n_bins=5):
    """Bin frames by breathing DISPLACEMENT (signal amplitude) within each
    valley<->peak half-cycle, instead of by time position (assign_breathing_bins).

    Frames are ranked by breathing signal value *within their half-cycle* and
    split into n_bins equal-sized quantile groups (bin 0 = lowest values, i.e.
    nearest rest/valley; bin n_bins-1 = nearest peak) -- not by slicing the
    amplitude range into equal-width thresholds. A half-cycle isn't traversed
    at constant speed (it can linger, overshoot or plateau near one end), so
    equal-width amplitude bins end up wildly unbalanced in frame count;
    equal-population (quantile) bins guarantee ~even counts regardless of the
    local curve shape.
    """
    frames = np.asarray(frames)
    anchors = sorted([(int(v), 'v') for v in valleys] + [(int(p), 'p') for p in peaks])
    bins = np.full(len(frames), -1, dtype=int)
    if len(anchors) < 2:
        return bins

    for (start_frame, _), (end_frame, _) in zip(anchors, anchors[1:]):
        if end_frame <= start_frame:
            continue
        in_segment = np.flatnonzero((frames >= start_frame) & (frames <= end_frame))
        if len(in_segment) == 0:
            continue
        values = np.array([breath_signal[int(frames[i])] for i in in_segment])
        # rank ascending by value: lowest value (nearest valley/rest) -> bin 0,
        # highest value (nearest peak) -> bin n_bins-1, regardless of whether
        # this half-cycle is ascending (v->p) or descending (p->v).
        ranks = np.argsort(np.argsort(values))
        quantile = ranks / max(len(in_segment) - 1, 1)
        bin_idx = np.clip(np.floor(quantile * n_bins).astype(int), 0, n_bins - 1)
        bins[in_segment] = bin_idx
    return bins


def plot_breathing_bins(ax, bins, title):
    ax.plot(breath, label='breathing signal')
    ax.plot(frame_indices, areas, color='gray', alpha=0.5, label='lumen area')
    ax.scatter(dia_idx, dia_areas, color='blue', label='diastole', zorder=3)
    ax.scatter(sys_idx, sys_areas, color='red', label='systole', zorder=3)
    ax.scatter(
        breathing_peaks,
        [breath[i] for i in breathing_peaks],
        color='blue',
        marker='x',
        s=100,
        label='breathing peak',
        zorder=4,
    )
    ax.scatter(
        breathing_valley,
        [breath[i] for i in breathing_valley],
        color='red',
        marker='x',
        s=100,
        label='breathing valley',
        zorder=4,
    )
    for frame, bin_idx in zip(frame_indices, bins):
        if bin_idx < 0:
            continue
        ax.axvline(frame, color=bin_cmap(bin_idx / (n_bins - 1)), linestyle='--', alpha=0.4, linewidth=1)

    for bin_idx in range(n_bins):
        in_bin = bins == bin_idx
        if in_bin.sum() < 2:
            continue
        bin_frames = frame_indices_arr[in_bin]
        bin_areas = areas_arr[in_bin]
        coeffs = np.polyfit(bin_frames, bin_areas, 1)
        x_fit = np.array([bin_frames.min(), bin_frames.max()])
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color=bin_cmap(bin_idx / (n_bins - 1)), linewidth=2, label=f'bin {bin_idx} fit')

    ax.set_title(title)
    ax.legend(fontsize='small', loc='upper right')


def plot_register_phase_sort(ax, frames_arr, areas_arr_phase, title):
    """Reproduce the actual Filtered-viewer sort (register_phase, per phase):
    bin -> fit a ground-truth area(frame) curve on bin-0 (rest) frames -> slide
    every other bin to the shift that best matches that curve -> sort by the
    resulting corrected position. This is the real algorithm from
    breathing_sort_viewer.py's _recompute, not a plain group-by-bin.
    """
    result = register_phase(
        frames_arr, areas_arr_phase, breathing_peaks, breathing_valley, n_bins=n_bins, n_total=len(breath)
    )
    corrected, order, bins_r, shifts = result['corrected'], result['order'], result['bins'], result['shifts']
    print(f'{title}: per-bin shifts (frames) = {[round(s, 1) for s in shifts]}')

    assigned = bins_r >= 0
    ax.plot(corrected[order], areas_arr_phase[order], color='gray', alpha=0.5, label='corrected order')
    sc = ax.scatter(
        corrected[assigned],
        areas_arr_phase[assigned],
        c=bins_r[assigned],
        cmap='viridis',
        vmin=0,
        vmax=n_bins - 1,
        s=25,
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, label='bin')
    ax.set_title(title)
    ax.set_xlabel('corrected position (breathing-registered)')
    ax.set_ylabel('lumen area')
    ax.legend(fontsize='small', loc='upper right')


bins_time = assign_breathing_bins(frame_indices_arr, breathing_peaks, breathing_valley, n_bins=n_bins)
bins_amplitude = assign_breathing_bins_by_amplitude(
    frame_indices_arr, breath, breathing_peaks, breathing_valley, n_bins=n_bins
)

dia_frames_arr = np.array(dia_idx, dtype=float)
dia_areas_arr = np.array(dia_areas, dtype=float)
sys_frames_arr = np.array(sys_idx, dtype=float)
sys_areas_arr = np.array(sys_areas, dtype=float)

fig, (ax_top, ax_bottom, ax_dia, ax_sys) = plt.subplots(4, 1, figsize=(14, 18))
plot_breathing_bins(ax_top, bins_time, 'assign_breathing_bins (time-based progress)')
plot_breathing_bins(ax_bottom, bins_amplitude, 'assign_breathing_bins_by_amplitude (signal-value-based progress)')
plot_register_phase_sort(ax_dia, dia_frames_arr, dia_areas_arr, 'register_phase sort - diastole (actual app algorithm)')
plot_register_phase_sort(ax_sys, sys_frames_arr, sys_areas_arr, 'register_phase sort - systole (actual app algorithm)')
plt.tight_layout()
plt.show()

full_distance = 1604 - 279
