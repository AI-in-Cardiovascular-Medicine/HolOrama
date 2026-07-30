"""Tests for the retained respiratory-signal helpers in gating.gating_pipeline.

The breathing-sort/en-bloc reordering was removed; these cover what remains and
feeds the (future) sorting: the detrend + respiratory extraction and the
peak/valley phase computation with manual hard-anchor override.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from gating.breathing_pipeline import (
    assign_breathing_bins,
    compute_breathing_phases,
    compute_breathing_signal,
    register_phase,
)


def test_breathing_signal_shapes_and_trend():
    frames = np.arange(300, dtype=float)
    areas = 3.0 + 0.01 * frames + 0.5 * np.sin(2 * np.pi * frames / 120.0)
    out = compute_breathing_signal(frames, areas, gated_frames=set(range(0, 300, 30)), fs=30.0)
    for key in ("frames", "areas", "trend", "slope", "residual", "smoothed", "f_resp"):
        assert key in out
    assert out["trend"].shape == frames.shape
    assert out["slope"].shape == frames.shape
    assert abs(np.median(out["slope"]) - 0.01) < 5e-3


def test_breathing_signal_degenerate():
    out = compute_breathing_signal(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert out["f_resp"] == 0.0
    assert np.allclose(out["residual"], 0.0)


def test_manual_anchors_override_auto():
    t = np.arange(600)
    sig = np.sin(2 * np.pi * t / 150.0)
    phase, peaks, valleys = compute_breathing_phases(
        sig, manual_peaks=[40], manual_valleys=[75], frames_arr=t, anchor_gap=20
    )
    assert 40 in peaks and 75 in valleys
    assert not [p for p in peaks if p != 40 and abs(p - 40) < 20]
    assert not [v for v in valleys if v != 75 and abs(v - 75) < 20]
    assert np.all((phase >= 0.0) & (phase < 1.0))


def test_manual_only_uses_labels_exclusively():
    t = np.arange(600)
    sig = np.sin(2 * np.pi * t / 150.0)  # would auto-detect several extrema
    peaks_frames = [50, 200, 350]
    valleys_frames = [125, 275, 425]
    phase, peaks, valleys = compute_breathing_phases(
        sig, manual_peaks=peaks_frames, manual_valleys=valleys_frames, frames_arr=t, manual_only=True
    )
    # only the labelled anchors survive
    assert sorted(peaks.tolist()) == peaks_frames
    assert sorted(valleys.tolist()) == valleys_frames
    assert np.all((phase >= 0.0) & (phase < 1.0))


def test_phases_without_anchors_are_valid():
    t = np.arange(600)
    sig = np.sin(2 * np.pi * t / 150.0)
    phase, peaks, valleys = compute_breathing_phases(sig, frames_arr=t)
    assert len(peaks) > 0 and len(valleys) > 0
    assert np.all((phase >= 0.0) & (phase < 1.0))


# ─────────────────────────── bin-registration sort ─────────────────────────


def _synthetic_gated_phase(seed=0):
    """Gated frames whose imaged area carries a one-directional breathing offset."""
    rng = np.random.default_rng(seed)
    N = 3000
    A_true = lambda x: 12.0 - 8.0 * (np.clip(x, 0, N) / N)  # noqa: E731  monotonic taper

    valleys = [0]
    while valleys[-1] < N:
        valleys.append(valleys[-1] + int(rng.integers(150, 230)))
    valleys = [v for v in valleys if v < N]
    peaks = [(valleys[i] + valleys[i + 1]) // 2 for i in range(len(valleys) - 1)]
    D = 200.0

    def disp(frames):
        frames = np.asarray(frames, float)
        out = np.zeros_like(frames)
        for i in range(len(valleys) - 1):
            v0, v1, p = valleys[i], valleys[i + 1], peaks[i]
            asc = (frames >= v0) & (frames < p)
            des = (frames >= p) & (frames < v1)
            out[asc] = D * (frames[asc] - v0) / max(1, (p - v0))
            out[des] = D * (v1 - frames[des]) / max(1, (v1 - p))
        return out

    dia = np.array([f for f in range(20, N - 20, 26)])
    area = A_true(dia + disp(dia)) + rng.normal(0, 0.05, len(dia))
    true_pos = dia + disp(dia)
    return dia, area, peaks, valleys, true_pos, N


def _inv(seq):
    seq = np.asarray(seq, float)
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def test_assign_breathing_bins_ranges():
    valleys = [0, 200]
    peaks = [100]
    frames = np.array([0, 25, 50, 75, 100, 125, 150, 175])
    bins = assign_breathing_bins(frames, peaks, valleys, n_bins=4)
    # valley (0) and just-before-next-valley (175) are bin 0; peak (100) is bin 3
    assert bins[0] == 0
    assert bins[frames.tolist().index(100)] == 3
    assert set(bins.tolist()) <= {0, 1, 2, 3}


def test_register_phase_recovers_order():
    dia, area, peaks, valleys, true_pos, _ = _synthetic_gated_phase()
    R = register_phase(dia, area, peaks, valleys, n_bins=4)
    # per-bin shifts are monotonic and increase away from rest
    assert R['shifts'][0] == 0
    assert np.all(np.diff(np.abs(R['shifts'])) >= -1e-9)
    # sorting strongly reduces disorder of the true anatomical positions
    inv_raw = _inv(true_pos)
    inv_sorted = _inv(true_pos[R['order']])
    assert inv_sorted < 0.5 * inv_raw
    # resulting area sequence is near-monotonic
    mono = abs(np.corrcoef(area[R['order']], np.arange(len(area)))[0, 1])
    assert mono > 0.98


def test_register_phase_degenerate():
    R = register_phase(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), [], [], n_bins=4)
    assert list(R['order']) == [0, 1, 2]


# ─────────────────────────── patient case ─────────────────────────
class TestBreathingPatient:
    """Manual playground for exploring one patient's pullback -- not an automated test.

    Instantiate directly (e.g. in a scratch script or notebook) to load and prepare
    the data, then poke at the attributes / call .plot() interactively.

    run by:
    ```bash
        $env:PYTHONPATH = "src;tests"
        python -i -c "from test_breathing_sort import TestBreathingPatient; p = TestBreathingPatient()"
    ```
    """

    __test__ = False  # pytest would otherwise collect this despite having no test_* methods

    json_path = r"E:\PostDoc_Anselm-Stark\06_ivus_data\NARCO_122\Run1\PDWN7A4I_contours_ho_0_4_0.json"

    def __init__(self, json_path: str | None = None):
        self.json_path = json_path or self.json_path
        self.load()
        self.prepare()

    def load(self):
        with open(self.json_path, "r") as file:
            self.raw = json.load(file)
        self.gating_signal = self.raw["gating_signal"]
        self.frame_keys = sorted((k for k in self.raw if k != "gating_signal"), key=int)
        self.frames = np.array([int(k) for k in self.frame_keys])
        self.areas = np.array([self.raw[k]["lumen"]["measurements"]["area"] for k in self.frame_keys])
        self.minor_axes = np.array([self.raw[k]["lumen"]["measurements"]["minor_axis"] for k in self.frame_keys])
        self.phases = [self.raw[k]["phase"] for k in self.frame_keys]
        self.manual_peaks = np.array(self.gating_signal["breathing_manual_peaks"])
        self.manual_valleys = np.array(self.gating_signal["breathing_manual_valleys"])

    def prepare(self):
        self.dia_mask = np.array([p == "D" for p in self.phases])
        self.sys_mask = np.array([p == "S" for p in self.phases])
        self.gated_frames = set(self.frames[self.dia_mask | self.sys_mask].tolist())

        self.breathing_display_signal = np.array(self.gating_signal["breathing_display_signal"])
        self.f_heart = self.gating_signal.get("f_heart")

        self.signal = compute_breathing_signal(
            self.frames, self.areas, gated_frames=self.gated_frames, f_heart=self.f_heart
        )
        self.phase, self.peaks, self.valleys = compute_breathing_phases(
            self.signal["smoothed"],
            manual_peaks=self.manual_peaks.tolist(),
            manual_valleys=self.manual_valleys.tolist(),
            frames_arr=self.frames,
            manual_only=True,
        )
        self.bins = assign_breathing_bins(self.frames, self.peaks.tolist(), self.valleys.tolist())
        self.registration = register_phase(self.frames, self.areas, self.peaks.tolist(), self.valleys.tolist())
        self.registration_dia = register_phase(
            self.dia_frames, self.dia_areas, self.peaks.tolist(), self.valleys.tolist()
        )
        self.registration_sys = register_phase(
            self.sys_frames, self.sys_areas, self.peaks.tolist(), self.valleys.tolist()
        )

    @property
    def dia_frames(self):
        return self.frames[self.dia_mask]

    @property
    def dia_areas(self):
        return self.areas[self.dia_mask]

    @property
    def sys_frames(self):
        return self.frames[self.sys_mask]

    @property
    def sys_areas(self):
        return self.areas[self.sys_mask]

    def plot(self):
        fig, (ax_area, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

        ax_area.plot(self.frames, self.areas, color="lightgray", lw=1, label="lumen area")
        ax_area.plot(self.frames, self.signal["trend"], color="black", lw=1, label="trend")
        ax_area.scatter(self.dia_frames, self.dia_areas, color="tab:blue", s=15, label="diastole")
        ax_area.scatter(self.sys_frames, self.sys_areas, color="tab:red", s=15, label="systole")
        ax_area.set_ylabel("area [mm²]")
        ax_area.legend(loc="upper right")

        ax_phase.plot(self.frames, self.breathing_display_signal, color="tab:green", lw=1, label="breathing signal")
        ax_phase.scatter(
            self.frames[self.peaks],
            self.breathing_display_signal[self.peaks],
            color="tab:orange",
            marker="^",
            label="peaks",
        )
        ax_phase.scatter(
            self.frames[self.valleys],
            self.breathing_display_signal[self.valleys],
            color="tab:purple",
            marker="v",
            label="valleys",
        )
        ax_phase.set_xlabel("frame")
        ax_phase.legend(loc="upper right")

        fig.tight_layout()
        data = {
            "frames": self.frames,
            "areas": self.areas,
            "trend": self.signal["trend"],
            "dia_frames": self.dia_frames,
            "dia_areas": self.dia_areas,
            "sys_frames": self.sys_frames,
            "sys_areas": self.sys_areas,
            "breathing_signal": self.breathing_display_signal,
            "peak_frames": self.frames[self.peaks],
            "valley_frames": self.frames[self.valleys],
        }
        return fig, data

    def _phase_frames_areas(self, phase: str):
        if phase == "dia":
            return self.dia_frames, self.dia_areas
        if phase == "sys":
            return self.sys_frames, self.sys_areas
        raise ValueError("phase must be 'dia' or 'sys'")

    def plot_binned_diastole(self, n_bins: int = 5, phase: str = "dia"):
        """Reproduce 'binned_diastole.png': the gated frames for one cardiac phase,
        binned by breathing displacement, each with its own per-bin linear fit --
        plotted against raw frame number, before any shift is applied.
        """
        frames, areas = self._phase_frames_areas(phase)
        bins = assign_breathing_bins(frames, self.peaks.tolist(), self.valleys.tolist(), n_bins=n_bins)
        cmap = plt.get_cmap("viridis")
        label = "Diastolic" if phase == "dia" else "Systolic"

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(self.frames, self.areas, color="lightgray", label="lumen area")
        ax.plot(
            self.frames, self.breathing_display_signal, color="tab:green", lw=1, alpha=0.8, label="breathing signal"
        )
        ax.scatter(
            self.frames[self.peaks],
            self.breathing_display_signal[self.peaks],
            color="tab:orange",
            marker="^",
            s=60,
            zorder=4,
            label="peak",
        )
        ax.scatter(
            self.frames[self.valleys],
            self.breathing_display_signal[self.valleys],
            color="tab:purple",
            marker="v",
            s=60,
            zorder=4,
            label="valley",
        )
        for bin_idx in range(n_bins):
            in_bin = bins == bin_idx
            if in_bin.sum() == 0:
                continue
            color = cmap(bin_idx / (n_bins - 1))
            bin_frames, bin_areas = frames[in_bin], areas[in_bin]
            ax.scatter(bin_frames, bin_areas, color=color, s=25, zorder=3, label=f"bin {bin_idx}")
            if in_bin.sum() >= 2:
                coeffs = np.polyfit(bin_frames, bin_areas, 1)
                x_fit = np.array([bin_frames.min(), bin_frames.max()])
                ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, lw=2)

        ax.set_title(f"{label} frames by breathing bin")
        ax.set_xlabel("frame / sample index")
        ax.set_ylabel("lumen area")
        ax.legend(fontsize="small", loc="upper right")
        fig.tight_layout()

        data = {
            bin_idx: {"frames": frames[bins == bin_idx], "areas": areas[bins == bin_idx]}
            for bin_idx in range(n_bins)
            if (bins == bin_idx).any()
        }
        data["peak_frames"] = self.frames[self.peaks]
        data["valley_frames"] = self.frames[self.valleys]
        return fig, data

    def plot_shifted_registration(self, n_bins: int = 5, phase: str = "dia"):
        """Reproduce 'shifted_breathing.png': the same per-bin fits as
        plot_binned_diastole, but at each frame's breathing-corrected position
        (register_phase) instead of its raw frame number -- re-based so 0 sits at
        the peak (largest-shift) end and position increases toward the valley
        (rest) end. A correctly registered phase collapses the per-bin fit lines
        onto one shared line.
        """
        frames, areas = self._phase_frames_areas(phase)
        result = self.registration_dia if phase == "dia" else self.registration_sys
        corrected, bins_r, shifts = result["corrected"], result["bins"], result["shifts"]
        corrected = corrected - corrected.min()
        cmap = plt.get_cmap("viridis")
        label = "Diastolic" if phase == "dia" else "Systolic"

        fig, ax = plt.subplots(figsize=(12, 3))
        for bin_idx in range(n_bins):
            in_bin = bins_r == bin_idx
            if in_bin.sum() == 0:
                continue
            color = cmap(bin_idx / (n_bins - 1))
            bin_pos, bin_areas = corrected[in_bin], areas[in_bin]
            ax.scatter(bin_pos, bin_areas, color=color, s=25, zorder=3, label=f"bin {bin_idx}")
            if in_bin.sum() >= 2:
                coeffs = np.polyfit(bin_pos, bin_areas, 1)
                x_fit = np.array([bin_pos.min(), bin_pos.max()])
                ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, lw=2)

        ax.set_title(f"{label} bins after shift (peak-anchored) -- shifts: {[round(s, 1) for s in shifts]}")
        ax.set_xlabel("corrected position (0 near peak bin, increasing toward valley)")
        ax.set_ylabel("lumen area")
        ax.legend(fontsize="small", loc="upper right")
        fig.tight_layout()

        data = {
            bin_idx: {
                "frames": frames[bins_r == bin_idx],
                "corrected_position": corrected[bins_r == bin_idx],
                "areas": areas[bins_r == bin_idx],
            }
            for bin_idx in range(n_bins)
            if (bins_r == bin_idx).any()
        }
        data["shifts"] = shifts
        return fig, data

    def test_correct_bin_assignment():
        # visually picked bin 0
        pass
