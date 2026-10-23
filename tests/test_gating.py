from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from signal_processing.automatic_gating import (
    AutomaticGating,
    _enforce_spacing,
    _fill_gaps,
    find_diastole,
    find_systole,
    is_auto_gated,
    refine_to_raw,
    walk_extrema,
)
from signal_processing.gating_pipeline import (
    bandpass_filter,
    compute_correlation_signal,
    compute_frequency_sweep,
)
from signal_processing.gating_pipeline import compute_lumen_signal as compute_lumen_area_signal
from signal_processing.gating_pipeline import (
    detect_heart_rate,
    lowpass_filter,
    normalize_data,
)


def _ag(lower_limit=0):
    return AutomaticGating(Mock(), lower_limit=lower_limit)


def _sine(f_hz, fs=100.0, n=1000):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f_hz * t)


# ─────────────────────────────── normalize_data ────────────────────────────


class TestNormalizeData:
    def test_global_zscore_zero_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_data(data, step=0)
        assert result.mean() == pytest.approx(0.0, abs=1e-10)

    def test_global_zscore_unit_std(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_data(data, step=0)
        assert result.std() == pytest.approx(1.0, abs=1e-6)

    def test_windowed_each_window_zero_mean(self):
        data = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        result = normalize_data(data, step=3)
        assert result[:3].mean() == pytest.approx(0.0, abs=1e-10)
        assert result[3:].mean() == pytest.approx(0.0, abs=1e-10)

    def test_constant_signal_returns_zeros(self):
        data = np.array([7.0, 7.0, 7.0, 7.0])
        result = normalize_data(data, step=0)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_output_length_preserved(self):
        data = np.arange(17, dtype=float)
        assert normalize_data(data, step=0).shape == (17,)
        assert normalize_data(data, step=5).shape == (17,)

    def test_nan_values_do_not_crash(self):
        data = np.array([1.0, np.nan, 3.0, 4.0])
        result = normalize_data(data, step=0)
        assert result.shape == (4,)
        assert np.isfinite(result[0])


# ──────────────────────────── compute_correlation_signal ────────────────────


class TestComputeCorrelationSignal:
    def test_identical_consecutive_frames_give_zero_signal(self):
        frame = np.arange(400, dtype=np.float32).reshape(20, 20)
        frames = np.stack([frame, frame, frame])
        result = compute_correlation_signal(frames)
        # NCC of identical non-constant frames = 1 → signal = 0
        np.testing.assert_allclose(result[:2], 0.0, atol=1e-6)

    def test_output_length_equals_frame_count(self):
        frames = np.arange(10 * 20 * 20, dtype=np.float32).reshape(10, 20, 20)
        assert compute_correlation_signal(frames).shape == (10,)

    def test_last_element_equals_second_to_last(self):
        frames = (np.arange(5 * 8 * 8, dtype=np.float32) + 1).reshape(5, 8, 8)
        result = compute_correlation_signal(frames)
        assert result[-1] == result[-2]

    def test_single_frame_returns_zeros(self):
        frame = np.ones((5, 5), dtype=np.float32)
        result = compute_correlation_signal(np.stack([frame]))
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_constant_frame_no_crash(self):
        # std = 0 branch: skips correlation computation, leaves 0
        frames = np.ones((4, 10, 10), dtype=np.float32)
        result = compute_correlation_signal(frames)
        np.testing.assert_array_equal(result, np.zeros(4))


# ─────────────────────────────── detect_heart_rate ──────────────────────────


class TestDetectHeartRate:
    def test_detects_known_frequency(self):
        # FFT resolution = 100/1000 = 0.1 Hz; 1.5 is an exact bin
        sig = _sine(1.5, fs=100.0, n=1000)
        result = detect_heart_rate(sig, fs=100.0, f_min=1.0, f_max=3.5)
        assert result == pytest.approx(1.5, abs=0.15)

    def test_ignores_frequency_below_f_min(self):
        # Strong signal at 0.5 Hz (outside range), weak at 2.0 Hz (inside)
        t = np.arange(1000) / 100.0
        sig = 10.0 * np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 2.0 * t)
        result = detect_heart_rate(sig, fs=100.0, f_min=1.0, f_max=3.5)
        assert result == pytest.approx(2.0, abs=0.15)

    def test_prefers_fundamental_over_stronger_second_harmonic(self):
        # Like the 1 - NCC signal: two motion events per beat put most power at 2 x HR
        t = np.arange(1000) / 100.0
        sig = 0.5 * np.sin(2 * np.pi * 1.2 * t) + np.sin(2 * np.pi * 2.4 * t) + 0.3 * np.sin(2 * np.pi * 3.6 * t)
        result = detect_heart_rate(sig, fs=100.0, f_min=1.0, f_max=3.5)
        assert result == pytest.approx(1.2, abs=0.15)

    def test_returns_midpoint_when_no_peak_in_range(self):
        # N=4, fs=1.0 → FFT bins at 0, 0.25, 0.5 Hz — none in [1.0, 3.5]
        sig = np.array([0.0, 1.0, 0.0, -1.0])
        result = detect_heart_rate(sig, fs=1.0, f_min=1.0, f_max=3.5)
        assert result == pytest.approx(2.25, abs=1e-6)


# ──────────────────────────────── bandpass_filter ───────────────────────────


class TestBandpassFilter:
    def test_output_same_length(self):
        sig = _sine(1.5)
        out = bandpass_filter(sig, f_heart=1.5, fs=100.0)
        assert out.shape == sig.shape

    def test_returns_raw_when_bounds_degenerate(self):
        # lo_frac > hi_frac → lo > hi after normalisation
        sig = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        out = bandpass_filter(sig, f_heart=1.5, fs=100.0, lo_frac=2.0, hi_frac=0.5)
        np.testing.assert_array_equal(out, sig.astype(float))

    def test_no_nan_in_output(self):
        # Signal must exceed filtfilt padlen; NaN positions filled via nan_to_num
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(200)
        sig[50] = np.nan
        sig[120] = np.nan
        out = bandpass_filter(sig, f_heart=1.5, fs=100.0)
        assert not np.any(np.isnan(out))

    def test_attenuates_out_of_band_signal(self):
        fs = 100.0
        f_heart = 1.5
        N = 500
        t = np.arange(N) / fs
        # In-band: 1.5 Hz ∈ [0.7×1.5, 2.2×1.5] = [1.05, 3.3]
        sig_in = np.sin(2 * np.pi * 1.5 * t)
        # Out-of-band: 0.2 Hz, well below lower cutoff
        sig_out = np.sin(2 * np.pi * 0.2 * t)
        assert np.std(bandpass_filter(sig_in, f_heart, fs)) > 3 * np.std(bandpass_filter(sig_out, f_heart, fs))


# ──────────────────────────────── lowpass_filter ────────────────────────────


class TestLowpassFilter:
    def test_output_same_length(self):
        sig = _sine(1.0)
        assert lowpass_filter(sig, f_cutoff=5.0, fs=100.0).shape == sig.shape

    def test_low_frequency_passes(self):
        sig = _sine(0.5, fs=100.0, n=500)
        out = lowpass_filter(sig, f_cutoff=5.0, fs=100.0)
        assert np.std(out) > 0.4

    def test_high_frequency_attenuated(self):
        sig = _sine(30.0, fs=100.0, n=1000)
        out = lowpass_filter(sig, f_cutoff=5.0, fs=100.0)
        assert np.std(out) < 0.1


# ─────────────────────────── compute_lumen_area_signal ──────────────────────


class TestComputeLumenAreaSignal:
    def test_full_coverage_exact_values(self):
        df = pd.DataFrame({'frame': [1, 2, 3, 4, 5], 'lumen_area': [10.0, 12.0, 11.0, 13.0, 10.5]})
        result = compute_lumen_area_signal(df, N=5, lower_limit=0)
        np.testing.assert_allclose(result, [10.0, 12.0, 11.0, 13.0, 10.5])

    def test_low_coverage_returns_all_nan(self):
        # 3 contours in 10 frames → 30% < 50%
        df = pd.DataFrame({'frame': [1, 2, 3], 'lumen_area': [10.0, 12.0, 11.0]})
        result = compute_lumen_area_signal(df, N=10, lower_limit=0)
        assert np.all(np.isnan(result))

    def test_gaps_linearly_interpolated(self):
        # Frames 1, 3, 5 have data → indices 0, 2, 4
        df = pd.DataFrame({'frame': [1, 3, 5], 'lumen_area': [10.0, 12.0, 14.0]})
        result = compute_lumen_area_signal(df, N=5, lower_limit=0)
        np.testing.assert_allclose(result, [10.0, 11.0, 12.0, 13.0, 14.0])

    def test_lower_limit_offset_applied(self):
        # lower_limit=5 → frame 6 maps to index 0
        df = pd.DataFrame({'frame': list(range(6, 12)), 'lumen_area': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        result = compute_lumen_area_signal(df, N=6, lower_limit=5)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_nan_lumen_area_rows_skipped(self):
        # Frame 2 has NaN area; remaining 5/6 frames have data → coverage OK
        df = pd.DataFrame(
            {
                'frame': [1, 2, 3, 4, 5, 6],
                'lumen_area': [10.0, np.nan, 12.0, 11.0, 13.0, 10.5],
            }
        )
        result = compute_lumen_area_signal(df, N=6, lower_limit=0)
        assert not np.any(np.isnan(result))
        assert result[0] == pytest.approx(10.0)
        assert result[2] == pytest.approx(12.0)

    def test_out_of_range_frames_ignored(self):
        # Frame 0 and frame 10 fall outside [lower_limit, lower_limit+N)
        df = pd.DataFrame({'frame': [0, 1, 2, 3, 4, 5, 10], 'lumen_area': [99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 99.0]})
        result = compute_lumen_area_signal(df, N=5, lower_limit=0)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])


# ──────────────────────────── compute_frequency_sweep ───────────────────────


class TestComputeFrequencySweep:
    def test_output_shapes(self):
        sig = _sine(1.5, fs=30.0, n=100)
        bpm_cuts, sweep = compute_frequency_sweep(sig, fs=30.0, n_steps=10)
        assert bpm_cuts.shape == (10,)
        assert sweep.shape == (10, 100)

    def test_bpm_range_starts_at_bpm_lo(self):
        sig = _sine(2.0, fs=50.0, n=200)
        bpm_cuts, _ = compute_frequency_sweep(sig, fs=50.0, n_steps=15, bpm_lo=40.0, bpm_hi=200.0)
        assert bpm_cuts[0] == pytest.approx(40.0, rel=0.01)

    def test_bpm_cuts_monotonically_increasing(self):
        sig = _sine(1.5, fs=30.0, n=100)
        bpm_cuts, _ = compute_frequency_sweep(sig, fs=30.0, n_steps=10)
        assert np.all(np.diff(bpm_cuts) > 0)


# ────────────────────────────── walk_extrema ────────────────────────────────


class TestWalkExtrema:
    def test_flat_signal_returns_empty(self):
        sig = np.ones(50)
        all_e, maxima, minima = walk_extrema(sig)
        assert len(all_e) == 0
        assert len(maxima) == 0
        assert len(minima) == 0

    def test_sawtooth_alternating_sequence(self):
        # Deterministic: up to 1 → down to -1, repeated
        sig = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0], dtype=float)
        # ptp=2, threshold=0.30 → each unit step triggers direction change
        all_e, maxima, minima = walk_extrema(sig, swing_fraction=0.15)
        np.testing.assert_array_equal(maxima, [1, 5, 9])
        np.testing.assert_array_equal(minima, [3, 7])

    def test_all_extrema_is_sorted_union_of_maxima_and_minima(self):
        sig = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0], dtype=float)
        all_e, maxima, minima = walk_extrema(sig)
        expected = np.sort(np.concatenate([maxima, minima]))
        np.testing.assert_array_equal(all_e, expected)

    def test_initial_downswing_finds_minimum_first(self):
        # Signal starts going down: first extremum should be a minimum
        sig = np.array([0, -1, 0, 1, 0], dtype=float)
        _, maxima, minima = walk_extrema(sig, swing_fraction=0.15)
        assert len(minima) > 0
        assert minima[0] < maxima[0]

    def test_nan_and_inf_handled(self):
        sig = np.array([np.nan, 1.0, np.nan, -1.0, 0.0, 1.0, -1.0, np.inf])
        # Should not raise; result may be degenerate but must return 3 arrays
        result = walk_extrema(sig)
        assert len(result) == 3

    def test_single_element_returns_empty(self):
        all_e, maxima, minima = walk_extrema(np.array([5.0]))
        assert len(all_e) == 0

    def test_higher_swing_fraction_finds_fewer_extrema(self):
        # Noisy signal: large oscillation with micro-wiggles
        t = np.linspace(0, 4 * np.pi, 400)
        sig = np.sin(t) + 0.1 * np.sin(20 * t)
        _, maxima_loose, _ = walk_extrema(sig, swing_fraction=0.05)
        _, maxima_strict, _ = walk_extrema(sig, swing_fraction=0.40)
        assert len(maxima_loose) >= len(maxima_strict)

    def test_sine_wave_finds_expected_peak_count(self):
        # 4 full cycles: 4 maxima and 4 minima expected
        t = np.linspace(0, 4, 400, endpoint=False)
        sig = np.sin(2 * np.pi * t)
        _, maxima, minima = walk_extrema(sig, swing_fraction=0.15)
        assert len(maxima) == 4
        assert len(minima) == 4


# ──────────────────────────── two-step gating ───────────────────────────────

FS = 30.0
BEAT = 24  # frames per beat, 75 BPM at 30 fps
DIA = np.arange(12, 600, BEAT)
SYS = DIA[:-1] + 11


def _valley_signal():
    """Synthetic 1 - NCC signal: deep wide valley at diastole, shallow narrow one at systole."""
    x = np.arange(624, dtype=float)
    sig = np.ones_like(x)
    for d in DIA:
        sig -= 1.0 * np.exp(-0.5 * ((x - d) / 2.5) ** 2)
    for s_ in SYS:
        sig -= 0.5 * np.exp(-0.5 * ((x - s_) / 1.5) ** 2)
    return sig


def _gating_signal(raw, f_heart=FS / BEAT):
    return normalize_data(bandpass_filter(raw, f_heart, FS, 0.7, 2.2), step=100)


def _max_offset(found, expected):
    return max(int(np.min(np.abs(np.asarray(found) - e))) for e in expected)


class TestFindDiastole:
    def test_one_anchor_per_beat_on_the_deep_valley(self):
        raw = _valley_signal()
        dia = find_diastole(raw, _gating_signal(raw), FS / BEAT, FS, step=100)
        inner = DIA[2:-2]  # filter edge effects
        assert _max_offset(dia, inner) <= 1
        assert all(np.diff(dia) > 0.7 * BEAT)


class TestDiastoleRepair:
    def test_close_pair_keeps_frame_on_beat_grid(self):
        # 30 sits half a beat off, e.g. an anchor snapped onto the systolic valley
        assert _enforce_spacing([0, 24, 30, 48, 72], beat=24) == [0, 24, 48, 72]

    def test_long_gap_split_into_equal_beats(self):
        raw = _valley_signal()
        filled = _fill_gaps([int(DIA[1]), int(DIA[4])], _gating_signal(raw), beat=BEAT)
        assert len(filled) == 4
        assert _max_offset(filled[1:3], DIA[2:4]) <= 1


class TestFindSystole:
    def test_shallow_valley_between_diastoles(self):
        raw = _valley_signal()
        sys_ = find_systole(list(DIA), _gating_signal(raw))
        assert len(sys_) == len(DIA) - 1
        assert _max_offset(sys_, SYS[2:-2]) <= 1

    def test_falls_back_to_lowest_point_without_valley(self):
        sys_ = find_systole([0, 20], np.linspace(1.0, 0.0, 21))
        assert sys_ == [15]  # end of the 25-75 % window on a falling ramp


class TestRefineToRaw:
    def test_moves_to_lowest_neighbour(self):
        raw = np.array([3.0, 2.0, 1.0, 2.0, 3.0, 0.5, 3.0])
        assert refine_to_raw([1, 4], raw) == [2, 5]

    def test_merges_frames_landing_on_same_minimum(self):
        raw = np.array([3.0, 1.0, 3.0])
        assert refine_to_raw([0, 2], raw) == [1]


def _main_window(n_frames, lower_limit=0):
    rt = SimpleNamespace(
        frame_data_dct={i: SimpleNamespace(phase='-') for i in range(lower_limit + n_frames)},
        metadata={'frame_rate': FS},
        gating_signal={},
        gated_frames_dia=[],
        gated_frames_sys=[],
    )
    return SimpleNamespace(
        runtime_data=rt,
        config=SimpleNamespace(gating=SimpleNamespace(normalize_step=100)),
        diastolic_frame_box=Mock(),
        systolic_frame_box=Mock(),
        save_contours_soon=Mock(),
    )


class TestAutomaticGating:
    def test_gates_alternating_phases_with_offset(self):
        raw = _valley_signal()
        mw = _main_window(len(raw), lower_limit=50)
        assert AutomaticGating(mw, lower_limit=50).automatic_gating(raw, _gating_signal(raw), FS / BEAT)
        rt = mw.runtime_data
        assert _max_offset(rt.gated_frames_dia, DIA[2:-2] + 50) <= 1
        assert _max_offset(rt.gated_frames_sys, SYS[2:-2] + 50) <= 1
        assert rt.frame_data_dct[rt.gated_frames_dia[3]].phase == 'D'
        assert rt.frame_data_dct[rt.gated_frames_sys[3]].phase == 'S'

    def test_returns_false_without_heart_rate(self):
        raw = _valley_signal()
        mw = _main_window(len(raw))
        assert not AutomaticGating(mw).automatic_gating(raw, _gating_signal(raw), 0.0)
        assert mw.runtime_data.gated_frames_dia == []


class TestSystoleInEveryInterval:
    @pytest.mark.parametrize('seed', range(5))
    def test_noisy_signal_never_leaves_diastoles_without_systole(self, seed):
        rng = np.random.default_rng(seed)
        raw = _valley_signal()
        raw = raw + rng.normal(0, 0.4, len(raw))
        mw = _main_window(len(raw))
        AutomaticGating(mw).automatic_gating(raw, _gating_signal(raw), FS / BEAT)
        dia, sys_ = mw.runtime_data.gated_frames_dia, mw.runtime_data.gated_frames_sys
        assert all(np.diff(dia) >= 0.7 * BEAT - 2)
        assert all(any(a < x < b for x in sys_) for a, b in zip(dia[:-1], dia[1:]))


class TestIsAutoGated:
    def test_true_until_manual_edit(self):
        raw = _valley_signal()
        mw = _main_window(len(raw))
        AutomaticGating(mw).automatic_gating(raw, _gating_signal(raw), FS / BEAT)
        rt = mw.runtime_data
        assert is_auto_gated(rt)
        rt.gated_frames_sys.remove(rt.gated_frames_sys[0])
        assert not is_auto_gated(rt)

    def test_false_without_automatic_gating(self):
        rt = SimpleNamespace(gating_signal={}, gated_frames_dia=[1], gated_frames_sys=[2])
        assert not is_auto_gated(rt)


# ──────────────────────────── AutomaticGating ───────────────────────────────


class TestAutomaticGatingSigToFrameKey:
    def test_with_zero_offset(self):
        ag = _ag(lower_limit=0)
        assert ag._signal_to_frame_key(7) == 7

    def test_with_nonzero_offset(self):
        ag = _ag(lower_limit=10)
        assert ag._signal_to_frame_key(5) == 15
        assert ag._signal_to_frame_key(0) == 10
