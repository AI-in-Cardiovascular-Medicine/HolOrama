"""
IVUS Gating Signal Processing
==============================
Two-signal hybrid algorithm:

Image signal  (always available)
    s[n] = 1 - NCC(frame_n, frame_{n+1})        (CCB s0, Maso Talou 2015)
    Bandpass at [0.7, 2.2] x f_heart removes both the slow pullback
    trend (DC) and high-frequency speckle noise.
    Peaks of filtered signal = maximum-motion frames (mid-systole and
    mid-diastole); used as timing landmarks, not stable-phase gating points.

Contour signal  (when ≥50% of frames have drawn contours)
    s[n] = lumen_area[n]  (mm², from report_data)
    Same bandpass filter.
    Peaks of filtered signal = diastole (large, relaxed lumen).
    Troughs = systole (small, compressed lumen).
    Classifier SNR vs labelled frames: 1.54 (vs 0.04 for centroid vector).

Heart rate detection
    FFT spectral peak of the correlation signal in the configured range.
    No grid-search optimisation - the FFT estimate is already correct for
    both rest (~60-100 BPM) and stress (~100-210 BPM) imaging.

Notes
-----
- DT-CWT, blur signal, centroid vector, and weight optimisation removed:
  they add noise for real IVUS data and degrade the clean correlation signal.
- The 2 x f_heart trick is unnecessary when the lumen area is available:
  it already separates sys and dia at a single f_heart.
- Centroid vector analysis showed SNR = 0.04 (noise-dominated by catheter
  rotation) versus lumen area SNR = 1.54; vector signals not used.
"""

import time
import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import butter, filtfilt


# ──────────────────────────────────────────────────────────── utilities ────


def _timing(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} took {time.time()-t0:.3f}s")
        return result

    return wrapper


# ──────────────────────────────── normalisation ────


def normalize_data(data, step: int) -> np.ndarray:
    """Z-score normalisation in non-overlapping windows of *step* frames."""
    data = np.asarray(data, dtype=float)
    if step == 0:
        std = np.nanstd(data)
        return (data - np.nanmean(data)) / std if std != 0 else np.zeros_like(data)
    out = np.zeros(len(data), dtype=float)
    for i in range(0, len(data), step):
        seg = data[i : i + step]
        std = np.nanstd(seg)
        if std != 0:
            out[i : i + step] = (seg - np.nanmean(seg)) / std
    return out


# ─────────────────────────────── image signal ────


def compute_correlation_signal(frames: np.ndarray) -> np.ndarray:
    """1 - normalised cross-correlation between consecutive frames (CCB s0).

    High when frames differ (motion); low at stable end-systole/end-diastole.
    Frames should be [0,1]-normalised beforehand.
    """
    N = len(frames)
    s = np.zeros(N)
    for n in range(N - 1):
        u = frames[n].ravel().astype(float)
        v = frames[n + 1].ravel().astype(float)
        sig_u, sig_v = u.std(), v.std()
        if sig_u > 0 and sig_v > 0:
            corr = np.dot(u - u.mean(), v - v.mean()) / (sig_u * sig_v * len(u))
            s[n] = 1.0 - float(np.clip(corr, -1.0, 1.0))
    s[-1] = s[-2] if N > 1 else 0.0
    return s


# ─────────────────────────────── heart rate detection ────


def detect_heart_rate(
    signal: np.ndarray,
    fs: float,
    f_min: float = 1.0,
    f_max: float = 3.5,
) -> float:
    """Dominant heart rate via FFT spectral peak in physiological range [Hz].

    Works reliably because the correlation signal has strong spectral power
    at f_heart (validated: correctly finds 1.55 Hz = 93 BPM on test data).
    No grid-search is applied - the FFT estimate is sufficiently accurate.
    """
    sig = np.nan_to_num(signal - np.nanmean(signal))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(sig))
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        logger.warning(f"No spectral peak in [{f_min}, {f_max}] Hz - using midpoint")
        return (f_min + f_max) / 2
    f_heart = float(freqs[mask][np.argmax(spectrum[mask])])
    logger.info(f"Heart rate: {f_heart:.3f} Hz  ({f_heart * 60:.0f} BPM)")

    # The 1-NCC signal has strong power at 2xf_heart (two motion events per cycle).
    # If the actual heart rate is below f_min/2, the harmonic rather than the
    # fundamental may be the dominant peak in [f_min, f_max].
    if f_heart > f_max / 2:
        logger.warning(
            f"Detected f_heart={f_heart:.2f} Hz is above half of f_max={f_max} Hz. "
            "The 2xf_heart harmonic of a slower heart rate may have been picked up. "
            "If gating looks wrong, try halving f_cardiac_max in config."
        )

    return f_heart


# ─────────────────────────────── bandpass filter ────


def bandpass_filter(
    signal: np.ndarray,
    f_heart: float,
    fs: float,
    lo_frac: float = 0.7,
    hi_frac: float = 2.2,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass at [lo_frac, hi_frac] x f_heart.

    Default passband [0.7, 2.2] x f_heart:
    - Lower cut removes slow pullback trend (DC and sub-cardiac drift).
    - Upper cut removes high-frequency speckle noise.
    - Passes up to 2nd harmonic so both sys and dia phases are visible
      as two distinct features per cardiac cycle in the image signal.
    """
    nyq = fs / 2.0
    lo = max(0.01, lo_frac * f_heart / nyq)
    hi = min(0.99, hi_frac * f_heart / nyq)
    if lo >= hi:
        logger.warning(f"Bandpass bounds invalid ({lo:.3f}, {hi:.3f}) - returning raw signal")
        return np.array(signal, dtype=float)
    b, a = butter(order, [lo, hi], btype='band')
    try:
        return filtfilt(b, a, np.nan_to_num(signal))
    except Exception as exc:
        logger.warning(f"Bandpass filter failed: {exc} - returning raw signal")
        return np.array(signal, dtype=float)


def lowpass_filter(signal: np.ndarray, f_cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (used only for frequency sweep display)."""
    nyq = fs / 2.0
    Wn = min(f_cutoff / nyq, 0.99)
    b, a = butter(order, Wn, btype='low')
    try:
        return filtfilt(b, a, np.nan_to_num(signal))
    except Exception:
        return np.array(signal, dtype=float)


# ────────────────────────── contour: lumen area signal ────


def compute_lumen_area_signal(
    report_data,
    N: int,
    lower_limit: int,
) -> np.ndarray:
    """Extract lumen_area [mm²] from report_data into a length-N array.

    Frames without contours → NaN; gaps are linearly interpolated so the
    bandpass filter operates on a complete signal.

    Returns NaN-filled array (all NaN) when coverage < 50%.
    """
    area = np.full(N, np.nan)

    for _, row in report_data.iterrows():
        idx = int(row['frame']) - 1 - lower_limit
        if 0 <= idx < N:
            v = row.get('lumen_area', np.nan)
            if not pd.isna(v):
                area[idx] = float(v)

    coverage = float(np.sum(~np.isnan(area))) / N
    if coverage < 0.5:
        logger.info(f"Lumen area coverage {coverage:.0%} < 50% - contour signal omitted")
        return np.full(N, np.nan)

    logger.info(f"Lumen area coverage: {coverage:.0%}")

    # Linear interpolation over NaN gaps
    valid = np.where(~np.isnan(area))[0]
    if len(valid) < 2:
        return np.full(N, np.nan)
    area = np.interp(np.arange(N), valid, area[valid])
    return area


# ─────────────────────── frequency sweep for visualisation ────


def compute_frequency_sweep(
    signal: np.ndarray,
    fs: float,
    f_heart: float,
    n_steps: int = 30,
) -> tuple:
    """Low-pass filtered signal at n_steps cutoff frequencies (BPM labelled).

    Sweeps from 0.5 x f_heart to 4 x f_heart so the user can see the
    transition from over-smoothed (single hump/cycle) to noisy signal.
    Returns (bpm_cuts, sweep) where sweep[i] is z-scored signal at bpm_cuts[i].
    """
    f_lo = 0.5 * f_heart
    f_hi = min(fs / 2 * 0.9, 4.0 * f_heart)
    f_cuts = np.linspace(f_lo, f_hi, n_steps)  # Hz
    bpm_cuts = f_cuts * 60  # BPM for display

    N = len(signal)
    sweep = np.zeros((n_steps, N))
    for i, f_c in enumerate(f_cuts):
        try:
            filt = lowpass_filter(signal, f_c, fs)
            std = filt.std()
            sweep[i] = (filt - filt.mean()) / std if std > 0 else filt
        except Exception:
            sweep[i] = 0.0
    return bpm_cuts, sweep


# ─────────────────────────── extrema / turning-point detection ────


def walk_extrema(
    signal: np.ndarray,
    swing_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hysteresis-gated turning-point detector.

    Walks the signal and registers a local maximum only after the value has
    dropped more than *swing_fraction* x peak-to-peak below the running high
    since the last confirmed turning point; vice versa for minima.

    Advantages over scipy find_peaks:
    - No global height threshold: amplitude-agnostic, adapts to signal level.
    - No minimum-distance parameter: the hysteresis swing alone suppresses
      micro-wiggles caused by residual noise on the bandpass-filtered signal.
    - Produces a naturally alternating max / min / max / min sequence that
      maps directly to cardiac phases without post-hoc alternation heuristics.

    Parameters
    ----------
    signal         : 1-D array (bandpass-filtered, normalised).
    swing_fraction : fraction of peak-to-peak range a reversal must exceed
                     before a turning point is registered.  0.15 (15%) works
                     well for IVUS bandpass-filtered signals.

    Returns
    -------
    all_extrema_idx : sorted union of maxima and minima indices
    maxima_idx      : indices of local maxima (high motion / end-phase peaks)
    minima_idx      : indices of local minima
    """
    sig = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    ptp = float(sig.max() - sig.min())
    empty: np.ndarray = np.array([], dtype=int)
    if ptp == 0.0:
        return empty, empty, empty

    threshold = swing_fraction * ptp
    maxima: list[int] = []
    minima: list[int] = []

    # direction: +1 = currently tracking a rising run (looking for maximum)
    #            -1 = currently tracking a falling run (looking for minimum)
    #            None = not yet determined (waiting for first significant swing)
    direction: int | None = None
    extreme_val = sig[0]
    extreme_idx = 0

    for i in range(1, len(sig)):
        v = sig[i]
        if direction is None:
            if v - sig[0] >= threshold:
                direction = 1
                extreme_val, extreme_idx = v, i
            elif sig[0] - v >= threshold:
                direction = -1
                extreme_val, extreme_idx = v, i
        elif direction == 1:
            if v > extreme_val:
                extreme_val, extreme_idx = v, i
            elif extreme_val - v >= threshold:
                maxima.append(extreme_idx)
                direction = -1
                extreme_val, extreme_idx = v, i
        else:  # direction == -1
            if v < extreme_val:
                extreme_val, extreme_idx = v, i
            elif v - extreme_val >= threshold:
                minima.append(extreme_idx)
                direction = 1
                extreme_val, extreme_idx = v, i

    maxima_idx = np.array(maxima, dtype=int)
    minima_idx = np.array(minima, dtype=int)
    all_extrema = np.sort(np.concatenate([maxima_idx, minima_idx]))
    return all_extrema, maxima_idx, minima_idx


def filter_by_period(
    indices: np.ndarray,
    expected_interval: float,
    tolerance: float = 0.4,
) -> np.ndarray:
    """Drop peaks whose inter-peak gap violates the expected cardiac interval.

    Peaks closer than (1 - tolerance) x expected_interval to the previous
    kept peak are discarded as noise ripple or walk-algorithm duplicates.
    Peaks further than (1 + tolerance) x expected_interval are retained but
    logged as potential missed beats (no automatic gap-filling).

    Parameters
    ----------
    expected_interval : expected frames between consecutive peaks of this type
                        (e.g. fs / (2 x f_heart) for image-signal maxima).
    tolerance         : fractional tolerance; 0.4 → ±40 %.
    """
    if len(indices) < 2:
        return indices

    t_min = (1.0 - tolerance) * expected_interval
    t_max = (1.0 + tolerance) * expected_interval
    kept = [int(indices[0])]
    for idx in indices[1:]:
        gap = int(idx) - kept[-1]
        if gap < t_min:
            logger.debug(f"Period filter: dropped peak at {idx} (gap {gap:.1f} < min {t_min:.1f})")
        else:
            if gap > t_max:
                logger.debug(
                    f"Period filter: large gap {gap:.1f} > {t_max:.1f} before peak at {idx} - possible missed beat"
                )
            kept.append(int(idx))
    return np.array(kept, dtype=int)


# ──────────────────────────────────────────────── main entry point ────


@_timing
def prepare_data(
    main_window,
    frames: np.ndarray,
    report_data,
    lower_limit: int = 0,
    x1: int = 50,
    x2: int = 450,
    y1: int = 50,
    y2: int = 450,
):
    """Full gating pipeline.

    Image signal  : 1 - NCC(frame_n, frame_{n+1}), bandpass filtered.
    Contour signal: lumen_area from report_data, bandpass filtered.
                    NaN-filled zeros when coverage < 50%.

    Returns (image_raw, contour_raw, image_filtered, contour_filtered).
    """
    cfg = main_window.config.gating
    fs = main_window.runtime_data.metadata['frame_rate']
    N = len(frames)

    # ── Cache ──────────────────────────────────────────────────────────────
    try:
        gs = main_window.runtime_data.gating_signal
        if gs and gs.get('gating_config') == dict(vars(cfg)) and len(gs.get('image_based_gating', [])) == N:
            return (
                np.array(gs['image_based_gating']),
                np.array(gs['contour_based_gating']),
                np.array(gs['image_based_gating_filtered']),
                np.array(gs['contour_based_gating_filtered']),
            )
    except Exception:
        pass

    # ── Pre-process frames: crop → [0,1] ──────────────────────────────────
    frames_crop = frames[:, x1:x2, y1:y2].astype(np.float32)
    f_lo = frames_crop.min(axis=(1, 2), keepdims=True)
    f_hi = frames_crop.max(axis=(1, 2), keepdims=True)
    f_rng = np.where(f_hi > f_lo, f_hi - f_lo, 1.0)
    frames_norm = (frames_crop - f_lo) / f_rng

    # ── Image signal: 1 - NCC ─────────────────────────────────────────────
    logger.info("Computing cross-correlation image signal …")
    image_raw = compute_correlation_signal(frames_norm)
    image_raw = normalize_data(image_raw, step=cfg.normalize_step)

    # ── Heart rate detection via FFT ───────────────────────────────────────
    f_heart_min = getattr(cfg, 'f_cardiac_min', 1.0)
    f_heart_max = getattr(cfg, 'f_cardiac_max', 3.5)
    f_heart = detect_heart_rate(image_raw, fs, f_heart_min, f_heart_max)

    # ── Bandpass filter ────────────────────────────────────────────────────
    lo_frac = getattr(cfg, 'bandpass_lo_frac', 0.7)
    hi_frac = getattr(cfg, 'bandpass_hi_frac', 2.2)

    image_filtered = bandpass_filter(image_raw, f_heart, fs, lo_frac, hi_frac)
    image_filtered = normalize_data(image_filtered, step=cfg.normalize_step)

    # ── Contour signal: lumen area ─────────────────────────────────────────
    contour_raw = np.zeros(N)
    contour_filtered = np.zeros(N)

    if report_data is not None and len(report_data) > 0:
        area_signal = compute_lumen_area_signal(report_data, N, lower_limit)
        has_contour = not np.all(np.isnan(area_signal))

        if has_contour:
            contour_raw = normalize_data(area_signal, step=cfg.normalize_step)
            area_filtered = bandpass_filter(area_signal, f_heart, fs, lo_frac, hi_frac)
            contour_filtered = normalize_data(area_filtered, step=cfg.normalize_step)

    # ── Frequency sweep for interactive visualisation ──────────────────────
    bpm_cuts, sweep = compute_frequency_sweep(image_raw, fs, f_heart)

    # ── Cache ─────────────────────────────────────────────────────────────
    main_window.runtime_data.gating_signal = {
        'image_based_gating': image_raw.tolist(),
        'contour_based_gating': contour_raw.tolist(),
        'image_based_gating_filtered': image_filtered.tolist(),
        'contour_based_gating_filtered': contour_filtered.tolist(),
        'gating_config': dict(vars(cfg)),
        'f_heart': f_heart,
        'f_heart_bpm': f_heart * 60,
        'freq_sweep_bpm_cuts': bpm_cuts.tolist(),
        'freq_sweep_signals': sweep.tolist(),
    }

    return image_raw, contour_raw, image_filtered, contour_filtered
