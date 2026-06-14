"""
IVUS Gating Signal Processing
==============================
Two-signal hybrid algorithm:

Image signal  (always available)
    s[n] = 1 - NCC(frame_n, frame_{n+1})        (CCB s0, Maso Talou 2015)
    Bandpass at [0.7, 2.2] × f_heart removes both the slow pullback
    trend (DC) and high-frequency speckle noise.
    Minima of filtered signal = stable cardiac phases.

Contour signal  (when ≥50% of frames have drawn contours)
    s[n] = lumen_area[n]  (mm², from report_data)
    Same bandpass filter.
    Peaks of filtered signal = diastole (large, relaxed lumen).
    Troughs = systole (small, compressed lumen).
    Classifier SNR vs labelled frames: 1.54 (vs 0.04 for centroid vector).

Heart rate detection
    FFT spectral peak of the correlation signal in the configured range.
    No grid-search optimisation — the FFT estimate is already correct for
    both rest (~60-100 BPM) and stress (~100-210 BPM) imaging.

Notes
-----
- DT-CWT, blur signal, centroid vector, and weight optimisation removed:
  they add noise for real IVUS data and degrade the clean correlation signal.
- The 2 × f_heart trick is unnecessary when the lumen area is available:
  it already separates sys and dia at a single f_heart.
- Centroid vector analysis showed SNR = 0.04 (noise-dominated by catheter
  rotation) versus lumen area SNR = 1.54; vector signals not used.
"""

import time
import numpy as np
from loguru import logger
from scipy.signal import find_peaks, butter, filtfilt


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
    No grid-search is applied — the FFT estimate is sufficiently accurate.
    """
    sig = np.nan_to_num(signal - np.nanmean(signal))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(sig))
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        logger.warning(f"No spectral peak in [{f_min}, {f_max}] Hz — using midpoint")
        return (f_min + f_max) / 2
    f_heart = float(freqs[mask][np.argmax(spectrum[mask])])
    logger.info(f"Heart rate: {f_heart:.3f} Hz  ({f_heart * 60:.0f} BPM)")
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
    """Zero-phase Butterworth bandpass at [lo_frac, hi_frac] × f_heart.

    Default passband [0.7, 2.2] × f_heart:
    - Lower cut removes slow pullback trend (DC and sub-cardiac drift).
    - Upper cut removes high-frequency speckle noise.
    - Passes up to 2nd harmonic so both sys and dia phases are visible
      as two distinct features per cardiac cycle in the image signal.
    """
    nyq = fs / 2.0
    lo = max(0.01, lo_frac * f_heart / nyq)
    hi = min(0.99, hi_frac * f_heart / nyq)
    if lo >= hi:
        logger.warning(f"Bandpass bounds invalid ({lo:.3f}, {hi:.3f}) — returning raw signal")
        return np.array(signal, dtype=float)
    b, a = butter(order, [lo, hi], btype='band')
    try:
        return filtfilt(b, a, np.nan_to_num(signal))
    except Exception as exc:
        logger.warning(f"Bandpass filter failed: {exc} — returning raw signal")
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
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                area[idx] = float(v)

    coverage = float(np.sum(~np.isnan(area))) / N
    if coverage < 0.5:
        logger.info(f"Lumen area coverage {coverage:.0%} < 50% — contour signal omitted")
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

    Sweeps from 0.5 × f_heart to 4 × f_heart so the user can see the
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


# ─────────────────────────── extrema / combined signal helpers ────


def identify_extrema(main_window, signal: np.ndarray, x_lim_override: int | None = None) -> tuple:
    """Find maxima and minima via scipy find_peaks.

    x_lim_override: if provided, overrides config extrema_x_lim.  Used when
    the heart rate is known so the minimum inter-peak distance can be set to
    roughly half the expected peak-to-peak spacing (= fs / (2 * f_heart) frames).
    """
    y_lim = main_window.config.gating.extrema_y_lim
    x_lim = x_lim_override if x_lim_override is not None else main_window.config.gating.extrema_x_lim

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    min_h = np.percentile(signal, y_lim)

    maxima_idx, _ = find_peaks(signal, distance=x_lim, height=min_h)
    minima_idx, _ = find_peaks(-signal, distance=x_lim, height=-min_h)

    extrema_idx = np.sort(np.concatenate((maxima_idx, minima_idx)))
    return extrema_idx, maxima_idx


def combined_signal(main_window, signal_list: list, maxima_only: bool = False) -> np.ndarray:
    """Inverse-variability weighted combination of signals."""
    extrema_indices = []
    for sig in signal_list:
        all_ext, maxima = identify_extrema(main_window, sig)
        extrema_indices.append((maxima if maxima_only else all_ext)[::2])

    variability = [float(np.std(np.diff(e))) if len(e) > 1 else np.nan for e in extrema_indices]
    n = len(signal_list)
    valid = [v for v in variability if not np.isnan(v) and v > 0]
    if len(valid) < n:
        weights = [1.0 / n] * n
    else:
        inv = [1.0 / v for v in variability]
        total = sum(inv)
        weights = [i / total for i in inv]

    out = np.zeros(len(signal_list[0]))
    for w, s in zip(weights, signal_list):
        out += w * np.nan_to_num(s, nan=0.0)
    return out


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
        if gs and gs.get('gating_config') == vars(cfg) and len(gs.get('image_based_gating', [])) == N:
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
        'gating_config': vars(cfg),
        'f_heart': f_heart,
        'f_heart_bpm': f_heart * 60,
        'freq_sweep_bpm_cuts': bpm_cuts.tolist(),
        'freq_sweep_signals': sweep.tolist(),
    }

    return image_raw, contour_raw, image_filtered, contour_filtered
