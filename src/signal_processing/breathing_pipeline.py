import numpy as np
from loguru import logger

from signal_processing.automatic_gating import track_diastole, walk_extrema
from signal_processing.gating_pipeline import (
    adjust_for_elliptic_deformation,
    fft_peak_freq,
    lowpass_filter,
)

# Breath tracker: the gating beat tracker with one breath as the beat. Breathing changes
# rate more than the heart, so the steps are wider than in gating (tested on 10 NARCO
# pullbacks against the breathing wave of the aortic pressure)
BREATH_GAP = (0.6, 1.5)  # valley-valley gap of a regular breath, in breaths
BREATH_SKIP_GAP = (1.7, 2.5)  # gap of a missed breath, in breaths
EDGE_PEAK = 0.75  # furthest peak outside the first or last valley, in breaths


def adjusted_areas_by_frame(
    area_of: dict[int, float],
    minor_axis_of: dict[int, float],
) -> dict[int, float]:
    """Frame-dict wrapper around gating_pipeline.adjust_for_elliptic_deformation
    for the breathing sort's sparse (gated-frames-only) data.

    No-op (returns ``area_of`` unchanged) unless every frame has a minor-axis
    measurement too, so partial coverage can't leave some frames adjusted and
    others not (which would mix incompatible scales in the same signal).
    """
    frames = sorted(area_of.keys())
    if not frames or any(f not in minor_axis_of for f in frames):
        return dict(area_of)

    area = np.array([area_of[f] for f in frames])
    minor_axis = np.array([minor_axis_of[f] for f in frames])
    adjusted = adjust_for_elliptic_deformation(area, minor_axis, step=0)
    return dict(zip(frames, adjusted))


def compute_breathing_signal(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None = None,
    fs: float = 30.0,
    f_heart: float | None = None,
    f_resp_override: float | None = None,
    poly_deg: int = 2,
    gated_weight: float = 10.0,
    cache: dict | None = None,
) -> dict:
    """
    Detrend the lumen-area sequence and extract the respiratory oscillation.

    Steps
    -----
    1. Fit a low-order polynomial trend (the pullback taper) to area vs. frame,
       giving *gated* (manually reviewed) frames ``gated_weight`` times the weight so
       the trend is anchored by reliable points and not dragged by ostial noise.
    2. Residual = area - trend.
    3. Detect the respiratory rate on the residual (or use ``f_resp_override``)
       and low-pass filter the residual at 2x that rate to isolate breathing.

    Parameters
    ----------
    frames_arr : (N,) array of frame numbers
    areas_arr : (N,) array of lumen areas (mm²)
    gated_frames : set of frame numbers to be given extra weight
    fs : sampling frequency (frames/sec)
    f_heart : heart rate (Hz) to expand the search range for breathing rate
    f_resp_override : override the detected breathing rate (Hz)
    poly_deg : degree of polynomial to fit the trend
    gated_weight : weight for gated frames
    cache : optional dict (typically ``runtime_data.gating_signal``) to read/write
        a cached result from. Callers like plot redraws invoke this on nearly
        every user action, so a signature over the inputs lets an unchanged
        pullback skip the polyfit + FFT + filtfilt work entirely.

    Returns
    -------
    dict with keys:
        frames : (N,) array of frame numbers
        areas : (N,) array of lumen areas (mm²)
        trend : (N,) polynomial trend fit to area vs. frame
        slope : (N,) derivative of the trend (mm²/frame)
        residual : (N,) area - trend
        smoothed : (N,) low-pass filtered residual at 2x breathing rate
        f_resp : detected or overridden breathing rate (Hz)
    """
    frames_arr = np.asarray(frames_arr, dtype=float)
    areas_arr = np.asarray(areas_arr, dtype=float)
    n_samples = len(frames_arr)

    signature = _build_signal_signature(
        frames_arr, areas_arr, gated_frames, fs, f_heart, f_resp_override, poly_deg, gated_weight
    )
    cached_result = _read_cached_signal(cache, signature)
    if cached_result is not None:
        return cached_result

    if n_samples < 3:
        result = _degenerate_signal_result(frames_arr, areas_arr)
    else:
        trend, slope, residual = _fit_weighted_trend(frames_arr, areas_arr, gated_frames, poly_deg, gated_weight)

        if f_resp_override is not None:
            f_resp = float(f_resp_override)
        else:
            f_resp = _detect_breathing_rate(residual, fs, f_heart)
        smoothed = _extract_breathing_signal(residual, f_resp, fs)

        result = {
            'frames': frames_arr,
            'areas': areas_arr,
            'trend': trend,
            'slope': slope,
            'residual': residual,
            'smoothed': smoothed,
            'f_resp': f_resp,
        }

    if cache is not None:
        cache['breathing_cache_signature'] = signature
        cache['breathing_cache_result'] = result

    return result


def _build_signal_signature(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None,
    fs: float,
    f_heart: float | None,
    f_resp_override: float | None,
    poly_deg: int,
    gated_weight: float,
) -> dict:
    """Inputs that fully determine ``compute_breathing_signal``'s output, used to invalidate the cache."""
    return {
        'version': 2,  # bump when the computation changes, so results cached in saved files are recomputed
        'frames': frames_arr.tolist(),
        'areas': areas_arr.tolist(),
        'gated_frames': sorted(int(f) for f in gated_frames) if gated_frames else [],
        'fs': fs,
        'f_heart': f_heart,
        'f_resp_override': f_resp_override,
        'poly_deg': poly_deg,
        'gated_weight': gated_weight,
    }


def _read_cached_signal(cache: dict | None, signature: dict) -> dict | None:
    """Return the cached result if `signature` still matches, else None."""
    if cache is None or cache.get('breathing_cache_signature') != signature:
        return None
    cached_result = cache.get('breathing_cache_result')
    if cached_result is None:
        return None
    return {k: np.asarray(v) if isinstance(v, (list, np.ndarray)) else v for k, v in cached_result.items()}


def _degenerate_signal_result(frames_arr: np.ndarray, areas_arr: np.ndarray) -> dict:
    """Fallback result when there are too few points (<3) to fit a trend."""
    zeros = np.zeros(len(frames_arr))
    return {
        'frames': frames_arr,
        'areas': areas_arr,
        'trend': areas_arr.copy(),
        'slope': zeros,
        'residual': zeros,
        'smoothed': zeros,
        'f_resp': 0.0,
    }


def _fit_weighted_trend(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None,
    poly_deg: int,
    gated_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polynomial pullback-taper trend, its slope, and the area residual.

    Gated (manually reviewed) frames get ``gated_weight`` times the weight so
    the trend is anchored by reliable points and not dragged by ostial noise.
    """
    n_samples = len(frames_arr)
    weights: np.ndarray = np.ones(n_samples)
    if gated_frames:
        weights = np.where(np.isin(frames_arr.astype(int), list(gated_frames)), gated_weight, 1.0)

    deg = min(poly_deg, n_samples - 1)
    # polyfit squares w with the residual, so sqrt gives gated frames gated_weight times the squared error
    coeffs = np.polyfit(frames_arr, areas_arr, deg=deg, w=np.sqrt(weights))
    trend = np.polyval(coeffs, frames_arr)
    # Analytic derivative of the polynomial trend -> local taper slope (mm²/frame).
    slope = np.polyval(np.polyder(coeffs), frames_arr)
    residual = areas_arr - trend
    return trend, slope, residual


def _detect_breathing_rate(
    signal: np.ndarray,
    fs: float,
    f_heart_hz: float | None = None,
) -> float:
    """Dominant respiratory rate via FFT spectral peak [Hz].

    Searches 12-30 BrPM (0.20-0.50 Hz) by default.
    Expands upper bound to 1.0 Hz (60 BrPM) when heart rate > 100 BPM.
    """
    f_resp_min = 12 / 60  # 0.20 Hz
    f_resp_max = 30 / 60  # 0.50 Hz
    if f_heart_hz is not None and f_heart_hz * 60 > 100:
        f_resp_max = 60 / 60  # 1.00 Hz

    f_resp = fft_peak_freq(signal, fs, f_resp_min, f_resp_max, label="respiratory")
    logger.info(f"Respiratory rate: {f_resp:.3f} Hz  ({f_resp * 60:.1f} BrPM)")
    return f_resp


def _extract_breathing_signal(
    area_signal: np.ndarray,
    f_resp_hz: float,
    fs: float,
) -> np.ndarray:
    """Low-pass filter area signal to isolate the respiratory component.

    Cuts off at 2x the respiratory rate to retain the breathing oscillation
    while suppressing cardiac motion (~1 Hz and above).
    """
    f_cutoff = min(2.0 * f_resp_hz, fs / 2.0 * 0.9)
    return lowpass_filter(area_signal, f_cutoff, fs)


def track_breaths(breathing_signal: np.ndarray, breath: float) -> tuple[np.ndarray, np.ndarray]:
    """Peaks and valleys of the breathing signal, valleys chosen as a chain like gating's diastoles.

    The valleys are the track_diastole chain with one breath as the beat, so
    they stay about one breath apart even where the breathing is shallow. The
    chain is extended breath by breath towards both ends, where the tracker
    may stop up to 1.5 breaths short. Each peak is the highest point between
    two valleys, plus one within EDGE_PEAK breaths outside each end valley.
    """
    sig = np.nan_to_num(breathing_signal)
    valleys, _ = track_diastole(sig, breath, BREATH_GAP, BREATH_SKIP_GAP)
    empty = np.array([], dtype=int)
    if not valleys:
        return empty, empty

    minima = _turning_points(-sig)
    lo, hi = BREATH_GAP[0] * breath, BREATH_GAP[1] * breath
    while len(c := minima[(minima >= valleys[0] - hi) & (minima <= valleys[0] - lo)]):
        valleys.insert(0, int(c[np.argmin(sig[c])]))
    while len(c := minima[(minima >= valleys[-1] + lo) & (minima <= valleys[-1] + hi)]):
        valleys.append(int(c[np.argmin(sig[c])]))

    peaks = [a + int(np.argmax(sig[a:b])) for a, b in zip(valleys[:-1], valleys[1:]) if b - a > 2]
    maxima = _turning_points(sig)
    before = maxima[(maxima < valleys[0]) & (maxima >= valleys[0] - EDGE_PEAK * breath)]
    after = maxima[(maxima > valleys[-1]) & (maxima <= valleys[-1] + EDGE_PEAK * breath)]
    if len(before):
        peaks.insert(0, int(before[np.argmax(sig[before])]))
    if len(after):
        peaks.append(int(after[np.argmax(sig[after])]))
    return np.array(peaks, dtype=int), np.array(valleys, dtype=int)


def _turning_points(x: np.ndarray) -> np.ndarray:
    """Indices of the local maxima of x (first sample of a flat top)."""
    return np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1


def compute_breathing_phases(
    breathing_signal: np.ndarray,
    manual_peaks: list[int] | None = None,
    manual_valleys: list[int] | None = None,
    frames_arr: np.ndarray | None = None,
    swing_fraction: float = 0.10,
    anchor_gap: int = 15,
    manual_only: bool = False,
    breath: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame breathing phase in [0, 1), peaks, and valley indices.

    Phase anchors: valleys (end-of-breath, vessel at rest) -> 0.0 / 1.0,
    peaks (mid-breath, vessel displaced) -> 0.5.  Phase between anchors is
    linearly interpolated from a monotonically increasing cumulative-phase
    sequence and then wrapped to [0, 1).

    The automatic anchors come from track_breaths when the breath length is
    known, else from walk_extrema (a swing of ``swing_fraction`` of the
    peak-to-peak range), which leaves holes where the breathing is shallow.

    Manual anchors override the automatic detection (hard anchors): any
    auto-detected extremum within ``anchor_gap`` samples of a manual anchor is
    discarded, so the automatic detector only *fills the gaps* between the
    frames the user labelled.  This is important in noisy ostial regions where
    the automatic detector is unreliable.

    Parameters
    ----------
    manual_peaks/manual_valleys : peak / valley locations **as frame numbers**
        (matched against ``frames_arr``).  If ``frames_arr`` is None they are
        treated as direct indices into ``breathing_signal``.
    breath : samples per breath (fs / f_resp), None or 0 if unknown.

    Returns
    -------
    phase       : (N,) float array in [0, 1)
    peaks_idx   : indices of breathing peaks (auto + manual, vessel displaced)
    valleys_idx : indices of breathing valleys (auto + manual, end-of-breath)
    """
    sig_centered = np.nan_to_num(breathing_signal - np.nanmean(breathing_signal))
    n_samples = len(sig_centered)
    phase = np.zeros(n_samples)

    if manual_only:
        auto_peaks = auto_valleys = np.array([], dtype=int)
    elif breath and np.isfinite(breath) and 2 < breath < n_samples:
        auto_peaks, auto_valleys = track_breaths(sig_centered, breath)
    else:
        _, auto_peaks, auto_valleys = walk_extrema(sig_centered, swing_fraction=swing_fraction)
    manual_peak_idx = _frames_to_indices(manual_peaks, frames_arr, n_samples)
    manual_valley_idx = _frames_to_indices(manual_valleys, frames_arr, n_samples)

    peaks_idx, valleys_idx = _reconcile_extrema(
        auto_peaks, auto_valleys, manual_peak_idx, manual_valley_idx, manual_only, anchor_gap
    )

    anchors = _alternate_anchors(_sorted_anchor_points(peaks_idx, valleys_idx))
    if len(anchors) < 2:
        return phase, peaks_idx, valleys_idx

    anchor_frames, anchor_cum_phase = _cumulative_phase_anchors(anchors)
    phase_unwrapped = np.interp(
        np.arange(n_samples, dtype=float),
        anchor_frames,
        anchor_cum_phase,
        left=anchor_cum_phase[0],
        right=anchor_cum_phase[-1],
    )
    phase = phase_unwrapped % 1.0
    return phase, peaks_idx, valleys_idx


def _frames_to_indices(
    frame_numbers: list[int] | None,
    frames_arr: np.ndarray | None,
    n_samples: int,
) -> list[int]:
    """Map frame numbers to indices into the breathing signal (nearest match)."""
    if not frame_numbers:
        return []
    if frames_arr is None:
        return [int(f) for f in frame_numbers if 0 <= int(f) < n_samples]
    frames_arr = np.asarray(frames_arr)
    return [int(np.argmin(np.abs(frames_arr - f))) for f in frame_numbers]


def _reconcile_extrema(
    auto_peaks: np.ndarray,
    auto_valleys: np.ndarray,
    manual_peak_idx: list[int],
    manual_valley_idx: list[int],
    manual_only: bool,
    anchor_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine auto-detected and manual extrema, letting manual anchors win.

    ``manual_only=True``: use only the user's labels.  Otherwise, auto extrema
    within ``anchor_gap`` samples of a manual anchor are dropped so manual
    anchors override the automatic detector locally, which only fills the
    gaps elsewhere.
    """
    if manual_only:
        peaks_idx = np.array(sorted(set(manual_peak_idx)), dtype=int)
        valleys_idx = np.array(sorted(set(manual_valley_idx)), dtype=int)
        return peaks_idx, valleys_idx

    manual_all = manual_peak_idx + manual_valley_idx

    def _drop_near_manual(auto_list):
        return [int(a) for a in auto_list if not manual_all or min(abs(int(a) - m) for m in manual_all) >= anchor_gap]

    peaks_idx = np.array(sorted(set(_drop_near_manual(auto_peaks) + manual_peak_idx)), dtype=int)
    valleys_idx = np.array(sorted(set(_drop_near_manual(auto_valleys) + manual_valley_idx)), dtype=int)
    return peaks_idx, valleys_idx


def _sorted_anchor_points(peaks_idx: np.ndarray, valleys_idx: np.ndarray) -> list[tuple[int, str]]:
    """Merge peak/valley indices into one (index, type) list sorted by index."""
    return sorted(
        [(int(f), 'v') for f in valleys_idx] + [(int(f), 'p') for f in peaks_idx],
        key=lambda point: point[0],
    )


def _cumulative_phase_anchors(anchors: list[tuple[int, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Assign monotonically increasing cumulative phase to alternating anchors.

    Each peak<->valley step is a half breathing cycle (+0.5); the running
    total is wrapped to [0, 1) by the caller after interpolation.
    """
    anchor_frames: list[float] = []
    cum_phase: list[float] = []
    cum = 0.0 if anchors[0][1] == 'v' else 0.5
    for k, (frame, _) in enumerate(anchors):
        if k > 0:
            cum += 0.5
        anchor_frames.append(float(frame))
        cum_phase.append(cum)
    return np.array(anchor_frames), np.array(cum_phase)


def _alternate_anchors(points: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Force a peak/valley point list into a strictly alternating sequence.

    Consecutive same-type anchors get a synthetic opposite anchor inserted at
    their midpoint, so the half-cycle phase model (0.5 per step) stays valid
    even when the user labels two peaks with no valley between them.
    """
    if len(points) < 2:
        return points
    out: list[tuple[int, str]] = [points[0]]
    for frame, kind in points[1:]:
        prev_frame, prev_kind = out[-1]
        if kind == prev_kind and frame != prev_frame:
            mid = (prev_frame + frame) // 2
            if mid not in (prev_frame, frame):
                out.append((mid, 'p' if kind == 'v' else 'v'))
        out.append((frame, kind))
    return out


# ─────────────── breathing-bin registration sort (gated frames only) ────
#
# Idea (per phase, diastole and systole are handled independently):
#   * The labelled valleys are the vessel at rest (displacement 0); peaks are max
#     displacement. Each breathing half-cycle (valley->peak and peak->valley) is
#     split into the same number of bins, and by symmetry the ascending and
#     descending bin at a given displacement are pooled together.
#   * Bin 0 (the valleys) is the ground truth: fit lumen-area (divided by ) vs frame position
#     through those rest frames.
#   * Every other bin's frames sample the SAME vessel profile but shifted along
#     the pullback by the breathing displacement.  Slide that bin's area profile
#     until it best matches the ground-truth curve; the winning shift is the
#     displacement, which is subtracted to recover each frame's true position.
#   * Frames are then re-ordered by corrected position (gaps ignored).


def assign_breathing_bins(
    frames: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    n_bins: int = 5,
) -> np.ndarray:
    """Assign each frame to a displacement bin 0..n_bins-1 (0 = valley / rest).
    Bins are needed to adjust for unregular breathing, were longer and shorter half-cycles
    can occur. The binning harmonizes the displacement.

    Ascending (valley->peak) maps displacement 0->max as bin 0->n_bins-1; descending
    (peak->valley) is mirrored so it shares the same bins by displacement.  Frames
    outside any labelled half-cycle get -1.
    """
    anchors = sorted([(int(v), 'v') for v in valleys] + [(int(p), 'p') for p in peaks])
    if len(anchors) < 2:
        return np.full(len(frames), -1, dtype=int)
    anchor_frames = np.array([frame for frame, _ in anchors])
    bins = np.full(len(frames), -1, dtype=int)
    for i, frame in enumerate(frames):
        anchor_idx = int(np.searchsorted(anchor_frames, frame, side='right')) - 1
        if anchor_idx < 0 or anchor_idx >= len(anchors) - 1:
            continue
        start_frame, start_kind = anchors[anchor_idx]
        end_frame, _ = anchors[anchor_idx + 1]
        if end_frame <= start_frame:
            continue
        progress = (frame - start_frame) / (end_frame - start_frame)  # position within the half-cycle, 0..1
        bin_idx = int(np.floor(progress * n_bins)) if start_kind == 'v' else int(np.floor((1.0 - progress) * n_bins))
        bins[i] = min(max(bin_idx, 0), n_bins - 1)
    return bins


def _fit_line(frames: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(frames, values, 1)
    return float(slope), float(intercept)


def register_phase(
    frames: np.ndarray,
    areas: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    n_bins: int = 5,
) -> dict:
    """Breathing-corrected positions via valley-anchored per-bin line-fit registration.

    See the module comment above.  For every bin, fits ``area = slope*frame +
    intercept`` through that bin's own points, then solves directly for the
    horizontal shift ``s`` on the bin-0 (valley/rest) line that makes it agree
    with the bin's own line at the bin's mean frame::

        valley_line(f_mid + s) == bin_line(f_mid)
        s = (bin_line(f_mid) - valley_line(f_mid)) / valley_slope

    Same method for every bin (including the peak bin), then the existing
    monotonic-magnitude enforcement gives the valley (fixed at 0) and peak
    (typically largest, so left untouched) bins priority while smoothing
    noisier in-between estimates.

    (This replaced an earlier grid-search MSE shift estimate against a fitted
    ground-truth curve, which was noisy for thin bins.)

    Returns a dict with ``bins`` (per-frame bin), ``shifts`` (per-bin
    displacement in frames), ``corrected`` (per-frame corrected position),
    ``order`` (frame indices sorted by corrected position), plus
    ``valley_span``/``peak_shift``/``true_length``: the corrected-position span
    of one full valley->peak half-cycle, since raw frame number stops being a
    meaningful position once each bin has been shifted independently.
    """
    frames = np.asarray(frames, dtype=float)
    areas = np.asarray(areas, dtype=float)
    bins = assign_breathing_bins(frames, peaks, valleys, n_bins)

    valley_mask = bins == 0
    if valley_mask.sum() < 2:
        return _passthrough_registration(frames, bins, n_bins)
    m_v, b_v = _fit_line(frames[valley_mask], areas[valley_mask])
    if m_v == 0:
        return _passthrough_registration(frames, bins, n_bins)

    valley_span = float(frames[valley_mask].max() - frames[valley_mask].min())

    shifts = np.zeros(n_bins)
    for b in range(1, n_bins):
        in_bin = bins == b
        if in_bin.sum() < 2:
            shifts[b] = shifts[b - 1]
            continue
        m_b, b_b = _fit_line(frames[in_bin], areas[in_bin])
        f_mid = frames[in_bin].mean()
        valley_at_mid = m_v * f_mid + b_v
        bin_at_mid = m_b * f_mid + b_b
        # s such that valley_line(f_mid + s) == bin_line(f_mid): the corrected
        # position sits on the ground-truth (valley) axis, not the bin's own.
        shifts[b] = (bin_at_mid - valley_at_mid) / m_v

    shifts = _enforce_monotonic_shifts(shifts)
    corrected = _apply_bin_shifts(frames, bins, shifts, n_bins)
    corrected = _extrapolate_out_of_range(frames, bins, corrected, peaks, valleys, shifts, n_bins)
    order = np.argsort(corrected, kind='stable')

    return {
        'bins': bins,
        'shifts': shifts,
        'corrected': corrected,
        'order': order,
        'valley_span': valley_span,
        'peak_shift': float(shifts[-1]),
        'true_length': valley_span + abs(float(shifts[-1])),
    }


def _extrapolate_out_of_range(
    frames: np.ndarray,
    bins: np.ndarray,
    corrected: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    shifts: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Correct frames before the first / after the last labelled anchor.

    ``assign_breathing_bins`` gives these ``bin == -1`` (they're outside any
    labelled half-cycle), and ``_apply_bin_shifts`` leaves them untouched at
    their raw frame value — which can misorder them relative to their now
    breathing-corrected neighbours. There's no labelled data to estimate their
    own displacement, so instead they inherit the shift in force at the
    boundary anchor they sit past (valley anchor -> bin 0 / no shift, peak
    anchor -> the peak bin's shift), which is a better estimate than none.
    """
    out_of_range = bins == -1
    if not np.any(out_of_range):
        return corrected
    anchors = sorted([(int(v), 'v') for v in valleys] + [(int(p), 'p') for p in peaks])
    if len(anchors) < 2:
        return corrected

    first_frame, first_kind = anchors[0]
    last_frame, last_kind = anchors[-1]
    first_bin = n_bins - 1 if first_kind == 'p' else 0
    last_bin = n_bins - 1 if last_kind == 'p' else 0

    before = out_of_range & (frames < first_frame)
    after = out_of_range & (frames > last_frame)
    corrected[before] = frames[before] + shifts[first_bin]
    corrected[after] = frames[after] + shifts[last_bin]
    return corrected


def _passthrough_registration(frames: np.ndarray, bins: np.ndarray, n_shift_bins: int) -> dict:
    """No-correction fallback (identity mapping) for when there isn't enough data to register."""
    return {
        'bins': bins,
        'shifts': np.zeros(n_shift_bins),
        'corrected': frames.copy(),
        'order': np.argsort(frames, kind='stable'),
    }


def _enforce_monotonic_shifts(shifts: np.ndarray) -> np.ndarray:
    """Keep |shift| non-decreasing and of one consistent sign across bins.

    Displacement grows away from the valley (bin 0); the sign used is the
    dominant direction across the bins.
    """
    sign = 1.0 if np.sum(shifts) >= 0 else -1.0
    magnitudes = np.maximum.accumulate(np.abs(shifts))
    return sign * magnitudes


def _apply_bin_shifts(frames: np.ndarray, bins: np.ndarray, shifts: np.ndarray, n_bins: int) -> np.ndarray:
    """Shift each frame's position by its bin's displacement to recover the true pullback position."""
    corrected = frames.copy()
    for bin_idx in range(n_bins):
        corrected[bins == bin_idx] = frames[bins == bin_idx] + shifts[bin_idx]
    return corrected


# ─────────────── shape sort (gated frames only) ────────────────────────
#
# The catheter shifts along the vessel by a displacement that is 0 at the
# breathing valley and largest at the peak, so frames from one breath can land
# far from their acquisition neighbours. Within each valley-to-valley breath
# the shift is that breath's amplitude times the breathing signal above the
# valley line, scaled to 1 at the peak. The amplitudes are chosen so lumen area
# and elliptic ratio oscillate least along the diastolic and systolic stacks,
# since a vessel cannot switch back and forth between round and elliptic.
# A second pass moves single frames a short way to remove leftover islands.

SHAPE_MAX_SHIFT_S = 24.0  # largest peak displacement tried, in seconds of pullback
SHAPE_GRID_STEPS = 29  # candidate amplitudes from 0 to the largest
SHAPE_STARTS = (0.0, 0.2, 0.4, 0.7)  # starting amplitudes, as fractions of the largest
SHAPE_SWEEPS = 4
REFINE_WINDOW = 0.5  # breaths, how far the fine-tuning may move one frame
REFINE_COST = 0.3  # variation a move across the whole window must save


def breath_profile(
    frames: np.ndarray, signal_frames: np.ndarray, signal: np.ndarray, valleys: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Breath index and the 0-at-valley, 1-at-peak breathing profile at each frame."""
    frames = np.asarray(frames, dtype=float)
    signal_frames = np.asarray(signal_frames, dtype=float)
    signal = np.asarray(signal, dtype=float)
    v = np.sort(np.asarray(valleys, dtype=float))
    level = np.interp(frames, signal_frames, signal)
    at_valley = np.interp(v, signal_frames, signal)
    breath = np.clip(np.searchsorted(v, frames) - 1, 0, len(v) - 2)
    profile = np.zeros(len(frames))
    for j in range(len(v) - 1):
        slope = (at_valley[j + 1] - at_valley[j]) / (v[j + 1] - v[j])
        inside = (signal_frames >= v[j]) & (signal_frames <= v[j + 1])
        if not inside.any():
            continue
        top = (signal[inside] - (at_valley[j] + slope * (signal_frames[inside] - v[j]))).max()
        mine = breath == j
        profile[mine] = (level[mine] - (at_valley[j] + slope * (frames[mine] - v[j]))) / max(top, 1e-9)
    return breath, profile


def shape_variation(positions: np.ndarray, features: list[np.ndarray]) -> float:
    """Total variation of each feature along the stack, scaled by the feature's range."""
    order = np.argsort(positions, kind='stable')
    return float(sum(np.abs(np.diff(f[order])).sum() / (np.ptp(f) + 1e-9) for f in features))


def _stack_features(frames: list[int], area_of: dict[int, float], ell_of: dict[int, float]) -> list[np.ndarray]:
    features = [np.log([area_of[f] for f in frames])]
    if all(ell_of.get(f, 0) > 0 for f in frames):
        features.append(np.log([ell_of[f] for f in frames]))
    return features


def refine_positions(
    positions: np.ndarray,
    features: list[np.ndarray],
    window: float,
    cost: float = REFINE_COST,
) -> np.ndarray:
    """Move single frames within +-window to where the stack's shape variation drops most."""
    z = np.asarray(positions, dtype=float).copy()
    n = len(z)
    if n < 3 or window <= 0:
        return z
    F = np.array([f / (np.ptp(f) + 1e-9) for f in features])

    def step(a: int, b: int) -> float:
        return float(np.abs(F[:, a] - F[:, b]).sum())

    for _ in range(5):
        moved = 0
        for i in np.argsort(z, kind='stable'):
            others = [j for j in np.argsort(z, kind='stable') if j != i]
            zo = z[others]
            k = int(np.searchsorted(zo, z[i]))
            if 0 < k < n - 1:
                saved = step(i, others[k - 1]) + step(i, others[k]) - step(others[k - 1], others[k])
            else:
                saved = step(i, others[0] if k == 0 else others[-1])
            best, best_z = 0.0, z[i]
            lo, hi = np.searchsorted(zo, z[i] - window), np.searchsorted(zo, z[i] + window)
            for m in range(lo, min(hi, n - 1) + 1):
                if m == 0:
                    added, new_z = step(i, others[0]), zo[0] - 1.0
                elif m == n - 1:
                    added, new_z = step(i, others[-1]), zo[-1] + 1.0
                else:
                    a, b = others[m - 1], others[m]
                    added, new_z = step(i, a) + step(i, b) - step(a, b), 0.5 * (zo[m - 1] + zo[m])
                if abs(new_z - z[i]) > window:
                    continue
                delta = added - saved + cost * abs(new_z - z[i]) / window
                if delta < best - 1e-9:
                    best, best_z = delta, new_z
            if best_z != z[i]:
                z[i] = best_z
                moved += 1
        if not moved:
            break
    return z


def sort_by_shape(
    dia: list[int],
    sys: list[int],
    area_of: dict[int, float],
    ell_of: dict[int, float],
    signal_frames: np.ndarray,
    signal: np.ndarray,
    valleys: list[int],
    fs: float,
    breath: float | None = None,
) -> dict:
    """Breathing-corrected positions of the gated frames from shape smoothness.

    Returns ``amplitudes`` (peak displacement per breath in frames, positive = distal),
    ``dia_pos`` and ``sys_pos`` (corrected position per frame on the frame axis, proximal = larger).
    """
    frames = np.array(list(dia) + list(sys), dtype=float)
    out: dict = {'amplitudes': [], 'dia_pos': {f: float(f) for f in dia}, 'sys_pos': {f: float(f) for f in sys}}
    if len(valleys) < 2 or len(dia) < 3:
        return out
    breath_idx, profile = breath_profile(frames, signal_frames, signal, valleys)
    n_breaths = len(valleys) - 1
    stacks = [(np.arange(len(dia)), _stack_features(dia, area_of, ell_of))]
    if len(sys) >= 3:
        stacks.append((len(dia) + np.arange(len(sys)), _stack_features(sys, area_of, ell_of)))

    def variation(amplitudes: np.ndarray) -> float:
        z = frames - amplitudes[breath_idx] * profile
        return sum(shape_variation(z[idx], feats) for idx, feats in stacks)

    # one direction for the whole pullback: peaks move distal (+) or proximal (-)
    best, best_cost = np.zeros(n_breaths), variation(np.zeros(n_breaths))
    for sign in (1.0, -1.0):
        grid = sign * np.linspace(0.0, SHAPE_MAX_SHIFT_S * fs, SHAPE_GRID_STEPS)
        for start in SHAPE_STARTS:
            amps = np.full(n_breaths, grid[int(round(start * (len(grid) - 1)))])
            cost = variation(amps)
            for _ in range(SHAPE_SWEEPS):
                for j in range(n_breaths):
                    trial = amps.copy()
                    costs = []
                    for a in grid:
                        trial[j] = a
                        costs.append(variation(trial))
                    i = int(np.argmin(costs))
                    if costs[i] < cost:
                        amps[j], cost = grid[i], costs[i]
            if cost < best_cost:
                best, best_cost = amps.copy(), cost

    z = frames - best[breath_idx] * profile
    window = REFINE_WINDOW * breath if breath and np.isfinite(breath) else 0.0
    for idx, feats in stacks:
        z[idx] = refine_positions(z[idx], feats, window)
    out['amplitudes'] = [float(a) for a in best]
    out['dia_pos'] = {f: float(z[i]) for i, f in enumerate(dia)}
    if len(stacks) > 1:
        out['sys_pos'] = {f: float(z[len(dia) + i]) for i, f in enumerate(sys)}
    return out


def breathing_shift(
    frames: np.ndarray, signal_frames: np.ndarray, signal: np.ndarray, valleys: list[int], amplitudes: list[float]
) -> np.ndarray:
    """Shift of each frame (negative = distal) for stored per-breath amplitudes."""
    frames = np.asarray(frames, dtype=float)
    if len(valleys) < 2 or len(amplitudes) != len(valleys) - 1:
        return np.zeros(len(frames))
    breath_idx, profile = breath_profile(frames, signal_frames, signal, valleys)
    return -np.asarray(amplitudes, dtype=float)[breath_idx] * profile
