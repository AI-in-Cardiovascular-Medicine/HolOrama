import numpy as np
from loguru import logger
from scipy.signal import find_peaks

from signal_processing.gating_pipeline import bandpass_filter, normalize_data

# Diastolic anchors: band around f_heart narrow enough to leave one valley per beat
ANCHOR_LO_FRAC = 0.7
ANCHOR_HI_FRAC = 1.3
ANCHOR_MIN_DISTANCE = 0.7  # x beat length
SNAP_FRAMES = 3  # anchor -> nearest valley of the gating signal
SYSTOLE_WINDOW = (0.25, 0.75)  # part of each D-D interval searched for systole
REFINE_FRAMES = 1  # final shift to the lowest raw value


def walk_extrema(
    signal: np.ndarray,
    swing_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hysteresis-gated turning-point detector.

    Walks the signal and registers a local maximum only after the value has
    dropped more than *swing_fraction* x peak-to-peak below the running high
    since the last confirmed turning point; vice versa for minima.
    *Note: Tested performance against scipy.signal.find_peaks()
    and found it more robust for IVUS gating signals.*

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


def _local_minima(signal: np.ndarray, lo: int, hi: int) -> list[int]:
    lo, hi = max(lo, 1), min(hi, len(signal) - 1)
    return [i for i in range(lo, hi) if signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]]


def _deepest_minimum(signal: np.ndarray, lo: int, hi: int) -> int | None:
    minima = _local_minima(signal, lo, hi)
    return min(minima, key=lambda i: signal[i]) if minima else None


def _beat_misfit(frame: int, neighbours: list[int], beat: float) -> float:
    """Distance in frames from the nearest whole number of beats to each neighbour."""
    misfit = 0.0
    for n in neighbours:
        gap = abs(n - frame)
        misfit += abs(gap - beat * max(1, round(gap / beat)))
    return misfit


def _enforce_spacing(dia: list[int], beat: float) -> list[int]:
    """Drop one of two diastoles closer than ANCHOR_MIN_DISTANCE beats.

    Snapping can pull an anchor onto the systolic valley. Of the close pair
    the one that fits the beat grid of its neighbours is kept.
    """
    dia = sorted(dia)
    while len(dia) > 1:
        gaps = np.diff(dia)
        i = int(np.argmin(gaps))
        if gaps[i] >= ANCHOR_MIN_DISTANCE * beat:
            break
        neighbours = dia[max(0, i - 1) : i] + dia[i + 2 : i + 3]
        if not neighbours:
            del dia[i + 1]
            continue
        a, b = dia[i], dia[i + 1]
        del dia[i + 1 if _beat_misfit(a, neighbours, beat) <= _beat_misfit(b, neighbours, beat) else i]
    return dia


def _fill_gaps(dia: list[int], gating_signal: np.ndarray, beat: float) -> list[int]:
    """Split a gap spanning several beats into equal beats, with one diastole per split.

    Each inserted diastole is the deepest valley within a quarter beat of its split point.
    """
    filled = [dia[0]]
    for nxt in dia[1:]:
        prev = filled[-1]
        n_beats = int(round((nxt - prev) / beat))
        sub_beat = (nxt - prev) / max(1, n_beats)
        for k in range(1, n_beats):
            expected = prev + k * sub_beat
            lo, hi = int(expected - sub_beat / 4), int(np.ceil(expected + sub_beat / 4)) + 1
            valley = _deepest_minimum(gating_signal, lo, hi)
            filled.append(valley if valley is not None else lo + int(np.argmin(gating_signal[lo:hi])))
        filled.append(nxt)
    return filled


def find_diastole(image_raw: np.ndarray, gating_signal: np.ndarray, f_heart: float, fs: float, step: int) -> list[int]:
    """One valley per beat on the raw signal bandpassed tightly around f_heart.

    That valley is the longest still phase, i.e. diastole. Each anchor is
    snapped to the deepest valley of the gating signal within SNAP_FRAMES.
    Anchors closer than ANCHOR_MIN_DISTANCE beats are thinned out and missed
    beats are filled, so every D-D interval holds about one beat.
    """
    beat = fs / f_heart
    anchor_signal = normalize_data(bandpass_filter(image_raw, f_heart, fs, ANCHOR_LO_FRAC, ANCHOR_HI_FRAC), step=step)
    anchors, _ = find_peaks(-anchor_signal, distance=max(1.0, ANCHOR_MIN_DISTANCE * beat))
    dia = set()
    for i in anchors:
        snapped = _deepest_minimum(gating_signal, int(i) - SNAP_FRAMES, int(i) + SNAP_FRAMES + 1)
        dia.add(int(i) if snapped is None else snapped)
    if not dia:
        return []
    filled = _fill_gaps(_enforce_spacing(sorted(dia), beat), gating_signal, beat)
    return _enforce_spacing(filled, beat)


def find_systole(dia: list[int], gating_signal: np.ndarray) -> list[int]:
    """Deepest valley between each pair of diastoles, falling back to the lowest point so phases alternate."""
    sys_frames = []
    for a, b in zip(dia[:-1], dia[1:]):
        lo = a + int(SYSTOLE_WINDOW[0] * (b - a))
        hi = a + int(np.ceil(SYSTOLE_WINDOW[1] * (b - a))) + 1
        if hi <= lo:
            continue
        valley = _deepest_minimum(gating_signal, lo, hi)
        sys_frames.append(valley if valley is not None else lo + int(np.argmin(gating_signal[lo:hi])))
    return sys_frames


def _refine_one(i: int, image_raw: np.ndarray) -> int:
    lo = max(0, i - REFINE_FRAMES)
    return lo + int(np.argmin(image_raw[lo : i + REFINE_FRAMES + 1]))


def refine_to_raw(frames: list[int], image_raw: np.ndarray, blocked: set[int] | None = None) -> list[int]:
    """Shift each frame to the lowest raw value within REFINE_FRAMES.

    A frame whose shift would land on a blocked frame (e.g. a diastole) stays put.
    """
    blocked = blocked or set()
    out = set()
    for i in frames:
        refined = _refine_one(i, image_raw)
        out.add(i if refined in blocked else refined)
    return sorted(out)


class AutomaticGating:
    def __init__(self, main_window, lower_limit: int = 0) -> None:
        self.main_window = main_window
        self.lower_limit = lower_limit

    def _signal_to_frame_key(self, sig_idx: int) -> int:
        return self.lower_limit + int(sig_idx)

    def automatic_gating(self, image_raw: np.ndarray, gating_signal: np.ndarray, f_heart: float) -> bool:
        """Two-step gating on the image signal (1 - NCC, low = still).

        1. Diastole: one valley per beat after a narrow bandpass at f_heart,
           snapped to the gating signal (the filtered curve shown in the plot).
        2. Systole: deepest valley of the gating signal between two diastoles.
        Both are finally shifted to the lowest raw value within one frame.
        Returns False when the signal yields too few beats.
        """
        if not f_heart or f_heart <= 0:
            logger.warning('Auto-gating: no heart rate available')
            return False
        fs = float(self.main_window.runtime_data.metadata.get('frame_rate', 30))
        step = self.main_window.config.gating.normalize_step

        dia = find_diastole(image_raw, gating_signal, f_heart, fs, step)
        # Refine before the systole search so every D-D interval keeps room for one systole
        dia = _enforce_spacing(refine_to_raw(dia, image_raw), fs / f_heart)
        if len(dia) < 2:
            logger.warning('Auto-gating: too few diastolic valleys - check signal quality')
            return False
        sys_ = refine_to_raw(find_systole(dia, gating_signal), image_raw, blocked=set(dia))
        dia_frames = [self._signal_to_frame_key(i) for i in dia]
        sys_frames = [self._signal_to_frame_key(i) for i in sys_ if i not in set(dia)]
        logger.info(f'Auto-gating at {f_heart * 60:.0f} BPM: {len(dia_frames)} dia, {len(sys_frames)} sys')

        self._apply_gating(dia_frames, sys_frames)
        return True

    def _apply_gating(self, dia_frames: list, sys_frames: list) -> None:
        for fd in self.main_window.runtime_data.frame_data_dct.values():
            fd.phase = '-'
        self.main_window.runtime_data.gated_frames_dia = []
        self.main_window.runtime_data.gated_frames_sys = []
        self.main_window.diastolic_frame_box.setChecked(False)
        self.main_window.systolic_frame_box.setChecked(False)

        self.main_window.runtime_data.gated_frames_dia = sorted(dia_frames)
        self.main_window.runtime_data.gated_frames_sys = sorted(sys_frames)

        for frame in self.main_window.runtime_data.gated_frames_dia:
            if frame in self.main_window.runtime_data.frame_data_dct:
                self.main_window.runtime_data.frame_data_dct[frame].phase = 'D'
        for frame in self.main_window.runtime_data.gated_frames_sys:
            if frame in self.main_window.runtime_data.frame_data_dct:
                self.main_window.runtime_data.frame_data_dct[frame].phase = 'S'

        # Reference for is_auto_gated(), any later manual edit makes the lists differ
        gs = self.main_window.runtime_data.gating_signal
        if gs is not None:
            gs['auto_gating_dia'] = list(self.main_window.runtime_data.gated_frames_dia)
            gs['auto_gating_sys'] = list(self.main_window.runtime_data.gated_frames_sys)

        self.main_window.save_contours_soon()
        logger.info(f'Gating applied: {len(dia_frames)} diastolic, {len(sys_frames)} systolic')


def is_auto_gated(runtime_data) -> bool:
    """True while the current phases are exactly those of the last automatic gating."""
    gs = runtime_data.gating_signal or {}
    if 'auto_gating_dia' not in gs:
        return False
    return sorted(runtime_data.gated_frames_dia) == sorted(gs['auto_gating_dia']) and sorted(
        runtime_data.gated_frames_sys
    ) == sorted(gs.get('auto_gating_sys', []))
