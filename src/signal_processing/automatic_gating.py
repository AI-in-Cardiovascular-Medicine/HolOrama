import numpy as np
from loguru import logger

# Beat tracker, tuned on 10 NARCO pullbacks with verified labels (see research/gating_validation)
NORMAL_GAP = (0.7, 1.35)  # D-D gap of a regular beat, in beats
SKIP_GAP = (1.55, 2.3)  # gap of a skipped beat (extrasystole or missed beat), in beats
GAP_WEIGHT = 5.0  # penalty for a gap away from one beat
TEMPO_WEIGHT = 10.0  # penalty for a change between consecutive gaps
SKIP_PENALTY = 6.0  # cost of skipping a beat
SYSTOLE_MARGIN = 0.35  # minimum distance of systole from both diastoles, in beats
DELAY_WEIGHT = 64.0  # pull of systole towards the recording's median D->S delay


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


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Z-score from median and interquartile range, so a few extreme valleys do not dominate."""
    x = np.asarray(x, dtype=float)
    scale = np.subtract(*np.percentile(x, [75, 25])) / 1.349
    if scale <= 1e-9:
        scale = x.std() or 1.0
    return (x - np.median(x)) / scale


def track_diastole(gating_signal: np.ndarray, beat: float) -> tuple[list[int], list[tuple[int, int]]]:
    """Chain of diastoles among the valleys of the gating signal, chosen as a whole.

    Deeper valleys score higher (the heart is stillest in diastole). Steps
    between diastoles should stay near one beat and change gradually. A step
    of about two beats skips an irregular or missed beat at SKIP_PENALTY.
    The best chain is found by dynamic programming over pairs of consecutive
    diastoles. Where no step can bridge to the next valleys (e.g. a long
    stretch without motion), a new chain starts after the gap, which is
    flagged like a skipped beat. Returns the diastoles and the skipped
    (start, end) intervals.
    """
    g = gating_signal
    cand = np.array(_local_minima(g, 1, len(g) - 1))
    if len(cand) < 2:
        return [int(c) for c in cand], []
    node = _robust_z(-g[cand])

    dia: list[int] = []
    joins: list[tuple[int, int]] = []
    first = 0
    while first < len(cand) - 1:
        chain = _best_chain(cand, node, beat, first)
        if len(chain) < 2:
            first += 1
            continue
        if dia:
            joins.append((dia[-1], int(cand[chain[0]])))
        dia += [int(cand[k]) for k in chain]
        first = int(np.searchsorted(cand, dia[-1] + NORMAL_GAP[0] * beat))
    if not dia:
        return [int(cand[np.argmin(g[cand])])], []
    skipped = [(a, b) for a, b in zip(dia[:-1], dia[1:]) if (b - a) / beat >= SKIP_GAP[0] or (a, b) in joins]
    return dia, skipped


def _best_chain(cand: np.ndarray, node: np.ndarray, beat: float, first: int) -> list[int]:
    """Best chain (candidate indices) starting within 1.5 beats of cand[first].

    Prefers chains ending within 1.5 beats of the last candidate. If none
    gets there, returns the chain reaching furthest so the caller can
    continue after it.
    """
    steps_by_end: dict[int, list[tuple[int, bool, float]]] = {}
    for i in range(first, len(cand)):
        for j in range(i + 1, len(cand)):
            gap = (cand[j] - cand[i]) / beat
            if gap > SKIP_GAP[1]:
                break
            if NORMAL_GAP[0] <= gap <= NORMAL_GAP[1]:
                steps_by_end.setdefault(j, []).append((i, False, gap))
            elif gap >= SKIP_GAP[0]:
                steps_by_end.setdefault(j, []).append((i, True, gap))

    start_ok = cand <= cand[first] + 1.5 * beat
    best: dict[tuple[int, int], float] = {}
    back: dict[tuple[int, int], tuple[int, int] | None] = {}
    for j in range(first, len(cand)):
        for i, skip, gap in steps_by_end.get(j, []):
            gain = node[j] - (SKIP_PENALTY if skip else GAP_WEIGHT * (gap - 1) ** 2)
            options: list[tuple[float, tuple[int, int] | None]] = [(node[i] + gain, None)] if start_ok[i] else []
            for h, skip_h, gap_h in steps_by_end.get(i, []):
                if (h, i) in best:
                    tempo = 0.0 if skip or skip_h else TEMPO_WEIGHT * (gap - gap_h) ** 2
                    options.append((best[(h, i)] + gain - tempo, (h, i)))
            if options:
                best[(i, j)], back[(i, j)] = max(options, key=lambda o: o[0])
    if not best:
        return []

    end_states = [s for s in best if cand[s[1]] >= cand[-1] - 1.5 * beat]
    if end_states:
        last = max(end_states, key=lambda s: best[s])
    else:
        last = max(best, key=lambda s: (s[1], best[s]))
    chain = [last[1]]
    state: tuple[int, int] | None = last
    while state is not None:
        chain.append(state[0])
        state = back[state]
    return chain[::-1]


def place_systole(dia: list[int], skipped: list[tuple[int, int]], gating_signal: np.ndarray, beat: float) -> list[int]:
    """One systole per regular beat, near the recording's median D->S delay.

    Systole keeps SYSTOLE_MARGIN beats from both diastoles (at most 45 % of a
    short interval). The delay is learned from the deepest valley of each beat
    first, then every beat takes the valley balancing depth and that delay.
    Skipped beats get no systole.
    """
    g = gating_signal
    windows = []
    for a, b in zip(dia[:-1], dia[1:]):
        if (a, b) in skipped or b - a < 3:
            continue
        margin = max(1, min(int(np.ceil(SYSTOLE_MARGIN * beat)), int(0.45 * (b - a))))
        windows.append((a, a + margin, b - margin))

    first = []
    for a, lo, hi in windows:
        valleys = _local_minima(g, lo, hi + 1)
        if valleys:
            first.append(min(valleys, key=lambda i: g[i]) - a)
    delay = float(np.median(first)) if first else 0.45 * beat

    sys_frames = []
    for a, lo, hi in windows:
        valleys = _local_minima(g, lo, hi + 1) or list(range(lo, hi + 1))
        costs = [g[i] + DELAY_WEIGHT * ((i - a - delay) / beat) ** 2 for i in valleys]
        sys_frames.append(valleys[int(np.argmin(costs))])
    return sys_frames


class AutomaticGating:
    def __init__(self, main_window, lower_limit: int = 0) -> None:
        self.main_window = main_window
        self.lower_limit = lower_limit

    def _signal_to_frame_key(self, sig_idx: int) -> int:
        return self.lower_limit + int(sig_idx)

    def automatic_gating(self, gating_signal: np.ndarray, f_heart: float) -> bool:
        """Gate diastole and systole on the image gating signal (1 - NCC, low = still).

        1. Diastole: the chain of deep valleys with a steady beat (track_diastole).
        2. Systole: one valley per beat near the learned D->S delay (place_systole).
        Skipped beats are stored in gating_signal['gating_skipped'] for review.
        Returns False when the signal yields too few beats.
        """
        if not f_heart or f_heart <= 0:
            logger.warning('Auto-gating: no heart rate available')
            return False
        fs = float(self.main_window.runtime_data.metadata.get('frame_rate', 30))
        beat = fs / f_heart

        dia, skipped = track_diastole(gating_signal, beat)
        if len(dia) < 2:
            logger.warning('Auto-gating: too few diastolic valleys - check signal quality')
            return False
        sys_ = place_systole(dia, skipped, gating_signal, beat)

        dia_frames = [self._signal_to_frame_key(i) for i in dia]
        sys_frames = [self._signal_to_frame_key(i) for i in sys_]
        logger.info(
            f'Auto-gating at {f_heart * 60:.0f} BPM: {len(dia_frames)} dia, {len(sys_frames)} sys, '
            f'{len(skipped)} skipped beats'
        )
        self._apply_gating(dia_frames, sys_frames)
        gs = self.main_window.runtime_data.gating_signal
        if gs is not None:
            gs['gating_skipped'] = [[self._signal_to_frame_key(a), self._signal_to_frame_key(b)] for a, b in skipped]
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
