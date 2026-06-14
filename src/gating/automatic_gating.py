import numpy as np
from loguru import logger

from gating.signal_processing import identify_extrema


class AutomaticGating:
    def __init__(self, main_window, report_data, lower_limit: int = 0) -> None:
        self.main_window = main_window
        self.report_data = report_data
        self.lower_limit = lower_limit

    def _sig_to_frame_key(self, sig_idx: int) -> int:
        return self.lower_limit + int(sig_idx)

    # ── public API ──────────────────────────────────────────────────────────

    def automatic_gating(self, image_filtered: np.ndarray, contour_filtered: np.ndarray):
        """Classify frames as systole or diastole.

        Combined path (contour available):
            1. Find PEAKS of image signal — these are maximum-motion frames that
               alternate between mid-systole and mid-diastole.
            2. Classify each peak by the gradient of the area signal:
                 d(area)/dt > 0  →  area increasing  →  diastolic filling    →  diastole
                 d(area)/dt < 0  →  area decreasing  →  systolic compression →  systole
               No alternation needed; gradient signs the phase directly.

        Image-only (no contour):
            Peaks of image signal, alternate into two groups, classify by
            mean amplitude (lower = more stable = diastole).
        """
        # Dynamic minimum peak separation: aim for ~half the inter-peak interval
        # (two peaks per cardiac cycle = mid-sys and mid-dia motion peaks).
        # Without this, at high heart rates (e.g. 162 BPM at 18 fps → 6.7 frames/cycle)
        # the default config x_lim=6 would collapse to one peak per cycle.
        gs = getattr(self.main_window.runtime_data, 'gating_signal', None) or {}
        f_heart = gs.get('f_heart')
        fs = self.main_window.runtime_data.metadata.get('frame_rate', 18)
        if f_heart and f_heart > 0:
            x_lim = max(2, int(0.7 * fs / (2.0 * f_heart)))
        else:
            x_lim = None  # use config default

        _, maxima = identify_extrema(self.main_window, image_filtered, x_lim_override=x_lim)

        if len(maxima) < 2:
            logger.warning('Auto-gating: too few image signal peaks — check extrema config')
            return

        contour_flat = np.all(np.abs(contour_filtered) < 1e-9)

        if not contour_flat:
            dia_frames, sys_frames = self._classify_by_area_gradient(maxima, contour_filtered)
            # If gradient classification collapsed to one phase, fall back to alternating
            if not dia_frames or not sys_frames:
                logger.warning('Gradient sign uniform across all peaks — falling back to alternating')
                dia_frames, sys_frames = self._alternate_by_amplitude(maxima, image_filtered)
        else:
            dia_frames, sys_frames = self._alternate_by_amplitude(maxima, image_filtered)

        if not dia_frames and not sys_frames:
            logger.warning('Auto-gating produced no frames — check signal quality')
            return

        self._apply_gating(dia_frames, sys_frames)

    # ── combined: image peaks + area gradient ─────────────────────────────────

    def _classify_by_area_gradient(self, maxima: np.ndarray, area_filtered: np.ndarray):
        """Classify each image-signal peak by the sign of d(area)/dt.

        Mid-systolic peak  → area is falling  → d(area)/dt < 0 → systole.
        Mid-diastolic peak → area is rising   → d(area)/dt > 0 → diastole.

        np.gradient gives central differences, which are smooth enough on a
        bandpass-filtered signal.
        """
        d_area = np.gradient(area_filtered)
        dia_frames = []
        sys_frames = []
        for i in maxima:
            frame_key = self._sig_to_frame_key(i)
            if d_area[i] >= 0:
                dia_frames.append(frame_key)
            else:
                sys_frames.append(frame_key)

        logger.info(f'Gradient classification: {len(dia_frames)} dia peaks, ' f'{len(sys_frames)} sys peaks')
        return dia_frames, sys_frames

    # ── image-only: alternating peaks ────────────────────────────────────────

    def _alternate_by_amplitude(self, maxima: np.ndarray, image_filtered: np.ndarray):
        """Alternate image signal peaks; lower mean amplitude → diastole."""
        first_sig = maxima[::2].tolist()
        second_sig = maxima[1::2].tolist()

        first_frames = [self._sig_to_frame_key(i) for i in first_sig]
        second_frames = [self._sig_to_frame_key(i) for i in second_sig]

        N = len(image_filtered)
        amp_f = self._mean_amp(image_filtered, first_frames, N)
        amp_s = self._mean_amp(image_filtered, second_frames, N)

        if amp_f <= amp_s:
            dia_frames, sys_frames = first_frames, second_frames
        else:
            dia_frames, sys_frames = second_frames, first_frames

        logger.info(
            f'Image-only gating: {len(dia_frames)} dia, {len(sys_frames)} sys '
            f'(amp_dia={min(amp_f,amp_s):.3f}, amp_sys={max(amp_f,amp_s):.3f})'
        )
        return dia_frames, sys_frames

    def _mean_amp(self, signal: np.ndarray, frame_keys: list, N: int) -> float:
        idxs = [k - self.lower_limit for k in frame_keys if 0 <= k - self.lower_limit < N]
        return float(np.mean([signal[i] for i in idxs])) if idxs else 0.0

    # ── apply ────────────────────────────────────────────────────────────────

    def _apply_gating(self, dia_frames: list, sys_frames: list):
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

        logger.info(f'Gating applied: {len(dia_frames)} diastolic, {len(sys_frames)} systolic')
