from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from domain.intravascular.io_types import FrameData
from domain.undo import UndoStack


class GatingSignal(TypedDict, total=False):
    image_based_gating: list[float]
    contour_based_gating: list[float]
    image_based_gating_filtered: list[float]
    contour_based_gating_filtered: list[float]
    gating_config: dict[str, Any]
    f_heart: float
    f_heart_bpm: float
    freq_sweep_bpm_cuts: list[float]
    freq_sweep_signals: list[float]
    f_resp: float
    f_resp_override: float
    breathing_cache_signature: dict[str, Any]
    breathing_cache_result: dict[str, Any]
    breathing_residual: list[float]
    breathing_frames: list[int]
    breathing_display_signal: list[float]
    breathing_phase: list[float]
    breathing_auto_peaks: list[int]
    breathing_auto_valleys: list[int]
    breathing_manual_mode: bool
    breathing_manual_peaks: list[int]
    breathing_manual_valleys: list[int]
    has_breathing_artefact: bool
    sort_signature: dict[str, Any]
    sort_peaks: list[int]
    sort_valleys: list[int]
    sort_n_bins: int
    sort_dia_order: list[int]
    sort_sys_order: list[int]
    sort_dia_pos: list[list[float]]
    sort_sys_pos: list[list[float]]
    sort_dia_shifts: list[float]
    sort_sys_shifts: list[float]


class RuntimeData:
    def __init__(self):
        self.frame_data_dct: dict[int, FrameData] | None = None
        self.metadata: dict = {}
        self.images: np.ndarray | None = None
        self.images_rgb: np.ndarray | None = None
        self.gated_frames: list[int] = []
        self.gated_frames_dia: list[int] = []
        self.gated_frames_sys: list[int] = []
        self.tagged_frames: list[int] = []
        self.contour_undo: UndoStack = UndoStack()  # last 5 contour-edit snapshots, for Ctrl+Z
        self.gating_signal: GatingSignal = {}
        # Set by every operation that edits frame data; the page saves shortly after and
        # clears it again (see IntravascularPage.save_contours_soon).
        self.unsaved_changes: bool = False

    def mark_unsaved(self) -> None:
        """Flag the frame data as edited, so the pending-changes save picks it up.

        Call this from anything that writes to frame_data_dct — contours, phases, quality
        labels, measurements — including batch operations that touch frames the user is
        not looking at.
        """
        self.unsaved_changes = True
