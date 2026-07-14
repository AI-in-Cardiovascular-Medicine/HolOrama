from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from domain.io_types import FrameData
from domain.undo import UndoStack


class CctaRuntimeData:
    def __init__(self):
        self.metadata: dict = {}
        self.volume: np.ndarray | None = None  # (Z, Y, X) int16 HU
        self.voxel_spacing: tuple[float, float, float] | None = None  # (dz, dy, dx) mm
        self.mask: np.ndarray | None = None  # (Z, Y, X) uint8 label values
        self.labels: list[int] = []  # non-background labels present in mask
        self.mask_undo: UndoStack = UndoStack()  # last 5 full-mask snapshots, for Ctrl+Z


class FusionRuntimeData:
    """Holds the multimodars objects produced while working through the fusion pipeline.

    Fields are grouped by the right-half column that produces them: geometry/centerline
    labeling (column 1), intravascular alignment (column 2), fusion/scaling/stitching
    (column 3). Nothing here is serialized as-is -> each stage writes its own output file
    (STL, VTP, JSON) via pipeline.py, and this container just keeps the in-memory objects
    needed to feed the next stage and to redraw the 3-D viewer.
    """

    def __init__(self):
        self.case_dir: str | None = None  # last directory used for file/save dialogs

        # -- Column 1: CCTA geometry + centerlines --------------------------------------
        self.centerline_aorta: Any | None = None  # PyCenterline
        self.centerline_rca: Any | None = None
        self.centerline_lca: Any | None = None
        self.results: dict | None = None  # multimodars "results" dict (mesh, *_points, ...)
        self.vessel_tree: Any | None = None  # PyDiscretizedVesselTree
        self.selected_rca_reference_index: int = 0  # index into vessel_tree.rca_references

        # -- Column 2: intravascular alignment -------------------------------------------
        self.iv_geometry_pair: Any | None = None  # PyGeometryPair from from_file_singlepair
        self.iv_align_logs: tuple | None = None
        self.aligned: Any | None = None  # PyGeometryPair | PyGeometry from align_combined
        self.resampled_centerline: Any | None = None

        # -- Column 3: fusion / scaling / stitching --------------------------------------
        self.prox_scaling: float | None = None
        self.distal_scaling: float | None = None
        self.aortic_scaling: float | None = None
        self.aortic_wall_scaling: float | None = None
        self.stitched: dict | None = None  # result of stitch_ccta_to_intravascular
        self.final_mesh: Any | None = None  # trimesh.Trimesh after remesh/smoothing


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
