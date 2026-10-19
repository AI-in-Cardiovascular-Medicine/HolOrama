from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CutSpec:
    """One flat cut at a vessel-tree slice (multimodars cut_at_slice). Only the slice plane
    comes from the tree, so a spec applies to any mesh the tree fits."""

    vessel: str  # 'aorta' / 'rca' / 'lca'
    branch: int  # branch id of that vessel's centerline, 0 for the aorta
    slice_idx: int  # index into that branch's tree contours
    keep: str  # 'larger' / 'proximal' / 'distal'
    name: str  # user-facing, also names the cap's export file

    @property
    def geometry(self) -> tuple[str, int, int, str]:
        """Everything but the name: what the cut result depends on."""
        return self.vessel, self.branch, self.slice_idx, self.keep


@dataclass
class CutResult:
    """The specs replayed on a source mesh (see pages/fusion/cutting.apply_cuts)."""

    mesh: Any  # trimesh.Trimesh, watertight with the caps in it
    centerlines: dict[str, Any]  # vessel -> PyCenterline, cut where a cut hit that vessel
    caps: dict[int, Any] = field(default_factory=dict)  # spec index -> cap trimesh
    dropped: set[tuple[str, int, int]] = field(default_factory=set)  # (vessel, branch, slice) cut off
    failed: dict[int, str] = field(default_factory=dict)  # spec index -> why it was skipped
    key: tuple = ()  # source mesh + specs this was computed from, see CutController


class FusionRuntimeData:
    """In-memory multimodars objects of the fusion pipeline, grouped by the right-half
    column that produces them."""

    def __init__(self):
        self.case_dir: str | None = None  # last directory used for file/save dialogs

        # -- Column 1: CCTA geometry + centerlines --------------------------------------
        self.centerline_aorta: Any | None = None  # PyCenterline
        self.centerline_rca: Any | None = None
        self.centerline_lca: Any | None = None
        self.results: dict | None = None  # multimodars "results" dict (mesh, *_points, ...)
        self.vessel_tree: Any | None = None  # PyDiscretizedVesselTree
        # Per vessel: selected branch (0 = main) and index into its reference list.
        self.selected_branch: dict[str, int] = {'rca': 0, 'lca': 0}
        self.selected_reference_index: dict[str, int] = {'rca': 0, 'lca': 0}
        # User-given branch names per vessel, by branch id (missing = default name).
        # Split/merge reassign branch ids and reset that vessel's names.
        self.branch_names: dict[str, dict[int, str]] = {'rca': {}, 'lca': {}}

        # -- Column 2: intravascular alignment -------------------------------------------
        self.iv_geometry_pair: Any | None = None  # loaded pullback (PyGeometryPair or PyGeometry)
        self.iv_align_logs: tuple | None = None
        self.aligned: Any | None = None  # PyGeometryPair | PyGeometry from align_combined
        self.aligned_centerline: Any | None = None  # single-branch rca/lca centerline passed into the alignment
        # The full vessel centerline (all branches) and branch id the pullback was aligned
        # onto. label_anomalous_region needs every branch, not just aligned_centerline.
        self.aligned_vessel: str | None = None
        self.aligned_branch: int = 0
        self.aligned_vessel_centerline: Any | None = None

        # -- Column 3: fusion / scaling / stitching --------------------------------------
        self.prox_scaling: float | None = None
        self.distal_scaling: float | None = None
        self.aortic_scaling: float | None = None
        # Snapshot of `results` after Remove Labeled Points. Stitch works on a fresh copy
        # (stitch_ccta_to_intravascular mutates in place), so it can be re-run.
        self.results_points_removed: dict | None = None
        # Copies of `results` before each Remove Labeled Points (oldest first), for undo.
        # Cleared whenever `results` is replaced otherwise, since they would be stale.
        self.removal_undo: list[dict] = []
        self.stitched: dict | None = None  # result of stitch_ccta_to_intravascular
        self.final_mesh: Any | None = None  # trimesh.Trimesh after remesh/smoothing
        # Previous final_mesh vertices (oldest first) for undoing smoothing, which keeps faces.
        # Reset on a new topology (Fix and Remesh) or when the final mesh is dropped.
        self.final_mesh_undo: list[np.ndarray] = []

        # -- Inlets & outlets: flat cuts of the last geometry ----------------------------
        self.cuts: list[CutSpec] = []
        self.cut_source: str | None = None  # 'final' / 'stitched' / 'ccta', None = latest there is
        self.cut_selection: tuple[str, int, int] | None = None  # (vessel, branch, slice) picked
        self.cut_result: CutResult | None = None  # cache, recomputed when its key goes stale
