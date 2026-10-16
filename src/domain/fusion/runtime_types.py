from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CutSpec:
    """One flat cut of the final geometry at a slice of the discretized vessel tree (see
    multimodars' cut_at_slice). Only the slice's plane comes from the tree, so a spec holds
    on any mesh the tree fits (CCTA, stitched, remeshed, smoothed) and the cut geometry is
    always these specs replayed in order on the chosen source mesh."""

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
    """Holds the multimodars objects produced while working through the fusion pipeline.

    Fields are grouped by the right-half column that produces them: geometry/centerline
    labeling (column 1), intravascular alignment (column 2), fusion/scaling/stitching
    (column 3). Nothing here is serialized as-is -> each stage writes its own output file
    (STL, VTP, JSON) via pipeline.py, and this container just keeps the in-memory objects
    needed to feed the next stage and to redraw the 3D viewer.
    """

    def __init__(self):
        self.case_dir: str | None = None  # last directory used for file/save dialogs

        # -- Column 1: CCTA geometry + centerlines --------------------------------------
        self.centerline_aorta: Any | None = None  # PyCenterline
        self.centerline_rca: Any | None = None
        self.centerline_lca: Any | None = None
        self.results: dict | None = None  # multimodars "results" dict (mesh, *_points, ...)
        self.vessel_tree: Any | None = None  # PyDiscretizedVesselTree
        # Per vessel ('rca'/'lca'): selected branch (0 = main, n = side branch n) and the
        # index into that branch's reference list (see FusionPage._branch_references).
        self.selected_branch: dict[str, int] = {'rca': 0, 'lca': 0}
        self.selected_reference_index: dict[str, int] = {'rca': 0, 'lca': 0}
        # User-given branch names per vessel, by branch id (missing = default 'RCA branch 2').
        # One name for a branch's centerline and its labelled vertices. Branch ids survive
        # remove_branch but not split/merge, which reset that vessel's names.
        self.branch_names: dict[str, dict[int, str]] = {'rca': {}, 'lca': {}}

        # -- Column 2: intravascular alignment -------------------------------------------
        self.iv_geometry_pair: Any | None = None  # PyGeometryPair from from_file_singlepair
        self.iv_align_logs: tuple | None = None
        self.aligned: Any | None = None  # PyGeometryPair | PyGeometry from align_combined
        self.aligned_centerline: Any | None = None  # single-branch rca/lca centerline passed into the alignment
        # The whole prepared vessel (all branches) the pullback was aligned onto, which one
        # ('rca'/'lca') and which branch id — label_anomalous_region needs every branch plus
        # the pullback branch id, not just aligned_centerline.
        self.aligned_vessel: str | None = None
        self.aligned_branch: int = 0
        self.aligned_vessel_centerline: Any | None = None

        # -- Column 3: fusion / scaling / stitching --------------------------------------
        self.prox_scaling: float | None = None
        self.distal_scaling: float | None = None
        self.aortic_scaling: float | None = None
        # Deep-copied snapshot of `results` right after Remove Labeled Points. Stitching
        # always starts from a fresh copy of this (stitch_ccta_to_intravascular mutates the
        # results dict + mesh in place), so Stitch can be re-run with different parameters.
        self.results_points_removed: dict | None = None
        # Deep copies of `results` from just before each Remove Labeled Points (oldest
        # first), for undoing removals. Cleared whenever `results` is replaced by anything
        # else (relabel, scaling, branch delete/relabel), which would make them stale.
        self.removal_undo: list[dict] = []
        self.stitched: dict | None = None  # result of stitch_ccta_to_intravascular
        self.final_mesh: Any | None = None  # trimesh.Trimesh after remesh/smoothing
        # Previous final_mesh vertex arrays (oldest first) for undoing smoothing steps
        # smoothing never changes faces, so vertices alone restore a mesh. Reset whenever a
        # new final_mesh topology arrives (Fix & Remesh) or the final mesh is dropped.
        self.final_mesh_undo: list[np.ndarray] = []

        # -- Inlets & outlets: flat cuts of the last geometry ----------------------------
        self.cuts: list[CutSpec] = []
        self.cut_source: str | None = None  # 'final' / 'stitched' / 'ccta', None = latest there is
        self.cut_selection: tuple[str, int, int] | None = None  # (vessel, branch, slice) picked
        self.cut_result: CutResult | None = None  # cache, recomputed when its key goes stale
