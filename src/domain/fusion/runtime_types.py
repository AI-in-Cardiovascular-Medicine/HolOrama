from __future__ import annotations

from typing import Any

import numpy as np


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

        # -- Column 2: intravascular alignment -------------------------------------------
        self.iv_geometry_pair: Any | None = None  # PyGeometryPair from from_file_singlepair
        self.iv_align_logs: tuple | None = None
        self.aligned: Any | None = None  # PyGeometryPair | PyGeometry from align_combined
        self.aligned_centerline: Any | None = None  # single-branch rca/lca centerline passed into the alignment

        # -- Column 3: fusion / scaling / stitching --------------------------------------
        self.prox_scaling: float | None = None
        self.distal_scaling: float | None = None
        self.aortic_scaling: float | None = None
        # Deep-copied snapshot of `results` right after Remove Labeled Points. Stitching
        # always starts from a fresh copy of this (stitch_ccta_to_intravascular mutates the
        # results dict + mesh in place), so Stitch can be re-run with different parameters.
        self.results_points_removed: dict | None = None
        self.stitched: dict | None = None  # result of stitch_ccta_to_intravascular
        self.final_mesh: Any | None = None  # trimesh.Trimesh after remesh/smoothing
        # Previous final_mesh vertex arrays (oldest first) for undoing smoothing steps
        # smoothing never changes faces, so vertices alone restore a mesh. Reset whenever a
        # new final_mesh topology arrives (Fix & Remesh) or the final mesh is dropped.
        self.final_mesh_undo: list[np.ndarray] = []
