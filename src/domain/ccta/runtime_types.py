from __future__ import annotations

from typing import Any

import numpy as np

from domain.ccta.io_types import VolumeGeometry
from domain.undo import UndoStack


class CctaRuntimeData:
    def __init__(self):
        self.metadata: dict = {}
        self.volume: np.ndarray | None = None  # (Z, Y, X) int16 HU
        self.voxel_spacing: tuple[float, float, float] | None = None  # (dz, dy, dx) mm
        self.geometry: VolumeGeometry | None = None  # source file's voxel grid, for writing masks back
        self.mask: np.ndarray | None = None  # (Z, Y, X) uint8 label values
        self.labels: list[int] = []  # non-background labels present in mask
        self.mask_undo: UndoStack = UndoStack()  # last 5 full-mask snapshots, for Ctrl+Z

        # -- Post-cut geometry (Build Cut Geometry / Smooth / Calculate Centerlines) ----
        self.cut_mesh: Any | None = None  # trimesh.Trimesh built from the LVOT/aorta-top cut
        self.cut_mesh_inlet: np.ndarray | None = None  # world (x, y, z) mm — lower-Z cut plane centroid
        self.cut_mesh_outlet: np.ndarray | None = None  # world (x, y, z) mm — higher-Z cut plane centroid
        # Previous (cut_mesh vertices, inlet, outlet) snapshots (oldest first) for undoing
        # Smooth / Sphere Smooth steps — smoothing never changes faces, so vertices alone
        # restore a mesh. Reset whenever the topology changes (Build Cut Geometry, Reduce
        # Mesh, Remesh).
        self.cut_mesh_undo: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
