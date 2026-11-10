"""Inputs and settings of Calculate Centerlines (src/centerlines)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import trimesh


@dataclass(frozen=True)
class SmoothingParams:
    """vmtkcenterlinesmoothing's two parameters, a moving-average filter over the centerline
    points. More iterations or a higher factor smooth more."""

    iterations: int
    factor: float


# The aorta usually needs much heavier smoothing than the finer coronary branches.
AORTA_SMOOTHING = SmoothingParams(iterations=300, factor=0.1)
CORONARY_SMOOTHING = SmoothingParams(iterations=100, factor=0.1)


@dataclass(frozen=True)
class VmtkEnv:
    """Where the pixi-managed vmtk environment lives and which pixi manages it."""

    env_dir: Path
    pixi_path: str = ''  # explicit pixi executable from config.yaml, '' = auto

    @property
    def manifest(self) -> Path:
        return self.env_dir / 'pixi.toml'


@dataclass(frozen=True)
class CenterlineInputs:
    """What Calculate Centerlines runs on: the cut mesh with its aorta inlet and outlet, the
    RCA/LCA outlet points, and where the results go (<out_dir>/<label>_cl.vtp)."""

    cut_mesh: trimesh.Trimesh
    inlet: np.ndarray
    outlet: np.ndarray
    rca_points: list[np.ndarray]
    lca_points: list[np.ndarray]
    out_dir: str
    stem: str  # case file name, prefix of the exported surface
