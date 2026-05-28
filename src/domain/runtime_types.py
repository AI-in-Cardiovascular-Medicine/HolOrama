from __future__ import annotations
from typing import Any

import numpy as np

from domain.io_types import FrameData


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
        self.tmp_contours: dict[str, tuple[list[float], list[float]]] = {}
        self.gating_signal: dict[str, Any] = {}
