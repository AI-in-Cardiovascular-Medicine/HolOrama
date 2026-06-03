from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from domain.all_types import OCT_QUALITY_LABELS


@dataclass
class Measurements:
    area: Optional[float] = None
    circumference: Optional[float] = None
    major_axis: Optional[float] = None
    minor_axis: Optional[float] = None
    elliptic_ratio: Optional[float] = None


@dataclass
class Contour:
    contours: List[Tuple[List[float], List[float]]] = field(default_factory=list)
    measurements: Measurements = field(default_factory=Measurements)
    closed: List[bool] = field(default_factory=list)
    # Each entry is a list of (x, y) tuples for that contour index.
    # Open splines: always [(first_x, first_y)] / [(last_x, last_y)] (auto-set).
    # Closed splines: [] initially, grows as user labels knot points.
    start_coords: List[List[Tuple[float, float]]] = field(default_factory=list)
    end_coords: List[List[Tuple[float, float]]] = field(default_factory=list)


@dataclass
class Measure:
    points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    length: Optional[float] = None


@dataclass
class FrameData:
    phase: str = '-'
    quality: str = OCT_QUALITY_LABELS[-1]
    lumen: Contour = field(default_factory=Contour)
    eem: Contour = field(default_factory=Contour)
    calcium: Contour = field(default_factory=Contour)
    branch: Contour = field(default_factory=Contour)
    lipid: Contour = field(default_factory=Contour)
    macrophage: Contour = field(default_factory=Contour)
    measurement_1: Optional[Measure] = None
    measurement_2: Optional[Measure] = None
    reference: Optional[Tuple[float, float]] = None
    wire: Optional[Tuple[Tuple[float, float], ...]] = None
    centroid: Optional[Tuple[float, float]] = None
    closest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    farthest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


@dataclass
class MetaDataIntravascular:
    modality: Optional[str] = None
    patient_name: str = 'Unknown'
    birthdate: str = 'Unknown'
    sex: str = 'Unknown'
    pullback_speed: Optional[float] = None
    pullback_length: Optional[float | np.ndarray] = None
    resolution: Optional[float] = None
    dimension: Optional[int] = None
    manufacturer: str = 'Unknown'
    model: str = 'Unknown'
    pullback_start_frame: Optional[int] = None
    frame_rate: Optional[float] = None
    ...


@dataclass
class MetaDataCCTA:
    modality: str = 'CCTA'
    patient_name: str = 'Unknown'
    birthdate: str = 'Unknown'
    sex: str = 'Unknown'
    slice_thickness: float = 0.0
    pixel_spacing: Tuple[float, float] = (0.0, 0.0)
    manufacturer: str = 'Unknown'
    model: str = 'Unknown'
    raw_tags: dict = field(default_factory=dict)  # all remaining DICOM / NIfTI tags
    ...
