from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

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
    quality: str = 'Very Good'
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


class ProcessableModality(Enum):
    OCT = "oct",
    IVUS = "ivus",
    NIRS = "nirs",

@dataclass
class MetaData:
    """All necessary metadata for the programm to run with fall-backs to calculate missing values."""

    patient_name = str,
