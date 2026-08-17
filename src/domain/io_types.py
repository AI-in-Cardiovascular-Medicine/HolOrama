from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

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


def wire_points(wire: Contour, index: int) -> List[Tuple[float, float]]:
    """The (x, y) angle points of wire `index`, or [] if that wire does not exist."""
    if index < 0 or index >= len(wire.contours):
        return []
    entry = wire.contours[index]
    xs = entry[0] if entry else []
    ys = entry[1] if len(entry) > 1 else []
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def iter_wires(wire) -> List[List[Tuple[float, float]]]:
    """Every wire on a frame, each as its list of (x, y) angle points.

    Also accepts the pre-multi-wire shape (a single ((x, y), ...) tuple), so data
    that has not been through the loader's migration still reads correctly.
    """
    if wire is None:
        return []
    if isinstance(wire, Contour):
        return [pts for pts in (wire_points(wire, i) for i in range(len(wire.contours))) if pts]
    legacy = [(float(p[0]), float(p[1])) for p in wire if p is not None and len(p) >= 2]
    return [legacy] if legacy else []


def set_wire_points(wire: Contour, index: int, points: Sequence[Tuple[float, float]]) -> None:
    """Write `points` as wire `index`, growing the wire list and its aligned
    per-contour lists as needed."""
    while len(wire.contours) <= index:
        wire.contours.append(([], []))
    while len(wire.closed) <= index:
        wire.closed.append(False)
    while len(wire.start_coords) <= index:
        wire.start_coords.append([])
    while len(wire.end_coords) <= index:
        wire.end_coords.append([])
    wire.contours[index] = ([float(p[0]) for p in points], [float(p[1]) for p in points])


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
    # Guide wires are stored like any other multi-instance contour (calcium, lipid, ...):
    # one entry in Contour.contours per wire, holding that wire's 1-2 angle points as
    # ([x, ...], [y, ...]) — the radial lines bounding its shadow. A frame can carry
    # several wires. Read/write via iter_wires / wire_points / set_wire_points.
    wire: Contour = field(default_factory=Contour)
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


@dataclass
class MetaDataFusion:
    modality: str = 'Fusion'
    patient_name: str = 'Unknown'
    birthdate: str = 'Unknown'
    sex: str = 'Unknown'
    ...
