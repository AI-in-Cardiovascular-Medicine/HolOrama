from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from domain.all_types import ContourType


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


def sector_points(contour: Contour, index: int) -> List[Tuple[float, float]]:
    """The (x, y) angle points of sector `index`, or [] if that sector does not exist.

    See ANGLE_TYPES in domain.all_types for what a sector is and tools.angle for how its
    points describe it.
    """
    if index < 0 or index >= len(contour.contours):
        return []
    entry = contour.contours[index]
    xs = entry[0] if entry else []
    ys = entry[1] if len(entry) > 1 else []
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def iter_sectors(contour) -> List[List[Tuple[float, float]]]:
    """Every angular sector on a frame, each as its list of (x, y) angle points.

    Also accepts the pre-multi-wire shape (a single ((x, y), ...) tuple), so data
    that has not been through the loader's migration still reads correctly.
    """
    if contour is None:
        return []
    if isinstance(contour, Contour):
        return [pts for pts in (sector_points(contour, i) for i in range(len(contour.contours))) if pts]
    legacy = [(float(p[0]), float(p[1])) for p in contour if p is not None and len(p) >= 2]
    return [legacy] if legacy else []


def set_sector_points(contour: Contour, index: int, points: Sequence[Tuple[float, float]]) -> None:
    """Write `points` as sector `index`, growing the sector list and its aligned
    per-contour lists as needed."""
    while len(contour.contours) <= index:
        contour.contours.append(([], []))
    while len(contour.closed) <= index:
        contour.closed.append(False)
    while len(contour.start_coords) <= index:
        contour.start_coords.append([])
    while len(contour.end_coords) <= index:
        contour.end_coords.append([])
    contour.contours[index] = ([float(p[0]) for p in points], [float(p[1]) for p in points])


@dataclass
class Measure:
    points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    length: Optional[float] = None


@dataclass
class FrameData:
    phase: str = '-'
    # OCT frame rating: one of OCT_QUALITY_LABELS, or '' while the frame is unrated.
    quality: str = ''
    guiding_catheter: bool = False
    unanalyzable: bool = False
    # Mutually exclusive with `quality`: a frame is unlabeled until it gets a rating.
    unlabeled: bool = True
    lumen: Contour = field(default_factory=Contour)
    eem: Contour = field(default_factory=Contour)
    calcium: Contour = field(default_factory=Contour)
    branch: Contour = field(default_factory=Contour)
    lipid: Contour = field(default_factory=Contour)
    macrophage: Contour = field(default_factory=Contour)
    measurement_1: Optional[Measure] = None
    measurement_2: Optional[Measure] = None
    reference: Optional[Tuple[float, float]] = None
    # Angular sectors (ANGLE_TYPES) are stored like any other multi-instance contour
    # (calcium, lipid, ...): one entry in Contour.contours per sector, holding that
    # sector's 2-3 angle points as ([x, ...], [y, ...]) — the radial lines bounding it,
    # plus the interior marker that says which of the two arcs between them is meant
    # (see tools.angle). A frame can carry several of each. Read/write via
    # iter_sectors / sector_points / set_sector_points.
    wire: Contour = field(default_factory=Contour)  # guide-wire shadow
    blood: Contour = field(default_factory=Contour)  # blood artefact sector
    centroid: Optional[Tuple[float, float]] = None
    closest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    farthest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


# Everything the user can draw on one image: every contour type (which includes both
# measurements, the reference point and the angular sectors) plus the values derived from
# the lumen. Not the phase or the OCT label — those describe the frame, not the drawing.
FRAME_ANNOTATION_FIELDS = tuple(contour_type.value for contour_type in ContourType) + (
    'centroid',
    'closest_points',
    'farthest_points',
)


def clear_frame_annotations(frame_data: FrameData) -> None:
    """Reset every annotation on `frame_data` to the state of a frame nobody has touched."""
    blank = FrameData()  # one fresh instance hands out every default, contours included
    for field_name in FRAME_ANNOTATION_FIELDS:
        setattr(frame_data, field_name, getattr(blank, field_name))


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
