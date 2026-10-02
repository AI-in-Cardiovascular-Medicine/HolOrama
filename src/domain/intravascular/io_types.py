from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from domain.intravascular.types import ContourType


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
    # Per contour: an open spline's first/last point, or the knots the user labels on a closed one
    start_coords: List[List[Tuple[float, float]]] = field(default_factory=list)
    end_coords: List[List[Tuple[float, float]]] = field(default_factory=list)


def sector_points(contour: Contour, index: int) -> List[Tuple[float, float]]:
    """The (x, y) angle points of sector `index`, or [] if it does not exist (see tools.angle)."""
    if index < 0 or index >= len(contour.contours):
        return []
    entry = contour.contours[index]
    xs = entry[0] if entry else []
    ys = entry[1] if len(entry) > 1 else []
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def iter_sectors(contour) -> List[List[Tuple[float, float]]]:
    """Every angular sector on a frame as its (x, y) points; also reads the legacy single-tuple shape."""
    if contour is None:
        return []
    if isinstance(contour, Contour):
        return [pts for pts in (sector_points(contour, i) for i in range(len(contour.contours))) if pts]
    legacy = [(float(p[0]), float(p[1])) for p in contour if p is not None and len(p) >= 2]
    return [legacy] if legacy else []


def set_sector_points(contour: Contour, index: int, points: Sequence[Tuple[float, float]]) -> None:
    """Write `points` as sector `index`, growing the per-contour lists as needed."""
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
    quality: str = ''  # OCT rating (one of OCT_QUALITY_LABELS), '' while unrated
    guiding_catheter: bool = False
    unanalyzable: bool = False
    unlabeled: bool = True  # until the frame gets a quality rating
    # By type id, for any preset's types; angle types hold one entry per sector (see iter_sectors)
    contours: Dict[str, Contour] = field(default_factory=dict)  # access via contour()
    measurement_1: Optional[Measure] = None
    measurement_2: Optional[Measure] = None
    reference: Optional[Tuple[float, float]] = None
    centroid: Optional[Tuple[float, float]] = None
    closest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    farthest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    def contour(self, contour_type: ContourType | str) -> Contour:
        """The frame's contour of `contour_type` (type or id), created empty on first use."""
        key = contour_type if isinstance(contour_type, str) else contour_type.value
        if key in NON_CONTOUR_KEYS:
            raise KeyError(f'{key} is not a contour type')
        return self.contours.setdefault(key, Contour())

    @property
    def lumen(self) -> Contour:
        return self.contour(ContourType.LUMEN)

    @lumen.setter
    def lumen(self, contour: Contour) -> None:
        self.contours[ContourType.LUMEN.value] = contour

    @property
    def eem(self) -> Contour:
        return self.contour(ContourType.EEM)

    @eem.setter
    def eem(self, contour: Contour) -> None:
        self.contours[ContourType.EEM.value] = contour


# Drawn annotations that are not contours: the measurements and the reference point
NON_CONTOUR_KEYS = frozenset(
    contour_type.value for contour_type in (ContourType.MEASUREMENT_1, ContourType.MEASUREMENT_2, ContourType.REFERENCE)
)

# Ids a contour type can't take: FrameData's fields (contours are saved next to them) and 'wall' (a region key)
RESERVED_CONTOUR_IDS = frozenset(f.name for f in fields(FrameData)) | {'wall'}


def is_contour_key(key: str) -> bool:
    """Whether `key` names a contour type rather than a measurement or the reference point."""
    return key not in NON_CONTOUR_KEYS


def frame_to_dict(frame_data: FrameData) -> dict:
    """One frame as saved: its fields, with each contour as its own key (the layout earlier versions read)."""
    raw: dict = {}
    for key, value in asdict(frame_data).items():
        if key == 'contours':
            raw.update(value)
        else:
            raw[key] = value
    return raw


# Everything drawn on a frame and the values derived from the lumen; not the phase or OCT label
FRAME_ANNOTATION_FIELDS = (
    'contours',
    'measurement_1',
    'measurement_2',
    'reference',
    'centroid',
    'closest_points',
    'farthest_points',
)


def clear_frame_annotations(frame_data: FrameData) -> None:
    """Reset every annotation on `frame_data` to the state of a frame nobody has touched."""
    blank = FrameData()  # fresh defaults, contours included
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
