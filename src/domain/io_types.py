from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

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

    See ContourPreset.angle_types in domain.contour_presets for what a sector is and tools.angle for how its
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
    # Every contour on the frame, by contour type id (see domain.contour_presets). Read and
    # write them through contour(), which hands out an empty one for a type not drawn yet.
    # Kept for any id rather than only the active preset's, so a file annotated with other
    # types loses nothing on its way through.
    #
    # Angular sectors (a preset's angle types) are stored like any other multi-instance
    # contour: one entry in Contour.contours per sector, holding that sector's 2-3 angle
    # points as ([x, ...], [y, ...]) — the radial lines bounding it, plus the interior
    # marker that says which of the two arcs between them is meant (see tools.angle). A
    # frame can carry several of each. Read/write via iter_sectors / sector_points /
    # set_sector_points.
    contours: Dict[str, Contour] = field(default_factory=dict)
    measurement_1: Optional[Measure] = None
    measurement_2: Optional[Measure] = None
    reference: Optional[Tuple[float, float]] = None
    centroid: Optional[Tuple[float, float]] = None
    closest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    farthest_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    def contour(self, contour_type: ContourType | str) -> Contour:
        """The frame's contour of `contour_type` (a ContourType or its id), created empty
        on first use so that it can be written to like any other."""
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


# The annotations that are drawn but are not contours: the measurements and the reference
# point, which sit outside the contour presets (see ContourType).
NON_CONTOUR_KEYS = frozenset(
    contour_type.value for contour_type in (ContourType.MEASUREMENT_1, ContourType.MEASUREMENT_2, ContourType.REFERENCE)
)

# Names a contour type id cannot take. A frame is saved with each contour as a key of its
# own next to FrameData's other fields (see frame_to_dict), so it may not shadow one of
# them; 'wall' is the vessel wall's key among a frame's measured regions (see
# imgs_masks.frame_region_metrics).
RESERVED_CONTOUR_IDS = frozenset(f.name for f in fields(FrameData)) | {'wall'}


def is_contour_key(key: str) -> bool:
    """Whether `key` names a contour type rather than a measurement or the reference point."""
    return key not in NON_CONTOUR_KEYS


def frame_to_dict(frame_data: FrameData) -> dict:
    """One frame as it is saved: its fields, with every contour as a key of its own in
    place of the `contours` dict — the layout every earlier version wrote, so a file stays
    readable by them and by anything else that parses it."""
    raw: dict = {}
    for key, value in asdict(frame_data).items():
        if key == 'contours':
            raw.update(value)
        else:
            raw[key] = value
    return raw


# Everything the user can draw on one image: every contour (which includes the angular
# sectors), both measurements and the reference point, plus the values derived from the
# lumen. Not the phase or the OCT label — those describe the frame, not the drawing.
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


CANONICAL_ORIENTATION = 'LAS'

# Direction cosines of CANONICAL_ORIENTATION, in SimpleITK's row-major order.
CANONICAL_DIRECTION: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class VolumeGeometry:
    """Voxel-grid geometry of a loaded CCTA volume.

    ``origin`` / ``spacing`` / ``direction`` describe the canonicalized grid the app works
    in, so they can be attached to any array shaped like the loaded volume.
    ``source_orientation`` is the orientation code the file was stored in, kept so a mask
    drawn here can be written back the way the source image was laid out.
    """

    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # sitk order: (x, y, z)
    direction: Tuple[float, ...] = CANONICAL_DIRECTION
    source_orientation: str = CANONICAL_ORIENTATION


def geometry_from_spacing(voxel_spacing: Tuple[float, float, float]) -> VolumeGeometry:
    """A canonical-orientation geometry carrying just `voxel_spacing` (dz, dy, dx).

    Fallback for writing a mask when the source image's geometry is not at hand.
    """
    dz, dy, dx = voxel_spacing
    return VolumeGeometry(spacing=(dx, dy, dz))


@dataclass
class MetaDataFusion:
    modality: str = 'Fusion'
    patient_name: str = 'Unknown'
    birthdate: str = 'Unknown'
    sex: str = 'Unknown'
    ...
