from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Tuple, Union

OCT_QUALITY_LABELS = ['Very Bad', 'Bad', 'Ok', 'Good', 'Very Good']  # best needs to be on the right


class SupportedType(Enum):
    IVUS = "IVUS"
    NIRS = "NIRS"
    OCT = "OCT"


@dataclass(frozen=True)
class ContourType:
    """One kind of annotation, identified by its stable id (`value`).

    Which contour types exist is data rather than code: the active contour preset defines
    them (see domain.intravascular.contour_presets), and a user can add their own there. Only the ones
    the software itself relies on are named here (the lumen and the EEM), which every
    preset has to define, and the two measurements and the reference point, which are not
    contours at all and so sit outside the presets.
    """

    value: str

    LUMEN: ClassVar['ContourType']
    EEM: ClassVar['ContourType']
    MEASUREMENT_1: ClassVar['ContourType']
    MEASUREMENT_2: ClassVar['ContourType']
    REFERENCE: ClassVar['ContourType']


ContourType.LUMEN = ContourType('lumen')
ContourType.EEM = ContourType('eem')
ContourType.MEASUREMENT_1 = ContourType('measurement_1')
ContourType.MEASUREMENT_2 = ContourType('measurement_2')
ContourType.REFERENCE = ContourType('reference')


class SegmentationTool(Enum):
    CLOSED_SPLINE = "closed_spline"
    OPEN_SPLINE = "open_spline"
    BRUSH = "brush"
    ANGLE = "angle"
    LINE = "line"
    POINT = "point"


@dataclass
class ContourConfig:
    """Configuration for a specific contour type"""

    color: Union[
        str, Tuple[int, int, int], Any
    ]  # accept string names ('green'), hex ('#ff00ff'), or RGB tuples (255,0,0)
    thickness: int
    point_radius: int
    point_thickness: int
    alpha: int
    n_points_contour: int
    n_interactive_points: int


@dataclass(frozen=True)
class BranchMeasurement:
    """One side-branch contour's diameter on a frame (see branch_metrics.branch_diameters)."""

    key: str  # its contour type id
    index: int  # which of that type's contours on the frame
    closed: bool  # False once it was cut at the lumen it leaves (tools.branch_arc)
    diameter: float  # mm
    start: Tuple[float, float]  # the diameter line's ends, unscaled px
    end: Tuple[float, float]


@dataclass(frozen=True)
class BranchRun:
    """Neighbouring frames showing one side branch leaving the lumen (see branch_metrics.branch_runs)."""

    frames: Tuple[int, ...]
    widest: Dict[int, BranchMeasurement]  # per frame, its widest branch leaving the lumen
    reference_frame: int  # the frame where that branch is widest
    landmark: Tuple[float, float]  # where its reference point goes on that frame, unscaled px


@dataclass(frozen=True)
class FrameAnnotation:
    """One annotation of a frame as exported for multimodars (see input_output.output.annotations).

    `kind` and `role` are multimodars' AnnotationKind / AnnotationRole names, a kind it does not
    know is kept as a custom one. Points are unscaled px (x, y).
    """

    frame: int
    kind: str
    role: str
    group: int
    points: Tuple[Tuple[float, float], ...]
