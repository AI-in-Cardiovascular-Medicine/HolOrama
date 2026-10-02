from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Tuple, Union

OCT_QUALITY_LABELS = ['Very Bad', 'Bad', 'Ok', 'Good', 'Very Good']  # best needs to be on the right


class SupportedType(Enum):
    IVUS = "IVUS"
    NIRS = "NIRS"
    OCT = "OCT"


@dataclass(frozen=True)
class ContourType:
    """One kind of annotation, identified by its stable id (`value`).

    Which contour types exist is data rather than code: the active contour preset defines
    them (see domain.contour_presets), and a user can add their own there. Only the ones
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
