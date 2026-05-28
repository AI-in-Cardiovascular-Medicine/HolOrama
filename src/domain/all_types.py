from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Union, Any


OCT_QUALITY_LABELS = ['Very Bad', 'Bad', 'Ok', 'Good', 'Very Good']  # best needs to be on the right


class SupportedType(Enum):
    IVUS = "IVUS"
    NIRS = "NIRS"
    OCT = "OCT"


class ContourType(Enum):
    LUMEN = "lumen"
    EEM = "eem"
    CALCIUM = "calcium"
    BRANCH = "branch"
    LIPID = "lipid"
    MACROPHAGE = "macrophage"
    MEASUREMENT_1 = "measurement_1"
    MEASUREMENT_2 = "measurement_2"
    REFERENCE = "reference"
    WIRE = "wire"


class SegmentationTool(Enum):
    CLOSED_SPLINE = "closed_spline"
    OPEN_SPLINE = "open_spline"
    BRUSH = "brush"
    ANGLE = "angle"
    LINE = "line"
    POINT = "point"


def validate_tool(contour_type: ContourType, tool: SegmentationTool):
    if tool not in ALLOWED_TOOLS.get(contour_type, set()):
        raise ValueError(f"{tool} not allowed for {contour_type}")


ALLOWED_TOOLS = {
    ContourType.LUMEN: {
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.EEM: {
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.CALCIUM: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.BRANCH: {
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.LIPID: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.MACROPHAGE: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.MEASUREMENT_1: {SegmentationTool.LINE},
    ContourType.MEASUREMENT_2: {SegmentationTool.LINE},
    ContourType.REFERENCE: {SegmentationTool.POINT},
    ContourType.WIRE: {SegmentationTool.ANGLE},
}


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
