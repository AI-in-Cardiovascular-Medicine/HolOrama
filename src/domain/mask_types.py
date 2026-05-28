from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from domain.all_types import ContourType

MASK_ALPHA = 0.45  # overlay opacity (0 = transparent, 1 = opaque)


@dataclass(frozen=True)
class MaskSpec:
    label: int
    overlay_color: tuple[int, int, int]
    contour_type: ContourType
    paint_order: int  # lower = painted first (lower priority, can be overwritten)
    read_predicate: Callable[[np.ndarray], np.ndarray] | None = None

    def matches(self, mask_array: np.ndarray) -> np.ndarray:
        """Return boolean array of pixels belonging to this label."""
        if self.read_predicate is not None:
            return self.read_predicate(mask_array)
        return mask_array == self.label


MASK_SPECS: dict[ContourType, MaskSpec] = {
    ContourType.WIRE: MaskSpec(
        label=9,
        overlay_color=(255, 165, 0),
        contour_type=ContourType.WIRE,
        paint_order=0,
    ),
    ContourType.EEM: MaskSpec(
        label=2,
        overlay_color=(0, 180, 255),
        contour_type=ContourType.EEM,
        paint_order=1,
        read_predicate=lambda a: np.isin(a, [1, 2]),
    ),
    ContourType.LUMEN: MaskSpec(
        label=1,
        overlay_color=(0, 200, 80),
        contour_type=ContourType.LUMEN,
        paint_order=2,
    ),
    ContourType.CALCIUM: MaskSpec(
        label=3,
        overlay_color=(255, 215, 0),
        contour_type=ContourType.CALCIUM,
        paint_order=3,
    ),
    ContourType.LIPID: MaskSpec(
        label=4,
        overlay_color=(255, 100, 0),
        contour_type=ContourType.LIPID,
        paint_order=4,
    ),
    ContourType.MACROPHAGE: MaskSpec(
        label=5,
        overlay_color=(200, 0, 220),
        contour_type=ContourType.MACROPHAGE,
        paint_order=5,
    ),
    ContourType.BRANCH: MaskSpec(
        label=7,
        overlay_color=(0, 180, 255),
        contour_type=ContourType.BRANCH,
        paint_order=6,
    ),
}
