from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from domain.all_types import ContourType
from domain.colors import DEFAULT_MASK_ALPHA

MASK_ALPHA = DEFAULT_MASK_ALPHA  # overlay opacity (0 = transparent, 1 = opaque)


@dataclass(frozen=True)
class MaskSpec:
    label: int
    contour_type: ContourType
    # Order the overlay blends this label in, and — among the angular sectors, which are
    # painted bottom-up rather than ranked — the order contours_to_mask paints it in.
    # For every other label the mask *priority* is decided dynamically, see contours_to_mask.
    paint_order: int
    read_predicate: Callable[[np.ndarray], np.ndarray] | None = None

    def matches(self, mask_array: np.ndarray) -> np.ndarray:
        """Return boolean array of pixels belonging to this label."""
        if self.read_predicate is not None:
            return self.read_predicate(mask_array)
        return mask_array == self.label


MASK_SPECS: dict[ContourType, MaskSpec] = {
    # Both angular sectors are bottom-most layers of the overlay and of the exported
    # mask (see contours_to_mask): they mark image regions nothing can be read from,
    # so anything drawn anyway belongs on top. Blood is the very back of the stack —
    # where the two sectors overlap the guide-wire shadow is the more specific finding,
    # so it is the one that stays visible.
    ContourType.BLOOD: MaskSpec(
        label=10,
        contour_type=ContourType.BLOOD,
        paint_order=0,
    ),
    ContourType.WIRE: MaskSpec(
        label=9,
        contour_type=ContourType.WIRE,
        paint_order=1,
    ),
    ContourType.EEM: MaskSpec(
        label=2,
        contour_type=ContourType.EEM,
        paint_order=2,
        read_predicate=lambda a: np.isin(a, [1, 2]),
    ),
    ContourType.LUMEN: MaskSpec(
        label=1,
        contour_type=ContourType.LUMEN,
        paint_order=3,
    ),
    ContourType.CALCIUM: MaskSpec(
        label=3,
        contour_type=ContourType.CALCIUM,
        paint_order=4,
    ),
    ContourType.LIPID: MaskSpec(
        label=4,
        contour_type=ContourType.LIPID,
        paint_order=5,
    ),
    ContourType.MACROPHAGE: MaskSpec(
        label=5,
        contour_type=ContourType.MACROPHAGE,
        paint_order=6,
    ),
    ContourType.BRANCH: MaskSpec(
        label=7,
        contour_type=ContourType.BRANCH,
        paint_order=7,
    ),
}
