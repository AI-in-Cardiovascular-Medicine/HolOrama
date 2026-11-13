"""Every annotated structure that is not a pipe, as multimodars reads it: one
`{type}_{phase}_contours.csv` per type, rows `frame, x, y, z, contour, closed`.

Each frame's contours are numbered from 0, their points are the spline as the display draws it,
in drawing order, the first not repeated at the end.

- side branches: as drawn. An open one is the arc cut at the lumen it leaves (tools.branch_arc),
  both ends on the lumen border.
- in-wall structures (calcium, lipid, any type lying in another): always closed, as multimodars
  requires. A closed contour is written as drawn. An open arc is closed against its container, one
  contour at a time, as the region the exported mask holds. A ring drawn round the lumen means the
  ring between it and its container, which one closed contour cannot hold, so it is split in two
  halves through the lumen centroid.
"""

from __future__ import annotations

import cv2
import numpy as np

from domain.intravascular.contour_presets import ContourPreset, ContourTypeDef
from input_output.output.imgs_masks import closes_on_lumen, contour_regions, measured_types
from tools.spline import sample_spline

# Preset ids multimodars knows under another name. The file prefix keeps the preset id, multimodars
# maps it (calcium_ files are calcification), this is for reading its kinds back.
KIND_NAMES = {'calcium': 'calcification', 'branch': 'sidebranch'}


def structure_types(preset: ContourPreset) -> tuple[ContourTypeDef, ...]:
    """The types written this way: every spline type but the lumen and the EEM."""
    return measured_types(preset)


def frame_contours(
    frame_data, defn: ContourTypeDef, preset: ContourPreset, image_shape: tuple[int, int], samples: int
) -> list[tuple[np.ndarray, bool]]:
    """(points (N, 2) in unscaled px, closed) of each contour of `defn` on the frame, numbered in order."""
    contour_obj = frame_data.contour(defn.type)
    if not contour_obj.contours:
        return []
    as_drawn = closes_on_lumen(defn, preset) or defn.inside is None
    regions = None if as_drawn else contour_regions(frame_data, defn, image_shape, preset)

    out: list[tuple[np.ndarray, bool]] = []
    for index, entry in enumerate(contour_obj.contours):
        if not entry or len(entry[0]) < 2:
            continue
        closed = contour_obj.closed[index] if index < len(contour_obj.closed) else True
        region = None if regions is None else regions[index]
        ring = region is not None and _has_hole(region)
        if region is None or (closed and not ring):
            out.append((_spline(entry[0], entry[1], closed, samples), closed))
            continue
        parts = _halves(region, frame_data.centroid, image_shape) if ring else [region]
        out.extend((outline, True) for part in parts for outline in _outlines(part, samples))
    return out


def _spline(xs, ys, closed: bool, samples: int) -> np.ndarray:
    points = np.column_stack(sample_spline(xs, ys, closed, samples))
    return points[:-1] if closed else points  # a closed spline's last sample repeats its first


def _has_hole(mask: np.ndarray) -> bool:
    """Whether a part of `mask` encloses background, as a ring round the lumen does."""
    found, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return hierarchy is not None and bool(np.any(hierarchy[0][:, 3] >= 0))


def _halves(mask: np.ndarray, centroid, image_shape: tuple[int, int]) -> list[np.ndarray]:
    """`mask` cut along the horizontal line through `centroid` (the image centre without one)."""
    cy = centroid[1] if centroid is not None else image_shape[0] / 2.0
    above = np.zeros(image_shape, dtype=bool)
    above[: int(np.ceil(cy))] = True
    return [mask & above, mask & ~above]


def _outlines(mask: np.ndarray, samples: int) -> list[np.ndarray]:
    """Outline (N, 2) x, y of each connected part of `mask`, at most `samples` points each."""
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outlines = []
    for contour in found:
        points = contour[:, 0, :].astype(float)
        if len(points) < 3:
            continue
        if len(points) > samples:
            points = points[np.linspace(0, len(points) - 1, samples).round().astype(int)]
        outlines.append(points)
    return outlines
