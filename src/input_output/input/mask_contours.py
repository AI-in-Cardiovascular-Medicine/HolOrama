"""Reading a multi-label intravascular mask back into contours, after the contour preset
naming each label: the inverse of imgs_masks.contours_to_mask.

The mask is painted bottom to top by layer, so each type is partly hidden by what lies over
it: the EEM by the wire shadow and a side branch, a plaque by the lumen it is clipped out
of, a side branch by the lumen. Boundary stretches against those are *occluded*: the
contour goes on unseen underneath. Each type's region (its label plus those inside it, see
ContourPreset.mask_labels) is traced, occluded stretches are dropped, and a closed spline
bridges them. This keeps the EEM out of the wire shadow.

Two kinds of type need more:

* angular sectors (wire, blood) are wedges about the image centre. A sector is a run of
  directions its label is seen in, not broken where it would be hidden anyway.
* a contained type (a plaque in the EEM) may have been drawn open. An open arc fills out to
  its container's boundary, so a region reaching it over nearly all its directions was an
  arc (its luminal edge), one reaching it all round was a ring around the lumen (see
  imgs_masks._contained_mask), and anything else was closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure as sk_measure

from domain.intravascular.contour_fitting_types import (
    FIBROUS_CAP_LABEL,
    FIT_SAMPLES,
    FIT_TOLERANCE_PER_PX,
    FIT_TOLERANCE_PX,
    KNOT_PLACEMENTS,
    LIPID_ID,
    MAX_MERGE_GAP_DEG,
    MIN_COMPONENT_PX,
    OPEN_FRACTION,
    RING_GAP_DEG,
    TOUCH_TOLERANCE_PX,
)
from domain.intravascular.types import ContourType
from domain.intravascular.contour_presets import ContourPreset, ContourTypeDef
from domain.intravascular.io_types import Contour, set_sector_points
from tools.angle import MIN_SWEEP, TWO_PI, points_for_sector
from tools.spline import sample_spline

Point = tuple[float, float]  # (x, y) in image pixels
_BINS = 360  # directions are binned one per degree


@dataclass(frozen=True)
class _Frame:
    """A frame's mask and its reading context."""

    mask: np.ndarray
    preset: ContourPreset
    centroid: Point  # of the lumen, which open contours fill outwards from

    @property
    def centre(self) -> Point:
        """Image centre: the apex of every angular sector."""
        height, width = self.mask.shape
        return width / 2.0, height / 2.0

    def region(self, defn: ContourTypeDef) -> np.ndarray:
        return np.isin(self.mask, list(self.preset.mask_labels(defn.type)))

    def occluders(self, defn: ContourTypeDef) -> np.ndarray:
        """Labels that can hide `defn`: painted over it but not inside it."""
        order = self.preset.paint_order()
        own = self.preset.mask_labels(defn.type)
        above = order[order.index(defn) + 1 :]
        return np.array([other.label for other in above if other.label not in own])

    def below(self, defn: ContourTypeDef) -> np.ndarray:
        """Background and labels under `defn`: where it would show if present."""
        order = self.preset.paint_order()
        return np.array([0] + [other.label for other in order[: order.index(defn)]])


@dataclass
class _Arc:
    points: list[Point]  # its luminal edge, by increasing angle
    start: float  # angle of the first point about the centroid
    end: float  # angle of the last


def frame_contours(
    frame_mask: np.ndarray,
    preset: ContourPreset,
    n_knots: int,
    handle_radius: float,
) -> tuple[dict[str, Contour], Point | None]:
    """The contours of `preset` read off one frame's mask, by type id, and the lumen centroid
    for open contours (None without a lumen).

    Every contour gets `n_knots` knots (n_interactive_points, plaques too, so a large one keeps
    its shape). `handle_radius` places angular sector points (only their direction matters).
    """
    for alias, label in label_aliases(preset).items():
        frame_mask = np.where(frame_mask == alias, label, frame_mask)
    height, width = frame_mask.shape
    lumen = preset[ContourType.LUMEN]
    lumen_part = _largest(_components(np.isin(frame_mask, list(preset.mask_labels(lumen.type))), frame_mask, lumen))
    centroid: Point | None = None
    if lumen_part is not None:
        rows, cols = np.nonzero(lumen_part)
        centroid = (float(cols.mean()), float(rows.mean()))
    frame = _Frame(frame_mask, preset, centroid or (width / 2.0, height / 2.0))

    contours: dict[str, Contour] = {}
    for defn in preset.types:
        if np.count_nonzero(frame_mask == defn.label) < MIN_COMPONENT_PX:
            continue
        if defn.is_angle:
            contour = _sectors(frame, defn, handle_radius)
        elif defn.inside is None:
            contour = _closed_contours(frame, defn, n_knots)
        else:
            contour = _contained_contours(frame, defn, n_knots)
        if contour.contours:
            contours[defn.type.value] = contour
    return contours, centroid


def label_aliases(preset: ContourPreset) -> dict[int, int]:
    """Mask labels read as another (see FIBROUS_CAP_LABEL)."""
    lipid = preset.get(LIPID_ID)
    if lipid is None or lipid.is_angle or any(defn.label == FIBROUS_CAP_LABEL for defn in preset.types):
        return {}
    return {FIBROUS_CAP_LABEL: preset[ContourType.EEM].label}


def _components(region: np.ndarray, frame_mask: np.ndarray, defn: ContourTypeDef) -> list[np.ndarray]:
    """Connected parts of `region` with at least MIN_COMPONENT_PX pixels of `defn`'s own
    label: a part only of what lies inside (a lumen without EEM) is skipped."""
    labelled = sk_measure.label(region, connectivity=2)
    own = frame_mask == defn.label
    parts = []
    for index in range(1, int(labelled.max()) + 1):
        part = labelled == index
        if np.count_nonzero(part & own) >= MIN_COMPONENT_PX:
            parts.append(part)
    return parts


def _largest(parts: list[np.ndarray]) -> np.ndarray | None:
    return max(parts, key=np.count_nonzero) if parts else None


# -- angular sectors -----------------------------------------------------------------------


def _sectors(frame: _Frame, defn: ContourTypeDef, handle_radius: float) -> Contour:
    """Each sector of `defn`: a run of directions about the image centre its label is seen in.
    Directions where it would be hidden at every distance neither end nor start a run."""
    height, width = frame.mask.shape
    cx, cy = frame.centre
    rows, cols = np.mgrid[0:height, 0:width]
    angles = np.arctan2(rows - cy, cols - cx)
    bins = np.floor(np.degrees(angles)).astype(int) % _BINS

    own = frame.mask == defn.label
    seen = np.bincount(bins[own], minlength=_BINS) > 0
    open_view = np.isin(frame.mask, frame.below(defn)) & ~own
    visible = np.bincount(bins[open_view], minlength=_BINS) > 0  # where it would show
    unknown = ~seen & ~visible

    contour = Contour()
    for run in _runs(seen, unknown):
        in_run = np.isin(bins, run) & own
        if np.count_nonzero(in_run) < MIN_COMPONENT_PX:
            continue
        # Sub-degree bounds from the pixels' own directions, measured from the run start so a
        # run across the -pi/pi seam stays whole.
        reference = math.radians(run[0])
        relative = (angles[in_run] - reference) % TWO_PI
        start = reference + float(relative.min())
        sweep = float(relative.max() - relative.min())
        if sweep < MIN_SWEEP:
            continue
        set_sector_points(contour, len(contour.contours), points_for_sector(frame.centre, handle_radius, start, sweep))
    return contour


def _runs(present: np.ndarray, unknown: np.ndarray) -> list[list[int]]:
    """Circular runs of `present` bins, bridging `unknown` ones, as ordered bins (may wrap
    past 359)."""
    passable = present | unknown
    if passable.all():
        if not present.any():
            return []
        start = _after_widest(unknown) if unknown.any() else 0  # whole circle: start after the widest unknown stretch
        order = [(start + i) % _BINS for i in range(_BINS)]
        return [_trim(order, present)]
    first_gap = int(np.argmin(passable))  # begin scanning just after a bin that is neither
    runs: list[list[int]] = []
    current: list[int] = []
    for i in range(1, _BINS + 1):
        b = (first_gap + i) % _BINS
        if passable[b]:
            current.append(b)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [trimmed for trimmed in (_trim(run, present) for run in runs) if trimmed]


def _after_widest(flags: np.ndarray) -> int:
    """Bin after the widest circular run of set `flags`."""
    best_end, best_len, length = 0, -1, 0
    for i in range(2 * _BINS):
        if flags[i % _BINS]:
            length += 1
            if length > best_len:
                best_len, best_end = length, i % _BINS
        else:
            length = 0
    return (best_end + 1) % _BINS


def _trim(run: list[int], present: np.ndarray) -> list[int]:
    """`run` minus its unknown end bins (only inner ones are bridged)."""
    marks = [i for i, b in enumerate(run) if present[b]]
    return run[marks[0] : marks[-1] + 1] if marks else []


# -- closed contours -----------------------------------------------------------------------


def _closed_contours(frame: _Frame, defn: ContourTypeDef, n_knots: int) -> Contour:
    parts = _components(frame.region(defn), frame.mask, defn)
    if _single(defn):
        largest = _largest(parts)
        parts = [largest] if largest is not None else []
    contour = Contour()
    for part in parts:
        knots = _boundary_knots(part, frame, defn, n_knots)
        if knots is not None:
            contour.add(knots, closed=True)
    return contour


def _single(defn: ContourTypeDef) -> bool:
    """One lumen and one EEM per frame, any number of other types."""
    return not defn.appendable


def _boundary_knots(part: np.ndarray, frame: _Frame, defn: ContourTypeDef, n_knots: int) -> list[Point] | None:
    """`n_knots` knots for the closed contour around `part`, from the unoccluded stretches of
    its boundary (see the module docstring). The spline bridges the gaps."""
    line = _outer_line(part)
    if line is None:
        return None
    genuine = ~_occluded(line, part, frame.mask, frame.occluders(defn))
    # Bridging needs gaps: a mostly hidden boundary is kept whole (its region is all there is).
    kept = line[genuine] if np.count_nonzero(genuine) >= 0.5 * len(line) else line
    return _fitted(kept, n_knots)


def _outer_line(part: np.ndarray) -> np.ndarray | None:
    """Outer boundary of `part` as (row, col) points on the half-pixel line around it, so its
    polygon covers exactly the pixel centres."""
    rows, cols = np.nonzero(part)
    if len(rows) == 0:
        return None
    top, left = rows.min(), cols.min()
    crop = part[top : rows.max() + 1, left : cols.max() + 1]
    lines = sk_measure.find_contours(np.pad(crop, 1).astype(float), 0.5)
    if not lines:
        return None
    line = max(lines, key=len)
    return line + np.array([top - 1, left - 1])


def _occluded(line: np.ndarray, part: np.ndarray, frame_mask: np.ndarray, occluders: np.ndarray) -> np.ndarray:
    """Per boundary point, whether it borders an occluder outside `part`."""
    if len(occluders) == 0:
        return np.zeros(len(line), dtype=bool)
    height, width = frame_mask.shape
    hidden = np.zeros(len(line), dtype=bool)
    base_rows = np.floor(line[:, 0]).astype(int)
    base_cols = np.floor(line[:, 1]).astype(int)
    for d_row in (0, 1):
        for d_col in (0, 1):
            rows = np.clip(base_rows + d_row, 0, height - 1)
            cols = np.clip(base_cols + d_col, 0, width - 1)
            hidden |= ~part[rows, cols] & np.isin(frame_mask[rows, cols], occluders)
    return hidden


def _fitted(line: np.ndarray, n_knots: int) -> list[Point] | None:
    """`n_knots` knots along the closed (row, col) `line`, as many as a drawn contour of the
    type gets (n_interactive_points), so it edits like one.

    With the count fixed, the fit depends on placement: evenly by length first, then, while
    the spline strays from over 5% of the line by more than FIT_TOLERANCE_PX, shifted along
    it and drawn towards bends, keeping the best fit."""
    if len(line) < 3:
        return None
    count = min(max(n_knots, 3), len(line))
    # Knots are spread by length, not point count. A hidden stretch left out is a jump in it.
    steps = np.hypot(*np.diff(line, axis=0).T)
    closing = float(np.hypot(*(line[0] - line[-1])))
    total = float(steps.sum()) + closing
    tolerance = float(np.clip(total * FIT_TOLERANCE_PER_PX, *FIT_TOLERANCE_PX))
    lengths = np.append(steps, closing)  # from each point to the next, round the loop
    bend = _bend(line)

    def placement(weight: float, phase: float) -> np.ndarray:
        shares = lengths * (1.0 + weight * bend)  # each point's share: its length, more where the line bends
        along = np.concatenate([[0.0], np.cumsum(shares)[:-1]])
        targets = (np.arange(count) + phase) * (along[-1] + shares[-1]) / count
        return np.unique(np.searchsorted(along, targets).clip(0, len(line) - 1))

    def error(picks: np.ndarray) -> float:
        xs, ys = sample_spline(line[picks, 1], line[picks, 0], True, FIT_SAMPLES)  # the spline the mask is painted with
        distances, _ = cKDTree(np.column_stack([ys, xs])).query(line)
        return float(np.percentile(distances, 95))

    best, best_error = None, np.inf
    for weight, phase in KNOT_PLACEMENTS:
        picks = placement(weight, phase)
        if len(picks) < 3:
            continue
        fit = error(picks)
        if fit < best_error:
            best, best_error = picks, fit
        if best_error <= tolerance:
            break
    if best is None:
        return None
    return [(float(line[i, 1]), float(line[i, 0])) for i in best]


def _bend(line: np.ndarray) -> np.ndarray:
    """How sharply the closed `line` turns at each point, 1 on average (0 if it turns
    evenly). Measured over a few points either side to ignore the pixel staircase."""
    span = max(len(line) // 50, 2)
    ahead = np.roll(line, -span, axis=0) - line
    behind = line - np.roll(line, span, axis=0)
    turn = np.abs(np.arctan2(*ahead.T) - np.arctan2(*behind.T))
    turn = np.minimum(turn, 2 * np.pi - turn)
    mean = turn.mean()
    return turn / mean if mean > 0 else np.zeros(len(line))


# -- types lying inside another ------------------------------------------------------------


def _contained_contours(frame: _Frame, defn: ContourTypeDef, n_knots: int) -> Contour:
    """Each part of a contained type, read as an arc, ring or closed contour (see module docstring)."""
    assert defn.inside is not None
    container = frame.preset[defn.inside]
    container_reach = _reach(frame.region(container), frame.centroid)
    contour = Contour()
    arcs: list[_Arc] = []
    for part in _components(frame.region(defn), frame.mask, defn):
        rows, cols = np.nonzero(part)
        cx, cy = frame.centroid
        theta = np.arctan2(rows - cy, cols - cx)
        radius = np.hypot(cols - cx, rows - cy)
        bins = np.floor(np.degrees(theta)).astype(int) % _BINS
        spanned = np.unique(bins)
        reach = np.full(_BINS, -np.inf)
        np.maximum.at(reach, bins, radius)
        touching = reach[spanned] >= container_reach[spanned] - TOUCH_TOLERANCE_PX
        if np.mean(touching) < OPEN_FRACTION:
            knots = _boundary_knots(part, frame, defn, n_knots)
            if knots is not None:
                contour.add(knots, closed=True)
        elif _widest_gap(theta)[1] < math.radians(RING_GAP_DEG):
            ring = _inner_edge(theta, radius, bins, frame.centroid)
            knots = _evenly(np.array([(y, x) for x, y in ring]), n_knots)
            if knots is not None:
                contour.add(knots, closed=True)
        else:
            arcs.append(_arc(theta, radius, bins, frame.centroid))

    for arc in _merge_arcs(arcs, frame, defn):
        knots = _arc_knots(arc.points, n_knots)
        if knots is not None:
            contour.add(knots, closed=False)
    return contour


def _reach(region: np.ndarray, centroid: Point) -> np.ndarray:
    """`region`'s max radius per direction about `centroid` (-inf if none)."""
    rows, cols = np.nonzero(region)
    cx, cy = centroid
    bins = np.floor(np.degrees(np.arctan2(rows - cy, cols - cx))).astype(int) % _BINS
    reach = np.full(_BINS, -np.inf)
    np.maximum.at(reach, bins, np.hypot(cols - cx, rows - cy))
    return reach


def _widest_gap(theta: np.ndarray) -> tuple[float, float]:
    """(end, width) of the widest circular gap in directions `theta`."""
    ordered = np.sort(theta)
    gaps = np.diff(np.append(ordered, ordered[0] + TWO_PI))
    widest = int(np.argmax(gaps))
    return float(ordered[(widest + 1) % len(ordered)]), float(gaps[widest])


def _inner_edge(theta, radius, bins, centroid: Point, order: list[int] | None = None) -> list[Point]:
    """Luminal edge: per direction (in `order`, else all by angle), half a pixel inside the
    pixel nearest `centroid`."""
    cx, cy = centroid
    points = []
    for b in order if order is not None else sorted(set(bins.tolist())):
        in_bin = bins == b
        if not in_bin.any():
            continue
        nearest = int(np.argmin(np.where(in_bin, radius, np.inf)))
        r = max(float(radius[nearest]) - 0.5, 0.0)
        points.append((cx + r * math.cos(theta[nearest]), cy + r * math.sin(theta[nearest])))
    return points


def _evenly(line: np.ndarray, n_knots: int) -> list[Point] | None:
    """`n_knots` even (x, y) picks of (row, col) `line`."""
    if len(line) < 3:
        return None
    count = min(n_knots, len(line))
    picks = np.linspace(0, len(line), count, endpoint=False).astype(int)
    return [(float(line[i, 1]), float(line[i, 0])) for i in picks]


def _arc(theta, radius, bins, centroid: Point) -> _Arc:
    """The luminal edge of a part drawn as an open arc, end to end of its span.

    The span is all but the widest gap between its pixel directions. Near the lumen a
    plaque's pixels miss some one-degree bins, so runs of bins would cut it up."""
    start, _ = _widest_gap(theta)
    first = int(math.floor(math.degrees(start))) % _BINS
    present = set(bins.tolist())
    order = [b for b in ((first + i) % _BINS for i in range(_BINS)) if b in present]
    points = _inner_edge(theta, radius, bins, centroid, order)
    cx, cy = centroid
    return _Arc(
        points=points,
        start=math.atan2(points[0][1] - cy, points[0][0] - cx),
        end=math.atan2(points[-1][1] - cy, points[-1][0] - cx),
    )


def _merge_arcs(arcs: list[_Arc], frame: _Frame, defn: ContourTypeDef) -> list[_Arc]:
    """Join arc pieces cut apart by an occluder (the wire shadow): arcs whose gap is hidden
    all the way across at their meeting radius are one."""
    occluders = frame.occluders(defn)
    if len(arcs) < 2 or len(occluders) == 0:
        return arcs
    arcs = sorted(arcs, key=lambda arc: arc.start % TWO_PI)
    merged = [arcs[0]]
    for arc in arcs[1:]:
        if _hidden_between(merged[-1], arc, frame, occluders):
            previous = merged[-1]
            merged[-1] = _Arc(points=previous.points + arc.points, start=previous.start, end=arc.end)
        else:
            merged.append(arc)
    if len(merged) > 1 and _hidden_between(merged[-1], merged[0], frame, occluders):
        last = merged.pop()
        merged[0] = _Arc(points=last.points + merged[0].points, start=last.start, end=merged[0].end)
    return merged


def _hidden_between(first: _Arc, second: _Arc, frame: _Frame, occluders: np.ndarray) -> bool:
    gap = (second.start - first.end) % TWO_PI
    if gap > math.radians(MAX_MERGE_GAP_DEG):
        return False
    cx, cy = frame.centroid
    r_first = math.hypot(first.points[-1][0] - cx, first.points[-1][1] - cy)
    r_second = math.hypot(second.points[0][0] - cx, second.points[0][1] - cy)
    radius = (r_first + r_second) / 2 + 1.0  # just outside the edge, inside the region
    height, width = frame.mask.shape
    steps = max(int(math.degrees(gap) * 2), 2)
    for i in range(1, steps):
        angle = first.end + gap * i / steps
        col = int(round(cx + radius * math.cos(angle)))
        row = int(round(cy + radius * math.sin(angle)))
        if 0 <= row < height and 0 <= col < width and frame.mask[row, col] not in occluders:
            return False
    return True


def _arc_knots(points: list[Point], n_knots: int) -> list[Point] | None:
    """`n_knots` knots along an open arc, ends kept."""
    if len(points) < 2:
        return None
    count = min(max(n_knots, 2), len(points))
    picks = np.linspace(0, len(points) - 1, count).round().astype(int)
    return [points[i] for i in picks]
