"""Reading a multi-label intravascular mask back into contours, after the contour preset
that says what each label is — the inverse of imgs_masks.contours_to_mask.

The mask is painted bottom to top by layer, so every type is partly hidden by whatever is
painted over it: the EEM by the wire shadow and a side branch, a plaque by the lumen it is
clipped out of, a side branch by the lumen. Where a type's region ends against one of
those, its contour goes on underneath, somewhere unknown — that stretch of the boundary is
*occluded*. Every type is read the same way: its region (its own label and those of what
lies inside it, see ContourPreset.mask_labels) is traced, the occluded stretches are left
out, and a closed spline through what is left bridges them. That is what keeps the EEM
from following the lumen through the wire shadow.

Two kinds of type need more than that:

* angular sectors (wire, blood) are wedges about the image centre, so each boundary has a
  single unknown, its angle: a sector is a run of directions its label is seen in, and a
  direction where it would be hidden whatever it held does not break the run;
* a type lying inside another (a plaque in the EEM) may have been drawn open: an open arc
  fills outwards to its container's boundary, so a region reaching that boundary over
  nearly all the directions it spans was an arc — its luminal edge. One that reaches it all
  the way round was a ring around the lumen (see imgs_masks._contained_mask). Anything else
  was drawn closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure as sk_measure

from domain.intravascular.types import ContourType
from domain.intravascular.contour_presets import ContourPreset, ContourTypeDef
from domain.intravascular.io_types import Contour, set_sector_points
from tools.angle import MIN_SWEEP, TWO_PI, points_for_sector

MIN_COMPONENT_PX = 20  # anything smaller is noise, not a structure
OPEN_FRACTION = 0.85  # of its directions a region reaches its container's boundary in, to be an arc
TOUCH_TOLERANCE_PX = 2.0  # how close to the container's boundary counts as reaching it
RING_GAP_DEG = 10.0  # a region whose widest gap in direction is narrower goes all the way round
MAX_MERGE_GAP_DEG = 90.0  # widest hidden gap two pieces of one arc are bridged across
# How far a closed contour's spline may stray from (most of) its boundary: a pixel along a
# long outline, half of one around something small, where a pixel is a share of its area.
FIT_TOLERANCE_PX = (0.5, 1.0)
FIT_TOLERANCE_PER_PX = 1 / 300  # of the outline's length
_BINS = 360  # one per degree

# The fibrous cap of OCT segmentations: the tissue between the lumen and a lipid pool, which
# such masks label on its own. It is no contour of its own but vessel wall, so it is read as
# the EEM's — and the lipid's luminal edge, the arc it is drawn as, then runs along the cap's
# far side, where the pool begins, as a real boundary rather than one hidden under the cap.
# Only while the preset has a lipid type and gives this label no type of its own.
FIBROUS_CAP_LABEL = 6
LIPID_ID = 'lipid'

Point = tuple[float, float]  # (x, y) in image pixels


@dataclass(frozen=True)
class _Frame:
    """One frame's mask and what every contour of it is read against."""

    mask: np.ndarray
    preset: ContourPreset
    centroid: Point  # of the lumen, which open contours fill outwards from

    @property
    def centre(self) -> Point:
        """The image centre, which every angular sector is a wedge about."""
        height, width = self.mask.shape
        return width / 2.0, height / 2.0

    def region(self, defn: ContourTypeDef) -> np.ndarray:
        return np.isin(self.mask, list(self.preset.mask_labels(defn.type)))

    def occluders(self, defn: ContourTypeDef) -> np.ndarray:
        """The labels that can hide `defn`: those painted over it that do not lie inside it."""
        order = self.preset.paint_order()
        own = self.preset.mask_labels(defn.type)
        above = order[order.index(defn) + 1 :]
        return np.array([other.label for other in above if other.label not in own])

    def below(self, defn: ContourTypeDef) -> np.ndarray:
        """Background, and the labels painted under `defn`: where it would show if it were there."""
        order = self.preset.paint_order()
        return np.array([0] + [other.label for other in order[: order.index(defn)]])


def label_aliases(preset: ContourPreset) -> dict[int, int]:
    """Mask values read as another type's label (see FIBROUS_CAP_LABEL)."""
    lipid = preset.get(LIPID_ID)
    if lipid is None or lipid.is_angle or any(defn.label == FIBROUS_CAP_LABEL for defn in preset.types):
        return {}
    return {FIBROUS_CAP_LABEL: preset[ContourType.EEM].label}


def frame_contours(
    frame_mask: np.ndarray,
    preset: ContourPreset,
    knots_for: Callable[[ContourTypeDef], int],
    handle_radius: float,
) -> tuple[dict[str, Contour], Point | None]:
    """Every contour type of `preset` read off one frame's mask, by id, and the lumen
    centroid the open contours among them were read about (None without a lumen).

    `knots_for` says how many knots a contour of a type gets; `handle_radius` is where an
    angular sector's stored points go (only their direction means anything).
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
            contour = _closed_contours(frame, defn, knots_for(defn))
        else:
            contour = _contained_contours(frame, defn, knots_for(defn))
        if contour.contours:
            contours[defn.type.value] = contour
    return contours, centroid


# -- regions -------------------------------------------------------------------------------


def _components(region: np.ndarray, frame_mask: np.ndarray, defn: ContourTypeDef) -> list[np.ndarray]:
    """The connected parts of `region` holding at least MIN_COMPONENT_PX pixels of `defn`'s
    own label — a part made only of what lies inside it (a lumen without an EEM around it,
    say) is not one of it."""
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


def _single(defn: ContourTypeDef) -> bool:
    """A frame holds one lumen and one EEM; of any other type, as many as it shows."""
    return not defn.appendable


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
            _append(contour, knots, closed=True)
    return contour


def _boundary_knots(part: np.ndarray, frame: _Frame, defn: ContourTypeDef, n_knots: int) -> list[Point] | None:
    """`n_knots` knots for the closed contour around `part`, taken from the stretches of its
    boundary that are its own (see the module docstring); the gaps the occluded ones leave
    are for the spline to bridge."""
    line = _outer_line(part)
    if line is None:
        return None
    genuine = ~_occluded(line, part, frame.mask, frame.occluders(defn))
    # Bridging only works across gaps; a boundary mostly hidden is kept whole instead (the
    # region it encloses is all that can be told about it).
    kept = line[genuine] if np.count_nonzero(genuine) >= 0.5 * len(line) else line
    return _fitted(kept, n_knots)


def _fitted(line: np.ndarray, n_knots: int) -> list[Point] | None:
    """`n_knots` knots along the closed (row, col) `line` — exactly as many as a contour of
    the type gets when drawn (n_interactive_points), so it edits like one.

    The count is fixed, so the fit is won by where they go: spread evenly by length first,
    then — while the spline through them strays from more than 5% of the line by more than
    the tolerance for its length (see FIT_TOLERANCE_PX) — shifted along it and drawn
    towards where it bends, keeping whichever placement fits best."""
    from input_output.output.imgs_masks import _smooth_contour  # the spline the mask is painted with

    if len(line) < 3:
        return None
    count = min(max(n_knots, 3), len(line))
    # Distance along the line, so knots are spread by length and not by how many boundary
    # points a stretch happens to have; a hidden stretch left out of it is a jump in it.
    steps = np.hypot(*np.diff(line, axis=0).T)
    closing = float(np.hypot(*(line[0] - line[-1])))
    total = float(steps.sum()) + closing
    tolerance = float(np.clip(total * FIT_TOLERANCE_PER_PX, *FIT_TOLERANCE_PX))
    lengths = np.append(steps, closing)  # from each point to the next, round the loop
    bend = _bend(line)

    def placement(weight: float, phase: float) -> np.ndarray:
        # Each point's share of the line: its length, more of it where the line bends.
        shares = lengths * (1.0 + weight * bend)
        along = np.concatenate([[0.0], np.cumsum(shares)[:-1]])
        targets = (np.arange(count) + phase) * (along[-1] + shares[-1]) / count
        return np.unique(np.searchsorted(along, targets).clip(0, len(line) - 1))

    def error(picks: np.ndarray) -> float:
        xs, ys = _smooth_contour(line[picks, 1], line[picks, 0], is_closed=True)
        distances, _ = cKDTree(np.column_stack([ys, xs])).query(line)
        return float(np.percentile(distances, 95))

    best, best_error = None, np.inf
    for weight, phase in _PLACEMENTS:
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


# (bend weight, phase) of the knot placements _fitted tries, plain even spacing first
_PLACEMENTS = ((0.0, 0.0), (0.0, 0.5), (1.0, 0.0), (1.0, 0.5), (3.0, 0.0), (3.0, 0.5))


def _bend(line: np.ndarray) -> np.ndarray:
    """Per point of the closed `line`, how sharply it turns there, 1 on average (0 for a
    line that turns evenly). Measured over a few points either side, so the pixel
    staircase of a mask edge does not count as bending."""
    span = max(len(line) // 50, 2)
    ahead = np.roll(line, -span, axis=0) - line
    behind = line - np.roll(line, span, axis=0)
    turn = np.abs(np.arctan2(*ahead.T) - np.arctan2(*behind.T))
    turn = np.minimum(turn, 2 * np.pi - turn)
    mean = turn.mean()
    return turn / mean if mean > 0 else np.zeros(len(line))


def _outer_line(part: np.ndarray) -> np.ndarray | None:
    """The outer boundary of `part` as (row, col) points, on the half-pixel line between
    its pixels and the rest, so the polygon through it covers exactly its pixel centres."""
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
    """Per boundary point, whether a pixel just outside `part` next to it holds an occluder."""
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


def _evenly(line: np.ndarray, n_knots: int) -> list[Point] | None:
    """`n_knots` (row, col) points spread evenly along `line`, as (x, y)."""
    if len(line) < 3:
        return None
    count = min(n_knots, len(line))
    picks = np.linspace(0, len(line), count, endpoint=False).astype(int)
    return [(float(line[i, 1]), float(line[i, 0])) for i in picks]


def _append(contour: Contour, knots: list[Point], closed: bool) -> None:
    contour.contours.append(([x for x, _ in knots], [y for _, y in knots]))
    contour.closed.append(closed)
    if closed:
        contour.start_coords.append([])
        contour.end_coords.append([])
    else:
        contour.start_coords.append([knots[0]])
        contour.end_coords.append([knots[-1]])


# -- angular sectors -----------------------------------------------------------------------


def _sectors(frame: _Frame, defn: ContourTypeDef, handle_radius: float) -> Contour:
    """Every sector of `defn` on the frame: each run of directions about the image centre
    its label is seen in. A direction where it would be hidden at every distance by what is
    painted over it says nothing, so it neither ends a run nor starts one."""
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
        # Sub-degree boundaries: the extreme directions of the pixels themselves, measured
        # from the start of the run so a run across the -pi/pi seam stays in one piece.
        reference = math.radians(run[0])
        relative = (angles[in_run] - reference) % TWO_PI
        start = reference + float(relative.min())
        sweep = float(relative.max() - relative.min())
        if sweep < MIN_SWEEP:
            continue
        set_sector_points(contour, len(contour.contours), points_for_sector(frame.centre, handle_radius, start, sweep))
    return contour


def _runs(present: np.ndarray, unknown: np.ndarray) -> list[list[int]]:
    """The circular runs of `present` bins, bridging bins that are `unknown`, each as its
    bins in order. A run can wrap past bin 359 to bin 0."""
    passable = present | unknown
    if passable.all():
        if not present.any():
            return []
        # The whole circle: start where the widest stretch of unknown bins ends, if any.
        start = _after_widest(unknown) if unknown.any() else 0
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


def _trim(run: list[int], present: np.ndarray) -> list[int]:
    """`run` without the unknown bins at either end: only those in between were bridged."""
    marks = [i for i, b in enumerate(run) if present[b]]
    return run[marks[0] : marks[-1] + 1] if marks else []


def _after_widest(flags: np.ndarray) -> int:
    """The bin just after the widest circular stretch of set `flags`."""
    best_end, best_len, length = 0, -1, 0
    for i in range(2 * _BINS):
        if flags[i % _BINS]:
            length += 1
            if length > best_len:
                best_len, best_end = length, i % _BINS
        else:
            length = 0
    return (best_end + 1) % _BINS


# -- types lying inside another ------------------------------------------------------------


@dataclass
class _Arc:
    points: list[Point]  # its luminal edge, in order of increasing angle
    start: float  # angle of the first point about the centroid
    end: float  # angle of the last


def _contained_contours(frame: _Frame, defn: ContourTypeDef, n_knots: int) -> Contour:
    """Each part of a type lying inside another, as the arc, ring or closed contour it was
    drawn as (see the module docstring)."""
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
                _append(contour, knots, closed=True)
        elif _widest_gap(theta)[1] < math.radians(RING_GAP_DEG):
            ring = _inner_edge(theta, radius, bins, frame.centroid)
            knots = _evenly(np.array([(y, x) for x, y in ring]), n_knots)
            if knots is not None:
                _append(contour, knots, closed=True)
        else:
            arcs.append(_arc(theta, radius, bins, frame.centroid))

    for arc in _merge_arcs(arcs, frame, defn):
        knots = _arc_knots(arc.points, n_knots)
        if knots is not None:
            _append(contour, knots, closed=False)
    return contour


def _reach(region: np.ndarray, centroid: Point) -> np.ndarray:
    """Per direction about `centroid`, how far out `region` reaches (-inf where it does not)."""
    rows, cols = np.nonzero(region)
    cx, cy = centroid
    bins = np.floor(np.degrees(np.arctan2(rows - cy, cols - cx))).astype(int) % _BINS
    reach = np.full(_BINS, -np.inf)
    np.maximum.at(reach, bins, np.hypot(cols - cx, rows - cy))
    return reach


def _inner_edge(theta, radius, bins, centroid: Point, order: list[int] | None = None) -> list[Point]:
    """Per direction (in `order`, or all of them by angle), the point half a pixel inside
    the region's pixel nearest `centroid` — its luminal edge."""
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


def _widest_gap(theta: np.ndarray) -> tuple[float, float]:
    """(where it ends, how wide) the widest circular gap between the directions `theta` is."""
    ordered = np.sort(theta)
    gaps = np.diff(np.append(ordered, ordered[0] + TWO_PI))
    widest = int(np.argmax(gaps))
    return float(ordered[(widest + 1) % len(ordered)]), float(gaps[widest])


def _arc(theta, radius, bins, centroid: Point) -> _Arc:
    """The luminal edge of a part drawn as an open arc, from one end of its span to the other.

    The span is everything but the widest gap between the directions of its pixels: at the
    small radius of a plaque by the lumen, its pixels do not reach every one-degree bin, so
    runs of bins would cut it up."""
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
    """Join the pieces of one arc that something painted over it (the wire shadow) cut apart:
    two arcs whose gap is hidden all the way across, at the radius they meet at, are one."""
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
    """`n_knots` knots along an open arc, its two ends among them."""
    if len(points) < 2:
        return None
    count = min(max(n_knots, 2), len(points))
    picks = np.linspace(0, len(points) - 1, count).round().astype(int)
    return [points[i] for i in picks]
