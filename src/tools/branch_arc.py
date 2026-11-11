"""A side branch drawn over the lumen, kept as the open arc outside it.

A closed branch contour that crosses the lumen is cut at the two crossings and only the
part outside the lumen is kept, as an open arc from crossing to crossing. Its region is
then the arc closed back along the lumen border (see close_against_lumen).
"""

from typing import Sequence

import numpy as np

from tools.spline import sample_spline


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd test of `points` (N, 2) against the closed `polygon` (M, 2)."""
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    px, py = points[:, :1], points[:, 1:]
    straddles = (y0 > py) != (y1 > py)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return (straddles & (px < x_cross)).sum(axis=1) % 2 == 1


def _crossing(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Where segment a->b crosses the closed `polygon`, the crossing nearest to a. b if none is found."""
    s = np.roll(polygon, -1, axis=0) - polygon
    r = b - a
    d = polygon - a
    denom = r[0] * s[:, 1] - r[1] * s[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (d[:, 0] * s[:, 1] - d[:, 1] * s[:, 0]) / denom
        u = (d[:, 0] * r[1] - d[:, 1] * r[0]) / denom
    hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return a + t[hit].min() * r if hit.any() else b


def _length(points: np.ndarray) -> np.ndarray:
    """Cumulative length along the polyline, starting at 0."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])


def _area(polygon: np.ndarray) -> float:
    """Shoelace area of the closed `polygon` (N, 2)."""
    x, y = polygon[:, 0], polygon[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def trim_outside_lumen(
    xs: Sequence[float],
    ys: Sequence[float],
    lumen_xs: Sequence[float],
    lumen_ys: Sequence[float],
    samples: int,
    knot_range: tuple[int, int],
) -> tuple[list[float], list[float]] | None:
    """Knots of the open arc of closed contour (xs, ys) outside the lumen, None if it does not cross it.

    Both splines are sampled at `samples` points (config: n_points_contour). The arc starts and
    ends on the crossings with the lumen border. If the contour crosses the lumen more than
    twice, the longest part outside is kept. Its knot count is the contour's own, scaled by how
    much of the contour that part is, within `knot_range` (config: n_interactive_points_range).
    """
    if len(xs) < 3 or len(lumen_xs) < 3:
        return None
    branch = np.column_stack(sample_spline(xs, ys, True, samples))[:-1]  # the last sample repeats the first
    lumen = np.column_stack(sample_spline(lumen_xs, lumen_ys, True, samples))[:-1]

    outside = ~_inside(branch, lumen)
    if outside.all() or not outside.any():
        return None

    # Start at an outside sample following an inside one, so no outside run wraps round the end.
    n = len(branch)
    first = int(np.flatnonzero(outside & ~np.roll(outside, 1))[0])
    order = np.roll(np.arange(n), -first)
    edges = np.diff(np.concatenate([[0], outside[order].astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    longest = int(np.argmax(ends - starts))
    idx = order[starts[longest] : ends[longest]]
    if len(idx) < 2:
        return None

    start = _crossing(branch[idx[0]], branch[(idx[0] - 1) % n], lumen)
    end = _crossing(branch[idx[-1]], branch[(idx[-1] + 1) % n], lumen)
    arc = np.vstack([start, branch[idx], end])

    arc_length = _length(arc)
    closed_length = _length(np.vstack([branch, branch[:1]]))[-1]
    count = int(np.clip(round(len(xs) * arc_length[-1] / closed_length), *knot_range))
    targets = np.linspace(0.0, arc_length[-1], count)
    return np.interp(targets, arc_length, arc[:, 0]).tolist(), np.interp(targets, arc_length, arc[:, 1]).tolist()


def close_against_lumen(arc: np.ndarray, lumen: np.ndarray) -> np.ndarray:
    """Closed polygon (N, 2) of a dense open arc and the dense lumen border between its ends.

    Of the two ways round the lumen from end to start, the one giving the smaller region is
    taken, so the polygon is the strip between arc and lumen rather than that strip plus the
    lumen. Ends off the border are joined to their nearest border point by a straight edge.
    """
    n = len(lumen)
    i_start = int(np.argmin(np.hypot(*(lumen - arc[0]).T)))
    i_end = int(np.argmin(np.hypot(*(lumen - arc[-1]).T)))
    forward = (i_end + np.arange((i_start - i_end) % n + 1)) % n
    backward = (i_end - np.arange((i_end - i_start) % n + 1)) % n
    return min((np.vstack([arc, lumen[path]]) for path in (forward, backward)), key=_area)
