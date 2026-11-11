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


def branch_region(
    xs: Sequence[float],
    ys: Sequence[float],
    closed: bool,
    lumen_xs: Sequence[float] | None,
    lumen_ys: Sequence[float] | None,
    samples: int,
) -> np.ndarray:
    """Closed polygon (N, 2) a branch contour fills, splines sampled at `samples` points.

    A closed contour is its own polygon. An open arc is closed against the lumen (see
    close_against_lumen), or by its own chord while there is no lumen.
    """
    arc = np.column_stack(sample_spline(xs, ys, closed, samples))
    if closed or lumen_xs is None or lumen_ys is None or len(lumen_xs) < 3:
        return arc
    lumen = np.column_stack(sample_spline(lumen_xs, lumen_ys, True, samples))
    return close_against_lumen(arc, lumen)


def farthest_pair(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Longest distance between any two of `points` (N, 2), and those two points."""
    diff = points[:, None, :] - points[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(dist[i, j]), points[i], points[j]


def orthogonal_width(
    region: np.ndarray, origin: np.ndarray, tip: np.ndarray, samples: int
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Widest chord across `region` orthogonal to the axis origin->tip, and its two ends.

    Chords are laid at `samples` positions along the axis, over the region's whole extent
    along it. Each is measured inside the region only, so a chord crossing a dent counts
    its longest inside stretch. None if the axis has no direction.
    """
    axis = tip - origin
    norm = float(np.hypot(*axis))
    if norm == 0:
        return None
    d = axis / norm
    n = np.array([-d[1], d[0]])
    rel = region - origin
    u, v = rel @ d, rel @ n
    u1, v1 = np.roll(u, -1), np.roll(v, -1)

    best: tuple[float, float, float, float] | None = None  # width, u, v_from, v_to
    for pos in np.linspace(u.min(), u.max(), samples)[1:-1]:
        straddles = (u <= pos) != (u1 <= pos)
        if not straddles.any():
            continue
        vs = np.sort(
            v[straddles] + (pos - u[straddles]) * (v1[straddles] - v[straddles]) / (u1[straddles] - u[straddles])
        )
        widths = vs[1::2] - vs[0::2][: len(vs) // 2]
        if not len(widths):
            continue
        k = int(np.argmax(widths))
        if best is None or widths[k] > best[0]:
            best = (
                float(widths[k]),
                float(pos),
                float(vs[2 * k]),
                float(vs[2 * k + 1]),
            )
    if best is None:
        return None
    width, pos, v_from, v_to = best
    return width, origin + pos * d + v_from * n, origin + pos * d + v_to * n


def branch_axis(
    xs: Sequence[float],
    ys: Sequence[float],
    lumen_xs: Sequence[float] | None,
    lumen_ys: Sequence[float] | None,
    centroid: tuple[float, float] | None,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """An open branch's region and its axis, from the lumen `centroid` to the middle of the arc.

    None without a lumen to close the arc against. The centroid falls back to the lumen's mean point.
    """
    if len(xs) < 2 or lumen_xs is None or lumen_ys is None or len(lumen_xs) < 3:
        return None
    region = branch_region(xs, ys, False, lumen_xs, lumen_ys, samples)
    if centroid is None:
        origin = np.column_stack(sample_spline(lumen_xs, lumen_ys, True, samples)).mean(axis=0)
    else:
        origin = np.asarray(centroid, dtype=float)
    arc = np.column_stack(sample_spline(xs, ys, False, samples))
    length = _length(arc)
    middle = np.array([np.interp(length[-1] / 2, length, arc[:, i]) for i in (0, 1)])
    return region, origin, middle


def branch_diameter(
    xs: Sequence[float],
    ys: Sequence[float],
    closed: bool,
    lumen_xs: Sequence[float] | None,
    lumen_ys: Sequence[float] | None,
    centroid: tuple[float, float] | None,
    samples: int,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """A side branch's diameter in pixels and the two ends of the line it is measured on.

    Closed: its longest distance, as the lumen's. Open: the widest chord across the region it
    fills, orthogonal to the axis from the lumen `centroid` to the middle of the arc (see
    orthogonal_width). None for an open arc without a lumen to close it against.
    """
    if closed:
        return farthest_pair(branch_region(xs, ys, True, None, None, samples)) if len(xs) >= 3 else None
    axis = branch_axis(xs, ys, lumen_xs, lumen_ys, centroid, samples)
    return orthogonal_width(*axis, samples) if axis is not None else None


def branch_landmark(
    xs: Sequence[float],
    ys: Sequence[float],
    lumen_xs: Sequence[float] | None,
    lumen_ys: Sequence[float] | None,
    centroid: tuple[float, float] | None,
    samples: int,
) -> np.ndarray | None:
    """Where a reference point goes on an open branch: the centre of its region along its axis.

    The axis is the diameter's (lumen `centroid` to the middle of the arc). Of the stretches of
    the axis inside the region, the one reaching furthest out (to the arc) is taken, and its
    midpoint returned. None without a lumen, or if the axis misses the region.
    """
    axis = branch_axis(xs, ys, lumen_xs, lumen_ys, centroid, samples)
    if axis is None:
        return None
    region, origin, tip = axis
    direction = tip - origin
    norm = float(np.hypot(*direction))
    if norm == 0:
        return None
    d = direction / norm
    rel = region - origin
    u, v = rel @ d, rel @ np.array([-d[1], d[0]])
    u1, v1 = np.roll(u, -1), np.roll(v, -1)
    crosses = (v <= 0) != (v1 <= 0)
    if not crosses.any():
        return None
    us = np.sort(u[crosses] + (0 - v[crosses]) * (u1[crosses] - u[crosses]) / (v1[crosses] - v[crosses]))
    if len(us) < 2:
        return None
    inner, outer = us[-2], us[-1]
    return origin + (inner + outer) / 2 * d
