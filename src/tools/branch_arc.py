"""A side branch drawn over the lumen, kept as the open arc outside it.

A closed branch contour that crosses the lumen is cut at the two crossings and only the
part outside the lumen is kept, as an open arc from crossing to crossing. Its region is
then the arc closed back along the lumen border (see close_against_lumen).
"""

from typing import Sequence

import numpy as np

from tools.spline import sample_spline

_N_DENSE = 500  # samples along each spline for the intersection
_MIN_KNOTS = 4


def _inside(px: np.ndarray, py: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Even-odd test of points (px, py) against the closed polygon (poly_x, poly_y)."""
    x0, y0 = poly_x[None, :], poly_y[None, :]
    x1, y1 = np.roll(poly_x, -1)[None, :], np.roll(poly_y, -1)[None, :]
    py_c, px_c = py[:, None], px[:, None]
    straddles = (y0 > py_c) != (y1 > py_c)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_cross = x0 + (py_c - y0) * (x1 - x0) / (y1 - y0)
    return (straddles & (px_c < x_cross)).sum(axis=1) % 2 == 1


def _crossing(a: np.ndarray, b: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Where segment a->b crosses the closed polygon, the crossing nearest to a. b if none is found."""
    p = np.column_stack([poly_x, poly_y])
    q = np.roll(p, -1, axis=0)
    r = b - a
    s = q - p
    denom = r[0] * s[:, 1] - r[1] * s[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((p[:, 0] - a[0]) * s[:, 1] - (p[:, 1] - a[1]) * s[:, 0]) / denom
        u = ((p[:, 0] - a[0]) * r[1] - (p[:, 1] - a[1]) * r[0]) / denom
    hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if not hit.any():
        return b
    return a + t[hit].min() * r


def _resample(xs: np.ndarray, ys: np.ndarray, n: int) -> tuple[list[float], list[float]]:
    """n points evenly spaced by arc length along the polyline, first and last kept."""
    seg = np.hypot(np.diff(xs), np.diff(ys))
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, dist[-1], n)
    return np.interp(targets, dist, xs).tolist(), np.interp(targets, dist, ys).tolist()


def trim_outside_lumen(
    xs: Sequence[float],
    ys: Sequence[float],
    lumen_xs: Sequence[float],
    lumen_ys: Sequence[float],
    n_knots: int,
) -> tuple[list[float], list[float]] | None:
    """Knots of the open arc of closed contour (xs, ys) outside the lumen, None if it does not cross it.

    The arc starts and ends on the crossings with the lumen border. If the contour crosses the
    lumen more than twice, the longest part outside is kept. Its knot count is `n_knots` scaled
    by how much of the contour that part is.
    """
    if len(xs) < 3 or len(lumen_xs) < 3:
        return None
    bx, by = sample_spline(xs, ys, True, _N_DENSE)
    lx, ly = sample_spline(lumen_xs, lumen_ys, True, _N_DENSE)
    bx, by = np.asarray(bx)[:-1], np.asarray(by)[:-1]  # the last sample repeats the first
    lx, ly = np.asarray(lx)[:-1], np.asarray(ly)[:-1]

    outside = ~_inside(bx, by, lx, ly)
    if outside.all() or not outside.any():
        return None

    # Rotate so index 0 is the first outside sample after an inside one, then take the longest outside run.
    first = int(np.flatnonzero(outside & ~np.roll(outside, 1))[0])
    order = np.roll(np.arange(len(bx)), -first)
    out = outside[order]
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(out):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(out)))
    run_start, run_end = max(runs, key=lambda r: r[1] - r[0])
    idx = order[run_start:run_end]
    if len(idx) < 2:
        return None

    pts = np.column_stack([bx, by])
    n = len(bx)
    start_pt = _crossing(pts[idx[0]], pts[(idx[0] - 1) % n], lx, ly)
    end_pt = _crossing(pts[idx[-1]], pts[(idx[-1] + 1) % n], lx, ly)
    arc = np.vstack([start_pt, pts[idx], end_pt])

    n_arc = max(_MIN_KNOTS, round(n_knots * len(idx) / n))
    return _resample(arc[:, 0], arc[:, 1], n_arc)


def close_against_lumen(
    xs: Sequence[float],
    ys: Sequence[float],
    lumen_xs: Sequence[float],
    lumen_ys: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Closed polygon of the open arc (xs, ys) and the lumen border between its ends.

    Of the two ways round the lumen from end to start, the one giving the smaller region is
    taken, so the polygon is the strip between arc and lumen rather than that strip plus the
    lumen. Ends off the border are joined to their nearest border point by a straight edge.
    """
    ax, ay = (np.asarray(v) for v in sample_spline(xs, ys, False, _N_DENSE))
    lx, ly = (np.asarray(v)[:-1] for v in sample_spline(lumen_xs, lumen_ys, True, _N_DENSE))
    n = len(lx)
    i_start = int(np.argmin(np.hypot(lx - ax[0], ly - ay[0])))
    i_end = int(np.argmin(np.hypot(lx - ax[-1], ly - ay[-1])))

    forward = (i_end + np.arange((i_start - i_end) % n + 1)) % n
    backward = (i_end - np.arange((i_end - i_start) % n + 1)) % n

    def polygon(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate([ax, lx[path]]), np.concatenate([ay, ly[path]])

    def area(px: np.ndarray, py: np.ndarray) -> float:
        return abs(float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))) / 2.0

    candidates = [polygon(forward), polygon(backward)]
    return min(candidates, key=lambda c: area(*c))
