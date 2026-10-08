"""The spline through a contour's knots, shared by the display, the masks and the resampling
so all three draw the same curve."""

from typing import Sequence

import numpy as np
from scipy.interpolate import splev, splprep


def sample_spline(
    xs: Sequence[float], ys: Sequence[float], closed: bool, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """`samples` points along the spline through knots `xs`, `ys`, end to end."""
    tck, u = fit_spline(xs, ys, closed)
    return splev(np.linspace(u.min(), u.max(), samples), tck)


def fit_spline(xs: Sequence[float], ys: Sequence[float], closed: bool):
    """The interpolating spline through knots `xs`, `ys`: its tck and each knot's parameter.

    A closed contour is closed by repeating its first knot unless it already ends on it, and
    that repeat gets a parameter too."""
    xs, ys = list(xs), list(ys)
    if closed and (xs[0] != xs[-1] or ys[0] != ys[-1]):
        xs, ys = xs + [xs[0]], ys + [ys[0]]
    tck, u = splprep(np.array([xs, ys]), s=0.0, k=min(3, len(xs) - 1), per=int(closed))
    return tck, u
