"""Changing how many knots a contour has while keeping its shape.

The shape kept is the contour as it was when the count was first changed: the spline the
display draws through its original knots (the *reference*). Every count is reached one
knot at a time from the one next to it, and remembered, so walking the count down and back
up returns the very knots passed on the way — the original ones included — instead of
resampling an already thinned contour and losing what was thinned away.

* One knot fewer: whichever knot the spline through the rest strays least from the
  reference without. Knots that must stay where they are — an open contour's ends, a knot
  labelled start or end — are never taken.
* One knot more: on the reference, where the spline through the current knots strays
  furthest from it; where it strays nowhere, halfway along the longest stretch between two
  knots.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

MIN_KNOTS = 3
MAX_KNOTS = 40
_SAMPLES = 400  # points along the reference the shape is compared at
_FLAT_PX = 0.25  # a spline this close to the reference everywhere already follows it
_PIN_PX = 1.0  # how close a label has to sit to a knot to pin it

Knots = tuple[list[float], list[float]]


def _fit(xs: Sequence[float], ys: Sequence[float], closed: bool):
    """The spline the display draws through the knots (see SplineGeometry.interpolate):
    its tck, and the parameter of each knot (without the closing repeat)."""
    xs, ys = list(xs), list(ys)
    if closed:
        xs, ys = xs + [xs[0]], ys + [ys[0]]
    k = min(3, len(xs) - 1)
    tck, u = splprep(np.array([xs, ys]), s=0.0, k=k, per=int(closed))
    return tck, (u[:-1] if closed else u)


def _curve(points: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    """`samples` points along the spline through `points` (n, 2)."""
    tck, _ = _fit(points[:, 0].tolist(), points[:, 1].tolist(), closed)
    xs, ys = splev(np.linspace(0.0, 1.0, samples, endpoint=not closed), tck)
    return np.column_stack([xs, ys])


def without_closing_repeat(xs: Sequence[float], ys: Sequence[float], closed: bool) -> Knots:
    """The knots of a contour, without the copy of the first a closed one may end on (see
    SplineGeometry._ensure_closed) — as many as it has handles."""
    xs, ys = list(xs), list(ys)
    if closed and len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
        xs, ys = xs[:-1], ys[:-1]
    return xs, ys


class KnotHistory:
    """Every knot count one contour has been resampled to, from its original knots."""

    def __init__(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        closed: bool,
        pinned: Iterable[tuple[float, float]] = (),
    ) -> None:
        xs, ys = without_closing_repeat(xs, ys, closed)
        if len(xs) < MIN_KNOTS - (0 if closed else 1):
            raise ValueError(f'a contour of {len(xs)} knots has no shape to keep')
        self.closed = closed
        self._original: Knots = (xs, ys)
        self._tck, params = _fit(xs, ys, closed)
        self._sample_params = np.linspace(0.0, 1.0, _SAMPLES, endpoint=not closed)
        self._reference = np.column_stack(splev(self._sample_params, self._tck))
        self._reference_tree = cKDTree(self._reference)

        knots = np.column_stack([xs, ys])
        keep = {0, len(xs) - 1} if not closed else set()
        for px, py in pinned:
            distances = np.hypot(knots[:, 0] - px, knots[:, 1] - py)
            if distances.min() <= _PIN_PX:
                keep.add(int(distances.argmin()))
        self._pinned = frozenset(float(params[i]) for i in keep)
        self._states: dict[int, np.ndarray] = {len(xs): np.asarray(params, dtype=float)}

    @property
    def original_count(self) -> int:
        return len(self._original[0])

    @property
    def min_count(self) -> int:
        return max(MIN_KNOTS, len(self._pinned))

    @property
    def max_count(self) -> int:
        return max(MAX_KNOTS, self.original_count)

    def knots(self, count: int) -> Knots:
        """The contour with `count` knots (clamped to what it can have)."""
        count = int(np.clip(count, self.min_count, self.max_count))
        while count < min(self._states):
            fewest = min(self._states)
            fewer = self._remove_one(self._states[fewest])
            if fewer is None:
                break
            self._states[fewest - 1] = fewer
        while count > max(self._states):
            most = max(self._states)
            self._states[most + 1] = self._insert_one(self._states[most])
        count = int(np.clip(count, min(self._states), max(self._states)))
        if count == self.original_count:
            return list(self._original[0]), list(self._original[1])
        xs, ys = splev(self._states[count], self._tck)
        return [float(x) for x in xs], [float(y) for y in ys]

    def holds(self, xs: Sequence[float], ys: Sequence[float], tolerance: float = 1e-4) -> bool:
        """Whether `xs`, `ys` are one of the contours this history has handed out — so
        the contour has not been edited in some other way since."""
        xs, ys = without_closing_repeat(xs, ys, self.closed)
        if len(xs) not in self._states:
            return False
        known_xs, known_ys = self.knots(len(xs))
        return bool(np.allclose(xs, known_xs, atol=tolerance) and np.allclose(ys, known_ys, atol=tolerance))

    def is_original(self, xs: Sequence[float], ys: Sequence[float]) -> bool:
        xs, ys = without_closing_repeat(xs, ys, self.closed)
        return len(xs) == self.original_count and self.holds(xs, ys)

    # -- one step --------------------------------------------------------------------------

    def _points(self, params: np.ndarray) -> np.ndarray:
        return np.column_stack(splev(params, self._tck))

    def _deviation(self, params: np.ndarray) -> np.ndarray:
        """Per reference sample, how far the spline through the knots at `params` is."""
        curve = _curve(self._points(params), self.closed, 2 * _SAMPLES)
        distances, _ = cKDTree(curve).query(self._reference)
        return distances

    def _remove_one(self, params: np.ndarray) -> np.ndarray | None:
        best, best_error = None, np.inf
        for i, param in enumerate(params):
            if float(param) in self._pinned:
                continue
            candidate = np.delete(params, i)
            try:
                error = float(np.mean(self._deviation(candidate) ** 2))
            except Exception:  # a spline through what is left could not be fitted
                continue
            if error < best_error:
                best, best_error = candidate, error
        return best

    def _insert_one(self, params: np.ndarray) -> np.ndarray:
        deviation = self._deviation(params)
        new = float(self._sample_params[int(deviation.argmax())])
        if deviation.max() <= _FLAT_PX or np.any(np.isclose(params, new)):  # never on top of a knot
            new = self._middle_of_longest_stretch(params)
        return np.sort(np.append(params, new))

    def _middle_of_longest_stretch(self, params: np.ndarray) -> float:
        """Halfway between the two knots furthest apart along the reference (a closed
        contour's last stretch runs on round to its first knot)."""
        ends = np.append(params[1:], params[0] + 1.0) if self.closed else params[1:]
        lengths = ends - params[: len(ends)]
        longest = int(lengths.argmax())
        return float((params[longest] + lengths[longest] / 2) % 1.0)
