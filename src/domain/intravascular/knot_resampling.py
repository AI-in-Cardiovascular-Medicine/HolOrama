"""Changing how many knots a contour has while keeping its shape.

The shape kept is the spline the display draws through the original knots (the
*reference*). Each count is reached one knot at a time from its neighbour and remembered,
so walking the count down and back up returns the same knots, originals included, instead
of resampling an already thinned contour.

* One knot fewer: drop the knot whose removal leaves the spline closest to the reference.
  Pinned knots (an open contour's ends, a knot labelled start or end) are never dropped.
* One knot more: on the reference, where the spline through the current knots strays
  furthest from it, or halfway along the longest stretch if it strays nowhere.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import splev
from scipy.spatial import cKDTree

from domain.intravascular.contour_fitting_types import FLAT_PX, PIN_PX, REFERENCE_SAMPLES
from tools.spline import fit_spline, sample_spline

Knots = tuple[list[float], list[float]]


class KnotHistory:
    """A contour's knots at every count reached from its originals, within `count_range`
    (config: n_interactive_points_range)."""

    def __init__(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        closed: bool,
        count_range: tuple[int, int],
        pinned: Iterable[tuple[float, float]] = (),
    ) -> None:
        xs, ys = without_closing_repeat(xs, ys, closed)
        if len(xs) < count_range[0] - (0 if closed else 1):
            raise ValueError(f'a contour of {len(xs)} knots has no shape to keep')
        self.closed = closed
        self._original: Knots = (xs, ys)
        self._tck, params = fit_spline(xs, ys, closed)
        params = params[:-1] if closed else params  # the closing repeat is no knot of its own
        self._sample_params = np.linspace(0.0, 1.0, REFERENCE_SAMPLES, endpoint=not closed)
        self._reference = np.column_stack(splev(self._sample_params, self._tck))
        self._pinned = frozenset(float(params[i]) for i in _pinned_indices(xs, ys, closed, pinned))
        self._states: dict[int, np.ndarray] = {len(xs): np.asarray(params, dtype=float)}
        self.min_count = max(count_range[0], len(self._pinned))
        self.max_count = max(count_range[1], len(xs))

    @property
    def original_count(self) -> int:
        return len(self._original[0])

    def knots(self, count: int) -> Knots:
        """The contour at `count` knots, clamped."""
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
        """Whether `xs`, `ys` is a contour this history handed out, so not edited
        otherwise since."""
        xs, ys = without_closing_repeat(xs, ys, self.closed)
        if len(xs) not in self._states:
            return False
        known_xs, known_ys = self.knots(len(xs))
        return bool(np.allclose(xs, known_xs, atol=tolerance) and np.allclose(ys, known_ys, atol=tolerance))

    def is_original(self, xs: Sequence[float], ys: Sequence[float]) -> bool:
        xs, ys = without_closing_repeat(xs, ys, self.closed)
        return len(xs) == self.original_count and self.holds(xs, ys)

    def _remove_one(self, params: np.ndarray) -> np.ndarray | None:
        best, best_error = None, np.inf
        for i, param in enumerate(params):
            if float(param) in self._pinned:
                continue
            candidate = np.delete(params, i)
            try:
                error = float(np.mean(self._deviation(candidate) ** 2))
            except Exception:  # no spline fits what is left
                continue
            if error < best_error:
                best, best_error = candidate, error
        return best

    def _deviation(self, params: np.ndarray) -> np.ndarray:
        """Per reference sample, distance to the spline through the knots at `params`."""
        xs, ys = splev(params, self._tck)
        curve = np.column_stack(sample_spline(xs, ys, self.closed, 2 * REFERENCE_SAMPLES))
        distances, _ = cKDTree(curve).query(self._reference)
        return distances

    def _insert_one(self, params: np.ndarray) -> np.ndarray:
        deviation = self._deviation(params)
        new = float(self._sample_params[int(deviation.argmax())])
        if deviation.max() <= FLAT_PX or np.any(np.isclose(params, new)):  # never on top of a knot
            new = self._middle_of_longest_stretch(params)
        return np.sort(np.append(params, new))

    def _middle_of_longest_stretch(self, params: np.ndarray) -> float:
        """Midpoint of the longest knot gap along the reference (closed: the last
        gap wraps to the first knot)."""
        ends = np.append(params[1:], params[0] + 1.0) if self.closed else params[1:]
        lengths = ends - params[: len(ends)]
        longest = int(lengths.argmax())
        return float((params[longest] + lengths[longest] / 2) % 1.0)


def without_closing_repeat(xs: Sequence[float], ys: Sequence[float], closed: bool) -> Knots:
    """A contour's knots without the closing repeat of the first (see
    SplineGeometry._ensure_closed), one per handle."""
    xs, ys = list(xs), list(ys)
    if closed and len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
        xs, ys = xs[:-1], ys[:-1]
    return xs, ys


def _pinned_indices(xs: list[float], ys: list[float], closed: bool, pinned: Iterable[tuple[float, float]]) -> set[int]:
    """The knots that must stay: an open contour's ends, and those a `pinned` label sits on."""
    knots = np.column_stack([xs, ys])
    keep = set() if closed else {0, len(xs) - 1}
    for px, py in pinned:
        distances = np.hypot(knots[:, 0] - px, knots[:, 1] - py)
        if distances.min() <= PIN_PX:
            keep.add(int(distances.argmin()))
    return keep
