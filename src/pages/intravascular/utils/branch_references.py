"""Reference points set from side branches.

A side branch leaving the lumen is a landmark, so it sets a reference point by itself: on
the middle of its region along its axis (see tools.branch_arc.branch_landmark). Only a
branch cut at the lumen (an open one) at least `branch_reference_min_diameter_mm` wide
counts, and of several on a frame only the widest.

A branch shows on several frames in a row, which should give it one reference, not one per
frame. Frames are grouped by phase (diastolic, systolic, tagged, each exported with its own
references), or taken all together when none are tagged. Within a group, neighbouring
frames with a branch form a run, and only the frame with the widest branch keeps the
reference. A reference placed by hand is never moved, and a run holding one gets none.
"""

from __future__ import annotations

from typing import Iterator

from domain.intravascular.contour_presets import active_preset
from domain.intravascular.io_types import FrameData
from input_output.output.imgs_masks import closes_on_lumen, lumen_knots
from input_output.output.reports import branch_diameters
from tools.branch_arc import branch_landmark

PHASES = ('D', 'S', 'T')

# Per frame: what its best branch was computed from, and the result (diameter mm, landmark).
_cache: dict[int, tuple[tuple, tuple[float, tuple[float, float]] | None]] = {}


def update_branch_references(main_window) -> bool:
    """Place, move or clear the automatic reference points of the whole pullback. True if any changed."""
    fdd: dict[int, FrameData] = main_window.runtime_data.frame_data_dct
    min_diameter = float(main_window.config.intravascular.branch_reference_min_diameter_mm)
    samples = main_window.display.n_points_contour
    keys = [defn.type.value for defn in active_preset().spline_types if closes_on_lumen(defn, active_preset())]

    best = {}
    for frame in sorted(fdd):
        if any(fdd[frame].contour(key).contours for key in keys):
            found = _best_branch(main_window, frame, keys, samples, min_diameter)
            if found is not None:
                best[frame] = found

    placed: set[int] = set()
    changed = False
    for run in _runs(fdd, set(best)):
        if any(fdd[f].reference is not None and not fdd[f].reference_auto for f in run):
            continue
        frame = max(run, key=lambda f: best[f][0])
        point = best[frame][1]
        placed.add(frame)
        fd = fdd[frame]
        if fd.reference is None or tuple(fd.reference) != point or not fd.reference_auto:
            fd.reference, fd.reference_auto = point, True
            changed = True

    for frame, fd in fdd.items():
        if fd.reference_auto and frame not in placed:
            fd.reference, fd.reference_auto = None, False
            changed = True
    return changed


def _best_branch(
    main_window, frame: int, keys: list[str], samples: int, min_diameter: float
) -> tuple[float, tuple[float, float]] | None:
    """(diameter mm, landmark) of the frame's widest branch leaving the lumen, None if none is wide enough."""
    fd = main_window.runtime_data.frame_data_dct[frame]
    lumen = lumen_knots(fd.lumen)
    signature = (
        samples,
        min_diameter,
        main_window.runtime_data.metadata['resolution'],
        _frozen(lumen),
        tuple(fd.centroid) if fd.centroid is not None else None,
        tuple((key, _frozen(fd.contour(key).contours), tuple(fd.contour(key).closed)) for key in keys),
    )
    cached = _cache.get(frame)
    if cached is not None and cached[0] == signature:
        return cached[1]

    result = None
    candidates = [
        b for b in branch_diameters(main_window, frame, samples) if not b.closed and b.diameter > min_diameter
    ]
    if candidates and lumen is not None:
        widest = max(candidates, key=lambda b: b.diameter)
        xs, ys = fd.contour(widest.key).contours[widest.index][:2]
        point = branch_landmark(xs, ys, lumen[0], lumen[1], fd.centroid, samples)
        if point is not None:
            result = (widest.diameter, (float(point[0]), float(point[1])))
    _cache[frame] = (signature, result)
    return result


def _frozen(value):
    """`value`'s nested lists as tuples, so it can be compared and kept."""
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def _runs(fdd: dict[int, FrameData], candidates: set[int]) -> Iterator[list[int]]:
    """Runs of neighbouring candidate frames, per phase group (all frames when none are tagged)."""
    ordered = sorted(fdd)
    tagged = any(fdd[f].phase in PHASES for f in ordered)
    groups = [[f for f in ordered if fdd[f].phase == phase] for phase in PHASES] if tagged else [ordered]
    for group in groups:
        run: list[int] = []
        for frame in group:
            if frame not in candidates:
                if run:
                    yield run
                run = []
            elif run and not tagged and frame != run[-1] + 1:
                yield run
                run = [frame]
            else:
                run.append(frame)
        if run:
            yield run
