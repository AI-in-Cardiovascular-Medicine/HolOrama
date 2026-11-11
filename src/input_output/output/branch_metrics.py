"""Side branches measured over the pullback: diameters, the widest branch per frame, and runs.

A side branch is any type closed against the lumen (see imgs_masks.closes_on_lumen). The runs
are what both the automatic reference points (pages.intravascular.utils.branch_references)
and the exported annotations (input_output.output.annotations) are built from, so both agree
on which frames show one branch and which of them is its reference frame.
"""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from domain.intravascular.contour_presets import ContourPreset, active_preset
from domain.intravascular.io_types import FrameData
from domain.intravascular.types import BranchMeasurement, BranchRun
from input_output.output.imgs_masks import closes_on_lumen, lumen_knots
from tools.branch_arc import branch_diameter, branch_landmark

PHASES = ('D', 'S', 'T')

# Per frame: what its widest branch was computed from, and the result.
_cache: dict[int, tuple[tuple, tuple[BranchMeasurement, tuple[float, float]] | None]] = {}


def branch_keys(preset: ContourPreset) -> list[str]:
    """Ids of the preset's side-branch types."""
    return [defn.type.value for defn in preset.spline_types if closes_on_lumen(defn, preset)]


def branch_diameters(main_window, frame, samples: int) -> list[BranchMeasurement]:
    """Every side branch's diameter (mm) on `frame`, with the two ends of its line (unscaled px).

    Measured by tools.branch_arc.branch_diameter on splines sampled at `samples` points. Each
    type's largest is stored as its measurements.major_axis, None if it has no contour to measure.
    """
    fd = main_window.runtime_data.frame_data_dct.get(frame)
    if fd is None:
        return []
    preset = active_preset()
    resolution = main_window.runtime_data.metadata['resolution']
    lumen = lumen_knots(fd.lumen)
    lumen_xs, lumen_ys = lumen if lumen is not None else (None, None)

    diameters = []
    for key in branch_keys(preset):
        contour_obj = fd.contour(key)
        largest = None
        for idx, entry in enumerate(contour_obj.contours):
            if not entry or not entry[0]:
                continue
            closed = contour_obj.closed[idx] if idx < len(contour_obj.closed) else True
            try:
                measured = branch_diameter(entry[0], entry[1], closed, lumen_xs, lumen_ys, fd.centroid, samples)
            except Exception as exc:
                logger.debug(f'Frame {frame}: {key} diameter failed: {exc}')
                continue
            if measured is None:
                continue
            diameter = measured[0] * resolution
            start, end = ((float(p[0]), float(p[1])) for p in measured[1:])
            diameters.append(BranchMeasurement(key, idx, closed, diameter, start, end))
            largest = diameter if largest is None else max(largest, diameter)
        contour_obj.measurements.major_axis = largest
    return diameters


def branch_runs(main_window) -> list[BranchRun]:
    """Every run of neighbouring frames whose widest branch leaves the lumen wider than
    `branch_reference_min_diameter_mm`, with the frame it is widest on.

    Frames are grouped by phase (diastolic, systolic, tagged), or taken all together when none
    are tagged. Within a group, consecutive frames of that group with such a branch form a run.
    """
    fdd: dict[int, FrameData] = main_window.runtime_data.frame_data_dct
    min_diameter = float(main_window.config.intravascular.branch_reference_min_diameter_mm)
    samples = main_window.display.n_points_contour
    keys = branch_keys(active_preset())

    best = {}
    for frame in sorted(fdd):
        if any(fdd[frame].contour(key).contours for key in keys):
            found = _widest_branch(main_window, frame, keys, samples, min_diameter)
            if found is not None:
                best[frame] = found

    runs = []
    for frames in _neighbouring(fdd, set(best)):
        reference = max(frames, key=lambda f: best[f][0].diameter)
        runs.append(
            BranchRun(
                frames=tuple(frames),
                widest={f: best[f][0] for f in frames},
                reference_frame=reference,
                landmark=best[reference][1],
            )
        )
    return runs


def _widest_branch(
    main_window, frame: int, keys: list[str], samples: int, min_diameter: float
) -> tuple[BranchMeasurement, tuple[float, float]] | None:
    """The frame's widest branch leaving the lumen and its landmark, None if none is wide enough."""
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
            result = (widest, (float(point[0]), float(point[1])))
    _cache[frame] = (signature, result)
    return result


def _frozen(value):
    """`value`'s nested lists as tuples, so it can be compared and kept."""
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def _neighbouring(fdd: dict[int, FrameData], candidates: set[int]) -> Iterator[list[int]]:
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
