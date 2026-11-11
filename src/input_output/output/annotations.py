"""Everything drawn on a frame that is not a pipe, as multimodars annotations.

multimodars carries these through every per-frame alignment with the lumen, so fusion can show
them in 3D (see multimodars' Annotation, read from `{phase}_annotations.csv`):

- side branches: every branch's region (the arc closed against the lumen) as `sidebranch` /
  `region`, grouped by run (see branch_metrics.branch_runs). On a run's reference frame also its
  `landmark`, `axis_tip` (the middle of the arc) and the two `diameter_end` points, from which
  fusion sweeps the branch.
- plaques and any other region type: each connected part of its filled region outline as
  `region`, the same region the exported mask holds (an open arc closed against its container).
- reference points: `reference` / `point`, group 1 if set from a side branch, 0 if by hand.
"""

from __future__ import annotations

import cv2
import numpy as np

from domain.intravascular.contour_presets import active_preset
from domain.intravascular.types import FrameAnnotation
from input_output.output.branch_metrics import branch_keys, branch_runs
from input_output.output.imgs_masks import closes_on_lumen, frame_regions, lumen_knots, measured_types
from tools.branch_arc import branch_axis, branch_region

# Preset ids whose multimodars kind is named differently. Any other id is used as it is.
KIND_NAMES = {'calcium': 'calcification'}
SIDEBRANCH = 'sidebranch'
REFERENCE_MANUAL, REFERENCE_AUTO = 0, 1


def pullback_annotations(main_window, frames, image_shape: tuple[int, int]) -> list[FrameAnnotation]:
    """The annotations of `frames`, in unscaled px, splines and outlines sampled at n_points_contour."""
    preset = active_preset()
    fdd = main_window.runtime_data.frame_data_dct
    samples = main_window.display.n_points_contour
    keys = branch_keys(preset)
    plaques = [defn.type.value for defn in measured_types(preset) if not closes_on_lumen(defn, preset)]

    runs = branch_runs(main_window)
    run_group = {
        (frame, run.widest[frame].key, run.widest[frame].index): group
        for group, run in enumerate(runs, start=1)
        for frame in run.frames
    }
    next_group = len(runs) + 1  # branches in no run get groups of their own

    out: list[FrameAnnotation] = []
    for frame in frames:
        fd = fdd.get(frame)
        if fd is None:
            continue
        lumen = lumen_knots(fd.lumen)
        lumen_xs, lumen_ys = lumen if lumen is not None else (None, None)

        for key in keys:
            contour_obj = fd.contour(key)
            for idx, entry in enumerate(contour_obj.contours):
                if not entry or len(entry[0]) < 2:
                    continue
                closed = contour_obj.closed[idx] if idx < len(contour_obj.closed) else True
                group = run_group.get((frame, key, idx))
                if group is None:
                    group, next_group = next_group, next_group + 1
                region = branch_region(entry[0], entry[1], closed, lumen_xs, lumen_ys, samples)
                out.append(_annotation(frame, SIDEBRANCH, 'region', group, region))

        for group, run in enumerate(runs, start=1):
            if run.reference_frame == frame:
                out.extend(_sweep_points(fd, run, group, lumen, samples))

        if any(fd.contour(key).contours for key in plaques):
            regions = frame_regions(fd, image_shape, preset)
            for key in plaques:
                for part, outline in enumerate(_outlines(regions[key], samples)):
                    out.append(_annotation(frame, KIND_NAMES.get(key, key), 'region', part, outline))

        if fd.reference is not None:
            group = REFERENCE_AUTO if fd.reference_auto else REFERENCE_MANUAL
            out.append(_annotation(frame, 'reference', 'point', group, np.asarray([fd.reference], dtype=float)))
    return out


def _sweep_points(fd, run, group: int, lumen, samples: int) -> list[FrameAnnotation]:
    """A run's landmark, axis tip and diameter ends on its reference frame, for the swept tube."""
    branch = run.widest[run.reference_frame]
    xs, ys = fd.contour(branch.key).contours[branch.index][:2]
    axis = branch_axis(xs, ys, lumen[0], lumen[1], fd.centroid, samples) if lumen is not None else None
    frame = run.reference_frame
    points = [
        _annotation(frame, SIDEBRANCH, 'landmark', group, np.asarray([run.landmark])),
        _annotation(frame, SIDEBRANCH, 'diameter_end', group, np.asarray([branch.start, branch.end])),
    ]
    if axis is not None:
        points.append(_annotation(frame, SIDEBRANCH, 'axis_tip', group, axis[2][None, :]))
    return points


def _outlines(mask: np.ndarray, samples: int) -> list[np.ndarray]:
    """Outline (N, 2) x, y of each connected part of `mask`, at most `samples` points each."""
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outlines = []
    for contour in found:
        points = contour[:, 0, :].astype(float)
        if len(points) < 3:
            continue
        if len(points) > samples:
            points = points[np.linspace(0, len(points) - 1, samples).round().astype(int)]
        outlines.append(points)
    return outlines


def _annotation(frame: int, kind: str, role: str, group: int, points: np.ndarray) -> FrameAnnotation:
    return FrameAnnotation(frame, kind, role, group, tuple((float(x), float(y)) for x, y in points))
