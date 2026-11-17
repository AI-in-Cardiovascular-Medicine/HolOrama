"""Reference points set from side branches.

A side branch leaving the lumen is a landmark, so it sets a reference point by itself: on
the middle of its region along its axis (see tools.branch_arc.branch_landmark). Only a
branch cut at the lumen (an open one) at least `branch_reference_min_diameter_mm` wide
counts, and of several on a frame only the widest.

A branch shows on several frames in a row, which should give it one reference, not one per
frame. Each run of such frames (see branch_metrics.branch_runs) keeps one reference, on the
frame where the branch is widest. A reference placed by hand is never moved, and a run
holding one gets none.
"""

from __future__ import annotations

from domain.intravascular.io_types import FrameData
from input_output.output.branch_metrics import branch_runs


def update_branch_references(main_window) -> bool:
    """Place, move or clear the automatic reference points of the whole pullback. True if any changed."""
    fdd: dict[int, FrameData] = main_window.runtime_data.frame_data_dct

    placed: set[int] = set()
    changed = False
    for run in branch_runs(main_window):
        if any(fdd[f].reference is not None and not fdd[f].reference_auto for f in run.frames):
            continue
        placed.add(run.reference_frame)
        fd = fdd[run.reference_frame]
        if fd.reference is None or tuple(fd.reference) != run.landmark or not fd.reference_auto:
            fd.reference, fd.reference_auto = run.landmark, True
            changed = True

    for frame, fd in fdd.items():
        if fd.reference_auto and frame not in placed:
            fd.reference, fd.reference_auto = None, False
            changed = True
    return changed
