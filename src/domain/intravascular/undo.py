from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from domain.intravascular.io_types import FRAME_ANNOTATION_FIELDS, is_contour_key

if TYPE_CHECKING:
    from domain.intravascular.io_types import Contour
    from domain.intravascular.runtime_types import RuntimeData


@dataclass
class ContourSnapshot:
    frame: int
    key: str
    contour: Contour
    active_index: int


@dataclass
class FrameAnnotationSnapshot:
    """One frame's annotations, for edits that clear them all."""

    frame: int
    fields: dict


@dataclass
class PullbackContoursSnapshot:
    """Contours and centroid of many frames, for mask import or delete on all frames.
    Undo returns to `frame`."""

    frame: int
    frames: dict  # frame -> (contours, centroid)


def push_pullback_contours_snapshot(runtime_data: RuntimeData, frame: int, frames: Iterable[int] | None = None) -> None:
    """Save the contours of `frames` (default: all) before replacing them."""
    runtime_data.mark_unsaved()
    if runtime_data.frame_data_dct is None:
        return
    indices = runtime_data.frame_data_dct.keys() if frames is None else frames
    runtime_data.contour_undo.push(
        PullbackContoursSnapshot(
            frame=frame,
            frames={
                index: (copy.deepcopy(fd.contours), fd.centroid)
                for index in indices
                if (fd := runtime_data.frame_data_dct.get(index)) is not None
            },
        )
    )


def push_frame_annotation_snapshot(runtime_data: RuntimeData, frame: int) -> None:
    """Record every contour, measurement and derived value on `frame` before it is wiped.

    One entry, not one per contour type: the stack keeps only the last few edits, and
    one click should take one Ctrl+Z to undo without evicting the rest of the history.
    """
    runtime_data.mark_unsaved()
    if runtime_data.frame_data_dct is None:
        return
    fd = runtime_data.frame_data_dct.get(frame)
    if fd is None:
        return
    runtime_data.contour_undo.push(
        FrameAnnotationSnapshot(
            frame=frame,
            fields={name: copy.deepcopy(getattr(fd, name)) for name in FRAME_ANNOTATION_FIELDS},
        )
    )


def push_contour_snapshot(runtime_data: RuntimeData, frame: int, key: str, active_index: int) -> None:
    """Record the contour `key` on `frame` before it is mutated.

    Also flags the frame data as unsaved, since every caller is about to change a contour.
    """
    runtime_data.mark_unsaved()
    if runtime_data.frame_data_dct is None:
        return
    fd = runtime_data.frame_data_dct.get(frame)
    if fd is None:
        return
    if not is_contour_key(key):
        return
    runtime_data.contour_undo.push(
        ContourSnapshot(frame=frame, key=key, contour=copy.deepcopy(fd.contour(key)), active_index=active_index)
    )
