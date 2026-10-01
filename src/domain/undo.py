from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from domain.io_types import FRAME_ANNOTATION_FIELDS, is_contour_key

if TYPE_CHECKING:
    from domain.io_types import Contour
    from domain.runtime_types import RuntimeData

T = TypeVar('T')


class UndoStack(Generic[T]):
    """Bounded LIFO history of the last `maxlen` snapshots."""

    def __init__(self, maxlen: int = 5) -> None:
        self._stack: deque[T] = deque(maxlen=maxlen)

    def push(self, snapshot: T) -> None:
        self._stack.append(snapshot)

    def pop(self) -> T | None:
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        self._stack.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._stack)


@dataclass
class ContourSnapshot:
    frame: int
    key: str
    contour: Contour
    active_index: int


@dataclass
class FrameAnnotationSnapshot:
    """Every annotation on one frame, for edits that clear the lot in one go."""

    frame: int
    fields: dict


@dataclass
class PullbackContoursSnapshot:
    """Every frame's contours and lumen centroid, for edits that replace them all at once
    (reading a mask). `frame` is where to go back to."""

    frame: int
    frames: dict  # frame -> (contours, centroid)


def push_pullback_contours_snapshot(runtime_data: RuntimeData, frame: int) -> None:
    runtime_data.mark_unsaved()
    if runtime_data.frame_data_dct is None:
        return
    runtime_data.contour_undo.push(
        PullbackContoursSnapshot(
            frame=frame,
            frames={
                index: (copy.deepcopy(fd.contours), fd.centroid) for index, fd in runtime_data.frame_data_dct.items()
            },
        )
    )


def push_frame_annotation_snapshot(runtime_data: RuntimeData, frame: int) -> None:
    """Record every contour, measurement and derived value on `frame` before it is wiped.

    One entry rather than one per contour type: the stack keeps only the last few edits,
    so ten separate snapshots would evict the rest of the history and still need ten
    Ctrl+Z presses to walk back what the user did with a single click.
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
    """Record the current state of the contour `key` on `frame` before it gets mutated.

    Taking a snapshot means an edit is about to happen, so this also flags the frame data
    as unsaved — every caller is by definition a contour-changing operation.
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
