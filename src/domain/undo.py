from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

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


def push_contour_snapshot(runtime_data: RuntimeData, frame: int, key: str, active_index: int) -> None:
    """Record the current state of `frame_data_dct[frame].<key>` before it gets mutated."""
    if runtime_data.frame_data_dct is None:
        return
    fd = runtime_data.frame_data_dct.get(frame)
    if fd is None:
        return
    contour_obj = getattr(fd, key, None)
    if contour_obj is None:
        return
    runtime_data.contour_undo.push(
        ContourSnapshot(frame=frame, key=key, contour=copy.deepcopy(contour_obj), active_index=active_index)
    )
