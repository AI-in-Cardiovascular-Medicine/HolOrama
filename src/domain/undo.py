from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

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
