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

    def items(self) -> list[T]:
        """Oldest first, for saving the history in another snapshot."""
        return list(self._stack)

    def replace(self, items: list[T]) -> None:
        """Restore a history saved with items()."""
        self._stack.clear()
        self._stack.extend(items)

    @property
    def can_undo(self) -> bool:
        return bool(self._stack)

    def __len__(self) -> int:
        return len(self._stack)
