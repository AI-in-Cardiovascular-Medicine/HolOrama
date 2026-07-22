from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QWidget


class SplitterPane(QWidget):
    """QWidget whose minimumSizeHint() always returns QSize(1, 1).

    QSplitter reads minimumSizeHint() — not minimumSize() — to compute the
    legal range for the handle.  Using a plain QWidget causes the 7-button row
    (~700-900 px minimum) or the NavigationToolbar (~480 px minimum) to lock
    the handle at those large positions.  This subclass removes that constraint
    so the handle can be positioned freely.
    """

    def minimumSizeHint(self) -> QSize:
        return QSize(1, 1)


def connect_consecutive_frames(missing: list) -> str:
    nums = sorted(set(missing))
    groups: list[list[int]] = []
    i = 0
    while i < len(nums):
        j = i
        while j < len(nums) - 1 and nums[j + 1] - nums[j] == 1:
            j += 1
        if i == j:
            groups.append([nums[i]])
        else:
            groups.append(nums[i : j + 1])
        i = j + 1
    connected = [
        (f'{sublist[0]}-{sublist[-1]}' if len(sublist) > 2 else ", ".join(map(str, sublist))) for sublist in groups
    ]
    return ", ".join(connected)
