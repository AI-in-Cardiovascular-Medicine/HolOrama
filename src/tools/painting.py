from dataclasses import dataclass

import numpy as np
from PyQt6.QtGui import QCursor, QPixmap, QPainter, QPen, QColor, QBrush


@dataclass
class BrushGeometry:
    """Pure brush description — no Qt dependency."""

    label: int = 1
    color: tuple[int, int, int] = (255, 60, 60)
    radius_px: int = 10

    def paint(self, mask_slice: np.ndarray, row: int, col: int) -> None:
        """Paint a filled disc of self.label onto a 2-D uint8 mask_slice in-place."""
        h, w = mask_slice.shape
        r = self.radius_px
        row_lo = max(0, row - r)
        row_hi = min(h, row + r + 1)
        col_lo = max(0, col - r)
        col_hi = min(w, col + r + 1)
        rows = np.arange(row_lo, row_hi)[:, None] - row
        cols = np.arange(col_lo, col_hi)[None, :] - col
        disc = rows**2 + cols**2 <= r**2
        mask_slice[row_lo:row_hi, col_lo:col_hi][disc] = self.label


class BrushCursor:
    """
    Builds OS-level QCursor pixmaps for the brush tool.

    Using a QCursor guarantees the circle renders on top of all scene content —
    the OS composites it above the application window.

    radius_px, color, and make_cursor() are public so intravascular can reuse them.
    Pass view_scale = QGraphicsView.transform().m11() so the circle tracks image pixels
    correctly at every zoom level.
    """

    def __init__(self) -> None:
        self._radius_px: int = 10
        self._color: tuple[int, int, int] = (255, 60, 60)

    @property
    def radius_px(self) -> int:
        return self._radius_px

    @property
    def color(self) -> tuple[int, int, int]:
        return self._color

    def update_from_geometry(self, geometry: BrushGeometry) -> None:
        self._radius_px = geometry.radius_px
        self._color = geometry.color

    def make_cursor(self, view_scale: float = 1.0) -> QCursor:
        """
        Return a QCursor whose circle equals radius_px image pixels at view_scale.
        The hotspot is at the circle centre.
        """
        screen_r = max(2, round(self._radius_px * view_scale))
        pad = 2  # extra pixels so the pen stroke isn't clipped
        size = screen_r * 2 + pad * 2

        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))  # fully transparent background

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r, g, b = self._color
        pen = QPen(QColor(r, g, b))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(r, g, b, 50)))
        painter.drawEllipse(pad, pad, screen_r * 2, screen_r * 2)
        painter.end()

        center = size // 2
        return QCursor(pixmap, center, center)


# ---------------------------------------------------------------------------
# Backward-compat alias — remove once intravascular is updated
# ---------------------------------------------------------------------------
BrushCursorItem = BrushCursor
