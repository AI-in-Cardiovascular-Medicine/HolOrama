import cv2
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor


_CROSSHAIR_COLOR = QColor(255, 255, 0)
_ZOOM_SENSITIVITY = 0.01  # same magnitude as intravascular display


class CctaDisplay(QGraphicsView):
    """
    Single-plane CT viewer for one of: 'axial', 'coronal', 'sagittal'.

    Volume convention: volume[z, y, x], all axes in voxel indices.
    Coronal/sagittal images are aspect-corrected using voxel_spacing so anatomy
    looks proportional (slice_thickness often differs from pixel_spacing).

    Scroll wheel        → steps the slice axis for this view
    Left click          → updates the crosshair / cursor in orthogonal axes
    Left drag (up/down) → zoom in / out (anchored under the cursor)
    Double-click        → reset zoom to fit
    Right drag          → window/level (horizontal=level, vertical=width)

    cursor_moved(z, y, x) is emitted by scroll and click events.
    set_cursor(z, y, x)   is called by CctaPage to synchronise all views;
                          it never re-emits to avoid signal loops.
    """

    cursor_moved = pyqtSignal(int, int, int)  # z, y, x

    _DEFAULT_LEVEL = 200  # HU center — cardiac soft tissue
    _DEFAULT_WIDTH = 700  # HU range

    def __init__(self, orientation: str, parent=None) -> None:
        super().__init__(parent)
        assert orientation in ('axial', 'coronal', 'sagittal')
        self.orientation = orientation
        self.volume: np.ndarray | None = None  # (Z, Y, X) int16 HU
        self.voxel_spacing: tuple[float, float, float] | None = None  # (dz, dy, dx) mm
        self.cursor_z = 0
        self.cursor_y = 0
        self.cursor_x = 0
        self.window_level: int = self._DEFAULT_LEVEL
        self.window_width: int = self._DEFAULT_WIDTH
        self._mouse_x: float = 0.0
        self._mouse_y: float = 0.0
        self._press_pos: QPointF = QPointF(0.0, 0.0)
        self._is_dragging: bool = False
        self._user_zoomed: bool = False
        self._render_buf: np.ndarray | None = None  # kept alive for QImage

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ public

    def set_volume(self, volume: np.ndarray, voxel_spacing: tuple[float, float, float]) -> None:
        self.volume = volume
        self.voxel_spacing = voxel_spacing
        Z, Y, X = volume.shape
        self.cursor_z = Z // 2
        self.cursor_y = Y // 2
        self.cursor_x = X // 2
        self.window_level = self._DEFAULT_LEVEL
        self.window_width = self._DEFAULT_WIDTH
        self._user_zoomed = False
        self.resetTransform()
        self._render()

    def reset_zoom(self) -> None:
        self._user_zoomed = False
        self.resetTransform()
        if self.volume is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_cursor(self, z: int, y: int, x: int) -> None:
        if z == self.cursor_z and y == self.cursor_y and x == self.cursor_x:
            return
        self.cursor_z = z
        self.cursor_y = y
        self.cursor_x = x
        self._render()

    # ----------------------------------------------------------------- private

    def _get_slice(self) -> np.ndarray:
        """
        Extract and aspect-correct the 2D slice for the current orientation.

        Coronal  (ZxX image): vertical axis is z-depth with spacing dz,
                              horizontal is x-width with spacing dx.
                              Scale so heights represent real mm.
        Sagittal (ZxY image): same logic with dy.
        Axial pixels are nearly square so no scaling is applied.
        """
        assert self.volume is not None and self.voxel_spacing is not None
        dz, dy, dx = self.voxel_spacing
        Z, Y, X = self.volume.shape

        if self.orientation == 'axial':
            return self.volume[self.cursor_z, :, :]

        elif self.orientation == 'coronal':
            raw = np.ascontiguousarray(self.volume[::-1, self.cursor_y, :])  # (Z, X) sup-up
            new_h = max(1, round(Z * dz / dx))
            return cv2.resize(raw.astype(np.float32), (X, new_h), interpolation=cv2.INTER_LINEAR).astype(np.int16)

        else:  # sagittal
            raw = np.ascontiguousarray(self.volume[::-1, :, self.cursor_x])  # (Z, Y) sup-up
            new_h = max(1, round(Z * dz / dy))
            return cv2.resize(raw.astype(np.float32), (Y, new_h), interpolation=cv2.INTER_LINEAR).astype(np.int16)

    def _crosshair_pos(self, img_h: int, img_w: int) -> tuple[int, int]:
        """Return (row, col) of the crosshair in the displayed (possibly scaled) image."""
        assert self.voxel_spacing is not None and self.volume is not None
        dz, dy, dx = self.voxel_spacing
        Z, Y, X = self.volume.shape

        if self.orientation == 'axial':
            return self.cursor_y, self.cursor_x

        elif self.orientation == 'coronal':
            row = round(((Z - 1) - self.cursor_z) * dz / dx)
            return max(0, min(row, img_h - 1)), max(0, min(self.cursor_x, img_w - 1))

        else:  # sagittal
            row = round(((Z - 1) - self.cursor_z) * dz / dy)
            return max(0, min(row, img_h - 1)), max(0, min(self.cursor_y, img_w - 1))

    def _scene_to_cursor(self, row: int, col: int) -> tuple[int, int, int]:
        """Map a left-click at (row, col) in scene coordinates to a (z, y, x) cursor."""
        assert self.voxel_spacing is not None and self.volume is not None
        dz, dy, dx = self.voxel_spacing
        Z, Y, X = self.volume.shape

        if self.orientation == 'axial':
            return (
                self.cursor_z,
                max(0, min(row, Y - 1)),
                max(0, min(col, X - 1)),
            )

        elif self.orientation == 'coronal':
            row_orig = int(row * dx / dz)
            z = max(0, min((Z - 1) - row_orig, Z - 1))
            return z, self.cursor_y, max(0, min(col, X - 1))

        else:  # sagittal
            row_orig = int(row * dy / dz)
            z = max(0, min((Z - 1) - row_orig, Z - 1))
            return z, max(0, min(col, Y - 1)), self.cursor_x

    def _render(self) -> None:
        if self.volume is None:
            return
        img = self._get_slice()
        lo = self.window_level - self.window_width / 2
        hi = self.window_level + self.window_width / 2
        norm = np.clip(img.astype(np.float32), lo, hi)
        self._render_buf = np.ascontiguousarray(((norm - lo) / (hi - lo) * 255).astype(np.uint8))
        h, w = self._render_buf.shape
        q_img = QImage(self._render_buf.data, w, h, w, QImage.Format.Format_Grayscale8)  # type: ignore[call-overload]
        pixmap = QPixmap.fromImage(q_img)

        self._scene.clear()
        self._scene.addPixmap(pixmap)

        ch_row, ch_col = self._crosshair_pos(h, w)
        pen = QPen(_CROSSHAIR_COLOR)
        pen.setCosmetic(True)  # 1 px regardless of view transform
        self._scene.addLine(0, ch_row, w, ch_row, pen)
        self._scene.addLine(ch_col, 0, ch_col, h, pen)

        self._scene.setSceneRect(0, 0, w, h)
        if not self._user_zoomed:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ---------------------------------------------------------- Qt overrides

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.volume is not None and not self._user_zoomed:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if self.volume is None:
            return
        self.setFocus()
        Z, Y, X = self.volume.shape
        delta = 1 if event.angleDelta().y() > 0 else -1
        if self.orientation == 'axial':
            self.cursor_moved.emit(
                max(0, min(self.cursor_z + delta, Z - 1)),
                self.cursor_y,
                self.cursor_x,
            )
        elif self.orientation == 'coronal':
            self.cursor_moved.emit(
                self.cursor_z,
                max(0, min(self.cursor_y + delta, Y - 1)),
                self.cursor_x,
            )
        else:  # sagittal
            self.cursor_moved.emit(
                self.cursor_z,
                self.cursor_y,
                max(0, min(self.cursor_x + delta, X - 1)),
            )

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
            self._mouse_y = event.position().y()
            self._is_dragging = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._mouse_x = event.position().x()
            self._mouse_y = event.position().y()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta_y = self._mouse_y - event.position().y()
            self._mouse_y = event.position().y()
            drag_dist = (event.position() - self._press_pos).manhattanLength()
            if drag_dist > 5:
                self._is_dragging = True
            if self._is_dragging:
                zoom_factor = 1.0 + delta_y * _ZOOM_SENSITIVITY
                if zoom_factor > 0:
                    self._user_zoomed = True
                    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                    self.scale(zoom_factor, zoom_factor)
                    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        elif event.buttons() == Qt.MouseButton.RightButton:
            dx = event.position().x() - self._mouse_x
            dy_px = event.position().y() - self._mouse_y
            self._mouse_x = event.position().x()
            self._mouse_y = event.position().y()
            self.window_level = int(self.window_level + dx)
            self.window_width = max(1, int(self.window_width + dy_px))
            self._render()
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._is_dragging and self.volume is not None:
                pos = self.mapToScene(event.position().toPoint())
                z, y, x = self._scene_to_cursor(int(pos.y()), int(pos.x()))
                self.cursor_moved.emit(z, y, x)
            self._is_dragging = False
        super().mouseReleaseEvent(event)
