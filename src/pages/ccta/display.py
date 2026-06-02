import cv2
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor


_CROSSHAIR_COLOR = QColor(255, 255, 0)
_ZOOM_SENSITIVITY = 0.01
_DEFAULT_MASK_ALPHA = 0.45

# Public — imported by tab_gui to build colour swatches
LABEL_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # red
    (60, 220, 60),  # green
    (60, 60, 255),  # blue
    (255, 220, 0),  # yellow
    (220, 60, 220),  # magenta
    (0, 210, 210),  # cyan
    (255, 140, 0),  # orange
    (160, 60, 255),  # purple
    (0, 180, 255),  # sky blue
    (255, 60, 140),  # pink
    (0, 200, 120),  # mint
    (180, 255, 0),  # lime
    (255, 180, 100),  # peach
    (140, 140, 255),  # lavender
)


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
    windowing_changed = pyqtSignal(int, int)  # level, width

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

        self._mask: np.ndarray | None = None  # (Z, Y, X) uint8 label values
        self._mask_lut: np.ndarray | None = None  # (256, 3) uint8; row = 0 → invisible
        self._mask_labels: list[int] = []
        self._hidden_labels: set[int] = set()
        self._mask_alpha: float = _DEFAULT_MASK_ALPHA

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def set_mask(self, mask: np.ndarray, labels: list[int]) -> None:
        self._mask = mask
        self._mask_labels = labels
        self._hidden_labels = set()
        self._rebuild_lut()
        self._render()

    def clear_mask(self) -> None:
        self._mask = None
        self._mask_lut = None
        self._mask_labels = []
        self._hidden_labels = set()
        self._render()

    def set_mask_alpha(self, alpha: float) -> None:
        self._mask_alpha = max(0.0, min(1.0, alpha))
        self._render()

    def set_label_visible(self, label: int, visible: bool) -> None:
        if visible:
            self._hidden_labels.discard(label)
        else:
            self._hidden_labels.add(label)
        self._rebuild_lut()
        self._render()

    def reset_zoom(self) -> None:
        self._user_zoomed = False
        self.resetTransform()
        if self.volume is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def reset_windowing(self) -> None:
        self.window_level = self._DEFAULT_LEVEL
        self.window_width = self._DEFAULT_WIDTH
        self._render()

    def set_windowing(self, level: int, width: int) -> None:
        if level == self.window_level and width == self.window_width:
            return
        self.window_level = level
        self.window_width = width
        self._render()

    def set_cursor(self, z: int, y: int, x: int) -> None:
        if z == self.cursor_z and y == self.cursor_y and x == self.cursor_x:
            return
        self.cursor_z = z
        self.cursor_y = y
        self.cursor_x = x
        self._render()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_lut(self) -> None:
        """Rebuild the 256-entry colour LUT respecting current hidden-label state."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i, label in enumerate(self._mask_labels):
            if 0 < label < 256 and label not in self._hidden_labels:
                lut[label] = LABEL_COLORS[i % len(LABEL_COLORS)]
        self._mask_lut = lut

    # ------------------------------------------------------------------
    # Slice extraction
    # ------------------------------------------------------------------

    def _get_slice(self) -> np.ndarray:
        assert self.volume is not None and self.voxel_spacing is not None
        dz, dy, dx = self.voxel_spacing
        Z, Y, X = self.volume.shape

        if self.orientation == 'axial':
            return self.volume[self.cursor_z, :, :]
        elif self.orientation == 'coronal':
            raw = np.ascontiguousarray(self.volume[::-1, self.cursor_y, :])
            new_h = max(1, round(Z * dz / dx))
            return cv2.resize(raw.astype(np.float32), (X, new_h), interpolation=cv2.INTER_LINEAR).astype(np.int16)
        else:  # sagittal
            raw = np.ascontiguousarray(self.volume[::-1, :, self.cursor_x])
            new_h = max(1, round(Z * dz / dy))
            return cv2.resize(raw.astype(np.float32), (Y, new_h), interpolation=cv2.INTER_LINEAR).astype(np.int16)

    def _get_mask_slice(self) -> np.ndarray | None:
        if self._mask is None or self.voxel_spacing is None or self.volume is None:
            return None
        if self._mask.shape != self.volume.shape:
            return None

        dz, dy, dx = self.voxel_spacing
        Z, Y, X = self.volume.shape

        if self.orientation == 'axial':
            return self._mask[self.cursor_z, :, :]
        elif self.orientation == 'coronal':
            raw = np.ascontiguousarray(self._mask[::-1, self.cursor_y, :])
            new_h = max(1, round(Z * dz / dx))
            return cv2.resize(raw.astype(np.float32), (X, new_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:  # sagittal
            raw = np.ascontiguousarray(self._mask[::-1, :, self.cursor_x])
            new_h = max(1, round(Z * dz / dy))
            return cv2.resize(raw.astype(np.float32), (Y, new_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self.volume is None:
            return
        img = self._get_slice()
        lo = self.window_level - self.window_width / 2
        hi = self.window_level + self.window_width / 2
        norm = np.clip(img.astype(np.float32), lo, hi)
        gray = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)

        mask_slice = self._get_mask_slice()
        if mask_slice is not None and self._mask_lut is not None:
            rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
            colors = self._mask_lut[mask_slice]  # (H, W, 3)
            has_label = np.any(colors > 0, axis=-1)  # True where label is visible
            if has_label.any():
                rgb[has_label] = (1 - self._mask_alpha) * rgb[has_label] + self._mask_alpha * colors[has_label].astype(
                    np.float32
                )
            self._render_buf = np.ascontiguousarray(rgb.astype(np.uint8))
            h, w = gray.shape
            q_img = QImage(self._render_buf.data, w, h, w * 3, QImage.Format.Format_RGB888)  # type: ignore[call-overload]
        else:
            self._render_buf = np.ascontiguousarray(gray)
            h, w = gray.shape
            q_img = QImage(self._render_buf.data, w, h, w, QImage.Format.Format_Grayscale8)  # type: ignore[call-overload]

        pixmap = QPixmap.fromImage(q_img)
        self._scene.clear()
        self._scene.addPixmap(pixmap)

        ch_row, ch_col = self._crosshair_pos(h, w)
        pen = QPen(_CROSSHAIR_COLOR)
        pen.setCosmetic(True)
        self._scene.addLine(0, ch_row, w, ch_row, pen)
        self._scene.addLine(ch_col, 0, ch_col, h, pen)

        self._scene.setSceneRect(0, 0, w, h)
        if not self._user_zoomed:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _crosshair_pos(self, img_h: int, img_w: int) -> tuple[int, int]:
        assert self.voxel_spacing is not None and self.volume is not None
        dz, dy, dx = self.voxel_spacing
        Z = self.volume.shape[0]

        if self.orientation == 'axial':
            return self.cursor_y, self.cursor_x
        elif self.orientation == 'coronal':
            row = round(((Z - 1) - self.cursor_z) * dz / dx)
            return max(0, min(row, img_h - 1)), max(0, min(self.cursor_x, img_w - 1))
        else:  # sagittal
            row = round(((Z - 1) - self.cursor_z) * dz / dy)
            return max(0, min(row, img_h - 1)), max(0, min(self.cursor_y, img_w - 1))

    def _scene_to_cursor(self, row: int, col: int) -> tuple[int, int, int]:
        assert self.voxel_spacing is not None and self.volume is not None
        dz, dy, dx = self.voxel_spacing
        Z, Y, X = self.volume.shape

        if self.orientation == 'axial':
            return self.cursor_z, max(0, min(row, Y - 1)), max(0, min(col, X - 1))
        elif self.orientation == 'coronal':
            row_orig = int(row * dx / dz)
            z = max(0, min((Z - 1) - row_orig, Z - 1))
            return z, self.cursor_y, max(0, min(col, X - 1))
        else:  # sagittal
            row_orig = int(row * dy / dz)
            z = max(0, min((Z - 1) - row_orig, Z - 1))
            return z, max(0, min(col, Y - 1)), self.cursor_x

    # ------------------------------------------------------------------
    # Qt event handlers
    # ------------------------------------------------------------------

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
            self.cursor_moved.emit(max(0, min(self.cursor_z + delta, Z - 1)), self.cursor_y, self.cursor_x)
        elif self.orientation == 'coronal':
            self.cursor_moved.emit(self.cursor_z, max(0, min(self.cursor_y + delta, Y - 1)), self.cursor_x)
        else:
            self.cursor_moved.emit(self.cursor_z, self.cursor_y, max(0, min(self.cursor_x + delta, X - 1)))

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
            self.windowing_changed.emit(self.window_level, self.window_width)
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
