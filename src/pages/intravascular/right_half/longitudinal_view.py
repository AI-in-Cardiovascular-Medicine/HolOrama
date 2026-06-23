import cv2

import numpy as np
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen, QBrush

from tools.geometry import Marker


class LongitudinalView(QGraphicsView):
    """
    Displays the longitudinal view of the IVUS pullback with lumen area overlay.
    """

    DOT_RADIUS = 3
    MARGIN_TOP = 0.05  # fraction of image_height reserved at top
    MARGIN_BOTTOM = 0.05  # fraction of image_height reserved at bottom

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.graphics_scene = QGraphicsScene()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setScene(self.graphics_scene)

        self._area_items: list[QGraphicsEllipseItem] = []
        self._phase_line_items: list = []
        self._current_marker = None
        self._areas_hidden = False
        self.num_frames = 0
        self.image_height = 0
        self.color = getattr(main_window.config.display, "color_contour", "green")

    def set_data(self, images):
        self.graphics_scene.clear()
        self._area_items = []
        self.num_frames = images.shape[0]
        self.image_height = images.shape[1]
        center_col = images.shape[2] // 2

        if self.main_window.runtime_data.images_rgb is not None:
            slice_data = self.main_window.runtime_data.images_rgb[:, :, center_col, :]
        else:
            gray = self.main_window.runtime_data.images[:, :, center_col]  # (frames, height)
            if gray.dtype != np.uint8:
                # Normalize to uint8 using the current display windowing so the
                # longitudinal view matches what the main display shows.
                lo = self.main_window.display.window_level - self.main_window.display.window_width / 2
                hi = self.main_window.display.window_level + self.main_window.display.window_width / 2
                gray = np.clip(gray, lo, hi)
                span = hi - lo
                gray = ((gray - lo) / span * 255).astype(np.uint8) if span > 0 else np.zeros_like(gray, dtype=np.uint8)
            slice_data = np.stack([gray, gray, gray], axis=-1)  # (frames, height, 3)
        slice_data = np.transpose(slice_data, (1, 0, 2)).copy()
        q_format = QImage.Format.Format_RGB888
        bytes_per_line = self.num_frames * 3

        if getattr(self.main_window, 'colormap_enabled', False):
            if len(slice_data.shape) == 3:
                gray_temp = cv2.cvtColor(slice_data, cv2.COLOR_RGB2GRAY)
                slice_data = cv2.applyColorMap(gray_temp, cv2.COLORMAP_COOL)
            else:
                slice_data = cv2.applyColorMap(slice_data, cv2.COLORMAP_COOL)
            slice_data = cv2.cvtColor(slice_data, cv2.COLOR_BGR2RGB)
            q_format = QImage.Format.Format_RGB888
            bytes_per_line = self.num_frames * 3

        longitudinal_image = QImage(slice_data.data, self.num_frames, self.image_height, bytes_per_line, q_format)
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(longitudinal_image))
        self.graphics_scene.addItem(pixmap_item)
        self.setSceneRect(pixmap_item.boundingRect())

        self.stretch_to_fit()
        self.plot_areas()

    def plot_areas(self):
        """Read lumen areas from main_window.runtime_data.frame_data_dct and draw one dot per frame."""
        for item in self._area_items:
            if item.scene() == self.graphics_scene:
                self.graphics_scene.removeItem(item)
        self._area_items = []

        if not self.main_window.runtime_data.frame_data_dct or self.image_height == 0:
            return

        areas: dict[int, float] = {}
        phases: dict[int, float] = {}
        for frame, fd in self.main_window.runtime_data.frame_data_dct.items():
            area = fd.lumen.measurements.area
            phase = fd.phase
            if area is not None and area > 0:
                areas[frame] = area
                phases[frame] = phase

        if not areas:
            return

        max_area = max(areas.values())
        usable_height = self.image_height * (1.0 - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        top_offset = self.image_height * self.MARGIN_TOP
        r = self.DOT_RADIUS

        brush = QBrush(QColor(self.color))
        no_pen = QPen(Qt.PenStyle.NoPen)

        for frame, area in areas.items():
            y = top_offset + (1.0 - area / max_area) * usable_height
            item = QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            item.setPos(frame, y)
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
            new_brush = None
            if phases[frame] == '-':
                new_brush = brush
            elif phases[frame] == 'D':
                new_brush = QBrush(QColor(*self.main_window.diastole_color))
            elif phases[frame] == 'S':
                new_brush = QBrush(QColor(*self.main_window.systole_color))
            else:
                new_brush = QBrush(QColor('orange'))
                new_brush = new_brush
            item.setBrush(new_brush)
            item.setPen(no_pen)
            if self._areas_hidden:
                item.setVisible(False)
            self.graphics_scene.addItem(item)
            self._area_items.append(item)

    def hide_lview_contours(self):
        self._areas_hidden = True
        for item in self._area_items:
            item.setVisible(False)

    def show_lview_contours(self):
        self._areas_hidden = False
        for item in self._area_items:
            item.setVisible(True)

    def remove_contours(self, lower_limit, upper_limit):
        """Called when contours are deleted for a range; refresh area overlay."""
        self.plot_areas()

    def _update_phase_lines(self):
        for item in self._phase_line_items:
            if item.scene() == self.graphics_scene:
                self.graphics_scene.removeItem(item)
        self._phase_line_items = []

        mw = self.main_window
        dia_frames = getattr(mw.runtime_data, 'gated_frames_dia', [])
        sys_frames = getattr(mw.runtime_data, 'gated_frames_sys', [])

        for frame, rgb in [(f, mw.diastole_color) for f in dia_frames] + [(f, mw.systole_color) for f in sys_frames]:
            color = QColor(*rgb)
            color.setAlpha(200)
            pen = QPen(color, 0.5, Qt.PenStyle.DotLine)
            item = QGraphicsLineItem(frame, 0, frame, self.image_height)
            item.setPen(pen)
            self.graphics_scene.addItem(item)
            self._phase_line_items.append(item)

    def update_marker(self, frame):
        if self._current_marker is not None and self._current_marker.scene() == self.graphics_scene:
            self.graphics_scene.removeItem(self._current_marker)

        self._update_phase_lines()

        self._current_marker = Marker(frame, 0, frame, self.image_height)
        self.graphics_scene.addItem(self._current_marker)

    def stretch_to_fit(self):
        if self.graphics_scene.items():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.IgnoreAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.stretch_to_fit()
