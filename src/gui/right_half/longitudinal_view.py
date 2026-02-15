import cv2

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen

from gui.utils.geometry import Point


class LongitudinalView(QGraphicsView):
    """
    Displays the longitudinal view of the IVUS pullback.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.lview_contour_size = 2
        self.graphics_scene = QGraphicsScene()

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setScene(self.graphics_scene)

    def set_data(
            self, 
            images, 
            contours):
        self.graphics_scene.clear()
        self.num_frames = images.shape[0]
        self.image_height = images.shape[1]
        self.points_on_marker = [None] * self.num_frames

        if hasattr(self.main_window, 'images_display') and self.main_window.images_display is not None:
            # RGB Path: Extract center slice from (Frames, H, W, 3)
            slice_data = self.main_window.dicom.pixel_array[:, :, self.image_height // 2, :]
            slice_data = np.transpose(slice_data, (1, 0, 2)).copy()
            q_format = QImage.Format.Format_RGB888
            bytes_per_line = self.num_frames * 3
        else:
            # Grayscale Path: Extract center slice from (Frames, H, W)
            slice_data = images[:, :, self.image_height // 2]
            slice_data = np.transpose(slice_data, (1, 0)).copy()
            q_format = QImage.Format.Format_Grayscale8
            bytes_per_line = self.num_frames

        if self.main_window.colormap_enabled:
            # If data is RGB, convert to Gray for OpenCV's applyColorMap
            if len(slice_data.shape) == 3:
                gray_temp = cv2.cvtColor(slice_data, cv2.COLOR_RGB2GRAY)
                slice_data = cv2.applyColorMap(gray_temp, cv2.COLORMAP_COOL)
            else:
                slice_data = cv2.applyColorMap(slice_data, cv2.COLORMAP_COOL)
            
            slice_data = cv2.cvtColor(slice_data, cv2.COLOR_BGR2RGB)
            q_format = QImage.Format.Format_RGB888
            bytes_per_line = self.num_frames * 3

        longitudinal_image = QImage(
            slice_data.data, 
            self.num_frames, 
            self.image_height, 
            bytes_per_line, 
            q_format
        )
        
        image = QGraphicsPixmapItem(QPixmap.fromImage(longitudinal_image))
        self.graphics_scene.addItem(image)

        for frame, contour in enumerate(contours):
            self.lview_contour(frame, contour)

        # Scale the view to fit the scene contents
        self.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_marker(self, frame):
        for item in self.graphics_scene.items():
            if isinstance(item, Marker):
                if item.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(item)
        
        marker = Marker(frame, 0, frame, self.image_height)
        self.graphics_scene.addItem(marker)

    def lview_contour(self, frame, contour, update=False):
        index = None

        if self.points_on_marker[frame] is not None:
            for point in self.points_on_marker[frame]:
                # ONLY remove if the point actually belongs to the current scene
                if point and point.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(point)

        if contour is None:  # skip frames without contour (but still remove previous points)
            self.points_on_marker[frame] = None
            return
        else:
            contour_x = np.array(contour[0])
            contour_y = np.array(contour[1])

        if update or self.points_on_marker[frame] is None:  # need to find the two closest points to the marker
            distances = contour_x - self.image_height // 2
            num_points_to_collect = len(contour_x) // 10
            point_indices = np.argpartition(np.abs(distances), num_points_to_collect)[:num_points_to_collect]
            for i in range(len(point_indices)):
                if (
                    np.abs(contour_y[point_indices[0]] - contour_y[point_indices[i]])
                    > self.image_height / 10
                ):  # ensure the two points are from different sides of the contour
                    index = i
                    break
            if index is None:  # no suitable points found
                return
            self.points_on_marker[frame] = (
                Point(
                    (frame, contour_y[point_indices[0]]),
                    line_thickness=self.lview_contour_size,
                    point_radius=self.lview_contour_size,
                    color='green',
                ),
                Point(
                    (frame, contour_y[point_indices[index]]),
                    line_thickness=self.lview_contour_size,
                    point_radius=self.lview_contour_size,
                    color='green',
                ),
            )
        for point in self.points_on_marker[frame]:
                if point.scene() is None:
                    self.graphics_scene.addItem(point)

    def hide_lview_contours(self):
        for item in self.graphics_scene.items():
            if isinstance(item, Point):
                if item.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(item)

    def show_lview_contours(self):
        for point in self.points_on_marker:
            try:
                self.graphics_scene.addItem(point[0])
                self.graphics_scene.addItem(point[1])
            except TypeError:
                pass

    def remove_contours(self, lower_limit, upper_limit):
        for frame in range(lower_limit, upper_limit):
            if self.points_on_marker[frame] is not None:
                for point in self.points_on_marker[frame]:
                    if point and point.scene() == self.graphics_scene:
                        self.graphics_scene.removeItem(point)
                
                self.points_on_marker[frame] = None


class Marker(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, color=Qt.GlobalColor.white):
        super().__init__()
        pen = QPen(QColor(color), 1)
        pen.setDashPattern([1, 6])
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
