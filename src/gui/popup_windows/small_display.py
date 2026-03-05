from loguru import logger
from PyQt6.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QPen
from shapely.geometry import Polygon

from gui.utils.geometry import SplineGeometry, Spline, Point
from report.report import farthest_points, closest_points
import numpy as np


class SmallDisplay(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_size = main_window.config.display.image_size
        self.n_points_contour = main_window.config.display.n_points_contour
        self.contour_thickness = main_window.config.display.contour_thickness
        self.point_thickness = main_window.config.display.point_thickness
        self.point_radius = main_window.config.display.point_radius
        self.scaling_factor = self.image_size / self.main_window.images[0].shape[0]
        self.window_to_image_ratio = 1.5
        self.window_size = int(self.image_size / self.window_to_image_ratio)
        self.resize(self.window_size, self.window_size)

        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCentralWidget(self.view)
        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

    def calculate_correlation(self, frame):
        """Calculates correlation coefficients with the previous 20 to 10 frames."""
        correlations = []
        frame_indices = []
        start_frame = max(0, frame - 30)
        end_frame = max(0, frame - 5)
        
        for i in range(start_frame, end_frame):
            corr = np.corrcoef(self.main_window.images[frame].ravel(), self.main_window.images[i].ravel())[0, 1]
            correlations.append(corr)
            frame_indices.append(i)

        # If less than 10 frames, pad with 0s to maintain the length
        while len(correlations) < 10:
            correlations.insert(0, 0)  # Prepend zeros if necessary
            frame_indices.insert(0, None)  # Prepend None for frame indices

        return correlations, frame_indices

    def find_best_correlation(self, correlations, frame_indices):
        """Finds the frame with the highest correlation."""
        if not correlations:
            return None, None

        max_corr = max(correlations)
        max_index = correlations.index(max_corr)
        best_frame_index = frame_indices[max_index]

        return best_frame_index, max_corr

    def update_frame(self, frame, update_image=False, update_contours=False, update_text=False):
        if update_image:
            if frame is None:
                self.pixmap.setPixmap(QPixmap())
                self.setWindowTitle("No Frame to Display")
                items = [item for item in self.scene.items() if not isinstance(item, QGraphicsPixmapItem)]
                for item in items:
                    if item.scene() == self.scene:
                        self.scene.removeItem(item)
                return
            
            self.pixmap.setPixmap(
                QPixmap.fromImage(
                    QImage(
                        self.main_window.images[frame],
                        self.main_window.images[frame].shape[1],
                        self.main_window.images[frame].shape[0],
                        self.main_window.images[frame].shape[1],
                        QImage.Format.Format_Grayscale8,
                    ).scaled(
                        self.image_size, 
                        self.image_size, 
                        Qt.AspectRatioMode.IgnoreAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                )
            )

        if update_contours:
            contour_types = (Spline, Point, QGraphicsLineItem)
            items = [item for item in self.scene.items() if isinstance(item, contour_types)]
            for item in items:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)

            key = self.main_window.display.contour_key()
            contour_data = self.main_window.data.get(key)
            if contour_data and contour_data[0][frame] and not self.main_window.hide_contours:
                lumen_x = [point * self.scaling_factor for point in contour_data[0][frame]]
                lumen_y = [point * self.scaling_factor for point in contour_data[1][frame]]
                geometry = SplineGeometry(
                    lumen_x,
                    lumen_y,
                    self.n_points_contour,
                    None,
                    None,
                )
                geometry.interpolate()
                current_contour = Spline(
                    geometry,
                    'green',
                    self.contour_thickness,
                )

                if current_contour.geometry.full_contour[0] is not None:
                    self.contour_points = [
                        Point(
                            (current_contour.knot_points[0][i], current_contour.knot_points[1][i]),
                            self.point_thickness,
                            self.point_radius,
                            'green',
                        )
                        for i in range(len(current_contour.knot_points[0]) - 1)
                    ]
                    [self.scene.addItem(point) for point in self.contour_points]
                    self.scene.addItem(current_contour)
                    polygon = Polygon(
                        [(x, y) for x, y in zip(current_contour.geometry.full_contour[0], current_contour.geometry.full_contour[1])]
                    )
                    self.view.centerOn(polygon.centroid.x, polygon.centroid.y)
                    _, farthest_x, farthest_y = farthest_points(self.main_window, polygon.exterior.coords, frame)
                    _, closest_x, closest_y = closest_points(self.main_window, polygon, frame)
                    self.scene.addLine(
                        farthest_x[0],
                        farthest_y[0],
                        farthest_x[1],
                        farthest_y[1],
                        QPen(Qt.GlobalColor.yellow, self.point_thickness * 2),
                    )
                    self.scene.addLine(
                        closest_x[0],
                        closest_y[0],
                        closest_x[1],
                        closest_y[1],
                        QPen(Qt.GlobalColor.yellow, self.point_thickness * 2),
                    )

        current_phase = 'Diastolic' if self.main_window.use_diastolic_button.isChecked() else 'Systolic'
        self.setWindowTitle(f"Next {current_phase} Frame {frame + 1}")

        if update_text:
            # Remove previous correlation text items
            text_items = [item for item in self.scene.items() if isinstance(item, QGraphicsTextItem)]
            for item in text_items:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            
            # Calculate correlation for this frame
            correlations, frame_indices = self.calculate_correlation(frame)
            best_frame_index, best_correlation = self.find_best_correlation(correlations, frame_indices)

            # Update window title with correlation information
            if best_frame_index is not None:
                distance = frame - self.main_window.display.frame
                text = f"Frame {frame + 1} (+{distance})\n Correlation Frame: {best_frame_index} ({best_correlation:.2f})"
            else:
                text = f"Frame {frame + 1} \n No Previous Frames Available"

            # Create and position the text item centered at the top of the view
            text_item = self.scene.addText(text)
            text_item.setDefaultTextColor(Qt.GlobalColor.white)
            font = text_item.font()
            font.setPointSize(font.pointSize() * 2)
            text_item.setFont(font)
            
            # Calculate centered position
            text_item_width = text_item.boundingRect().width()
            text_item_height = text_item.boundingRect().height()
            text_item.setPos((self.image_size - text_item_width) / 2, (self.image_size - text_item_height) / 2)