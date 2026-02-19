from typing import Tuple

import numpy as np
from loguru import logger
from shapely.geometry import Polygon
from PyQt6.QtWidgets import QGraphicsTextItem
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt, QLineF

# Internal imports based on your file structure
from gui.utils.geometry import get_qt_pen
from report.report import compute_polygon_metrics, farthest_points, closest_points

class MetricsMixin:
    """
    Mixin class for IVUSDisplay to handle plaque/lumen metrics 
    and UI text overlays.
    """

    def _maybe_compute_metrics(self, unscaled_lumen: Tuple[np.ndarray, np.ndarray] | None = None, unscaled_eem: Tuple[np.ndarray, np.ndarray] | None = None):
        if unscaled_lumen is None:
            return
        
        try:
            x, y = unscaled_lumen
            # Ensure we have valid arrays and they match in length
            if x is None or y is None or len(x) != len(y):
                logger.warning("Mismatch or None in unscaled_lumen coordinates.")
                return
            if len(x) < 3:
                # Not enough points to form a polygon
                return
        except (ValueError, TypeError):
            logger.warning("unscaled_lumen is not in the expected (x_coords, y_coords) format.")
            return

        # list(zip(x, y)) converts ( [x1, x2], [y1, y2] ) -> [ (x1, y1), (x2, y2) ]
        poly = Polygon(list(zip(x, y)))

        if not poly.is_valid or poly.area == 0:
            logger.warning("Invalid or zero-area polygon created. Skipping metrics.")
            return            

        lumen_area, lumen_circumf, _, _ = compute_polygon_metrics(self.main_window, poly, self.frame)
        longest_d, far_x, far_y = farthest_points(self.main_window, poly.exterior.coords, self.frame)
        shortest_d, close_x, close_y = closest_points(self.main_window, poly, self.frame)
        
        eem_area, pct = self.compute_eem_and_percent_stenosis(self.frame, lumen_area, unscaled_eem)

        if not self.main_window.hide_special_points:
            pen = get_qt_pen('yellow', self.point_thickness * 2, self.alpha_contour)
            self.graphics_scene.addLine(
                QLineF(far_x[0] * self.scaling_factor, far_y[0] * self.scaling_factor,
                    far_x[1] * self.scaling_factor, far_y[1] * self.scaling_factor),
                pen,
            )
            self.graphics_scene.addLine(
                QLineF(close_x[0] * self.scaling_factor, close_y[0] * self.scaling_factor,
                    close_x[1] * self.scaling_factor, close_y[1] * self.scaling_factor),
                pen,
            )

            ell = (longest_d / shortest_d) if shortest_d else 0
            self.build_frame_metrics_text(lumen_area, lumen_circumf, ell, longest_d, shortest_d, eem_area, pct, update_phase=False)

    def update_phase_text(self):
        code = self.main_window.data["phases"][self.frame]
        if code == "D":
            text = "Diastole"
            color = QColor(*self.main_window.diastole_color)
        elif code == "S":
            text = "Systole"
            color = QColor(*self.main_window.systole_color)
        else:
            text = ""
            color = Qt.GlobalColor.white

        # replace previous phase_text if present
        phase_text = getattr(self, "phase_text", None)
        if phase_text:
            if phase_text.scene() == self.graphics_scene:
                try:
                    self.graphics_scene.removeItem(phase_text)
                except Exception:
                    pass

        self.phase_text = QGraphicsTextItem(text)
        self.phase_text.setDefaultTextColor(color)
        self.phase_text.setX(self.image_size - self.image_size / 3.75)
        self.phase_text.setFont(QFont("Helvetica", int(self.image_size / 50), QFont.Weight.Bold))
        self.graphics_scene.addItem(self.phase_text)

    def update_active_contour(self):
        active_type = self.active_contour_type
        config = self.contour_configs.get(active_type)
        
        old_text = getattr(self, "active_contour_text", None)
        if old_text and old_text.scene() == self.graphics_scene:
            self.graphics_scene.removeItem(old_text)
        
        text = active_type.value.upper()
        self.active_contour_text = QGraphicsTextItem(text)
        
        if isinstance(config.color, str):
            color = QColor(config.color)
        elif isinstance(config.color, (tuple, list)):
            color = QColor(*config.color)
        else:
            color = Qt.GlobalColor.white
            
        self.active_contour_text.setDefaultTextColor(color)
        
        font_size = max(8, int(self.image_size / 45))
        self.active_contour_text.setFont(QFont("Helvetica", font_size, QFont.Weight.Bold))
        
        self.active_contour_text.setPos(10, self.image_size - (font_size * 2.5))
        
        self.graphics_scene.addItem(self.active_contour_text)

    def compute_eem_and_percent_stenosis(self, frame: int, lumen_area: float, eem_full: Tuple[float, float] | None = None):
        """
        Return (eem_area, percent_stenosis_text).
        Robust to numpy arrays and malformed data structures.
        """
        eem_area = None
        percent_text = "n/a"
        
        try:
            # Preferred: use prepared full_contours for EEM (display coords)
            if eem_full is not None:
                try:
                    eem_x, eem_y = eem_full
                except Exception:
                    eem_x = eem_y = None

                try:
                    has_eem_coords = (eem_x is not None and len(eem_x) > 0) and (eem_y is not None and len(eem_y) > 0)
                except Exception:
                    has_eem_coords = False

                if has_eem_coords:
                    polygon_eem = Polygon([(float(x), float(y)) for x, y in zip(eem_x, eem_y)])
                    eem_area, _, _, _ = compute_polygon_metrics(self.main_window, polygon_eem, frame)
        except Exception:
            logger.exception("Failed while computing EEM area")

        try:
            if lumen_area is not None and eem_area not in (None, 0):
                percent = ((eem_area - lumen_area)/ eem_area) * 100.0
                percent = max(0.0, min(100.0, percent))  # clamp 0..100
                percent_text = f"{round(percent, 2)} %"
        except Exception:
            logger.exception("Failed to compute percent stenosis")

        return eem_area, percent_text
            
    def build_frame_metrics_text(
        self,
        lumen_area,
        lumen_circumf,
        elliptic_ratio,
        longest_distance,
        shortest_distance,
        eem_area,
        percent_stenosis_text,
        update_phase,
    ):
        """
        Build/add a single QGraphicsTextItem with lumen + EEM + percent-stenosis metrics.
        """
        try:
            prev = getattr(self, "frame_metrics_text", None)
            if prev is not None:
                try:
                    if hasattr(prev, "scene") and prev.scene() is self.graphics_scene:
                        self.graphics_scene.removeItem(prev)
                except Exception:
                    pass
        except Exception:
            pass

        lines = [
            f"Lumen area:\t\t{round(lumen_area, 2)} (mm\N{SUPERSCRIPT TWO})" if lumen_area is not None else "Lumen area:\t\tn/a",
            f"Lumen circ:\t\t{round(lumen_circumf, 2)} (mm)" if lumen_circumf is not None else "Lumen circ:\t\tn/a",
            f"Elliptic ratio:\t\t{round(elliptic_ratio, 2)}" if elliptic_ratio is not None else "Elliptic ratio:\t\tn/a",
            f"Longest distance:\t{round(longest_distance, 2)} (mm)" if longest_distance is not None else "Longest distance:\t\tn/a",
            f"Shortest distance:\t{round(shortest_distance, 2)} (mm)" if shortest_distance is not None else "Shortest distance:\t\tn/a",
            f"EEM area:\t\t{round(eem_area, 2)} (mm\N{SUPERSCRIPT TWO})" if eem_area is not None else "EEM area:\t\tn/a",
            f"Plaque burden:\t\t{percent_stenosis_text}",
        ]

        self.frame_metrics_text = QGraphicsTextItem("\n".join(lines))
        self.frame_metrics_text.setFont(QFont("Helvetica", int(self.image_size / 50)))

        self.frame_metrics_text.setPos(5, 5)
        self.graphics_scene.addItem(self.frame_metrics_text)

        if not update_phase:
            try:
                if hasattr(self, "phase_text") and self.phase_text is not None:
                    if self.phase_text.scene() is None:
                        self.graphics_scene.addItem(self.phase_text)
            except Exception:
                pass
