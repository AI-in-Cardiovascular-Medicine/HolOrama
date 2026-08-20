from typing import Any, Sequence, Tuple

from loguru import logger
from PyQt6.QtCore import QLineF, Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QGraphicsTextItem
from shapely.geometry import Polygon

from domain.io_types import FrameData, Measurements
from input_output.output.reports import (
    closest_points,
    compute_polygon_metrics,
    farthest_points,
)

# Internal imports based on your file structure
from tools.geometry import get_qt_pen

# Bundled and registered in main.py; hardcoded here (rather than read from the app-wide
# font) because only this overlay uses it - the rest of the UI stays on the default font.
_OVERLAY_FONT_FAMILY = 'JetBrains Mono'


def _overlay_font(point_size: int, bold: bool = False) -> QFont:
    """Font for the text overlaid on the intravascular display. The size stays
    caller-controlled because it scales with the image, not with the UI."""
    font = QFont(_OVERLAY_FONT_FAMILY, point_size)
    if bold:
        font.setWeight(QFont.Weight.Bold)
    return font


def clear_lumen_measurements(frame_data: FrameData) -> None:
    """Forget everything derived from a frame's lumen contour."""
    frame_data.lumen.measurements = Measurements()
    frame_data.centroid = None
    frame_data.closest_points = None
    frame_data.farthest_points = None


class MetricsMixin:
    """
    Mixin class for Display to handle plaque/lumen metrics
    and UI text overlays.
    """

    main_window: Any
    frame: int
    point_thickness: int
    alpha_contour: int
    graphics_scene: Any
    scaling_factor: float
    contour_configs: Any
    image_size: int
    n_points_contour: int
    active_contour_type: Any

    def _maybe_compute_metrics(
        self,
        unscaled_lumen: Tuple[Sequence[float], Sequence[float]] | None = None,
        unscaled_eem: Tuple[Sequence[float], Sequence[float]] | None = None,
    ):
        if unscaled_lumen is None:
            self._clear_lumen_metrics_if_uncontoured()
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

        if not poly.is_valid:
            poly = poly.buffer(0)  # fix self-intersecting splines without raising
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
                QLineF(
                    far_x[0] * self.scaling_factor,
                    far_y[0] * self.scaling_factor,
                    far_x[1] * self.scaling_factor,
                    far_y[1] * self.scaling_factor,
                ),
                pen,
            )
            self.graphics_scene.addLine(
                QLineF(
                    close_x[0] * self.scaling_factor,
                    close_y[0] * self.scaling_factor,
                    close_x[1] * self.scaling_factor,
                    close_y[1] * self.scaling_factor,
                ),
                pen,
            )

            ell = (longest_d / shortest_d) if shortest_d else 0
            self.build_frame_metrics_text(
                lumen_area, lumen_circumf, ell, longest_d, shortest_d, eem_area, pct, update_phase=False
            )

    def _clear_lumen_metrics_if_uncontoured(self):
        """Drop the current frame's lumen measurements once its contour is gone.

        The measurements are what the longitudinal view and the OCT schematic plot, so
        leaving them behind after a delete keeps drawing a frame that no longer has a
        contour. Only clears when frame_data agrees the contour is gone — a contour that
        merely failed to interpolate keeps its numbers.
        """
        frame_data = self.main_window.runtime_data.frame_data_dct.get(self.frame)
        if frame_data is None or frame_data.lumen.contours:
            return
        clear_lumen_measurements(frame_data)

    def refresh_all_frame_metrics(self):
        """Recompute every contoured frame's metrics, then redraw the pullback overviews.

        For contour changes made across many frames at once — automatic segmentation,
        mask import, file load — where per-frame measurements would otherwise only appear
        as the user visits each frame, leaving both overviews blank in the meantime.
        """
        self.compute_all_frame_metrics()
        try:
            self.main_window.longitudinal_view.plot_areas()
        except Exception as exc:
            logger.debug(f'Could not refresh the pullback overviews: {exc}')

    def compute_all_frame_metrics(self):
        """Batch-compute lumen area and minor axis for every frame that has a contour.

        Called once after image + contour load so plot_areas() has data
        for all frames immediately (not just frames the user has navigated to).
        Minor axis is needed alongside area for the breathing signal's
        elliptic-deformation adjustment (see breathing_pipeline.adjust_for_elliptic_deformation).
        """
        from tools.geometry import (
            SplineGeometry,  # local import avoids circular at module level
        )

        sf = self.scaling_factor
        n_pts = self.n_points_contour
        frame_data_dct = self.main_window.runtime_data.frame_data_dct

        for frame_idx, fd in frame_data_dct.items():
            if fd is None or fd.lumen is None or not fd.lumen.contours:
                continue
            contour = fd.lumen.contours[0]
            if not contour or not contour[0]:
                continue
            x_raw = contour[0]
            y_raw = contour[1] if len(contour) > 1 else []
            if len(x_raw) < 3:
                continue
            try:
                geom = SplineGeometry(
                    [p * sf for p in x_raw],
                    [p * sf for p in y_raw],
                    n_pts,
                    None,
                    None,
                )
                x_full, y_full = geom.to_unscaled(sf)
                if len(x_full) < 3:
                    continue
                poly = Polygon(list(zip(x_full, y_full)))
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and poly.area > 0:
                    compute_polygon_metrics(self.main_window, poly, frame_idx)
                    closest_points(self.main_window, poly, frame_idx)
            except Exception as exc:
                logger.debug(f'Frame {frame_idx}: batch metric computation failed: {exc}')

    def update_phase_text(self):
        fd = self.main_window.runtime_data.frame_data_dct.get(self.frame)
        code = fd.phase if fd else '-'
        if code == "D":
            text = "Diastole"
            color = QColor(*self.main_window.diastole_color)
        elif code == "S":
            text = "Systole"
            color = QColor(*self.main_window.systole_color)
        elif code == "T":
            text = "Tagged"
            color = QColor("yellow")
        else:
            text = ""
            color = QColor(Qt.GlobalColor.white)

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
        self.phase_text.setFont(_overlay_font(int(self.image_size / 50), bold=True))
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
            color = QColor(Qt.GlobalColor.white)

        self.active_contour_text.setDefaultTextColor(color)

        font_size = max(8, int(self.image_size / 45))
        self.active_contour_text.setFont(_overlay_font(font_size, bold=True))

        # Sit the label on the bottom edge using its real height rather than guessing it
        # from the point size: 'font_size * 2.5' overshot and clipped the descenders at
        # small image sizes (image_size is not fixed, and each font has its own line
        # height), which is why this reads the laid-out bounding rect instead.
        text_height = self.active_contour_text.boundingRect().height()
        self.active_contour_text.setPos(10, self.image_size - text_height - 5)

        self.graphics_scene.addItem(self.active_contour_text)

    def compute_eem_and_percent_stenosis(self, frame: int, lumen_area: float, eem_full: Tuple[Any, Any] | None = None):
        """
        Return (eem_area, percent_stenosis_text).
        Robust to numpy arrays and malformed data structures.
        """
        eem_area: float | None = None
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
                    eem_area = polygon_eem.area * self.main_window.runtime_data.metadata['resolution'] ** 2
        except Exception:
            logger.exception("Failed while computing EEM area")

        try:
            if lumen_area is not None and eem_area is not None and eem_area != 0:
                percent = ((eem_area - lumen_area) / eem_area) * 100.0
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

        # Label/value pairs padded to a common width rather than separated by tabs. The
        # old '\t'/'\t\t' mix relied on the 80px default tab stop landing just past each
        # label, which only held for the proportional font this used to hardcode; in the
        # monospace UI font the labels straddle that stop and the values end up in two
        # different columns. Padding aligns them exactly, whatever the label lengths.
        squared = '(mm\N{SUPERSCRIPT TWO})'
        metrics = [
            ('Lumen area:', f'{round(lumen_area, 2)} {squared}' if lumen_area is not None else 'n/a'),
            ('Lumen circ:', f'{round(lumen_circumf, 2)} (mm)' if lumen_circumf is not None else 'n/a'),
            ('Elliptic ratio:', f'{round(elliptic_ratio, 2)}' if elliptic_ratio is not None else 'n/a'),
            ('Longest distance:', f'{round(longest_distance, 2)} (mm)' if longest_distance is not None else 'n/a'),
            ('Shortest distance:', f'{round(shortest_distance, 2)} (mm)' if shortest_distance is not None else 'n/a'),
            ('EEM area:', f'{round(eem_area, 2)} {squared}' if eem_area is not None else 'n/a'),
            ('Plaque burden:', f'{percent_stenosis_text}'),
        ]
        label_width = max(len(label) for label, _ in metrics) + 2
        lines = [f'{label:<{label_width}}{value}' for label, value in metrics]

        self.frame_metrics_text = QGraphicsTextItem("\n".join(lines))
        self.frame_metrics_text.setFont(_overlay_font(int(self.image_size / 50)))

        self.frame_metrics_text.setPos(5, 5)
        self.graphics_scene.addItem(self.frame_metrics_text)

        if not update_phase:
            try:
                if hasattr(self, "phase_text") and self.phase_text is not None:
                    if self.phase_text.scene() is None:
                        self.graphics_scene.addItem(self.phase_text)
            except Exception:
                pass
