import cv2
import math

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Union, Any

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, QLineF, QPointF
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont, QPen
from shapely.geometry import Polygon

from gui.utils.geometry import Point, Spline, SplineGeometry, get_qt_pen
from gui.right_half.longitudinal_view import Marker
from report.report import compute_polygon_metrics, farthest_points, closest_points
from segmentation.segment import downsample


class ContourType(Enum):
    LUMEN = "lumen"
    EEM = "eem"
    CALCIUM = "calcium"
    BRANCH = "branch"


@dataclass
class ContourConfig:
    """Configuration for a specific contour type"""

    color: Union[
        str, Tuple[int, int, int], Any
    ]  # accept string names ('green'), hex ('#ff00ff'), or RGB tuples (255,0,0)
    thickness: int
    point_radius: int
    point_thickness: int
    alpha: int
    n_points_contour: int
    n_interactive_points: int


class IVUSDisplay(QGraphicsView):
    """
    Displays images and contours and allows the user to add and manipulate contours.
    """

    def __init__(self, main_window):
        super(IVUSDisplay, self).__init__()
        self.main_window = main_window
        config = main_window.config

        self.n_interactive_points: int = config.display.n_interactive_points
        self.n_points_contour: int = config.display.n_points_contour
        self.image_size: int = config.display.image_size # image display in pixel (square)
        self.windowing_sensitivity: float = config.display.windowing_sensitivity
        self.contour_thickness: int = config.display.contour_thickness
        self.point_thickness: int = config.display.point_thickness
        self.point_radius: int = config.display.point_radius

        self.color_contour = getattr(config.display, "color_contour", (255, 255, 255))
        self.alpha_contour = getattr(config.display, "alpha_contour", 255)  # config uses 0..255

        _default_colors = {
            ContourType.LUMEN: getattr(config.display, "color_contour", "green"),
            ContourType.EEM: getattr(config.display, "color_eem", "red"),
            ContourType.CALCIUM: getattr(config.display, "color_calcium", "white"),
            ContourType.BRANCH: getattr(config.display, "color_branch", "green"),
        }

        self.contour_configs = {}
        for ct in ContourType:
            self.contour_configs[ct] = ContourConfig(
                color=_default_colors.get(ct, self.color_contour),
                thickness=self.contour_thickness,
                point_radius=self.point_radius,
                point_thickness=self.point_thickness,
                alpha=self.alpha_contour,
                n_points_contour=self.n_points_contour,
                n_interactive_points=self.n_interactive_points,
            )

        self.graphics_scene = QGraphicsScene(self)
        self.points_to_draw: list[Point] = []
        self.contour_points: list[Point] = []
        self.frame: int = 0
        self.contour_mode: bool = False
        self.contour_drawn: bool = False
        self.full_contours: dict = {}

        self.current_spline: Spline | Tuple[Spline, Spline] = None  # entire contour (not only knotpoints), needed for elliptic ratio
        self.lumen_spline: Spline | Tuple[Spline, Spline] = None
        self.new_spline: Spline | Tuple[Spline, Spline] = None
        self.current_geometry: list[SplineGeometry] = []

        self.active_point: Point = None
        self.active_point_index: int = None
        self.measure_index: int = None
        self.measure_colors = self.main_window.measure_colors
        self.reference_mode: bool = False
        self.active_contour_type: ContourType = ContourType.LUMEN
        self.split_index_dict: dict = {}

        self.initial_window_level = 128  # window level is the center which determines the brightness of the image
        self.initial_window_width = 256  # window width is the range of pixel values that are displayed
        self.window_level = self.initial_window_level
        self.window_width = self.initial_window_width

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        image = QGraphicsPixmapItem(QPixmap(self.image_size, self.image_size))
        self.graphics_scene.addItem(image)
        self.setScene(self.graphics_scene)

    def cleanup_temporary_drawing(self):
        """Safely removes un-finalized points and splines from the scene."""
        if hasattr(self, 'new_spline') and self.new_spline:
            if self.new_spline.scene() is not None:
                self.graphics_scene.removeItem(self.new_spline)
            self.new_spline = None

        if hasattr(self, 'points_to_draw'):
            for point in self.points_to_draw:
                if point.scene() is not None:
                    self.graphics_scene.removeItem(point)
            self.points_to_draw = []

        self.contour_drawn = False
        self.contour_mode = False 
        
        self.main_window.tmp_contours = {}

    def set_frame(self, value):
        self.frame = value
        self.current_spline = None
        self.stop_contour()
        if self.measure_index is not None:
            self.stop_measure(self.measure_index)

    def contour_key(self, contour_type: ContourType = None) -> str:
        """Return the string key for the given contour type (defaults to active)."""
        return (contour_type or self.active_contour_type).value

    def _get_contour_data(self, contour_type: ContourType = None) -> list[list[Any], list[Any]]:
        """Return main_window.data[...] for the given/the active contour type (or None)."""
        key = self.contour_key(contour_type)
        return self.main_window.data.get(key, None)

    def set_active_contour_type(self, contour_type: ContourType):
        """Set active contour type and refresh transient state for editing that contour."""
        if contour_type == self.active_contour_type:
            return
        self.active_contour_type = contour_type

        self.current_spline = None
        self.contour_points = []
        self.active_point = None
        self.active_point_index = None

        self.display_image(update_contours=True, update_image=False, update_phase=False)

    def set_data(self, lumen, images):
        """
        Initialize display data. 'lumen' is the legacy argument (first contour),
        but we create entries for all ContourType members in main_window.data
        and prepare self.full_contours dict with per-frame placeholders.
        """
        num_frames = images.shape[0]
        self.image_width = images.shape[1]
        self.scaling_factor = self.image_size / images.shape[1] # image_size in config

        if not hasattr(self.main_window, "data") or self.main_window.data is None:
            self.main_window.data = {}

        self.main_window.data[ContourType.LUMEN.value] = lumen
        self._initialize_contour_data(num_frames=num_frames)
        self.full_contours = {ct.value: [None] * num_frames for ct in ContourType}

        self._build_spline_from_contour(num_frames=num_frames)

        self.images = images
        self.main_window.longitudinal_view.set_data(self.images, self.get_full_contour_list(self.active_contour_type))
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def _initialize_contour_data(self, num_frames: int):
        """
        Ensure every contour type has a [ [x per frame], [y per frame] ] structure
        initialize with number of frames
        """
        for ct in ContourType:
            key = ct.value
            if key not in self.main_window.data:
                self.main_window.data[key] = [[] for _ in range(2)]
                self.main_window.data[key][0] = [[] for _ in range(num_frames)]
                self.main_window.data[key][1] = [[] for _ in range(num_frames)]
            else:
                # make sure existing entries have per-frame lists of correct length (defensive)
                try:
                    if len(self.main_window.data[key][0]) < num_frames:
                        missing = num_frames - len(self.main_window.data[key][0])
                        self.main_window.data[key][0].extend([[] for _ in range(missing)])
                    if len(self.main_window.data[key][1]) < num_frames:
                        missing = num_frames - len(self.main_window.data[key][1])
                        self.main_window.data[key][1].extend([[] for _ in range(missing)])
                except Exception:
                    self.main_window.data[key] = [[] for _ in range(2)]
                    self.main_window.data[key][0] = [[] for _ in range(num_frames)]
                    self.main_window.data[key][1] = [[] for _ in range(num_frames)]

    def _build_spline_from_contour(self, num_frames):
        """
        For each contour type, try to build a Spline -> unscaled contour (if data exists)
        """
        for ct in ContourType:
            key = ct.value
            contour_data = self.main_window.data.get(key, [[], []])
            for frame in range(num_frames):
                try:
                    if contour_data and contour_data[0][frame]:
                        xs = contour_data[0][frame]
                        ys = contour_data[1][frame]
                        geometry = SplineGeometry(
                            xs,
                            ys,
                            self.n_points_contour,
                            (xs[0], ys[0]),
                            None,
                        )
                        spline = Spline(
                            geometry,
                            self.contour_configs[ct].color if ct in self.contour_configs else self.color_contour,
                            self.contour_thickness,
                            self.contour_configs[ct].alpha if ct in self.contour_configs else self.alpha_contour,
                        ).full_contours()
                        self.full_contours[key][frame] = spline.get_unscaled_contour(scaling_factor=1)
                    else:
                        self.full_contours[key][frame] = None
                except Exception:
                    self.full_contours[key][frame] = None

    def get_full_contour_list(self, contour_type: ContourType = None):
        """
        Return the list-of-frame full_contours for a contour type.
        If full_contours is still a list (legacy), return it directly.
        """
        # if legacy list
        if not hasattr(self, "full_contours"):
            return None
        if isinstance(self.full_contours, list):
            return self.full_contours
        # dict path (preferred): return list or None
        key = self.contour_key(contour_type)
        return self.full_contours.get(key, None)

    def _get_full_contour_for_frame(self, contour_type: ContourType = None, frame: int = None):
        """
        Return the contour (contour_x, contour_y) for a single frame and contour_type.
        Defensive: handles both old list-format and new dict-format.
        """
        frame = self.frame if frame is None else frame
        # try dict style first
        if isinstance(getattr(self, "full_contours", None), dict):
            key = self.contour_key(contour_type)
            contour_list = self.full_contours.get(key)
            if contour_list is None:
                return None
            if 0 <= frame < len(contour_list):
                return contour_list[frame]
            return None
        # fallback to legacy list style
        try:
            return self.full_contours[frame]
        except Exception:
            return None

    def ensure_main_window_contour_structure(self, key: str):
        """Create basic [ [x per frame], [y per frame] ] structure if missing."""
        if key not in self.main_window.data:
            if hasattr(self, 'images') and self.images is not None:
                nframes = self.images.shape[0]
            else:
                nframes = 0
            self.main_window.data[key] = [[] for _ in range(2)]
            # make per-frame lists for both x/y
            if nframes:
                self.main_window.data[key][0] = [[] for _ in range(nframes)]
                self.main_window.data[key][1] = [[] for _ in range(nframes)]

    def display_image(self, update_image=False, update_contours=False, update_phase=False):
        image_types = (QGraphicsPixmapItem, Marker)

        if update_image:
            self._remove_image_items(image_types)
            self.active_point = None
            self.active_point_index = None

            display_data, h, w, bpl, qfmt = self._prepare_display_data()
            display_data = self._apply_filter(display_data)
            display_data, bpl, qfmt = self._apply_colormap_if_enabled(display_data, w)

            q_image = QImage(display_data.data, w, h, bpl, qfmt).scaled(
                self.image_size, self.image_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.graphics_scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))
            self._add_center_marker(int(h))

        old_overlays = [it for it in self.graphics_scene.items() if not isinstance(it, image_types)]
        self._remove_non_image_items(image_types)

        if self.main_window.hide_contours:
            self.main_window.longitudinal_view.hide_lview_contours()
        else:
            if update_contours:
                self._draw_contours()
                self._draw_measure()
                self._draw_reference()
                self._maybe_compute_metrics()
            else:
                for it in old_overlays:
                    self.graphics_scene.addItem(it)

        if update_phase:
            self._update_phase_text()

    def update_display(self):
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def _remove_image_items(self, image_types):
        for it in list(self.graphics_scene.items()):
            if isinstance(it, image_types):
                if it.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(it)

    def _prepare_display_data(self):
        if hasattr(self.main_window, "images_display") and self.main_window.images_display is not None:
            img = self.main_window.dicom.pixel_array[self.frame].copy()
            h, w, ch = img.shape
            return img, h, w, ch * w, QImage.Format.Format_RGB888

        lo = self.window_level - self.window_width / 2
        hi = self.window_level + self.window_width / 2
        norm = np.clip(self.images[self.frame, :, :], lo, hi)
        g = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)
        h, w = g.shape
        return g, h, w, w, QImage.Format.Format_Grayscale8

    def _apply_filter(self, img):
        f = getattr(self.main_window, "filter", None)
        if f == 0:
            return cv2.medianBlur(img, 5)
        if f == 1:
            return cv2.GaussianBlur(img, (5, 5), 0)
        if f == 2:
            return cv2.bilateralFilter(img, 9, 75, 75)
        return img

    def _apply_colormap_if_enabled(self, img, width):
        if not getattr(self.main_window, "colormap_enabled", False):
            # return unchanged + appropriate bpl/qfmt inferred by caller
            if img.ndim == 2:
                return img, width, QImage.Format.Format_Grayscale8
            return img, img.shape[2] * width, QImage.Format.Format_RGB888

        # colormap expects gray; handle RGB->gray then map; convert BGR->RGB for Qt
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cmap = cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
        else:
            cmap = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
        rgb = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        return rgb, width * 3, QImage.Format.Format_RGB888

    def _add_center_marker(self, height):
        cx = int((self.image_width // 2) * self.scaling_factor)
        m = Marker(cx, 0, cx, int(height * self.scaling_factor))
        self.graphics_scene.addItem(m)
        if hasattr(self.main_window, "longitudinal_view"):
            self.main_window.longitudinal_view.update_marker(self.frame)

    def _remove_non_image_items(self, image_types):
        for it in list(self.graphics_scene.items()):
            if not isinstance(it, image_types):
                if it.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(it)

    def _maybe_compute_metrics(self):
        ls = getattr(self, "lumen_spline", None)
        if not ls or getattr(ls.geometry, "full_contour", None) is None:
            return

        x, y = ls.get_unscaled_contour(self.scaling_factor)

        if len(x) < 3:
            return

        poly = Polygon(list(zip(x, y)))

        if not poly.is_valid or poly.area == 0:
            logger.warning("Invalid or zero-area polygon created. Skipping metrics.")
            return            

        lumen_area, lumen_circumf, _, _ = compute_polygon_metrics(self.main_window, poly, self.frame)
        longest_d, far_x, far_y = farthest_points(self.main_window, poly.exterior.coords, self.frame)
        shortest_d, close_x, close_y = closest_points(self.main_window, poly, self.frame)
        eem_area, pct = self.compute_eem_and_percent_stenosis(self.frame, lumen_area)

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

    def _update_phase_text(self):
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

    def compute_eem_and_percent_stenosis(self, frame: int, lumen_area: float):
        """
        Return (eem_area, percent_stenosis_text).
        Robust to numpy arrays and malformed data structures.
        """
        eem_area = None
        percent_text = "n/a"
        try:
            # Preferred: use prepared full_contours for EEM (display coords)
            eem_full = self._get_full_contour_for_frame(ContourType.EEM, frame)
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

            # Fallback: use main_window.data (original image coords) and scale to display coords
            if eem_area is None:
                eem_data = self._get_contour_data(ContourType.EEM)
                if eem_data and isinstance(eem_data, (list, tuple)) and len(eem_data) >= 2:
                    xs_orig = ys_orig = None
                    try:
                        # defensive: ensure per-frame lists exist and frame index valid
                        if len(eem_data[0]) > frame:
                            xs_orig = eem_data[0][frame]
                        if len(eem_data[1]) > frame:
                            ys_orig = eem_data[1][frame]
                    except Exception:
                        xs_orig = ys_orig = None

                    try:
                        has_orig = (xs_orig is not None and len(xs_orig) > 0) and (ys_orig is not None and len(ys_orig) > 0)
                    except Exception:
                        has_orig = False

                    if has_orig:
                        xs_scaled = [float(x) * self.scaling_factor for x in xs_orig]
                        ys_scaled = [float(y) * self.scaling_factor for y in ys_orig]
                        polygon_eem = Polygon([(x, y) for x, y in zip(xs_scaled, ys_scaled)])
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
        Safely removes any previous metrics item only if it belongs to this scene.
        """
        # remove previous metrics text if present and belongs to our scene (prevents removeItem scene mismatch)
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

    def _draw_measure(self):
        for index in range(2):
            if (
                self.main_window.data['measures'][self.frame][index] is not None
                and len(self.main_window.data['measures'][self.frame][index]) == 4
            ):
                first_point = QPointF(
                    self.main_window.data['measures'][self.frame][index][0],
                    self.main_window.data['measures'][self.frame][index][1],
                )
                second_point = QPointF(
                    self.main_window.data['measures'][self.frame][index][2],
                    self.main_window.data['measures'][self.frame][index][3],
                )
                self.main_window.data['measures'][self.frame][index] = None
                self.add_measure(first_point, index=index, new=False)
                self.add_measure(second_point, index=index, new=False)

    def add_measure(self, point, index=None, new=True):
        index = index if index is not None else self.measure_index
        new_point = Point((point.x(), point.y()), self.point_thickness, self.point_radius, self.measure_colors[index])
        self.graphics_scene.addItem(new_point)

        if self.main_window.data['measures'][self.frame][index] is None:
            self.main_window.data['measures'][self.frame][index] = [point.x(), point.y()]
        else:  # second point
            self.main_window.data['measures'][self.frame][index] += [point.x(), point.y()]
            line = QLineF(
                self.main_window.data['measures'][self.frame][index][0],
                self.main_window.data['measures'][self.frame][index][1],
                self.main_window.data['measures'][self.frame][index][2],
                self.main_window.data['measures'][self.frame][index][3],
            )
            length = round(line.length() * self.main_window.metadata["resolution"] / self.scaling_factor, 2)
            self.main_window.data['measure_lengths'][self.frame][index] = length
            length_text = QGraphicsTextItem(f'{length} mm')
            length_text.setPos(point.x(), point.y())
            self.graphics_scene.addItem(length_text)
            self.graphics_scene.addLine(line, get_qt_pen(self.measure_colors[index], self.point_thickness))
            if new:
                self.measure_index = None
                self.main_window.setCursor(Qt.CursorShape.ArrowCursor)

    def start_measure(self, index: int):
        if self.contour_mode:
            self.stop_contour()
        self.main_window.data['measures'][self.frame][index] = None
        self.main_window.setCursor(Qt.CrossCursor)
        self.measure_index = index
        self.display_image(update_contours=True)

    def stop_measure(self, index):
        if self.main_window.image_displayed:
            self.measure_index = None
            self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)
            self.main_window.longitudinal_view.update_measure(
                self.frame, index, self.main_window['measures'][self.frame][index]
            )

    def _draw_reference(self):
        if self.main_window.data['reference'][self.frame] is not None:
            reference_point = self.main_window.data['reference'][self.frame]
            # Convert original coordinates to scaled display coordinates
            scaled_x = reference_point[0] * self.scaling_factor
            scaled_y = reference_point[1] * self.scaling_factor
            reference = Point(
                (scaled_x, scaled_y),
                self.point_thickness,
                self.point_radius,
                self.main_window.reference_color,
            )
            self.graphics_scene.addItem(reference)
            text = QGraphicsTextItem('Reference')
            text.setPos(scaled_x, scaled_y)  # Position text at scaled coordinates
            self.graphics_scene.addItem(text)

    def start_reference(self):
        self.reference_mode = True
        self.main_window.setCursor(Qt.CrossCursor)
        self.main_window.data['reference'][self.frame] = None
        self.display_image(update_contours=True)

    def _handle_reference_placement(self, pos):
        """Saves the reference point and exits reference mode."""
        original_x = pos.x() / self.scaling_factor
        original_y = pos.y() / self.scaling_factor
        self.main_window.data['reference'][self.frame] = [original_x, original_y]
        
        self.reference_mode = False
        self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
        self.display_image(update_contours=True)

    def _attempt_contour_switch(self, pos):
        """Switches active contour type if clicking near a different contour's knotpoint."""
        threshold_px = 20 
        min_dist = float('inf')
        nearest_ct = None

        for ct in ContourType:
            contour_data = self._get_contour_data(ct)
            if not contour_data or not contour_data[0][self.frame]:
                continue

            xs, ys = contour_data[0][self.frame], contour_data[1][self.frame]
            for x_orig, y_orig in zip(xs, ys):
                dist = math.hypot(pos.x() - (x_orig * self.scaling_factor), 
                                pos.y() - (y_orig * self.scaling_factor))
                if dist < min_dist:
                    min_dist = dist
                    nearest_ct = ct

        if nearest_ct and min_dist < threshold_px and nearest_ct != self.active_contour_type:
            self.set_active_contour_type(nearest_ct)
            self.display_image(update_contours=True)

    def _draw_contours(self):
        # lumen
        lumen = self._get_contour_data(ContourType.LUMEN)
        if lumen and lumen[0][self.frame]:
            self.draw_contour(lumen, contour_type=ContourType.LUMEN,
                            set_current=(self.active_contour_type == ContourType.LUMEN))
        else:
            self.lumen_spline = None
            if self.active_contour_type == ContourType.LUMEN:
                self.current_spline = None
                self.contour_points = []

        # other contours
        for ct in ContourType:
            if ct == ContourType.LUMEN:
                continue
            data = self._get_contour_data(ct)
            if data and data[0][self.frame]:
                self.draw_contour(data, contour_type=ct, set_current=(ct == self.active_contour_type))

    def draw_contour(self, contour_data, contour_type: ContourType = None, set_current: bool = False):
        """
        Draw contour_data for the specified contour_type.
        - If set_current is True, this spline becomes self.current_spline (editing target).
        - If contour_type == ContourType.LUMEN also set self.lumen_spline for metrics.
        """
        if not contour_data or not contour_data[0][self.frame]:
            return

        lumen_x = [point * self.scaling_factor for point in contour_data[0][self.frame]]
        lumen_y = [point * self.scaling_factor for point in contour_data[1][self.frame]]

        ct = contour_type or self.active_contour_type
        cfg = self.contour_configs.get(ct, None)
        thickness = cfg.thickness if cfg else self.contour_thickness
        color = cfg.color if cfg else self.color_contour
        alpha = cfg.alpha if cfg else self.alpha_contour

        geometry = SplineGeometry(
            lumen_x, 
            lumen_y, 
            self.n_points_contour, 
            (lumen_x[0], lumen_y[0]), 
            None
        )
        geometry.interpolate()

        if geometry.full_contour[0] is not None:
            knot_points = [
                Point(
                    (geometry.knot_points_x[i], geometry.knot_points_y[i]),
                    self.point_thickness,
                    self.point_radius,
                    color,
                    alpha,
                )
                for i in range(len(geometry.knot_points_x) - 1)
            ]

            for p in knot_points:
                self.graphics_scene.addItem(p)
            spline = Spline(geometry, color, thickness, alpha)
            self.graphics_scene.addItem(spline)

            self.full_contours[self.contour_key(ct)][self.frame] = spline.get_unscaled_contour(self.scaling_factor)

            if ct == ContourType.LUMEN:
                self.lumen_spline = spline

            if set_current:
                self.current_spline = spline
                self.contour_points = knot_points
        else:
            logger.warning(f'Spline for frame {self.frame + 1} could not be interpolated for {ct.value}')

    ###############################
    # Start contour drawing logic #
    ###############################
    def start_contour(self, contour_type: ContourType = None):
        """
        Start drawing a new contour of the specified type.
        
        Sets the active contour type, clears previous data for this frame,
        and switches to contour drawing mode.
        """
        if contour_type is not None:
            self.set_active_contour_type(contour_type)

        self.measure_index = None
        self.main_window.setCursor(Qt.CursorShape.CrossCursor)
        self.contour_mode = True
        self.points_to_draw = []

        key = self.contour_key()
        self.ensure_main_window_contour_structure(key)

        self.main_window.data[key][0][self.frame] = []
        self.main_window.data[key][1][self.frame] = []
        self.display_image(update_contours=True)  # clear previous contour

    def add_contour(self, click_pos):
        """Handles logic for adding a new point to a manual contour being drawn."""
        # 1. Validation: Handle cases where the drawing state is corrupted
        if not self._is_drawing_valid():
            self._reset_contour_drawing()
            return

        # 2. Closure Check: See if the user clicked near the start to finish the shape
        if self._should_close_contour(click_pos):
            self._close_current_spline()
            return

        # 3. Point Placement: Create and store the new knot point
        new_point = self._create_knot_point(click_pos)
        self.points_to_draw.append(new_point)
        self.graphics_scene.addItem(new_point)

        # 3. Spline Management: Draw or update the smooth curve
        if len(self.points_to_draw) >= 3:
            self._update_or_create_spline()

    def _is_drawing_valid(self) -> bool:
        """Checks if the first point is valid; returns False if drawing was interrupted."""
        if not self.points_to_draw:
            return True
        return self.points_to_draw[0].get_coords()[0] is not None
    
    def _reset_contour_drawing(self):
        """Resets the contour drawing state and refreshes the display."""
        self.points_to_draw = []
        self.new_spline = None
        self.contour_drawn = False
        self.display_image(update_contours=True)

    def _create_knot_point(self, pos) -> Point:
        """Helper to instantiate a Point with current config."""
        return Point((pos.x(), pos.y()), self.point_thickness, self.point_radius)

    def _update_or_create_spline(self):
        """Logic to draw the curve between knot points."""
        xs = [p.get_coords()[0] for p in self.points_to_draw]
        ys = [p.get_coords()[1] for p in self.points_to_draw]

        if not self.contour_drawn:
            cfg = self.contour_configs[self.active_contour_type]
            start_coords = (xs[0], ys[0])
            geometry = SplineGeometry(
                xs,
                ys,
                self.n_points_contour,
                start_coords,
                None,
            )
            self.new_spline = Spline(
                geometry, cfg.color, self.contour_thickness, cfg.alpha
            )
            self.contour_drawn = True
            self.graphics_scene.addItem(self.new_spline)
        elif self.new_spline:
            self.new_spline.geometry.knot_points_x = xs
            self.new_spline.geometry.knot_points_y = ys
            self.new_spline.geometry.start_coords = (xs[0], ys[0])
            self.new_spline._rebuild_path()

    def _should_close_contour(self, current_pos) -> bool:
            if len(self.points_to_draw) < 2: 
                return False
                
            start_x, start_y = self.points_to_draw[0].get_coords()
            dist = math.hypot(current_pos.x() - start_x, current_pos.y() - start_y)
            return dist < 20

    def _close_current_spline(self):
        """Close the current contour and save it."""
        if self.new_spline is not None:
            downsampled = downsample(
                ([self.new_spline.geometry.full_contour[0].tolist()], [self.new_spline.geometry.full_contour[1].tolist()]),
                self.n_interactive_points,
            )

            key = self.contour_key()
            self.ensure_main_window_contour_structure(key)

            self.main_window.data[key][0][self.frame] = [
                point / self.scaling_factor for point in downsampled[0]
            ]
            self.main_window.data[key][1][self.frame] = [
                point / self.scaling_factor for point in downsampled[1]
            ]

        self.stop_contour()

    def stop_contour(self):
        """
        Stop contour drawing mode, finalize the contour for the current frame, and update the display.
        
        This method exits contour drawing mode, resets the cursor, and refreshes the image display with the updated contour.
        If a contour was drawn for the current frame, it also updates the longitudinal view with the new contour.
        """
        if self.main_window.image_displayed:
            self.contour_mode = False
            self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)

            contour_for_frame: Tuple[np.array, np.array] = self._get_full_contour_for_frame(self.active_contour_type, self.frame)
            try:
                self.main_window.longitudinal_view.lview_contour(self.frame, contour_for_frame, update=True)
            except Exception as e:
                logger.debug(f"Could not update longitudinal view for frame {self.frame}: {e}")

    ######################
    # Mouse click events #
    ######################
    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.contour_mode:
                self.add_contour(pos)
            elif self.measure_index is not None:
                self.add_measure(pos)
            elif self.reference_mode:
                self._handle_reference_placement(pos)
            else:
                # First, try to switch active contour if user clicked near another one
                self._attempt_contour_switch(pos)
                # Then, handle interaction with points or the spline path
                self._handle_item_interaction(pos, event.pos())

        elif event.button() == Qt.MouseButton.RightButton:
            # Store for windowing/leveling or context menus
            self.mouse_x = event.position().x()
            self.mouse_y = event.position().y()

        super().mousePressEvent(event)

    def _handle_item_interaction(self, scene_pos, view_pos):
        """Handles clicking existing knotpoints or adding new ones to a spline."""
        items = self.items(view_pos)
        point_item = next((i for i in items if isinstance(i, Point)), None)
        spline_item = next((i for i in items if isinstance(i, Spline)), None)

        if point_item and point_item in self.contour_points:
            self._select_existing_point(point_item)
        elif spline_item:
            self._add_new_point_to_spline(scene_pos)

    def _select_existing_point(self, point_item):
        self.main_window.setCursor(Qt.CursorShape.BlankCursor) # remove cursor for precise contour changes
        # https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
        self.active_point_index = self.contour_points.index(point_item)
        point_item.update_color()
        self.active_point = point_item

    def _add_new_point_to_spline(self, pos):
        if not self.current_spline:
            return

        path_index = self.current_spline.on_path(pos)
        if path_index is None: # Safety check: only add if we actually clicked the path
            return

        self.main_window.setCursor(Qt.CursorShape.BlankCursor)
        
        cfg = self.contour_configs.get(self.active_contour_type)
        self.active_point = Point(
            (pos.x(), pos.y()),
            self.point_thickness,
            self.point_radius,
            cfg.color if cfg else self.color_contour,
            cfg.alpha if cfg else self.alpha_contour,
        )
        self.graphics_scene.addItem(self.active_point)
        self.active_point_index = self.current_spline.update(pos, -1, path_index)
        self.contour_points.insert(self.active_point_index, self.active_point)
        
        self.active_point.update_color()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None:
                item = self.active_point
                new_scene_pos = self.mapToScene(event.pos())

                item.update_pos(new_scene_pos)
                
                self.current_spline.update(new_scene_pos, self.active_point_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.position().x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.position().y() - self.mouse_y) * self.windowing_sensitivity
            self.display_image(update_image=True)
            self.setMouseTracking(False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # for some reason event.buttons() does not work here
            if self.active_point_index is not None:
                self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
                item = self.active_point
                item.reset_color()

                key = self.contour_key()
                self.ensure_main_window_contour_structure(key)

                self.main_window.data[key][0][self.frame] = [
                    point / self.scaling_factor for point in self.current_spline.knot_points[0]
                ]
                self.main_window.data[key][1][self.frame] = [
                    point / self.scaling_factor for point in self.current_spline.knot_points[1]
                ]
                contour_for_frame = self._get_full_contour_for_frame(self.active_contour_type, self.frame)
                self.display_image(update_contours=True)
                try:
                    self.main_window.longitudinal_view.lview_contour(self.frame, contour_for_frame, update=True)
                except Exception as e:
                    logger.debug(f"Could not update longitudinal view after mouse release for frame {self.frame}: {e}")
                self.active_point_index = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.contour_mode and self.new_spline is not None:
                # print("IVUSDisplay data:", self)
                # print(f"Frame: {self.frame}")
                # print(f"Image size: {self.image_size}")
                # print(f"Scaling factor: {self.scaling_factor}")
                # print(f"Active contour type: {self.active_contour_type}")
                # print(f"Contour mode: {self.contour_mode}")
                # print(f"Contour drawn: {self.contour_drawn}")
                # print(f"Points to draw: {[(p.get_coords()) for p in self.points_to_draw]}")
                # print(f"Contour points: {[(p.get_coords()) for p in self.contour_points]}")
                # print(f"Active point: {self.active_point.get_coords() if self.active_point else None}")
                # print(f"Active point index: {self.active_point_index}")
                # print(f"Current spline: {self.current_spline}")
                # print(f"New spline: {self.new_spline}")
                # print(f"Lumen spline: {self.lumen_spline}")
                # print(f"Full contours: {self.full_contours}")
                # print(f"Window level: {self.window_level}, Window width: {self.window_width}")
                # print(f"Measure index: {self.measure_index}")
                # print(f"Reference mode: {self.reference_mode}")
                pos = self.mapToScene(event.pos())
                self.new_spline.geometry.end_coords = (pos.x(), pos.y())
                self.new_spline.update(pos, len(self.points_to_draw) - 1, len(self.points_to_draw) - 1)
                self._close_current_spline()
                print(self.new_spline.geometry)
            else:
                print("Active Point:", self.active_point)
                print("Active Point Index:", self.active_point_index)
                print("Main window data:", self.main_window.data)
                pass