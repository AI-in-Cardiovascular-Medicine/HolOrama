import cv2
import math

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Union, Any

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, QLineF, QPointF
from PyQt6.QtGui import QPixmap, QImage

from gui.utils.geometry import Point, Spline, SplineGeometry, OpenSplineGeometry, OpenSpline, get_qt_pen
from gui.utils.metrics import MetricsMixin
from gui.right_half.longitudinal_view import Marker
from segmentation.segment import downsample
from segmentation.save_as_nifti import contours_to_mask
from input_output.contours_io import Measure


# RGB colour for each mask label (index = label value).
# Label 0 (background) is intentionally skipped during blending.
_MASK_OVERLAY_COLORS = np.array(
    [
        [0, 0, 0],  # 0 – background  (unused)
        [0, 180, 255],  # 1 – lumen        cyan-blue
        [0, 200, 80],  # 2 – EEM wall     green
        [255, 215, 0],  # 3 – calcium      gold
        [255, 100, 0],  # 4 – lipid        orange
        [200, 0, 220],  # 5 – macrophage   violet
        [220, 80, 80],  # 6 – adventitia   rose
        [0, 180, 255],  # 7 – branch       cyan-blue (same as lumen)
    ],
    dtype=np.float32,
)

_MASK_ALPHA = 0.45  # overlay opacity (0 = transparent, 1 = opaque)


class ContourType(Enum):
    LUMEN = "lumen"
    EEM = "eem"
    CALCIUM = "calcium"
    BRANCH = "branch"
    LIPID = "lipid"
    MACROPHAGE = "macrophage"
    MEASUREMENT_1 = "measurement_1"
    MEASUREMENT_2 = "measurement_2"
    REFERENCE = "reference"
    WIRE = "wire"


class SegmentationTool(Enum):
    CLOSED_SPLINE = "closed_spline"
    OPEN_SPLINE = "open_spline"
    BRUSH = "brush"
    ANGLE = "angle"
    LINE = "line"
    POINT = "point"


ALLOWED_TOOLS = {
    ContourType.LUMEN: {
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.EEM: {
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.CALCIUM: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.BRANCH: {
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.LIPID: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.MACROPHAGE: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.MEASUREMENT_1: {SegmentationTool.LINE},
    ContourType.MEASUREMENT_2: {SegmentationTool.LINE},
    ContourType.REFERENCE: {SegmentationTool.POINT},
    ContourType.WIRE: {SegmentationTool.ANGLE},
}


def validate_tool(contour_type: ContourType, tool: SegmentationTool):
    if tool not in ALLOWED_TOOLS.get(contour_type, set()):
        raise ValueError(f"{tool} not allowed for {contour_type}")


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


SENSITIVITY = 20  # pixels for closure detection


class Display(QGraphicsView, MetricsMixin):
    """
    Displays images and contours and allows the user to add and manipulate contours.
    """

    def __init__(self, main_window):
        super(Display, self).__init__()
        self.main_window = main_window
        config = main_window.config

        self.n_interactive_points: int = config.display.n_interactive_points
        self.n_points_contour: int = config.display.n_points_contour
        self.image_size: int = config.display.image_size  # image display in pixel (square)
        self.windowing_sensitivity: float = config.display.windowing_sensitivity
        self.zoom_sensitivity: float = config.display.zoom_sensitivity
        self.contour_thickness: int = config.display.contour_thickness
        self.point_thickness: int = config.display.point_thickness
        self.point_radius: int = config.display.point_radius
        self.start_color: str = config.display.color_start_point
        self.end_color: str = config.display.color_end_point
        self.color_angle: str = config.display.color_angle

        self.color_contour = getattr(config.display, "color_contour", (255, 255, 255))
        self.alpha_contour = getattr(config.display, "alpha_contour", 255)  # config uses 0..255

        _default_colors = {
            ContourType.LUMEN: getattr(config.display, "color_contour", "green"),
            ContourType.EEM: getattr(config.display, "color_eem", "red"),
            ContourType.CALCIUM: getattr(config.display, "color_calcium", "white"),
            ContourType.BRANCH: getattr(config.display, "color_branch", "green"),
            ContourType.LIPID: getattr(config.display, "color_lipid", "yellow"),
            ContourType.MACROPHAGE: getattr(config.display, "color_macrophage", "blue"),
            ContourType.WIRE: getattr(config.display, "color_angle", "#ffa500"),
            ContourType.REFERENCE: getattr(config.display, "color_reference", "yellow"),
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

        # scene data
        self.graphics_scene = QGraphicsScene(self)
        self.images: np.ndarray = None
        self.scaling_factor: float = 1.0
        self.image_width: int = 0
        image = QGraphicsPixmapItem(QPixmap(self.image_size, self.image_size))
        self.graphics_scene.addItem(image)
        self.setScene(self.graphics_scene)

        self.initial_window_level: int = 128  # window level is the center which determines the brightness of the image
        self.initial_window_width: int = 256  # window width is the range of pixel values that are displayed
        self.window_level: int = self.initial_window_level
        self.window_width: int = self.initial_window_width
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0
        self._panning: bool = False
        self._pan_last_pos = None

        self.frame: int = 0
        self.points_to_draw: list[Point] = []
        self.start_coords: Tuple[float, float] | None = None
        self.end_coords: Tuple[float, float] | None = None
        self.working_spline: Spline = None
        self.finalized_splines: dict[ContourType, Spline] = {}

        # flags and states
        self.active_contour_type: ContourType = ContourType.LUMEN
        self.active_contour_index: int = 0
        self.active_segmentation_tool: SegmentationTool = SegmentationTool.CLOSED_SPLINE
        self.drawing_mode: bool = False
        self.append_contour_mode: bool = False
        self._contour_close_committed: bool = False
        self.mask_mode: bool = False
        self.active_point: Point = None
        self.active_end_coords_flag = True  # Flag to switch, which point double click sets

        #####################################################################################################
        # legacy to be refactored
        self.active_point_index: int = None  # wtf is this legacy crap
        self.measure_index: int = None  # wtf is this legacy crap
        self.pending_measure_points: list = [None, None]  # first-click-only state per measure index
        self.reference_mode: bool = False
        self.angle_mode: bool = False
        self.angle_clicks: list[QPointF] = []
        #####################################################################################################

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # initialize data from main_window data
    def set_data(self, images):
        """Initialize display data from main_window.data (Dict[int, FrameData])."""
        self.images = images
        self.image_width = images.shape[1]
        self.scaling_factor = self.image_size / images.shape[1]

        self.finalized_splines = {ct.value: [] for ct in ContourType}

        self._draw_contours_frame()

        self.main_window.longitudinal_view.set_data(self.images)
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def _draw_contours_frame(self):
        # other contours
        closed_contour_types = {
            ct for ct in ContourType if SegmentationTool.CLOSED_SPLINE in ALLOWED_TOOLS.get(ct, set())
        }
        for ct in closed_contour_types:
            fd = self.main_window.data.get(self.frame)
            if fd is None:
                continue
            key = self.contour_key(ct)
            contour_obj = getattr(fd, key, None)
            if contour_obj is None or not contour_obj.contours:
                continue
            for i, contour in enumerate(contour_obj.contours):
                if not contour or not contour[0]:
                    continue
                data = (contour[0], contour[1] if len(contour) > 1 else [])
                self._draw_contour_frame(
                    data,
                    contour_type=ct,
                    set_current=(ct == self.active_contour_type and i == self.active_contour_index),
                    contour_index=i,
                )

    def _draw_contour_frame(
        self,
        contour_data: Tuple[List[float], List[float]],
        contour_type: ContourType = None,
        set_current: bool = False,
        contour_index: int = 0,
    ):
        """
        Draw contour_data for the specified contour_type.
        - If set_current is True, this spline becomes self.working_spline (editing target).
        - If contour_type == ContourType.LUMEN also set self.lumen_spline for metrics.
        """
        if not contour_data or not contour_data[0] or not contour_data[1]:
            return

        lumen_x = [point * self.scaling_factor for point in contour_data[0]]
        lumen_y = [point * self.scaling_factor for point in contour_data[1]]

        ct = contour_type
        cfg = self.contour_configs.get(ct, None)
        color = cfg.color if cfg else self.color_contour
        alpha = cfg.alpha if cfg else self.alpha_contour
        thickness = cfg.thickness if cfg else self.contour_thickness

        key = self.contour_key(ct)
        fd = self.main_window.data.get(self.frame)
        contour_obj = getattr(fd, key, None) if fd else None
        raw_start = (
            contour_obj.start_coords[contour_index]
            if (contour_obj and len(contour_obj.start_coords) > contour_index)
            else None
        )
        raw_end = (
            contour_obj.end_coords[contour_index]
            if (contour_obj and len(contour_obj.end_coords) > contour_index)
            else None
        )

        start_coords = (raw_start[0] * self.scaling_factor, raw_start[1] * self.scaling_factor) if raw_start else None
        end_coords = (raw_end[0] * self.scaling_factor, raw_end[1] * self.scaling_factor) if raw_end else None

        if start_coords is None and lumen_x:
            start_coords = (lumen_x[0], lumen_y[0])

        is_closed = (
            contour_obj.closed[contour_index] if (contour_obj and len(contour_obj.closed) > contour_index) else True
        )

        if is_closed:
            geometry = SplineGeometry(
                lumen_x,
                lumen_y,
                self.n_points_contour,
                start_coords,
                end_coords,
            )
            geometry._ensure_start_end_coords()
            geometry.interpolate()
            spline_cls = Spline
        else:
            geometry = OpenSplineGeometry(
                knot_points_x=lumen_x,
                knot_points_y=lumen_y,
                n_interpolated_points=self.n_points_contour,
                start_coords=start_coords,
                end_coords=end_coords,
            )
            geometry.interpolate()
            spline_cls = OpenSpline

        if geometry.full_contour[0] is not None:
            knot_points = []
            knot_range = range(len(geometry.knot_points_x) - 1) if is_closed else range(len(geometry.knot_points_x))
            for i in knot_range:
                curr_x = geometry.knot_points_x[i]
                curr_y = geometry.knot_points_y[i]
                knot_color = color
                brush = False
                if start_coords and math.hypot(curr_x - start_coords[0], curr_y - start_coords[1]) < SENSITIVITY:
                    knot_color = self.start_color
                    brush = True
                if end_coords and math.hypot(curr_x - end_coords[0], curr_y - end_coords[1]) < SENSITIVITY:
                    knot_color = self.end_color
                    brush = True
                knot_point = Point(
                    (curr_x, curr_y),
                    self.point_thickness,
                    self.point_radius,
                    i,
                    knot_color,
                    alpha,
                    brush,
                )
                knot_points.append(knot_point)

            for p in knot_points:
                self.graphics_scene.addItem(p)
            spline = spline_cls(geometry, color=color, line_thickness=thickness, transparency=alpha)
            self.graphics_scene.addItem(spline)

            key_str = self.contour_key(ct)
            lst = self._ensure_finalized_list_for_key(key_str, contour_index + 1)
            lst[contour_index] = spline

            if set_current and not (self.drawing_mode and self.append_contour_mode):
                self.working_spline = spline
                self.points_to_draw = knot_points
        else:
            logger.warning(f'Spline for frame {self.frame + 1} could not be interpolated for {ct.value}')

    def set_frame(self, value):
        self.frame = value
        self.active_contour_index = 0
        self._interrupt_drawing_mode()
        if self.measure_index is not None:
            self.stop_measure(self.measure_index)

        self.finalized_splines = {ct.value: None for ct in ContourType}
        self._draw_contours_frame()

        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def get_full_contour_list(self, contour_type: ContourType = None, unscaled: bool = False) -> List | None:
        """
        Return a list of length num_frames with interpolated contours (or None per frame).
        Reads from main_window.data (Dict[int, FrameData]).
        """
        key = self.contour_key(contour_type)
        num_frames = self.images.shape[0] if self.images is not None else 0
        full_contours = [None] * num_frames

        for frame_idx in range(num_frames):
            fd = self.main_window.data.get(frame_idx)
            if fd is None:
                continue
            contour_obj = getattr(fd, key, None)
            if contour_obj is None or not contour_obj.contours or not contour_obj.contours[0]:
                continue
            x_coords = list(contour_obj.contours[0][0])
            y_coords = list(contour_obj.contours[0][1])

            if len(x_coords) > 1:
                spline_geo = SplineGeometry(
                    knot_points_x=x_coords,
                    knot_points_y=y_coords,
                    n_interpolated_points=self.n_interactive_points,
                    start_coords=None,
                    end_coords=None,
                )
                if unscaled and hasattr(self, 'scaling_factor'):
                    spline_geo = spline_geo.scale(self.scaling_factor)
                full_contours[frame_idx] = spline_geo.interpolate()

        return full_contours

    def contour_key(self, contour_type: ContourType = None) -> str:
        """Return the string key for the given contour type (defaults to active)."""
        return (contour_type or self.active_contour_type).value

    def get_current_spline(self):
        """Returns the currently active spline based on self.active_contour_type."""
        return self.get_finalized_spline(self.active_contour_type, self.active_contour_index)

    def _ensure_finalized_list_for_key(self, key: str, length: int):
        """Ensure finalized_splines[key] exists and has at least `length` entries."""
        lst = self.finalized_splines.get(key)
        if lst is None:
            lst = []
            self.finalized_splines[key] = lst
        if len(lst) < length:
            lst.extend([None] * (length - len(lst)))
        return lst

    def get_finalized_spline(self, contour_type: ContourType = None, index: int | None = None):
        """Return the finalized spline for given contour type and index (or None)."""
        key = self.contour_key(contour_type)
        lst = self.finalized_splines.get(key, [])
        if not lst:
            return None
        idx = index if index is not None else self.active_contour_index
        if idx is None or idx >= len(lst):
            # fallback to first contour if requested index missing
            return lst[0] if lst else None
        return lst[idx]

    def _get_contour_data(self, contour_type: ContourType = None, frame: int | None = None):
        """
        Return (x_list, y_list) for the given contour type at the given frame,
        or ([], []) if absent. Reads from main_window.data (Dict[int, FrameData]).
        """
        key = self.contour_key(contour_type)
        if frame is None:
            frame = self.frame
        fd = self.main_window.data.get(frame)
        if fd is None:
            return ([], [])
        contour_obj = getattr(fd, key, None)
        if contour_obj is None or not contour_obj.contours or not contour_obj.contours[0]:
            return ([], [])
        c = contour_obj.contours[0]
        return (c[0] if c else []), (c[1] if len(c) > 1 else [])

    def set_active_contour_type(self, contour_type: ContourType):
        """Set active contour type and refresh transient state for editing that contour."""
        if contour_type == self.active_contour_type:
            return
        self.active_contour_type = contour_type
        self.active_contour_index = 0

        self.working_spline = None
        self.points_to_draw = []
        self.active_point = None
        self.active_point_index = None

        self.display_image(update_contours=True, update_image=False, update_phase=False)

    # image data handling methods
    def display_image(self, update_image=False, update_contours=False, update_phase=False):
        image_types = (QGraphicsPixmapItem, Marker)

        old_overlays = [it for it in self.graphics_scene.items() if not isinstance(it, image_types)]
        self._remove_non_image_items(image_types)

        if update_image:
            self._remove_image_items(image_types)
            self.active_point = None

            display_data, h, w, bpl, qfmt = self._prepare_display_data()
            display_data, bpl, qfmt = self._apply_colormap_if_enabled(display_data, w)

            if getattr(self.main_window, 'mask_mode_box', None) and self.main_window.mask_mode_box.isChecked():
                display_data, bpl, qfmt = self._apply_mask_overlay(display_data, w)

            q_image = QImage(display_data.data, w, h, bpl, qfmt).scaled(
                self.image_size,
                self.image_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.graphics_scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))
            self._add_center_marker(int(h))

        if self.main_window.hide_contours:
            self.main_window.longitudinal_view.hide_lview_contours()
        else:
            if update_contours:
                lumen_key = self.contour_key(ContourType.LUMEN)
                eem_key = self.contour_key(ContourType.EEM)
                lumen_contour = None
                lumen = self.finalized_splines.get(lumen_key)
                if lumen:
                    if isinstance(lumen, list):
                        first = lumen[0] if len(lumen) > 0 else None
                    else:
                        first = lumen
                    if first is not None:
                        lumen_contour = first.get_unscaled_contour(self.scaling_factor)

                eem_contour = None
                eem = self.finalized_splines.get(eem_key)
                if eem:
                    if isinstance(eem, list):
                        first = eem[0] if len(eem) > 0 else None
                    else:
                        first = eem
                    if first is not None:
                        eem_contour = first.get_unscaled_contour(self.scaling_factor)

                self._draw_contours_frame()
                self._draw_measure()
                self._draw_reference()
                self._draw_angles()
                self._draw_open_spline_edge_lines()

                self._maybe_compute_metrics(lumen_contour, eem_contour)
                self.update_active_contour()
            else:
                for it in old_overlays:
                    self.graphics_scene.addItem(it)

        if update_phase:
            self.update_phase_text()

    def update_display(self):
        """Syntax sugar method to update the entire display after changes to contours or image."""
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def _remove_non_image_items(self, image_types):
        for it in list(self.graphics_scene.items()):
            if not isinstance(it, image_types):
                if it.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(it)

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

    def _apply_mask_overlay(self, display_data, w):
        """
        Alpha-blend per-label segmentation colours into the display image array.
        Returns (rgb_array, bytes_per_line, QImage_format).
        """
        try:
            frame_mask = contours_to_mask(
                self.images[self.frame : self.frame + 1],
                [self.frame],
                self.main_window.data,
                self.main_window.metadata,
            )[
                0
            ]  # (H, W) uint8
        except Exception as e:
            logger.warning(f'Mask overlay failed for frame {self.frame}: {e}')
            if display_data.ndim == 2:
                return display_data, w, QImage.Format.Format_Grayscale8
            return display_data, w * 3, QImage.Format.Format_RGB888

        # Ensure RGB base
        if display_data.ndim == 2:
            rgb = np.stack([display_data, display_data, display_data], axis=-1).astype(np.float32)
        else:
            rgb = display_data.astype(np.float32)

        for label_idx in range(1, len(_MASK_OVERLAY_COLORS)):
            pixels = frame_mask == label_idx
            if not pixels.any():
                continue
            rgb[pixels] = rgb[pixels] * (1.0 - _MASK_ALPHA) + _MASK_OVERLAY_COLORS[label_idx] * _MASK_ALPHA

        result = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(result), w * 3, QImage.Format.Format_RGB888

    def _add_center_marker(self, height):
        cx = int((self.image_width // 2) * self.scaling_factor)
        m = Marker(cx, 0, cx, int(height * self.scaling_factor))
        self.graphics_scene.addItem(m)
        if hasattr(self.main_window, "longitudinal_view"):
            self.main_window.longitudinal_view.update_marker(self.frame)

    # contour drawing and manipulation methods
    def cleanup_temporary_drawing(self):
        """Safely removes un-finalized points and splines from the scene."""
        if hasattr(self, 'working_spline') and self.working_spline is not None:
            if self.working_spline.scene() is not None:
                self.graphics_scene.removeItem(self.working_spline)
            self.working_spline = None

        if hasattr(self, 'points_to_draw') and self.points_to_draw is not None:
            for point in self.points_to_draw:
                if point.scene() is not None:
                    self.graphics_scene.removeItem(point)
            self.points_to_draw = []

        was_drawing = self.drawing_mode
        self.drawing_mode = False
        self.reference_mode = False
        self.angle_mode = False
        self.append_contour_mode = False
        self._contour_close_committed = False
        self.active_point = None
        if was_drawing and (
            self.active_contour_type == ContourType.MEASUREMENT_1
            or self.active_contour_type == ContourType.MEASUREMENT_2
            or self.active_contour_type == ContourType.REFERENCE
            or self.active_contour_type == ContourType.WIRE
        ):
            self.active_contour_type = ContourType.LUMEN

    def _interrupt_drawing_mode(self):
        """Handles safe exit of drawing mode, returning to initial state."""
        self.cleanup_temporary_drawing()
        if self.measure_index is not None:
            self.pending_measure_points[self.measure_index] = None
            self.measure_index = None
        self.main_window.display.setCursor(Qt.CursorShape.ArrowCursor)
        self.display_image(update_contours=True)

    def start_contour(
        self,
        contour_type: ContourType = ContourType.LUMEN,
        segmentation_tool: SegmentationTool = None,
        append: bool = False,
    ):
        """
        Start drawing a new contour of the specified type and with the specified tool.

        Sets the active contour type, clears previous data for this frame,
        and switches to contour drawing mode (leaves temporary data in main_window.tmp_contours).
        If append=True, existing contours are preserved and the new one will be appended.
        """
        if contour_type is not None:
            self.set_active_contour_type(contour_type)

        self.drawing_mode = True
        self.append_contour_mode = append
        self._contour_close_committed = False
        if not append:
            self.active_contour_index = 0

        # save the current state of the contour to the tmp storage
        key = self.contour_key(contour_type)
        if not hasattr(self.main_window, 'tmp_contours'):
            self.main_window.tmp_contours = {}
        fd = self.main_window.data.get(self.frame)
        if fd:
            contour_obj = getattr(fd, key, None)
            if contour_obj and contour_obj.contours and contour_obj.contours[0]:
                xlist = list(contour_obj.contours[0][0]) if contour_obj.contours[0][0] else []
                ylist = list(contour_obj.contours[0][1]) if len(contour_obj.contours[0]) > 1 else []
                self.main_window.tmp_contours[key] = (xlist, ylist)

        self.active_segmentation_tool = segmentation_tool if segmentation_tool else self.active_segmentation_tool

        # Fall back to CLOSED_SPLINE if the active tool is not allowed for this contour type
        ct = contour_type or self.active_contour_type
        if self.active_segmentation_tool not in ALLOWED_TOOLS.get(ct, set()):
            self.active_segmentation_tool = SegmentationTool.CLOSED_SPLINE
            self.main_window.left_half.closed_spline_btn.setChecked(True)

        self.measure_index = None
        self.working_spline = None
        self.points_to_draw = []
        self.active_point = None
        self.active_end_coords_flag = True
        self.main_window.display.setCursor(Qt.CursorShape.CrossCursor)

        fd = self.main_window.data[self.frame]
        contour_obj = getattr(fd, key)
        if not append:
            contour_obj.contours = []
            contour_obj.start_coords = []
            contour_obj.end_coords = []
            contour_obj.closed = []
        self.display_image(update_contours=True)

    def add_contour(self, click_pos, segmentation_tool: SegmentationTool = SegmentationTool.CLOSED_SPLINE):
        """Handles logic for adding a new point to a manual contour being drawn."""
        # 1. Validation: Handle cases where the drawing state is corrupted
        if not self._is_drawing_valid():
            self._interrupt_drawing_mode()
            return

        if segmentation_tool == SegmentationTool.CLOSED_SPLINE:
            # 2. Closure Check: See if the user clicked near the start to finish the shape
            if self._should_close_contour(click_pos):
                self._close_current_spline()
                return

            # 3. Point Placement: Create and store the new knot point
            new_point = self._create_knot_point(click_pos)
            self.points_to_draw.append(new_point)
            self.graphics_scene.addItem(new_point)

            # 4. Spline Management: Draw or update the smooth curve
            if len(self.points_to_draw) >= 3:
                self._update_or_create_spline()

        elif segmentation_tool == SegmentationTool.OPEN_SPLINE:
            new_point = self._create_knot_point(click_pos)
            self.points_to_draw.append(new_point)
            self.graphics_scene.addItem(new_point)

            if len(self.points_to_draw) >= 2:
                self._update_or_create_spline(is_closed=False)

    def _is_drawing_valid(self) -> bool:
        """Checks if the first point is valid; returns False if drawing was interrupted."""
        if not self.points_to_draw:
            return True
        return self.points_to_draw[0].get_coords()[0] is not None

    def _create_knot_point(self, pos) -> Point:
        """Helper to instantiate a Point with current config."""
        ct = self.active_contour_type
        cfg = self.contour_configs.get(ct, None)
        color = cfg.color if cfg else self.color_contour
        alpha = cfg.alpha if cfg else self.alpha_contour
        return Point(
            pos=(pos.x(), pos.y()),
            line_thickness=self.point_thickness,
            point_radius=self.point_radius,
            index=0,
            color=color,
            transparency=alpha,
        )

    def _update_or_create_spline(self, is_closed=True):
        """Logic to draw the curve between knot points."""
        xs = [p.get_coords()[0] for p in self.points_to_draw]
        ys = [p.get_coords()[1] for p in self.points_to_draw]

        if self.working_spline is None:
            cfg = self.contour_configs[self.active_contour_type]
            start_coords = (xs[0], ys[0])
            if is_closed:
                geometry = SplineGeometry(
                    knot_points_x=xs,
                    knot_points_y=ys,
                    n_interpolated_points=self.n_points_contour,
                    start_coords=start_coords,
                    end_coords=None,
                    is_closed=True,
                )
                self.working_spline = Spline(geometry, cfg.color, self.contour_thickness, cfg.alpha)
            else:
                geometry = OpenSplineGeometry(
                    knot_points_x=xs,
                    knot_points_y=ys,
                    n_interpolated_points=self.n_points_contour,
                    start_coords=start_coords,
                    end_coords=None,
                )
                self.working_spline = OpenSpline(
                    geometry, color=cfg.color, line_thickness=self.contour_thickness, transparency=cfg.alpha
                )
            self.graphics_scene.addItem(self.working_spline)
        elif self.working_spline.scene() is not None:
            self.working_spline.geometry.knot_points_x = xs
            self.working_spline.geometry.knot_points_y = ys
            self.working_spline._rebuild_path()

    def _should_close_contour(self, current_pos) -> bool:
        if len(self.points_to_draw) < 2:
            return False

        start_x, start_y = self.points_to_draw[0].get_coords()
        dist = math.hypot(current_pos.x() - start_x, current_pos.y() - start_y)
        return dist < SENSITIVITY

    def _close_current_spline(self):
        """Close the current contour and save it."""
        if self.working_spline is not None:
            downsampled = downsample(
                (
                    [self.working_spline.geometry.full_contour[0].tolist()],
                    [self.working_spline.geometry.full_contour[1].tolist()],
                ),
                self.n_interactive_points,
            )
            key = self.contour_key(self.active_contour_type)
            x_list = [point / self.scaling_factor for point in downsampled[0]]
            y_list = [point / self.scaling_factor for point in downsampled[1]]
            contour_obj = getattr(self.main_window.data[self.frame], key)
            if self.append_contour_mode:
                contour_obj.contours.append([x_list, y_list])
                contour_obj.closed.append(True)
                self._contour_close_committed = True
            else:
                contour_obj.contours = [[x_list, y_list]]
                contour_obj.closed = [True]

        self.stop_contour()

    def _finish_open_spline(self):
        """Finish drawing an open spline on double-click and save it as open (closed=False).

        Qt fires a mousePressEvent just before mouseDoubleClickEvent, so one extra point
        is added at the double-click position. We discard that last point here.
        """
        # Remove the spurious last point added by the preceding mousePressEvent
        if self.points_to_draw:
            last_point = self.points_to_draw.pop()
            if last_point.scene() is not None:
                self.graphics_scene.removeItem(last_point)

        if self.working_spline is None or len(self.points_to_draw) < 2:
            self._interrupt_drawing_mode()
            return

        key = self.contour_key(self.active_contour_type)
        contour_obj = getattr(self.main_window.data[self.frame], key)
        if self.append_contour_mode:
            contour_obj.closed.append(False)
        else:
            contour_obj.closed = [False]

        self.stop_contour()

    def stop_contour(self):
        """
        Stop contour drawing mode, finalize the contour for the current frame, and update the display.

        This method exits contour drawing mode, resets the cursor, and refreshes the image display with the updated contour.
        If a contour was drawn for the current frame, it also updates the longitudinal view with the new contour.
        """
        if self.main_window.image_displayed:
            self.drawing_mode = False
            key = self.contour_key(self.active_contour_type)
            fd = self.main_window.data[self.frame]
            contour_obj = getattr(fd, key)

            if self.working_spline is not None:
                downsampled = downsample(
                    (
                        [self.working_spline.geometry.full_contour[0].tolist()],
                        [self.working_spline.geometry.full_contour[1].tolist()],
                    ),
                    self.n_interactive_points,
                )
                xs_sparse_origin = [x / self.scaling_factor for x in downsampled[0]]
                ys_sparse_origin = [y / self.scaling_factor for y in downsampled[1]]

                if not self.working_spline.geometry.is_closed and xs_sparse_origin:
                    xs_sparse_origin[-1] = self.working_spline.geometry.full_contour[0][-1] / self.scaling_factor
                    ys_sparse_origin[-1] = self.working_spline.geometry.full_contour[1][-1] / self.scaling_factor

                if self.append_contour_mode:
                    if not self._contour_close_committed:
                        contour_obj.contours.append([xs_sparse_origin, ys_sparse_origin])
                    start = self.working_spline.geometry.start_coords
                    end = self.working_spline.geometry.end_coords
                    contour_obj.start_coords.append(
                        (start[0] / self.scaling_factor, start[1] / self.scaling_factor) if start else None
                    )
                    contour_obj.end_coords.append(
                        (end[0] / self.scaling_factor, end[1] / self.scaling_factor) if end else None
                    )
                    self.active_contour_index = len(contour_obj.contours) - 1
                else:
                    contour_obj.contours = [[xs_sparse_origin, ys_sparse_origin]]
                    start = self.working_spline.geometry.start_coords
                    end = self.working_spline.geometry.end_coords
                    if start:
                        contour_obj.start_coords = [(start[0] / self.scaling_factor, start[1] / self.scaling_factor)]
                    if end:
                        contour_obj.end_coords = [(end[0] / self.scaling_factor, end[1] / self.scaling_factor)]
                lst = self._ensure_finalized_list_for_key(key, self.active_contour_index + 1)
                lst[self.active_contour_index] = self.working_spline

            self._interrupt_drawing_mode()

            try:
                self.main_window.longitudinal_view.plot_areas()
            except Exception as e:
                logger.debug(f"Could not update longitudinal view for frame {self.frame}: {e}")

    ################################################################################################
    # later to be refactored into contour manipulation methods (measure and reference point)
    def _draw_measure(self):
        fd = self.main_window.data.get(self.frame)
        if fd is None:
            return
        for index, attr in enumerate(['measurement_1', 'measurement_2']):
            measure = getattr(fd, attr)
            if measure is None or measure.points is None:
                continue
            (x1, y1), (x2, y2) = measure.points
            p1 = QPointF(x1 * self.scaling_factor, y1 * self.scaling_factor)
            p2 = QPointF(x2 * self.scaling_factor, y2 * self.scaling_factor)
            self.graphics_scene.addItem(
                Point(
                    (p1.x(), p1.y()),
                    self.point_thickness,
                    self.point_radius,
                    0,
                    self.main_window.left_half.measure_colors[index],
                )
            )
            self.graphics_scene.addItem(
                Point(
                    (p2.x(), p2.y()),
                    self.point_thickness,
                    self.point_radius,
                    1,
                    self.main_window.left_half.measure_colors[index],
                )
            )
            self.graphics_scene.addLine(
                QLineF(p1, p2), get_qt_pen(self.main_window.left_half.measure_colors[index], self.point_thickness)
            )
            length = measure.length
            if length is None:
                length = round(
                    QLineF(p1, p2).length() * self.main_window.metadata["resolution"] / self.scaling_factor, 2
                )
            length_text = QGraphicsTextItem(f'{length} mm')
            length_text.setPos(p2.x(), p2.y())
            self.graphics_scene.addItem(length_text)
        # Draw any pending first-click-only points
        for index, pending in enumerate(self.pending_measure_points):
            if pending is not None:
                px, py = pending
                self.graphics_scene.addItem(
                    Point(
                        pos=(px * self.scaling_factor, py * self.scaling_factor),
                        line_thickness=self.point_thickness,
                        point_radius=self.point_radius,
                        index=0,
                        color=self.main_window.left_half.measure_colors[index],
                    )
                )

    def add_measure(self, point, index=None, new=True):
        index = index if index is not None else self.measure_index
        orig_x = point.x() / self.scaling_factor
        orig_y = point.y() / self.scaling_factor
        self.graphics_scene.addItem(
            Point(
                pos=(point.x(), point.y()),
                line_thickness=self.point_thickness,
                point_radius=self.point_radius,
                index=index,
                color=self.main_window.left_half.measure_colors[index],
            )
        )
        if self.pending_measure_points[index] is None:
            # First click — store as pending
            self.pending_measure_points[index] = (orig_x, orig_y)
        else:
            # Second click — complete the measure
            p1_orig = self.pending_measure_points[index]
            p1 = QPointF(p1_orig[0] * self.scaling_factor, p1_orig[1] * self.scaling_factor)
            p2 = QPointF(orig_x * self.scaling_factor, orig_y * self.scaling_factor)
            line = QLineF(p1, p2)
            length = round(line.length() * self.main_window.metadata["resolution"] / self.scaling_factor, 2)
            attr = f'measurement_{index + 1}'
            setattr(self.main_window.data[self.frame], attr, Measure(points=(p1_orig, (orig_x, orig_y)), length=length))
            self.pending_measure_points[index] = None
            self.graphics_scene.addLine(
                line, get_qt_pen(self.main_window.left_half.measure_colors[index], self.point_thickness)
            )
            length_text = QGraphicsTextItem(f'{length} mm')
            length_text.setPos(point.x(), point.y())
            self.graphics_scene.addItem(length_text)
            if new:
                self.measure_index = None
                self.main_window.display.setCursor(Qt.CursorShape.ArrowCursor)

    def start_measure(self, index: int):
        if self.drawing_mode:
            self.stop_contour()
        fd = self.main_window.data.get(self.frame)
        if fd:
            setattr(fd, f'measurement_{index + 1}', None)
        self.pending_measure_points[index] = None
        self.main_window.display.setCursor(Qt.CursorShape.CrossCursor)
        self.measure_index = index
        self.display_image(update_contours=True)
        if self.active_segmentation_tool == SegmentationTool.OPEN_SPLINE:
            self.main_window.left_half.open_spline_btn.setChecked(True)
        elif self.active_segmentation_tool == SegmentationTool.BRUSH:
            self.main_window.left_half.brush_btn.setChecked(True)
        else:
            self.main_window.left_half.closed_spline_btn.setChecked(True)

    def stop_measure(self, index):
        if self.main_window.image_displayed:
            self.pending_measure_points[index] = None
            self.measure_index = None
            self.main_window.display.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)

    def _draw_reference(self):
        fd = self.main_window.data.get(self.frame)
        if fd is None or fd.reference is None:
            return
        scaled_x = fd.reference[0] * self.scaling_factor
        scaled_y = fd.reference[1] * self.scaling_factor
        self.graphics_scene.addItem(
            Point(
                pos=(scaled_x, scaled_y),
                line_thickness=self.point_thickness,
                point_radius=self.point_radius,
                index=0,
                color=self.main_window.left_half.reference_color,
            )
        )
        text = QGraphicsTextItem('Reference')
        text.setPos(scaled_x, scaled_y)
        self.graphics_scene.addItem(text)

    def start_reference(self):
        self.reference_mode = True
        self.main_window.display.setCursor(Qt.CursorShape.CrossCursor)
        fd = self.main_window.data.get(self.frame)
        if fd:
            fd.reference = None
        self.display_image(update_contours=True)
        if self.active_segmentation_tool == SegmentationTool.OPEN_SPLINE:
            self.main_window.left_half.open_spline_btn.setChecked(True)
        elif self.active_segmentation_tool == SegmentationTool.BRUSH:
            self.main_window.left_half.brush_btn.setChecked(True)
        else:
            self.main_window.left_half.closed_spline_btn.setChecked(True)

    def _handle_reference_placement(self, pos):
        """Saves the reference point and exits reference mode."""
        self.main_window.data[self.frame].reference = (
            pos.x() / self.scaling_factor,
            pos.y() / self.scaling_factor,
        )
        self.reference_mode = False
        self.main_window.display.setCursor(Qt.CursorShape.ArrowCursor)
        self.display_image(update_contours=True)

    ################################################################################################

    def start_angle(self):
        """Initializes the angle measurement mode."""
        if self.drawing_mode:
            self.stop_contour()
        self.angle_mode = True
        self.angle_clicks = []
        self.main_window.display.setCursor(Qt.CursorShape.CrossCursor)
        self.main_window.data[self.frame].wire = None
        self.display_image(update_contours=True)
        if self.active_segmentation_tool == SegmentationTool.OPEN_SPLINE:
            self.main_window.left_half.open_spline_btn.setChecked(True)
        elif self.active_segmentation_tool == SegmentationTool.BRUSH:
            self.main_window.left_half.brush_btn.setChecked(True)
        else:
            self.main_window.left_half.closed_spline_btn.setChecked(True)

    def _handle_angle_placement(self, pos: QPointF):
        """Handles the two clicks required to define an angle."""
        self.angle_clicks.append(pos)
        original_point = (pos.x() / self.scaling_factor, pos.y() / self.scaling_factor)
        fd = self.main_window.data[self.frame]

        if len(self.angle_clicks) == 1:
            fd.wire = (original_point,)
            self.display_image(update_contours=True)
        elif len(self.angle_clicks) == 2:
            fd.wire = (fd.wire[0], original_point)
            self.angle_mode = False
            self.main_window.display.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)

    def _draw_angles(self):
        """Draws lines from center through the stored wire/angle points, stopping at image edges."""
        fd = self.main_window.data.get(self.frame)
        if fd is None or not fd.wire:
            return

        half_size = self.image_size / 2
        center = QPointF(half_size, half_size)
        pen = get_qt_pen(self.color_angle, self.point_thickness)

        for pt_coords in fd.wire:
            target_pt = QPointF(pt_coords[0] * self.scaling_factor, pt_coords[1] * self.scaling_factor)
            dx = target_pt.x() - center.x()
            dy = target_pt.y() - center.y()
            if dx == 0 and dy == 0:
                continue
            t_x = abs(half_size / dx) if dx != 0 else float('inf')
            t_y = abs(half_size / dy) if dy != 0 else float('inf')
            t = min(t_x, t_y)
            edge_pt = QPointF(center.x() + t * dx, center.y() + t * dy)
            self.graphics_scene.addLine(QLineF(center, edge_pt), pen)
            self.graphics_scene.addItem(
                Point((target_pt.x(), target_pt.y()), self.point_thickness, self.point_radius, 0, self.color_angle)
            )

    def _draw_open_spline_edge_lines(self):
        """Draw lines from open spline start/end points to image edge, in direction away from contour centroid."""
        fd = self.main_window.data.get(self.frame)
        if fd is None:
            return

        half_size = self.image_size / 2
        image_center = QPointF(half_size, half_size)

        open_spline_types = {ct for ct in ContourType if SegmentationTool.OPEN_SPLINE in ALLOWED_TOOLS.get(ct, set())}
        for ct in open_spline_types:
            key = ct.value
            contour_obj = getattr(fd, key, None)
            if contour_obj is None or not contour_obj.contours:
                continue

            cfg = self.contour_configs.get(ct)
            color = cfg.color if cfg else self.color_contour
            pen = get_qt_pen(color, self.point_thickness)

            centroid = image_center
            if fd.centroid:
                centroid = QPointF(fd.centroid[0] * self.scaling_factor, fd.centroid[1] * self.scaling_factor)

            for i in range(len(contour_obj.contours)):
                is_closed = contour_obj.closed[i] if len(contour_obj.closed) > i else True
                if is_closed:
                    continue
                raw_start = contour_obj.start_coords[i] if len(contour_obj.start_coords) > i else None
                raw_end = contour_obj.end_coords[i] if len(contour_obj.end_coords) > i else None
                if raw_start is None and raw_end is None:
                    continue

                for raw_coord in [raw_start, raw_end]:
                    if raw_coord is None:
                        continue
                    endpoint = QPointF(raw_coord[0] * self.scaling_factor, raw_coord[1] * self.scaling_factor)
                    dx = endpoint.x() - centroid.x()
                    dy = endpoint.y() - centroid.y()
                    if dx == 0 and dy == 0:
                        continue
                    t_vals = []
                    if dx > 0:
                        t_vals.append((self.image_size - endpoint.x()) / dx)
                    elif dx < 0:
                        t_vals.append(-endpoint.x() / dx)
                    if dy > 0:
                        t_vals.append((self.image_size - endpoint.y()) / dy)
                    elif dy < 0:
                        t_vals.append(-endpoint.y() / dy)
                    if not t_vals:
                        continue
                    t = min(t_vals)
                    edge_pt = QPointF(endpoint.x() + t * dx, endpoint.y() + t * dy)
                    self.graphics_scene.addLine(QLineF(endpoint, edge_pt), pen)

    ######################
    # Mouse click events #
    ######################
    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_y = event.position().y()
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._panning = True
                self._pan_last_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            if self.drawing_mode:
                self.add_contour(pos, self.active_segmentation_tool)
            elif self.measure_index is not None:
                self.add_measure(pos)
            elif self.reference_mode:
                self._handle_reference_placement(pos)
            elif self.angle_mode:
                self._handle_angle_placement(pos)
            else:
                # First, try to switch active contour if user clicked near another one.
                # If a switch occurred the scene was redrawn; skip item interaction for this click.
                if not self._attempt_contour_switch(pos):
                    self._handle_item_interaction(pos, event.pos())

        elif event.button() == Qt.MouseButton.RightButton:
            # Check if we clicked on a knot point to delete it
            self._attempt_contour_switch(pos)

            items = self.items(event.pos())
            point_item = next((i for i in items if isinstance(i, Point)), None)

            if point_item and point_item in self.points_to_draw:
                self._delete_point(point_item)
                return  # Stop here so we don't trigger windowing/leveling drag

            # Original windowing/leveling logic
            self.mouse_x = event.position().x()
            self.mouse_y = event.position().y()
        super().mousePressEvent(event)

    def _attempt_contour_switch(self, pos) -> bool:
        """Switches active contour type or index if clicking near a different contour's knotpoint.
        Returns True if a switch occurred, False otherwise."""
        if self.drawing_mode:
            return False
        min_dist = float('inf')
        nearest_ct = None
        nearest_index = 0

        closed_contour_types = {
            ct for ct in ContourType if SegmentationTool.CLOSED_SPLINE in ALLOWED_TOOLS.get(ct, set())
        }
        for ct in closed_contour_types:
            fd = self.main_window.data.get(self.frame)
            if fd is None:
                continue
            key = self.contour_key(ct)
            contour_obj = getattr(fd, key, None)
            if contour_obj is None or not contour_obj.contours:
                continue
            for i, contour in enumerate(contour_obj.contours):
                if not contour or not contour[0]:
                    continue
                for x_orig, y_orig in zip(contour[0], contour[1] if len(contour) > 1 else []):
                    dist = math.hypot(
                        pos.x() - (x_orig * self.scaling_factor), pos.y() - (y_orig * self.scaling_factor)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ct = ct
                        nearest_index = i

        if nearest_ct and min_dist < SENSITIVITY:
            if nearest_ct != self.active_contour_type or nearest_index != self.active_contour_index:
                self.active_contour_type = nearest_ct
                self.active_contour_index = nearest_index
                self.working_spline = None
                self.points_to_draw = []
                self.active_point = None
                self.active_point_index = None
                self.display_image(update_contours=True)
                return True
        return False

    def _handle_item_interaction(self, scene_pos, view_pos):
        """Handles clicking existing knotpoints or adding new ones to a spline."""
        items = self.items(view_pos)
        point_item = next((i for i in items if isinstance(i, Point)), None)
        spline_item = next((i for i in items if isinstance(i, Spline)), None)

        current_spline = self.get_current_spline()

        if point_item and point_item in self.points_to_draw:
            self._select_existing_point(point_item)
        elif spline_item and current_spline:
            self._add_new_point_to_spline(scene_pos)

    def _select_existing_point(self, point_item: Point):
        # self.main_window.display.setCursor(Qt.CursorShape.BlankCursor)  # remove cursor for precise contour changes
        # https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
        try:
            self.active_point_index = self.points_to_draw.index(point_item)
        except ValueError:
            # Should not happen if point_item belongs to the active spline
            return
        self.active_point = point_item
        point_item.update_color()
        self.working_spline = self.get_current_spline()

    def _add_new_point_to_spline(self, pos):
        if not self.working_spline:
            return

        path_index = self.working_spline.on_path(pos)
        if path_index is None:  # Safety check: only add if we actually clicked the path
            return

        self.main_window.display.setCursor(Qt.CursorShape.BlankCursor)
        self.active_point_index = self.working_spline.update(pos, -1, path_index)

        cfg = self.contour_configs.get(self.active_contour_type)
        self.active_point = Point(
            (pos.x(), pos.y()),
            self.point_thickness,
            self.point_radius,
            cfg.color if cfg else self.color_contour,
            cfg.alpha if cfg else self.alpha_contour,
        )
        self.graphics_scene.addItem(self.active_point)
        self.active_point.update_color()

    def _delete_point(self, point_item: Point):
        """Removes a knot point from the scene and the data model."""
        try:
            idx = self.points_to_draw.index(point_item)
        except ValueError:
            return

        self.graphics_scene.removeItem(point_item)
        self.points_to_draw.pop(idx)

        key = self.contour_key(self.active_contour_type)
        fd = self.main_window.data.get(self.frame)
        if fd:
            contour_obj = getattr(fd, key, None)
            ci = self.active_contour_index
            if contour_obj and contour_obj.contours and len(contour_obj.contours) > ci and contour_obj.contours[ci]:
                contour_obj.contours[ci][0].pop(idx)
                if len(contour_obj.contours[ci]) > 1:
                    contour_obj.contours[ci][1].pop(idx)

                if point_item.color == self.start_color and ci < len(contour_obj.start_coords):
                    contour_obj.start_coords[ci] = None
                if point_item.color == self.end_color and ci < len(contour_obj.end_coords):
                    contour_obj.end_coords[ci] = None

        self.display_image(update_contours=True)

    def mouseMoveEvent(self, event):
        if self._panning and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.pos() - self._pan_last_pos
            self._pan_last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None:
                item = self.active_point
                new_scene_pos = self.mapToScene(event.pos())

                item.update_pos(new_scene_pos)

                self.working_spline.update(new_scene_pos, self.active_point_index)
            elif self.active_point_index is None and not self.drawing_mode:
                self.setMouseTracking(True)
                delta_y = self.mouse_y - event.position().y()
                self.mouse_y = event.position().y()
                zoom_factor = 1.0 + delta_y * self.zoom_sensitivity
                if zoom_factor > 0:
                    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                    self.scale(zoom_factor, zoom_factor)
                    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.position().x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.position().y() - self.mouse_y) * self.windowing_sensitivity
            self.display_image(update_image=True)
            self.setMouseTracking(False)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._pan_last_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None and self.working_spline:
                self.main_window.display.setCursor(Qt.CursorShape.ArrowCursor)
                self.active_point.reset_color()

                geom = self.working_spline.geometry
                key = self.contour_key(self.active_contour_type)
                new_pos = (geom.knot_points_x[self.active_point_index], geom.knot_points_y[self.active_point_index])

                if self.active_point.color == self.start_color:
                    geom.start_coords = new_pos
                elif self.active_point.color == self.end_color:
                    geom.end_coords = new_pos

                x_list = [p / self.scaling_factor for p in geom.knot_points_x]
                y_list = [p / self.scaling_factor for p in geom.knot_points_y]
                contour_obj = getattr(self.main_window.data[self.frame], key)
                ci = self.active_contour_index
                if ci < len(contour_obj.contours):
                    contour_obj.contours[ci] = [x_list, y_list]
                # Only persist start/end coords when the dragged point is the designated start or end point.
                # geom.start_coords always has a value (defaults to first knot), so we must not use it
                # as a condition — that would overwrite start_coords on every ordinary point drag.
                if self.active_point.color == self.start_color and geom.start_coords:
                    start_val = (geom.start_coords[0] / self.scaling_factor, geom.start_coords[1] / self.scaling_factor)
                    if ci < len(contour_obj.start_coords):
                        contour_obj.start_coords[ci] = start_val
                    else:
                        contour_obj.start_coords.append(start_val)
                if self.active_point.color == self.end_color and geom.end_coords:
                    end_val = (geom.end_coords[0] / self.scaling_factor, geom.end_coords[1] / self.scaling_factor)
                    if ci < len(contour_obj.end_coords):
                        contour_obj.end_coords[ci] = end_val
                    else:
                        contour_obj.end_coords.append(end_val)

                self.display_image(update_contours=True)
                self.active_point_index = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.drawing_mode:
                pos = self.mapToScene(event.pos())
                current_spline = self.get_current_spline()

                if current_spline and current_spline.on_path(pos) is not None:
                    scaled_coords = (pos.x(), pos.y())
                    orig_coords = (pos.x() / self.scaling_factor, pos.y() / self.scaling_factor)

                    key = self.contour_key(self.active_contour_type)
                    contour_obj = getattr(self.main_window.data[self.frame], key)

                    ci = self.active_contour_index
                    if self.active_end_coords_flag:
                        current_spline.geometry.end_coords = scaled_coords
                        if ci < len(contour_obj.end_coords):
                            contour_obj.end_coords[ci] = orig_coords
                        else:
                            contour_obj.end_coords.append(orig_coords)
                    else:
                        current_spline.geometry.start_coords = scaled_coords
                        if ci < len(contour_obj.start_coords):
                            contour_obj.start_coords[ci] = orig_coords
                        else:
                            contour_obj.start_coords.append(orig_coords)

                    self.active_end_coords_flag = not self.active_end_coords_flag

                    current_spline.geometry._ensure_start_end_coords()
                    current_spline._rebuild_path()
                    self.update_display()
            else:
                if self.active_segmentation_tool == SegmentationTool.OPEN_SPLINE:
                    pos = self.mapToScene(event.pos())
                    if self.working_spline is not None:
                        self.working_spline.geometry.end_coords = (pos.x(), pos.y())
                    if self.append_contour_mode:
                        key = self.contour_key(self.active_contour_type)
                        contour_obj = getattr(self.main_window.data[self.frame], key)
                        while len(contour_obj.end_coords) < len(contour_obj.contours):
                            contour_obj.end_coords.append(None)
                    self._finish_open_spline()
                    return  # prevent super() re-delivering event after drawing_mode is cleared
        super().mouseDoubleClickEvent(event)
