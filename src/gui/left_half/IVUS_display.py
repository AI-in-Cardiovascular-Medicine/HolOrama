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

from gui.utils.geometry import Point, Spline, SplineGeometry, get_qt_pen
from gui.utils.metrics import MetricsMixin
from gui.right_half.longitudinal_view import Marker
from segmentation.segment import downsample


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
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.EEM: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.CALCIUM: {
        SegmentationTool.OPEN_SPLINE,
        SegmentationTool.CLOSED_SPLINE,
        SegmentationTool.BRUSH,
    },
    ContourType.BRANCH: {
        SegmentationTool.OPEN_SPLINE,
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
    ContourType.WIRE: {SegmentationTool.ANGLE}
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


class IVUSDisplay(QGraphicsView, MetricsMixin):
    """
    Displays images and contours and allows the user to add and manipulate contours.
    """

    def __init__(self, main_window):
        super(IVUSDisplay, self).__init__()
        self.main_window = main_window
        config = main_window.config

        self.n_interactive_points: int = config.display.n_interactive_points
        self.n_points_contour: int = config.display.n_points_contour
        self.image_size: int = config.display.image_size  # image display in pixel (square)
        self.windowing_sensitivity: float = config.display.windowing_sensitivity
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

        self.frame: int = 0
        self.points_to_draw: list[Point] = []
        self.start_coords: Tuple[float, float] | None = None
        self.end_coords: Tuple[float, float] | None = None
        self.working_spline: Spline = None
        self.finalized_splines: dict[ContourType, Spline] = {}

        # flags and states
        self.active_contour_type: ContourType = ContourType.LUMEN
        self.active_segmentation_tool: SegmentationTool = SegmentationTool.CLOSED_SPLINE
        self.drawing_mode: bool = False
        self.active_point: Point = None
        self.active_end_coords_flag = True  # Flag to switch, which point double click sets

        #####################################################################################################
        # legacy to be refactored
        self.active_point_index: int = None # wtf is this legacy crap
        self.measure_index: int = None  # wtf is this legacy crap
        self.measure_colors = self.main_window.measure_colors
        self.reference_mode: bool = False
        self.angle_mode: bool = False
        self.angle_clicks: list[QPointF] = []
        #####################################################################################################

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # initialize data from main_window data
    def set_data(self, lumen, images):
        """
        Initialize display data. 'lumen' is the legacy argument (first contour),
        but we create entries for all ContourType members in main_window.data
        and prepare self.full_contours dict with per-frame placeholders.
        """
        self.images = images
        num_frames = images.shape[0]
        self.image_width = images.shape[1]
        self.scaling_factor = self.image_size / images.shape[1]  # image_size in config

        # for legacy reasons we still expect  'lumen argument'
        self.main_window.data[ContourType.LUMEN.value] = lumen

        if not hasattr(self.main_window, "data") or self.main_window.data is None:
            self.main_window.data = {}

        for ct in [
            ContourType.LUMEN, 
            ContourType.EEM, 
            ContourType.CALCIUM, 
            ContourType.BRANCH, 
            ContourType.LIPID,
            ContourType.MACROPHAGE,
            ]:
            self._init_main_window_data(num_frames, ct.value)

        self.finalized_splines = {ct.value: None for ct in ContourType}

        self._draw_contours_frame()

        contours = self.get_full_contour_list(self.active_contour_type)
        self.main_window.longitudinal_view.set_data(self.images, contours)
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def _init_main_window_data(self, num_frames: int, key: str = None):
        """
        Ensure every contour type has a [ [x per frame], [y per frame] ] structure
        initialize with number of frames
        """
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
        if f"{key}_start" not in self.main_window.data:
            self.main_window.data[f"{key}_start"] = [
            None
            ] * num_frames  # initialize start/end point storage for all contour types
        if f"{key}_end" not in self.main_window.data:
            self.main_window.data[f"{key}_end"] = [None] * num_frames

    def _draw_contours_frame(self):
        # other contours
        closed_contour_types = {ct for ct in ContourType if SegmentationTool.CLOSED_SPLINE in ALLOWED_TOOLS.get(ct, set())}
        for ct in closed_contour_types:
            data = self._get_contour_data(ct, self.frame)
            if data:
                self._draw_contour_frame(data, contour_type=ct, set_current=(ct == self.active_contour_type))

    def _draw_contour_frame(self, contour_data: Tuple[List[float], List[float]], contour_type: ContourType = None, set_current: bool = False):
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
        raw_start = self.main_window.data.get(f"{key}_start", {})[self.frame]
        raw_end = self.main_window.data.get(f"{key}_end", {})[self.frame]

        start_coords = (raw_start[0] * self.scaling_factor, raw_start[1] * self.scaling_factor) if raw_start else None
        end_coords = (raw_end[0] * self.scaling_factor, raw_end[1] * self.scaling_factor) if raw_end else None

        if start_coords is None and lumen_x:
            start_coords = (lumen_x[0], lumen_y[0])

        geometry = SplineGeometry(
            lumen_x, 
            lumen_y, 
            self.n_points_contour, 
            start_coords, 
            end_coords,
        )
        geometry._ensure_start_end_coords()
        geometry.interpolate()

        if geometry.full_contour[0] is not None:
            knot_points = []
            for i in range(len(geometry.knot_points_x) - 1):
                curr_x = geometry.knot_points_x[i]
                curr_y = geometry.knot_points_y[i]
                knot_color = color
                brush = False
                if start_coords and math.hypot(curr_x - start_coords[0], curr_y - start_coords[1]) < SENSITIVITY:
                    knot_color=self.start_color
                    brush = True
                if end_coords and math.hypot(curr_x - end_coords[0], curr_y - end_coords[1]) < SENSITIVITY:
                    knot_color=self.end_color
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
            spline = Spline(geometry, color, thickness, alpha)
            self.graphics_scene.addItem(spline)

            self.finalized_splines[self.contour_key(ct)] = spline

            if set_current:
                self.working_spline = spline
                self.points_to_draw = knot_points
        else:
            logger.warning(f'Spline for frame {self.frame + 1} could not be interpolated for {ct.value}')

    def set_frame(self, value):
        self.frame = value
        self._interrupt_drawing_mode()
        if self.measure_index is not None:
            self.stop_measure(self.measure_index)

        self.finalized_splines = {ct.value: None for ct in ContourType}
        self._draw_contours_frame()

        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def get_full_contour_list(self, contour_type: ContourType = None, unscaled: bool = False) -> List[Tuple[List[float], List[float]]] | None:
        """
        Return the list-of-frame full_contours for a contour type.
        Expects self.main_window.data[key] to be [ [x_frames], [y_frames] ]
        """
        key = self.contour_key(contour_type)
        
        if key in self.main_window.data:
            full_contours = []
            # Correctly access the x and y lists from the data structure
            x_frames_list = self.main_window.data[key][0]
            y_frames_list = self.main_window.data[key][1]

            for x_coords, y_coords in zip(x_frames_list, y_frames_list):
                # Initialize geometry (Ensure parameters match your SplineGeometry signature)
                if len(x_coords) > 1:
                    spline_geo = SplineGeometry(
                        knot_points_x=x_coords,
                        knot_points_y=y_coords,
                        n_interpolated_points=self.n_interactive_points,
                        start_coords=None,
                        end_coords=None
                    )

                    # Handle scaling if necessary
                    if unscaled and hasattr(self, 'scaling_factor'):
                        spline_geo = spline_geo.scale(self.scaling_factor)

                    interpolated_coords = spline_geo.interpolate()
                    full_contours.append(interpolated_coords)
                else:
                    continue

            return full_contours
        
        return None

    def contour_key(self, contour_type: ContourType = None) -> str:
        """Return the string key for the given contour type (defaults to active)."""
        return (contour_type or self.active_contour_type).value

    def get_current_spline(self):
        """Returns the currently active spline based on self.active_contour_type."""
        key = self.contour_key(self.active_contour_type)
        if key in self.finalized_splines:
            return self.finalized_splines[key]
        return None

    def _get_contour_data(self, contour_type: ContourType = None, frame: int | None = None) -> list[list[Any], list[Any]] | Tuple[List[float], List[float]]:
        """Return main_window.data[...] for the given/the active contour type (or None)."""
        key = self.contour_key(contour_type)
        if frame is None:
            return self.main_window.data.get(key, None)
        else:
            data = self.main_window.data.get(key, None)
            return (data[0][self.frame], data[1][self.frame])

    def set_active_contour_type(self, contour_type: ContourType):
        """Set active contour type and refresh transient state for editing that contour."""
        if contour_type == self.active_contour_type:
            return
        self.active_contour_type = contour_type

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
                if self.finalized_splines[lumen_key] is not None:
                    lumen_contour = self.finalized_splines[lumen_key].get_unscaled_contour(self.scaling_factor)

                eem_contour = None
                if self.finalized_splines[eem_key] is not None:
                    eem_contour = self.finalized_splines[eem_key].get_unscaled_contour(self.scaling_factor)
                
                self._draw_contours_frame()
                self._draw_measure()
                self._draw_reference()
                self._draw_angles()

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

        self.drawing_mode = False
        self.active_point = None
        self.active_contour_type = ContourType.LUMEN

        self.main_window.tmp_contours = {}

    def _interrupt_drawing_mode(self):
        """Handles safe exit of drawing mode, returning to initial state."""
        self.cleanup_temporary_drawing()
        self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
        self.display_image(update_contours=True)

    def start_contour(
        self, contour_type: ContourType = ContourType.LUMEN, segmentation_tool: SegmentationTool = SegmentationTool.CLOSED_SPLINE
    ):
        """
        Start drawing a new contour of the specified type and with the specified tool.

        Sets the active contour type, clears previous data for this frame,
        and switches to contour drawing mode (leaves temporary data in main_window.tmp_contours).
        """
        if contour_type is not None:
            self.set_active_contour_type(contour_type)

        self.drawing_mode = True        
        # save the current state of the contour to the tmp storage
        key = self.contour_key(contour_type)
        current_spline = self.get_current_spline()
        if current_spline is not None:
            current_full_contour = current_spline.geometry.full_contour
            self.main_window.tmp_contours[key] = current_full_contour

        self.active_segmentation_tool = segmentation_tool if segmentation_tool else self.active_segmentation_tool

        self.measure_index = None
        self.working_spline = None
        self.points_to_draw = []
        self.active_point = None
        self.active_end_coords_flag = True
        self.main_window.setCursor(Qt.CursorShape.CrossCursor)

        self.main_window.data[key][0][self.frame] = []
        self.main_window.data[key][1][self.frame] = []
        self.main_window.data[f"{key}_start"][self.frame] = None
        self.main_window.data[f"{key}_end"][self.frame] = None
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
            geometry = SplineGeometry(
                knot_points_x=xs,
                knot_points_y=ys,
                n_interpolated_points=self.n_points_contour,
                start_coords=start_coords,
                end_coords=None,
                is_closed=is_closed,
            )
            self.working_spline = Spline(geometry, cfg.color, self.contour_thickness, cfg.alpha)
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
            self.main_window.data[key][0][self.frame] = [point / self.scaling_factor for point in downsampled[0]]
            self.main_window.data[key][1][self.frame] = [point / self.scaling_factor for point in downsampled[1]]

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

            if self.working_spline is not None:
                downsampled = downsample(
                    (
                        [self.working_spline.geometry.full_contour[0].tolist()],
                        [self.working_spline.geometry.full_contour[1].tolist()],
                    ),
                    self.n_interactive_points,
                )
                xs_sparse_display = downsampled[0]
                ys_sparse_display = downsampled[1]

                xs_sparse_origin = [ x / self.scaling_factor for x in xs_sparse_display]
                ys_sparse_origin = [ y / self.scaling_factor for y in ys_sparse_display]
                
                self.main_window.data[key][0][self.frame] = xs_sparse_origin
                self.main_window.data[key][1][self.frame] = ys_sparse_origin

                start = self.working_spline.geometry.start_coords
                end = self.working_spline.geometry.end_coords
                if start:
                    self.main_window.data[f"{key}_start"][self.frame] = (
                        start[0] / self.scaling_factor,
                        start[1] / self.scaling_factor,
                    )
                if end:
                    self.main_window.data[f"{key}_end"][self.frame] = (
                        end[0] / self.scaling_factor,
                        end[1] / self.scaling_factor,
                    )
                self.finalized_splines[key] = self.working_spline

            self._interrupt_drawing_mode()

            contour_for_frame: Tuple[np.array, np.array] = (
                np.array(self.main_window.data[key][0][self.frame]),
                np.array(self.main_window.data[key][1][self.frame]),
            )
            try:
                self.main_window.longitudinal_view.lview_contour(self.frame, contour_for_frame, update=True)
            except Exception as e:
                logger.debug(f"Could not update longitudinal view for frame {self.frame}: {e}")

    ################################################################################################
    # later to be refactored into countour manipulation methods (measure and reference point)
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
        if self.drawing_mode:
            self.stop_contour()
        self.main_window.data['measures'][self.frame][index] = None
        self.main_window.setCursor(Qt.CursorShape.CrossCursor)
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
        self.main_window.setCursor(Qt.CursorShape.CrossCursor)
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
    ################################################################################################

    def start_angle(self):
        """Initializes the angle measurement mode."""
        if self.drawing_mode:
            self.stop_contour()

        if 'angles' not in self.main_window.data:
            num_frames = self.main_window.metadata.get('num_frames', 0)
            self.main_window.data['angles'] = [[None, None] for _ in range(num_frames)]
        
        self.angle_mode = True
        self.angle_clicks = []
        self.main_window.setCursor(Qt.CursorShape.CrossCursor)
        # Ensure the data structure exists for this frame
        self.main_window.data['angles'][self.frame] = [None, None]
        
        self.display_image(update_contours=True)

    def _handle_angle_placement(self, pos: QPointF):
        """Handles the two clicks required to define an angle."""
        self.angle_clicks.append(pos)
        
        # Store original coordinates (unscaled)
        original_point = [pos.x() / self.scaling_factor, pos.y() / self.scaling_factor]
        
        if len(self.angle_clicks) == 1:
            # Save first point and refresh to show feedback
            self.main_window.data['angles'][self.frame] = [original_point]
            self.display_image(update_contours=True)
        
        elif len(self.angle_clicks) == 2:
            # Save second point and exit mode
            self.main_window.data['angles'][self.frame].append(original_point)
            self.angle_mode = False
            self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)

    def _draw_angles(self):
        """Draws lines from center through the stored angle points, stopping at image edges."""
        try:
            angle_data = self.main_window.data['angles'][self.frame]
        except (IndexError, TypeError, KeyError):
            return

        if not angle_data or all(pt is None for pt in angle_data):
            return

        # Center point of the square image
        half_size = self.image_size / 2
        center = QPointF(half_size, half_size)
        
        pen = get_qt_pen(self.color_angle, self.point_thickness)

        for pt_coords in angle_data:
            target_pt = QPointF(pt_coords[0] * self.scaling_factor, pt_coords[1] * self.scaling_factor)
            
            # 2. Determine the direction vector from center to click
            dx = target_pt.x() - center.x()
            dy = target_pt.y() - center.y()

            # Avoid division by zero if clicking exactly on the center
            if dx == 0 and dy == 0:
                continue

            t_x = abs(half_size / dx) if dx != 0 else float('inf')
            t_y = abs(half_size / dy) if dy != 0 else float('inf')
            t = min(t_x, t_y)

            edge_pt = QPointF(center.x() + t * dx, center.y() + t * dy)
            line = QLineF(center, edge_pt)
            
            self.graphics_scene.addLine(line, pen)
            
            point_marker = Point(
                (target_pt.x(), target_pt.y()), 
                self.point_thickness, 
                self.point_radius,
                0, # Assuming this is an index or type ID for your Point class
                self.color_angle,
            )
            self.graphics_scene.addItem(point_marker)

    ######################
    # Mouse click events #
    ######################
    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing_mode:
                self.add_contour(pos)
            elif self.measure_index is not None:
                self.add_measure(pos)
            elif self.reference_mode:
                self._handle_reference_placement(pos)
            elif self.angle_mode:
                self._handle_angle_placement(pos)
            else:
                # First, try to switch active contour if user clicked near another one
                self._attempt_contour_switch(pos)
                # Then, handle interaction with points or the spline path
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

    def _attempt_contour_switch(self, pos):
        """Switches active contour type if clicking near a different contour's knotpoint."""
        min_dist = float('inf')
        nearest_ct = None

        closed_contour_types = {ct for ct in ContourType if SegmentationTool.CLOSED_SPLINE in ALLOWED_TOOLS.get(ct, set())}
        for ct in closed_contour_types:
            contour_data = self._get_contour_data(ct)
            if not contour_data or not contour_data[0][self.frame]:
                continue

            xs, ys = contour_data[0][self.frame], contour_data[1][self.frame]
            for x_orig, y_orig in zip(xs, ys):
                dist = math.hypot(pos.x() - (x_orig * self.scaling_factor), pos.y() - (y_orig * self.scaling_factor))
                if dist < min_dist:
                    min_dist = dist
                    nearest_ct = ct

        if nearest_ct and min_dist < SENSITIVITY and nearest_ct != self.active_contour_type:
            self.active_contour_type = nearest_ct
            self.display_image(update_contours=True)

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

    def _select_existing_point(self, point_item):
        # self.main_window.setCursor(Qt.CursorShape.BlankCursor)  # remove cursor for precise contour changes
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

        self.main_window.setCursor(Qt.CursorShape.BlankCursor)
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

    def _delete_point(self, point_item):
        """Removes a knot point from the scene and the data model."""
        try:
            idx = self.points_to_draw.index(point_item)
        except ValueError:
            return

        self.graphics_scene.removeItem(point_item)
        self.points_to_draw.pop(idx)

        key = self.contour_key(self.active_contour_type)
        if key in self.main_window.data:
            self.main_window.data[key][0][self.frame].pop(idx)
            self.main_window.data[key][1][self.frame].pop(idx)

        self.display_image(update_contours=True)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None:
                item = self.active_point
                new_scene_pos = self.mapToScene(event.pos())

                item.update_pos(new_scene_pos)

                self.working_spline.update(new_scene_pos, self.active_point_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.position().x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.position().y() - self.mouse_y) * self.windowing_sensitivity
            self.display_image(update_image=True)
            self.setMouseTracking(False)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None and self.working_spline:
                self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
                self.active_point.reset_color()

                geom = self.working_spline.geometry
                key = self.contour_key(self.active_contour_type)
                new_pos = (geom.knot_points_x[self.active_point_index],
                           geom.knot_points_y[self.active_point_index])
                
                if self.active_point.color == self.start_color:
                    geom.start_coords = new_pos
                elif self.active_point.color == self.end_color:
                    geom.end_coords = new_pos

                self.main_window.data[key][0][self.frame] = [p / self.scaling_factor for p in geom.knot_points_x]
                self.main_window.data[key][1][self.frame] = [p / self.scaling_factor for p in geom.knot_points_y]
                
                if geom.start_coords:
                    self.main_window.data[f"{key}_start"][self.frame] = (
                        geom.start_coords[0]/self.scaling_factor, 
                        geom.start_coords[1]/self.scaling_factor,
                        )
                if geom.end_coords:
                    self.main_window.data[f"{key}_end"][self.frame] = (
                        geom.end_coords[0]/self.scaling_factor, 
                        geom.end_coords[1]/self.scaling_factor,
                        )

                self.display_image(update_contours=True)
                self.active_point_index = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            print(self.main_window.data)
            if not self.drawing_mode:
                pos = self.mapToScene(event.pos())
                current_spline = self.get_current_spline()
                
                if current_spline and current_spline.on_path(pos) is not None:
                    scaled_coords = (pos.x(), pos.y())
                    orig_coords = (pos.x() / self.scaling_factor, pos.y() / self.scaling_factor)
                    
                    key = self.contour_key(self.active_contour_type)
                    
                    if self.active_end_coords_flag:
                        current_spline.geometry.end_coords = scaled_coords
                        self.main_window.data[f"{key}_end"][self.frame] = orig_coords
                    else:
                        current_spline.geometry.start_coords = scaled_coords
                        self.main_window.data[f"{key}_start"][self.frame] = orig_coords
                    
                    self.active_end_coords_flag = not self.active_end_coords_flag
                    
                    current_spline.geometry._ensure_start_end_coords()
                    current_spline._rebuild_path()
                    self.update_display()
        super().mouseDoubleClickEvent(event)
