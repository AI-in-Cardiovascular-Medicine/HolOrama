import bisect

import numpy as np
from loguru import logger
from scipy.interpolate import splprep, splev
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen, QPainterPath, QColor
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any

@dataclass
class SplineGeometry:
    """Pure geometric representation of a spline, no QT dependencies"""

    knot_points_x: List[float]
    knot_points_y: List[float]
    n_interpolated_points: int
    start_coords: Tuple[float, float] | None
    end_coords: Tuple[float, float] | None
    full_contour: Tuple[List[float], List[float]] = field(
        default_factory=lambda: ([], [])
    )
    is_closed: bool = True
    dashed: bool = False

    def __post_init__(self):
        """Validate and ensure the spline is properly set up."""
        if len(self.knot_points_x) != len(self.knot_points_y):
            raise ValueError("X and Y knot points must have same length")
        if self.is_closed and len(self.knot_points_x) > 0:
            self._ensure_closed()
            self.interpolate()
    
    def __str__(self):
        return (f"SplineGeometry(knot_points_x={self.knot_points_x}, "
                f"knot_points_y={self.knot_points_y}, "
                f"n_interpolated_points={self.n_interpolated_points}, "
                f"start_coords={self.start_coords}, "
                f"end_coords={self.end_coords}, "
                f"is_closed={self.is_closed})")

    def _ensure_start_end_coords(self):
        """Sync start/end coords with existing knots. Enforces pins exactly."""
        if not self.knot_points_x:
            return

        if self.start_coords:
            idx = self._get_closest_knot_index(self.start_coords[0], self.start_coords[1])
            self.knot_points_x[idx] = self.start_coords[0]
            self.knot_points_y[idx] = self.start_coords[1]
            # Ensure closing point matches if we moved the head
            if self.is_closed and idx == 0:
                self.knot_points_x[-1] = self.start_coords[0]
                self.knot_points_y[-1] = self.start_coords[1]

        if self.end_coords:
            idx = self._get_closest_knot_index(self.end_coords[0], self.end_coords[1])
            self.knot_points_x[idx] = self.end_coords[0]
            self.knot_points_y[idx] = self.end_coords[1]
            # If end is the same as start, and it's closed, update the other end too
            if self.is_closed and (idx == 0 or idx == len(self.knot_points_x) - 1):
                self.knot_points_x[0] = self.end_coords[0]
                self.knot_points_x[-1] = self.end_coords[0]
                self.knot_points_y[0] = self.end_coords[1]
                self.knot_points_y[-1] = self.end_coords[1]

    def _get_closest_knot_index(self, x: float, y: float) -> int:
        """Helper to find which knot index is physically closest to a coordinate."""
        distances = [np.sqrt((kx - x)**2 + (ky - y)**2) 
                    for kx, ky in zip(self.knot_points_x, self.knot_points_y)]
        return np.argmin(distances)

    def _ensure_closed(self):
        """Ensure first and last points match for closed splines."""
        if (self.knot_points_x[0] != self.knot_points_x[-1] or 
            self.knot_points_y[0] != self.knot_points_y[-1]):
            self.knot_points_x.append(self.knot_points_x[0])
            self.knot_points_y.append(self.knot_points_y[0])

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]],
                    n_interpolated_points: int,
                    is_closed: bool = True) -> 'SplineGeometry':
        """Create a spline from a list of (x, y) points."""
        if not points:
            return cls([], [], None, None, n_interpolated_points, is_closed)
        x_coords, y_coords = zip(*points)
        return cls(list(x_coords), list(y_coords), None, None, n_interpolated_points, is_closed)

    @classmethod
    def from_arrays(cls, x_coords: List[float], y_coords: List[float],
                    n_interpolated_points: int,
                    is_closed: bool = True) -> 'SplineGeometry':
        """Create a spline from separate x and y arrays."""
        return cls(list(x_coords), list(y_coords), None, None, n_interpolated_points, is_closed)
    
    def interpolate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the spline using B-splines.
        This method changes the state of the geometry by updating self.full_contour.
        and returns interpolated_x and interpolated_y.
        """
        try:
            # cubic splines (k=3) require at least k+1 points
            n_points = len(self.knot_points_x)
            if n_points < 2:
                logger.warning(f"Not enough points for spline interpolation: {len(self.knot_points_x)}")
                return np.array(self.knot_points_x), np.array(self.knot_points_y)
            
            # k is the degree. Cubic is 3. We need m > k.
            # If we have 2 points, k=1 (linear). 3 points, k=2 (quadratic).
            k = min(3, n_points - 1)

            points_array = np.array([self.knot_points_x, self.knot_points_y])
            
            tck, u = splprep(points_array, u=None, s=0.0, k=k, per=int(self.is_closed))

            u_new = np.linspace(u.min(), u.max(), self.n_interpolated_points)
            x_new, y_new = splev(u_new, tck, der=0)
            self.full_contour = (x_new, y_new)

            return x_new, y_new
        except Exception as  e:
            logger.error(f"Error in spline interpolation: {e}")
            return np.array(self.knot_points_x), np.array(self.knot_points_y)
        
    def insert_point(self, x: float, y: float, insert_idx: Optional[int] = None) -> int:
        """Insert a new point into the spline"""
        is_was_closed = self.is_closed and len(self.knot_points_x) > 1
        if is_was_closed:
            self.knot_points_x.pop()
            self.knot_points_y.pop()

        if insert_idx is None:
            insert_idx = len(self.knot_points_x)

        self.knot_points_x.insert(insert_idx, x)
        self.knot_points_y.insert(insert_idx, y)

        if self.is_closed:
            self._ensure_closed()

        return insert_idx
    
    def get_closest_contour_index(self, x: float, y: float, threshold: float = 20.0) -> Optional[int]:
        """
        Logic: Is the mouse (x, y) near the smooth curve?
        Returns the index of the closest point on the full_contour if within threshold.
        """
        if not self.full_contour or len(self.full_contour[0]) == 0:
            return None
        
        distances = np.sqrt(
            (self.full_contour[0] - x) ** 2 + 
            (self.full_contour[1] - y) ** 2
        )
        min_dist = np.min(distances)

        if min_dist < threshold:
            return np.argmin(distances)
        return None
    
    def _find_best_insertion_index(self, path_index: int) -> int:
        """Find the best index to insert a new point based on contour position."""
        if not self.knot_points_x:
            return 0
        
        knot_path_indices = []
        # Use [:-1] if closed to avoid the duplicate end point confusing the search
        search_x = self.knot_points_x[:-1] if self.is_closed else self.knot_points_x
        search_y = self.knot_points_y[:-1] if self.is_closed else self.knot_points_y

        for kx, ky in zip(search_x, search_y):
            dist = np.sqrt((self.full_contour[0] - kx)**2 + (self.full_contour[1] - ky)**2)
            knot_path_indices.append(np.argmin(dist))

        # Use bisect to find where the new path_index fits among the knot indices
        import bisect
        insertion_idx = bisect.bisect_left(knot_path_indices, path_index)
        return insertion_idx

    def scale(self, factor: float) -> 'SplineGeometry':
        """Return a scaled version of the spline."""
        scaled_x = [x * factor for x in self.knot_points_x]
        scaled_y = [y * factor for y in self.knot_points_y]
        return SplineGeometry(
            scaled_x, 
            scaled_y, 
            self.n_interpolated_points, 
            self.start_coords,
            self.end_coords)
    
    def to_unscaled(self, scaling_factor: float) -> Tuple[List[float], List[float]]:
        """Return unscaled knot points."""
        if self.full_contour is not None:
            return ([x / scaling_factor for x in self.full_contour[0]],
                [y / scaling_factor for y in self.full_contour[1]])
        else:
            return ([x / scaling_factor for x in self.knot_points_x],
                [y / scaling_factor for y in self.knot_points_y])
        
    def get_split_interpolated_points(self):
        """Logic: If no end_coords, the whole spline is the 'main' solid segment."""
        full_x, full_y = self.interpolate()
        
        # If no end point is defined, there is no 'tail' (dotted part)
        if self.end_coords is None:
            return (full_x, full_y), (np.array([]), np.array([]))

        # Find the index in the interpolated array closest to start and end
        start_idx = 0
        if self.start_coords:
            start_idx = np.argmin(np.sqrt((full_x - self.start_coords[0])**2 + (full_y - self.start_coords[1])**2))
            
        end_idx = np.argmin(np.sqrt((full_x - self.end_coords[0])**2 + (full_y - self.end_coords[1])**2))

        # Rotate the array so it logically begins at the start_idx
        if self.is_closed:
            full_x = np.roll(full_x, -start_idx)
            full_y = np.roll(full_y, -start_idx)
            end_idx = (end_idx - start_idx) % len(full_x)

        main_seg = (full_x[:end_idx + 1], full_y[:end_idx + 1])
        tail_seg = (full_x[end_idx:], full_y[end_idx:])
        
        return main_seg, tail_seg


class Point(QGraphicsEllipseItem):
    """Qt-specific point drawing class - only handles Qt interaction"""
    
    def __init__(self, pos, line_thickness=1, point_radius=10, index=0, color=None, transparency=255):
        super().__init__()
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.transparency = transparency
        
        self.color = color
        self.x, self.y = pos[0], pos[1]
        self.index = index
        
        self.default_color = get_qt_pen(color, line_thickness, transparency)
        self.setPen(self.default_color)
        self._update_qt_rect()
    
    def get_coords(self):
        """Get coordinates - simple wrapper for Qt compatibility"""
        return self.x, self.y
    
    def update_pos(self, pos):
        """Update point position from Qt event"""
        if isinstance(pos, QPointF):
            self.x, self.y = pos.x(), pos.y()
        else:
            # Handle case where pos might be a tuple or other type
            self.x, self.y = pos.x(), pos.y() if hasattr(pos, 'x') else pos
        return self._update_qt_rect()
    
    def _update_qt_rect(self):
        """Update Qt rectangle from internal coordinates"""
        self.setRect(
            self.x - self.point_radius * 0.5,
            self.y - self.point_radius * 0.5,
            self.point_radius,
            self.point_radius
        )
        return self.rect()
    
    def update_color(self):
        """Change appearance when selected"""
        self.setPen(QPen(Qt.GlobalColor.transparent, self.line_thickness))
    
    def reset_color(self):
        """Reset to default appearance"""
        self.setPen(self.default_color)


class Spline(QGraphicsPathItem):
    """Qt-specific spline drawing class initialized with SplineGeometry"""
    
    def __init__(self, 
                 geometry: SplineGeometry, 
                 color: Any = "blue", 
                 line_thickness: int = 1, 
                 transparency: int = 255):
        super().__init__()
        self.geometry = geometry
        
        self.main_pen = get_qt_pen(color, line_thickness, transparency)
        self.setPen(self.main_pen)

        self.tail_item = QGraphicsPathItem(self) 
        self.tail_pen = get_qt_pen(color, line_thickness, transparency)
        self.tail_pen.setStyle(Qt.PenStyle.DotLine)
        self.tail_item.setPen(self.tail_pen)
        
        self._rebuild_path()

    def __str__(self):
        return (f"Spline(geometry={self.geometry}, "
                f"color={self.main_pen.color().name()}, "
                f"line_thickness={self.main_pen.width()}, "
                f"transparency={self.main_pen.color().alpha()})")

    @property
    def full_contours(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility property for existing IVUSDisplay code"""
        return self.geometry.interpolate()
    
    @property
    def knot_points(self) -> Tuple[List[float], List[float]]:
        """Compatibility property for existing IVUSDisplay code"""
        return self.geometry.knot_points_x, self.geometry.knot_points_y
    
    @property
    def full_points(self) -> List[Point]:
        """Get all interpolated points as Qt Point items."""
        x_vals, y_vals = self.geometry.interpolate()

        points: List[Point] = []
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            pt = Point(
                pos=(x, y),
                line_thickness=self.main_pen.width(),
                index=i,
                color=self.main_pen.color(),
                transparency=self.main_pen.color().alpha(),
            )
            points.append(pt)

        return points

    def set_geometry(self, geometry: SplineGeometry):
        """Update the underlying geometry and redraw"""
        self.geometry = geometry
        self._rebuild_path()

    def update_style(self, dashed: Optional[bool] = None, color: Optional[Any] = None):
        """Update visual properties dynamically"""
        pen = self.pen()
        if dashed is not None:
            self.dashed = dashed
            pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        if color is not None:
            # Re-use existing get_qt_pen logic to parse color
            new_pen = get_qt_pen(color, pen.width(), pen.color().alpha())
            new_pen.setStyle(pen.style())
            pen = new_pen
            
        self.setPen(pen)

    def _rebuild_path(self):
            """Internal: Rebuild Qt path from the geometry object with dotted 'closure' logic."""
            # 1. Get the split interpolated points
            # If end_coords exist, main_seg is start->end, tail_seg is end->start
            main_seg, tail_seg = self.geometry.get_split_interpolated_points()
            
            # --- Handle Main Path (Solid) ---
            main_path = QPainterPath()
            if len(main_seg[0]) > 0:
                main_path.moveTo(QPointF(main_seg[0][0], main_seg[1][0]))
                for i in range(1, len(main_seg[0])):
                    main_path.lineTo(QPointF(main_seg[0][i], main_seg[1][i]))
            
            self.setPath(main_path)

            tail_path = QPainterPath()
            # Only draw dotted line if we have a tail AND the geometry is meant to be closed
            if self.geometry.is_closed and len(tail_seg[0]) > 1:
                tail_path.moveTo(QPointF(tail_seg[0][0], tail_seg[1][0]))
                for i in range(1, len(tail_seg[0])):
                    tail_path.lineTo(QPointF(tail_seg[0][i], tail_seg[1][i]))
                
                # If the geometry says it's closed but the tail doesn't quite reach 
                # the start, we can force a closeSubpath or a lineTo start_coords
                tail_path.lineTo(QPointF(main_seg[0][0], main_seg[1][0]))
                
                self.tail_item.setPath(tail_path)
                self.tail_item.setVisible(True)
            else:
                self.tail_item.setVisible(False)

    def update(self, pos: QPointF, index: int, path_index: Optional[int] = None) -> int:
        """
        Updates the geometry and redraws. 
        Matches the signature IVUSDisplay.mouseMoveEvent expects.
        """
        if path_index is not None:
            # Adding a new point
            new_knot_idx = self.geometry._find_best_insertion_index(path_index)
            new_idx = self.geometry.insert_point(pos.x(), pos.y(), new_knot_idx)
            self.geometry._ensure_start_end_coords()
            self._rebuild_path()
            return new_idx
        else:
            # Moving an existing point
            self.geometry.knot_points_x[index] = pos.x()
            self.geometry.knot_points_y[index] = pos.y()

            if self.geometry.is_closed:
                last_idx = len(self.geometry.knot_points_x) - 1
                if index == 0:
                    self.geometry.knot_points_x[last_idx] = pos.x()
                    self.geometry.knot_points_y[last_idx] = pos.y()
                elif index == last_idx:
                    self.geometry.knot_points_x[0] = pos.x()
                    self.geometry.knot_points_y[0] = pos.y()
                
            self._rebuild_path()
            return index
            
    def on_path(self, pos: QPointF) -> Optional[int]:
        """Qt Wrapper for the geometry logic"""
        return self.geometry.get_closest_contour_index(pos.x(), pos.y())
            
    def get_unscaled_contour(self, scaling_factor: float):
        """Compatibility method"""
        return self.geometry.to_unscaled(scaling_factor)


def get_qt_pen(color, thickness, transparency=255):
    """Create a QPen with the specified color, thickness, and transparency"""
    if isinstance(color, str):
        # Try to get color from Qt.GlobalColor
        try:
            color_enum = getattr(Qt.GlobalColor, color.lower())
            pen_color = QColor(color_enum)
        except AttributeError:
            # Try to parse as hex color
            if color.startswith('#'):
                pen_color = QColor(color)
            else:
                # Default to blue
                pen_color = QColor(Qt.GlobalColor.blue)
    elif isinstance(color, (tuple, list)) and len(color) >= 3:
        # RGB or RGBA tuple
        if len(color) == 3:
            pen_color = QColor(color[0], color[1], color[2])
        else:
            pen_color = QColor(color[0], color[1], color[2], color[3] if len(color) > 3 else 255)
    else:
        # Default to blue
        pen_color = QColor(Qt.GlobalColor.blue)
    
    if not isinstance(transparency, int):
        try:
            transparency = int(transparency)
        except (ValueError, TypeError):
            transparency = 255

    pen_color.setAlpha(transparency)
    return QPen(pen_color, thickness)