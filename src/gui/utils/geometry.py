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
    
    def __str__(self):
        return (f"SplineGeometry(knot_points_x={self.knot_points_x}, "
                f"knot_points_y={self.knot_points_y}, "
                f"n_interpolated_points={self.n_interpolated_points}, "
                f"start_coords={self.start_coords}, "
                f"end_coords={self.end_coords}, "
                f"is_closed={self.is_closed})")

    def _ensure_start_end_coords(self):
        """Ensure start and end coordinates are included as knot poitns if specified."""
        if self.start_coords:
            if (self.start_coords[0] not in self.knot_points_x or 
                self.start_coords[1] not in self.knot_points_y):
                self.knot_points_x.insert(0, self.start_coords[0])
                self.knot_points_y.insert(0, self.start_coords[1])

        if self.end_coords:
            if (self.end_coords[0] not in self.knot_points_x or 
                self.end_coords[1] not in self.knot_points_y):
                insert_idx = -1 if (self.is_closed and len(self.knot_points_x) > 0) else len(self.knot_points_x)
                self.knot_points_x.insert(insert_idx, self.end_coords[0])
                self.knot_points_y.insert(insert_idx, self.end_coords[1])

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
    
    def find_best_insertion_index(self, contour_index: int,
                                interpolated_x: np.ndarray,
                                interpolated_y: np.ndarray) -> int:
        """Find the best index to insert a new point based on contour position."""
        if not self.knot_points_x:
            return 0
        
        # Slice off the last point if closed to avoid duplicate-index confusion
        kx = self.knot_points_x[:-1]
        ky = self.knot_points_y[:-1]
        
        path_indices = []
        for i in range(len(kx)):
            distances = np.sqrt((kx[i] - interpolated_x) ** 2 + (ky[i] - interpolated_y) ** 2)
            path_indices.append(np.argmin(distances))
        
        # Find position using bisect
        idx = bisect.bisect_left(path_indices, contour_index)
        return idx

    def scale(self, factor: float) -> 'SplineGeometry':
        """Return a scaled version of the spline."""
        scaled_x = [x * factor for x in self.knot_points_x]
        scaled_y = [y * factor for y in self.knot_points_y]
        return SplineGeometry(scaled_x, scaled_y, self.n_interpolated_points, self.is_closed)
    
    def to_unscaled(self, scaling_factor: float) -> Tuple[List[float], List[float]]:
        """Return unscaled knot points."""
        if self.full_contour is not None:
            return ([x / scaling_factor for x in self.full_contour[0]],
                [y / scaling_factor for y in self.full_contour[1]])
        else:
            return ([x / scaling_factor for x in self.knot_points_x],
                [y / scaling_factor for y in self.knot_points_y])
    
    def split_at_two_indices(self, idx1: int, idx2: int) -> Tuple['SplineGeometry', 'SplineGeometry']:
        """Splits a spline into two separate open sections, handling closed-loop redundancy."""
        i, j = sorted([idx1, idx2])
        
        # If closed, the last point is a duplicate of the first.
        kx = self.knot_points_x[:-1] if self.is_closed else self.knot_points_x
        ky = self.knot_points_y[:-1] if self.is_closed else self.knot_points_y
        
        # inner segment
        x1, y1 = kx[i : j+1], ky[i : j+1]
        
        # outer segment
        # Note the +1 on the end slice to ensure the segments overlap at the knots
        x2 = kx[j:] + kx[:i+1]
        y2 = ky[j:] + ky[:i+1]

        return (
            SplineGeometry(x1, y1, None, None, self.n_interpolated_points, is_closed=False, dashed=False),
            SplineGeometry(x2, y2, None, None, self.n_interpolated_points, is_closed=False, dashed=True)
        )

    def stitch_with(self, other: 'SplineGeometry', close_final: bool = True) -> 'SplineGeometry':
        """Stitches two splines and cleans up the junction points."""
        new_x = self.knot_points_x + other.knot_points_x[1:]
        new_y = self.knot_points_y + other.knot_points_y[1:]
        
        # If we are closing it, the last point is already a duplicate of the first 
        # because of the way split_at_two_indices wraps. 
        # SplineGeometry.__post_init__ will call _ensure_closed, so we don't want 
        # TWO duplicates at the end. We strip one here.
        if close_final and len(new_x) > 1:
            if new_x[0] == new_x[-1] and new_y[0] == new_y[-1]:
                new_x.pop()
                new_y.pop()

        return SplineGeometry(new_x, new_y, None, None, self.n_interpolated_points, is_closed=close_final)

    def get_split_interpolated_points(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Returns ((main_x, main_y), (tail_x, tail_y)) split at the end_coords
        """
        full_x, full_y = self.interpolate()
        if self.end_coords is None:
            return (full_x, full_y), (np.array([]), np.array([]))

        distances = np.sqrt(
            (full_x - self.end_coords[0])**2 + 
            (full_y - self.end_coords[1])**2
        )
        split_idx = np.argmin(distances)

        main_segment = (full_x[:split_idx + 1], full_y[:split_idx + 1])
        tail_segment = (full_x[split_idx:], full_y[split_idx:])
        
        return main_segment, tail_segment

class Point(QGraphicsEllipseItem):
    """Qt-specific point drawing class - only handles Qt interaction"""
    
    def __init__(self, pos, line_thickness=1, point_radius=10, color=None, transparency=255):
        super().__init__()
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.transparency = transparency
        
        self.x, self.y = pos[0], pos[1]
        
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
            interpolated_x, interpolated_y = self.geometry.interpolate()
            self.geometry._ensure_start_end_coords()
            new_idx = self.geometry.insert_point(pos.x(), pos.y(), 
                        self.geometry.find_best_insertion_index(path_index, interpolated_x, interpolated_y))
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