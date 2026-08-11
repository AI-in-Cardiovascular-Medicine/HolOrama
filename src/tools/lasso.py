"""Screen-space lasso polygon for VTK 3-D views.

Shared by CCTA's mask-erase lasso (pages/ccta/left_half/display_3d.py) and Fusion's
point-reclassify lasso (pages/fusion/left_half/display_results.py): both let a user
click out a polygon over the render window, close it, then test which 3-D points
project inside it. What "inside" means to the caller — erase a mask voxel, move a
point from one label to another — is left entirely to the caller; this module only
owns the polygon capture/drawing/projection/containment mechanics.
"""

import numpy as np
from matplotlib.path import Path as MplPath
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkCoordinate,
    vtkPolyDataMapper2D,
    vtkRenderer,
)

from tools.geometry import SplineGeometry

CLOSE_PX = 15  # pixels — clicking within this radius of the first point closes the lasso


class Lasso2D:
    """One in-progress lasso polygon, captured in VTK display (screen-pixel, y from
    bottom) coordinates, plus its 2-D overlay actors drawn into `renderer`. Owns no
    mouse-event wiring — callers translate their own Qt events into add_point() calls
    and decide when a right-click (or similar) should force a close."""

    def __init__(self, renderer: vtkRenderer, color: tuple[float, float, float] = (1.0, 1.0, 0.0)) -> None:
        self._ren = renderer
        self._color = color
        self.points: list[tuple[int, int]] = []
        self._overlay: list[vtkActor2D] = []

    def add_point(self, sx: int, sy: int) -> bool:
        """Add a control point. Returns True if this click closed the lasso (landed
        within CLOSE_PX of the first point, with >=3 points already placed) — the
        closing click itself is not appended, mirroring how the first point is never
        duplicated as a last point."""
        if len(self.points) >= 3:
            fx, fy = self.points[0]
            if abs(sx - fx) <= CLOSE_PX and abs(sy - fy) <= CLOSE_PX:
                return True
        self.points.append((sx, sy))
        return False

    def clear(self) -> None:
        self.points.clear()
        for a in self._overlay:
            self._ren.RemoveActor2D(a)
        self._overlay.clear()

    def redraw(self) -> None:
        """Rebuild the dot + polyline overlay actors from the current control points."""
        for a in self._overlay:
            self._ren.RemoveActor2D(a)
        self._overlay.clear()

        for sx, sy in self.points:
            self._overlay.append(_make_dot2d(sx, sy, self._color))
        if len(self.points) >= 2:
            self._overlay.append(_make_polyline2d(self.spline_points(), self._color))

        for a in self._overlay:
            self._ren.AddActor2D(a)

    def spline_points(self) -> list[tuple[float, float]]:
        """Interpolated closed polygon in screen space (SplineGeometry over the control
        points), falling back to the raw closed control polygon if the spline fit fails
        (e.g. too few distinct points)."""
        pts = self.points
        if len(pts) >= 3:
            try:
                geom = SplineGeometry(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    300,
                    None,
                    None,
                    is_closed=True,
                )
                cx, cy = geom.full_contour
                return list(zip(cx.tolist(), cy.tolist()))
            except Exception:
                pass
        return [(float(x), float(y)) for x, y in pts] + [(float(pts[0][0]), float(pts[0][1]))]

    def contains(self, screen_pts: np.ndarray) -> np.ndarray:
        """Boolean mask over `screen_pts` (N, 2): which fall inside the closed polygon."""
        polygon = np.array(self.spline_points())
        return MplPath(polygon).contains_points(screen_pts)


def project_world_batch(renderer: vtkRenderer, window_size: tuple[int, int], wx, wy, wz) -> np.ndarray:
    """Vectorised world -> VTK display-pixel projection for many points at once, using
    the renderer's active camera. Returns an (N, 2) float array of (sx, sy)."""
    vtk_mat = renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(
        renderer.GetTiledAspectRatio(), -1.0, 1.0
    )
    m = np.array([[vtk_mat.GetElement(r, c) for c in range(4)] for r in range(4)])

    world_h = np.stack([wx, wy, wz, np.ones(len(wx))], axis=1)  # (N, 4)
    clip = (m @ world_h.T).T  # (N, 4)

    w = np.where(np.abs(clip[:, 3]) > 1e-10, clip[:, 3], 1.0)
    ndc_x = clip[:, 0] / w
    ndc_y = clip[:, 1] / w

    vp = renderer.GetViewport()
    W, H = window_size
    sx = (ndc_x + 1.0) * 0.5 * (vp[2] - vp[0]) * W + vp[0] * W
    sy = (ndc_y + 1.0) * 0.5 * (vp[3] - vp[1]) * H + vp[1] * H
    return np.column_stack([sx, sy])


def _make_dot2d(sx: float, sy: float, color: tuple[float, float, float], size: int = 8) -> vtkActor2D:
    pts = vtkPoints()
    pts.InsertNextPoint(float(sx), float(sy), 0.0)
    verts = vtkCellArray()
    verts.InsertNextCell(1)
    verts.InsertCellPoint(0)
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(verts)
    coord = vtkCoordinate()
    coord.SetCoordinateSystemToDisplay()
    mapper = vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)
    actor = vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetPointSize(size)
    return actor


def _make_polyline2d(pts: list[tuple[float, float]], color: tuple[float, float, float]) -> vtkActor2D:
    vtk_pts = vtkPoints()
    for x, y in pts:
        vtk_pts.InsertNextPoint(float(x), float(y), 0.0)
    n = len(pts)
    lines = vtkCellArray()
    for i in range(n - 1):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i)
        lines.InsertCellPoint(i + 1)
    poly = vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetLines(lines)
    coord = vtkCoordinate()
    coord.SetCoordinateSystemToDisplay()
    mapper = vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)
    actor = vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(2)
    return actor
