"""Dedicated 3-D view for the post-cut geometry (Build Cut Geometry / Smooth /
Calculate Centerlines / RCA-LCA outlet points), in its own tab next to the main
segmentation view (see CctaViewer3D in display_3d.py). Split out from that class so
this layer isn't a "ghost" sharing space (and the label picking ray-march) with the
raw segmentation labels — it has its own render window, its own mask (the combined
cut mask), and its own picking logic that only ever targets that mask.
"""

import numpy as np
import trimesh
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PyQt6.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkActor, vtkLightKit, vtkPolyDataMapper, vtkRenderer

from domain.ccta_display_types import (
    CENTERLINE_AO_COLOR,
    CENTERLINE_LCA_COLOR,
    CENTERLINE_RCA_COLOR,
    CUT_MESH_COLOR,
    INLET_COLOR,
    LCA_POINT_COLOR,
    OUTLET_COLOR,
    RCA_POINT_COLOR,
)


def _mesh_to_polydata(mesh: trimesh.Trimesh) -> vtkPolyData:
    """trimesh.Trimesh -> vtkPolyData (points + triangle cells). Duplicated from
    pages/fusion/left_half/display_results.py rather than imported, per CCTA-only scope."""
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64)))

    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    id_array = numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True)
    cell_array = vtkCellArray()
    cell_array.SetCells(len(faces), id_array)

    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cell_array)

    tri = vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    return tri.GetOutput()


def _points_to_polydata(points: np.ndarray) -> vtkPolyData:
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64)))
    verts = vtkCellArray()
    for i in range(len(points)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(verts)
    return poly


class CutGeometryViewer3D(QWidget):
    outlet_points_changed = pyqtSignal(str, int)  # category ('rca'/'lca'), point count
    smooth_requested = pyqtSignal(float)  # taubin lambda
    reduce_mesh_requested = pyqtSignal(float)  # target reduction fraction (0-1)
    calculate_centerlines_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)

        self._smooth_lambda_spin = QDoubleSpinBox()
        self._smooth_lambda_spin.setRange(0.0, 1.0)
        self._smooth_lambda_spin.setSingleStep(0.05)
        self._smooth_lambda_spin.setValue(0.6)
        self._smooth_lambda_spin.setToolTip('Taubin smoothing lambda')
        self._smooth_lambda_spin.setFixedWidth(60)

        self._smooth_btn = QPushButton('Smooth')
        self._smooth_btn.setToolTip('Taubin-smooth the cut geometry and re-locate inlet/outlet')
        self._smooth_btn.clicked.connect(lambda: self.smooth_requested.emit(self._smooth_lambda_spin.value()))

        self._reduce_pct_spin = QSpinBox()
        self._reduce_pct_spin.setRange(1, 95)
        self._reduce_pct_spin.setValue(50)
        self._reduce_pct_spin.setSuffix('%')
        self._reduce_pct_spin.setToolTip('Target face-count reduction (higher = faster centerlines, less detail)')
        self._reduce_pct_spin.setFixedWidth(60)

        self._reduce_btn = QPushButton('Reduce Mesh')
        self._reduce_btn.setToolTip('Decimate the cut geometry to speed up Calculate Centerlines')
        self._reduce_btn.clicked.connect(lambda: self.reduce_mesh_requested.emit(self._reduce_pct_spin.value() / 100.0))

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setFixedWidth(80)
        self._opacity_slider.setToolTip('Cut geometry opacity')
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)

        self._centerlines_btn = QPushButton('Calculate Centerlines')
        self._centerlines_btn.clicked.connect(self.calculate_centerlines_requested.emit)

        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(4, 2, 4, 2)
        btn_bar.addWidget(self._smooth_lambda_spin)
        btn_bar.addWidget(self._smooth_btn)
        btn_bar.addWidget(self._reduce_pct_spin)
        btn_bar.addWidget(self._reduce_btn)
        btn_bar.addStretch()
        btn_bar.addWidget(QLabel('Opacity:'))
        btn_bar.addWidget(self._opacity_slider)
        btn_bar.addWidget(self._centerlines_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._vtk_widget, 1)
        layout.addLayout(btn_bar)

        self._ren = vtkRenderer()
        self._ren.SetBackground(0.0, 0.0, 0.0)
        self._ren.AutomaticLightCreationOff()

        light_kit = vtkLightKit()
        light_kit.SetKeyLightElevation(30)
        light_kit.SetKeyLightAzimuth(-60)
        light_kit.SetKeyLightIntensity(1.0)
        light_kit.SetFillLightWarmth(0.4)
        light_kit.SetBackLightWarmth(0.35)
        light_kit.AddLightsToRenderer(self._ren)

        self._vtk_widget.GetRenderWindow().AddRenderer(self._ren)
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()
        trackball = vtkInteractorStyleTrackballCamera()
        trackball.SetMotionFactor(7.5)
        self._vtk_widget.SetInteractorStyle(trackball)

        self._voxel_spacing: tuple[float, float, float] | None = None
        self._cut_mask: np.ndarray | None = None  # the combined mask the cut mesh was built from
        self._cut_mesh_actor: vtkActor | None = None
        self._inlet_actor: vtkActor | None = None
        self._outlet_actor: vtkActor | None = None
        self._centerline_actors: dict[str, vtkActor] = {}  # 'ao'/'rca'/'lca' -> actor
        self._press_qt: QPoint = QPoint()

        # Outlet point picking (Add RCA/LCA Outlet)
        self._point_mode: str | None = None  # 'rca' | 'lca' | None
        self._rca_points: list[tuple[int, int, int]] = []  # voxel (z, y, x)
        self._lca_points: list[tuple[int, int, int]] = []
        self._rca_points_actor: vtkActor | None = None
        self._lca_points_actor: vtkActor | None = None

        self._vtk_widget.installEventFilter(self)

    # ── cut mesh + inlet/outlet markers ─────────────────────────────────────

    def set_cut_mesh(
        self,
        mesh: trimesh.Trimesh,
        inlet_world: np.ndarray,
        outlet_world: np.ndarray,
        cut_mask: np.ndarray,
        voxel_spacing: tuple[float, float, float],
    ) -> None:
        """Add/replace the cut-geometry surface and its inlet/outlet markers, and
        reset any outlet points from a previous cut (they'd no longer correspond to
        the new surface). Fits the camera the first time this layer appears.

        cut_mask/voxel_spacing are stored so outlet point picking can ray-march
        against the actual cut geometry, not the raw segmentation mask.
        """
        is_first = self._cut_mesh_actor is None
        self._voxel_spacing = voxel_spacing
        self._cut_mask = cut_mask
        self._set_cut_mesh_actor(mesh)
        self._inlet_actor = self._place_marker(self._inlet_actor, inlet_world, INLET_COLOR, radius=2.5)
        self._outlet_actor = self._place_marker(self._outlet_actor, outlet_world, OUTLET_COLOR, radius=2.5)
        for category in ('rca', 'lca'):
            self.clear_points(category)
        if is_first:
            self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def update_cut_mesh(self, mesh: trimesh.Trimesh, inlet_world: np.ndarray, outlet_world: np.ndarray) -> None:
        """Swap in a re-smoothed mesh + refreshed inlet/outlet markers without
        touching the camera or the outlet points (smoothing doesn't change the
        underlying combined mask)."""
        self._set_cut_mesh_actor(mesh)
        self._inlet_actor = self._place_marker(self._inlet_actor, inlet_world, INLET_COLOR, radius=2.5)
        self._outlet_actor = self._place_marker(self._outlet_actor, outlet_world, OUTLET_COLOR, radius=2.5)
        self._vtk_widget.GetRenderWindow().Render()

    def _set_cut_mesh_actor(self, mesh: trimesh.Trimesh) -> None:
        if self._cut_mesh_actor is not None:
            self._ren.RemoveActor(self._cut_mesh_actor)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(_mesh_to_polydata(mesh))
        actor = vtkActor()
        actor.SetMapper(mapper)
        r, g, b = CUT_MESH_COLOR
        actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
        actor.GetProperty().SetOpacity(self._opacity_slider.value() / 100.0)  # preserve the current slider setting
        actor.GetProperty().SetInterpolationToFlat()
        self._ren.AddActor(actor)
        self._cut_mesh_actor = actor

    def _on_opacity_changed(self, value: int) -> None:
        if self._cut_mesh_actor is not None:
            self._cut_mesh_actor.GetProperty().SetOpacity(value / 100.0)
            self._vtk_widget.GetRenderWindow().Render()

    def set_centerlines(self, paths: dict[str, str]) -> None:
        """Read and display the computed ao/rca/lca centerlines (.vtp, written by
        Calculate Centerlines) as colored polylines, so the result can be checked
        visually against the cut geometry before trusting it. Replaces any
        previously displayed centerlines."""
        colors = {'ao': CENTERLINE_AO_COLOR, 'rca': CENTERLINE_RCA_COLOR, 'lca': CENTERLINE_LCA_COLOR}
        for label, path in paths.items():
            actor = self._centerline_actors.pop(label, None)
            if actor is not None:
                self._ren.RemoveActor(actor)

            reader = vtkXMLPolyDataReader()
            reader.SetFileName(path)
            reader.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            new_actor = vtkActor()
            new_actor.SetMapper(mapper)
            r, g, b = colors.get(label, (255, 255, 255))
            new_actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
            new_actor.GetProperty().SetLineWidth(3)
            new_actor.GetProperty().SetRenderLinesAsTubes(True)
            self._ren.AddActor(new_actor)
            self._centerline_actors[label] = new_actor

        self._vtk_widget.GetRenderWindow().Render()

    def _place_marker(
        self, actor: vtkActor | None, world: np.ndarray, color: tuple[int, int, int], radius: float
    ) -> vtkActor:
        if actor is not None:
            self._ren.RemoveActor(actor)
        sphere = vtkSphereSource()
        sphere.SetCenter(float(world[0]), float(world[1]), float(world[2]))
        sphere.SetRadius(radius)
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        sphere.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        new_actor = vtkActor()
        new_actor.SetMapper(mapper)
        r, g, b = color
        new_actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
        new_actor.GetProperty().SetOpacity(1.0)
        self._ren.AddActor(new_actor)
        return new_actor

    def shutdown(self) -> None:
        """Finalize the VTK OpenGL context while the HWND is still valid."""
        rw = self._vtk_widget.GetRenderWindow()
        rw.Finalize()
        rw.SetInteractor(None)

    # ── voxel <-> world <-> screen (same conventions as CctaViewer3D) ───────

    def voxel_to_world(self, z: int, y_vox: int, x_vox: int) -> tuple[float, float, float]:
        assert self._voxel_spacing is not None and self._cut_mask is not None
        dz, dy, dx = self._voxel_spacing
        Y = self._cut_mask.shape[1]
        return x_vox * dx, (Y - 1 - y_vox) * dy, z * dz

    def screen_to_ray(self, sx: int, sy: int) -> tuple[np.ndarray, np.ndarray]:
        def _display_to_world(depth: float) -> np.ndarray:
            self._ren.SetDisplayPoint(float(sx), float(sy), depth)
            self._ren.DisplayToWorld()
            wp = self._ren.GetWorldPoint()
            w = wp[3] if wp[3] != 0.0 else 1.0
            return np.array([wp[0] / w, wp[1] / w, wp[2] / w])

        return _display_to_world(0.0), _display_to_world(1.0)

    def _project_world_batch(self, wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
        """Vectorised world -> VTK display-pixel projection. Returns (N, 2) float array."""
        vtk_mat = self._ren.GetActiveCamera().GetCompositeProjectionTransformMatrix(
            self._ren.GetTiledAspectRatio(), -1.0, 1.0
        )
        m = np.array([[vtk_mat.GetElement(r, c) for c in range(4)] for r in range(4)])

        world_h = np.stack([wx, wy, wz, np.ones(len(wx))], axis=1)
        clip = (m @ world_h.T).T

        w = np.where(np.abs(clip[:, 3]) > 1e-10, clip[:, 3], 1.0)
        ndc_x = clip[:, 0] / w
        ndc_y = clip[:, 1] / w

        vp = self._ren.GetViewport()
        W, H = self._vtk_widget.GetRenderWindow().GetSize()
        sx = (ndc_x + 1.0) * 0.5 * (vp[2] - vp[0]) * W + vp[0] * W
        sy = (ndc_y + 1.0) * 0.5 * (vp[3] - vp[1]) * H + vp[1] * H
        return np.column_stack([sx, sy])

    def _pick_nearest_on_cut_mask(self, sx: int, sy: int) -> tuple[int, int, int] | None:
        """Cast a ray through screen pixel (sx, sy) and return the first non-zero
        voxel hit in self._cut_mask, as (z, y_vox, x_vox). Returns None if it misses."""
        if self._cut_mask is None or self._voxel_spacing is None:
            return None
        dz, dy, dx = self._voxel_spacing
        Z, Y, X = self._cut_mask.shape

        near, far = self.screen_to_ray(sx, sy)
        ray = far - near
        length = float(np.linalg.norm(ray))
        if length < 1e-6:
            return None
        ray_dir = ray / length

        step = min(dx, dy, dz) * 0.5
        n_steps = int(length / step) + 1

        prev_ijk = (-1, -1, -1)
        for i in range(n_steps):
            wp = near + ray_dir * (i * step)
            xi = int(round(wp[0] / dx))
            yi = int((Y - 1) - round(wp[1] / dy))
            zi = int(round(wp[2] / dz))
            ijk = (zi, yi, xi)
            if ijk == prev_ijk:
                continue
            prev_ijk = ijk
            if 0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X and self._cut_mask[zi, yi, xi]:
                return zi, yi, xi

        return None

    # ── outlet point picking (Add RCA/LCA Outlet) ───────────────────────────

    _CLOSE_PX = 15  # pixels — right-click-to-remove tolerance

    def set_point_mode(self, category: str) -> None:
        """category is 'rca', 'lca', or '' to cancel picking mode.

        Caller (CctaPage._on_outlet_point_mode_requested) is expected to have already
        checked that the cut geometry exists (and reset the panel's toggle button if
        not) before calling this.
        """
        self._point_mode = category or None

    def clear_points(self, category: str) -> None:
        points = self._rca_points if category == 'rca' else self._lca_points
        points.clear()
        self._rebuild_points_actor(category)
        self.outlet_points_changed.emit(category, 0)

    def set_points(self, category: str, voxel_points: list[tuple[int, int, int]]) -> None:
        """Replace all points for a category (used to restore persisted outlet points
        on load) and re-render + notify the panel of the new count."""
        points = voxel_points[:]
        if category == 'rca':
            self._rca_points = points
        else:
            self._lca_points = points
        self._rebuild_points_actor(category)
        self.outlet_points_changed.emit(category, len(points))

    def rca_points_voxel(self) -> list[tuple[int, int, int]]:
        return list(self._rca_points)

    def lca_points_voxel(self) -> list[tuple[int, int, int]]:
        return list(self._lca_points)

    def rca_points_world(self) -> list[np.ndarray]:
        return [np.array(self.voxel_to_world(*p)) for p in self._rca_points]

    def lca_points_world(self) -> list[np.ndarray]:
        return [np.array(self.voxel_to_world(*p)) for p in self._lca_points]

    def _add_point(self, category: str, voxel: tuple[int, int, int]) -> None:
        points = self._rca_points if category == 'rca' else self._lca_points
        points.append(voxel)
        self._rebuild_points_actor(category)
        self.outlet_points_changed.emit(category, len(points))

    def _remove_nearest_point(self, category: str, sx: int, sy: int) -> None:
        points = self._rca_points if category == 'rca' else self._lca_points
        if not points:
            return
        world = np.array([self.voxel_to_world(*p) for p in points])
        screen = self._project_world_batch(world[:, 0], world[:, 1], world[:, 2])
        dists = np.hypot(screen[:, 0] - sx, screen[:, 1] - sy)
        idx = int(np.argmin(dists))
        if dists[idx] <= self._CLOSE_PX:
            points.pop(idx)
            self._rebuild_points_actor(category)
            self.outlet_points_changed.emit(category, len(points))

    def _rebuild_points_actor(self, category: str) -> None:
        is_rca = category == 'rca'
        points = self._rca_points if is_rca else self._lca_points
        actor = self._rca_points_actor if is_rca else self._lca_points_actor
        if actor is not None:
            self._ren.RemoveActor(actor)
            actor = None

        if points:
            world = np.array([self.voxel_to_world(*p) for p in points])
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(_points_to_polydata(world))
            actor = vtkActor()
            actor.SetMapper(mapper)
            r, g, b = RCA_POINT_COLOR if is_rca else LCA_POINT_COLOR
            actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
            actor.GetProperty().SetPointSize(12)
            actor.GetProperty().SetRenderPointsAsSpheres(True)
            self._ren.AddActor(actor)

        if is_rca:
            self._rca_points_actor = actor
        else:
            self._lca_points_actor = actor
        self._vtk_widget.GetRenderWindow().Render()

    # ── mouse handling: point-mode add/remove, else plain camera passthrough ─

    def eventFilter(self, obj, event) -> bool:
        if obj is self._vtk_widget:
            t = event.type()
            if self._point_mode is not None:
                if t == QEvent.Type.MouseButtonPress:
                    vtk_y = self._vtk_widget.height() - 1 - event.pos().y()
                    if event.button() == Qt.MouseButton.LeftButton:
                        hit = self._pick_nearest_on_cut_mask(event.pos().x(), vtk_y)
                        if hit is not None:
                            self._add_point(self._point_mode, hit)
                        return True  # block VTK camera move
                    if event.button() == Qt.MouseButton.RightButton:
                        self._remove_nearest_point(self._point_mode, event.pos().x(), vtk_y)
                        return True
        return super().eventFilter(obj, event)
