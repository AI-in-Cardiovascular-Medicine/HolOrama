"""Dedicated 3D view for the post-cut geometry (Build Cut Geometry / Smooth /
Calculate Centerlines / RCA-LCA outlet points), in its own tab next to the main
segmentation view (see CctaViewer3D in display_3d.py). Split out from that class so
this layer isn't a "ghost" sharing space (and the label picking ray-march) with the
raw segmentation labels — it has its own render window, its own mask (the combined
cut mask), and its own picking logic that only ever targets that mask.
"""

from typing import cast

import numpy as np
import trimesh
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
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
from pages.ccta.left_half.cut_geometry.geometry import mesh_to_vtk_polydata
from tools.sphere_smooth import SphereBrush


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
    remesh_requested = pyqtSignal(float, int)  # target edge length (mm), iterations
    calculate_centerlines_requested = pyqtSignal()
    # Sphere brush (local smoothing/remeshing, see set_sphere_mode / tools.sphere_smooth):
    # mode picked from the button bar ('smooth' | 'remesh' | '' = off), surface point
    # under the cursor / clicked, and undo.
    sphere_mode_toggled = pyqtSignal(str)
    sphere_hovered = pyqtSignal(float, float, float)
    sphere_clicked = pyqtSignal(float, float, float)
    undo_requested = pyqtSignal()

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
        self._reduce_pct_spin.setValue(10)
        self._reduce_pct_spin.setSuffix('%')
        self._reduce_pct_spin.setToolTip('Target face-count reduction (higher = faster centerlines, less detail)')
        self._reduce_pct_spin.setFixedWidth(60)

        self._reduce_btn = QPushButton('Reduce Mesh')
        self._reduce_btn.setToolTip('Decimate the cut geometry to speed up Calculate Centerlines')
        self._reduce_btn.clicked.connect(lambda: self.reduce_mesh_requested.emit(self._reduce_pct_spin.value() / 100.0))

        self._remesh_edge_spin = QDoubleSpinBox()
        self._remesh_edge_spin.setRange(0.01, 10.0)
        self._remesh_edge_spin.setSingleStep(0.05)
        self._remesh_edge_spin.setValue(0.5)
        self._remesh_edge_spin.setSuffix(' mm')
        self._remesh_edge_spin.setToolTip('Remesh target edge length')
        self._remesh_edge_spin.setFixedWidth(75)

        self._remesh_iter_spin = QSpinBox()
        self._remesh_iter_spin.setRange(1, 100)
        self._remesh_iter_spin.setValue(10)
        self._remesh_iter_spin.setToolTip('Remesh iterations')
        self._remesh_iter_spin.setFixedWidth(50)

        self._remesh_btn = QPushButton('Remesh')
        self._remesh_btn.setToolTip("Fix and isotropically remesh the cut geometry (same as Fusion's Fix and Remesh)")
        self._remesh_btn.clicked.connect(
            lambda: self.remesh_requested.emit(self._remesh_edge_spin.value(), self._remesh_iter_spin.value())
        )

        self.sphere_btn = QPushButton('Sphere Smooth')
        self.sphere_btn.setCheckable(True)
        self.sphere_btn.setToolTip(
            'Hover over the cut geometry to preview, click to smooth the highlighted patch.\n'
            'Ctrl+wheel changes the radius. Uses the Smooth lambda.'
        )
        self.sphere_btn.toggled.connect(lambda checked: self._on_sphere_btn_toggled('smooth', checked))

        self.sphere_remesh_btn = QPushButton('Sphere Remesh')
        self.sphere_remesh_btn.setCheckable(True)
        self.sphere_remesh_btn.setToolTip(
            'Hover to preview, click to remesh the highlighted patch evenly at the Remesh edge length\n'
            '(thins out dense point clusters behind spikes and holes). Ctrl+wheel changes the radius.\n'
            'Cannot be undone, and clears the smoothing undo history.'
        )
        self.sphere_remesh_btn.toggled.connect(lambda checked: self._on_sphere_btn_toggled('remesh', checked))
        self._sphere_btns = {'smooth': self.sphere_btn, 'remesh': self.sphere_remesh_btn}

        self.sphere_radius = QDoubleSpinBox()
        self.sphere_radius.setRange(0.2, 20.0)
        self.sphere_radius.setSingleStep(0.2)
        self.sphere_radius.setValue(2.0)
        self.sphere_radius.setSuffix(' mm')
        self.sphere_radius.setToolTip('Brush radius, measured along the surface (Ctrl+wheel in the view)')
        self.sphere_radius.valueChanged.connect(self._on_sphere_radius_changed)

        self.sphere_iterations = QSpinBox()
        self.sphere_iterations.setRange(1, 100)
        self.sphere_iterations.setValue(10)
        self.sphere_iterations.setToolTip('Sphere Smooth: Taubin iterations per click')

        self._undo_btn = QPushButton('Undo')
        self._undo_btn.setToolTip('Undo the last smoothing step (Ctrl+Z while this tab is shown)')
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo_requested.emit)

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
        btn_bar.addWidget(self._remesh_edge_spin)
        btn_bar.addWidget(self._remesh_iter_spin)
        btn_bar.addWidget(self._remesh_btn)
        btn_bar.addStretch()
        btn_bar.addWidget(QLabel('Opacity:'))
        btn_bar.addWidget(self._opacity_slider)
        btn_bar.addWidget(self._centerlines_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._vtk_widget, 1)
        layout.addLayout(btn_bar)

        sphere_bar = QHBoxLayout()
        sphere_bar.setContentsMargins(4, 0, 4, 2)
        sphere_bar.addWidget(self.sphere_btn)
        sphere_bar.addWidget(self.sphere_remesh_btn)
        sphere_bar.addWidget(QLabel('Radius:'))
        sphere_bar.addWidget(self.sphere_radius)
        sphere_bar.addWidget(QLabel('Iterations:'))
        sphere_bar.addWidget(self.sphere_iterations)
        sphere_bar.addWidget(self._undo_btn)
        sphere_bar.addStretch()
        layout.addLayout(sphere_bar)

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

        # Outlet point picking (Add RCA/LCA Outlet). Keyed by category ('rca'/'lca')
        # instead of a pair of parallel attributes, so every method below is a plain
        # dict lookup instead of an if/else repeated in five places.
        self._point_mode: str | None = None  # 'rca' | 'lca' | None
        self._points: dict[str, list[tuple[int, int, int]]] = {'rca': [], 'lca': []}  # voxel (z, y, x)
        self._points_actors: dict[str, vtkActor | None] = {'rca': None, 'lca': None}

        # Sphere brush: hover preview (sphere + the vertices that would move, which the
        # caller computes and hands back via set_sphere_highlight), debounced so a fast
        # mouse doesn't queue up a region search per pixel.
        self._sphere_mode = ''  # 'smooth' | 'remesh' | '' (off)
        self._sphere_brush = SphereBrush(self._ren)
        self._sphere_brush.set_radius(self.sphere_radius.value())
        self._sphere_hover_pos = QPoint()
        self._sphere_hover_timer = QTimer(self)
        self._sphere_hover_timer.setSingleShot(True)
        self._sphere_hover_timer.setInterval(30)
        self._sphere_hover_timer.timeout.connect(self._on_sphere_hover_timer)

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
        # Re-point the existing actor's mapper at the new geometry instead of building
        # a fresh vtkActor/vtkPolyDataMapper — Smooth/Reduce Mesh call this on every
        # click, and there's no need to churn VTK objects just to swap the polydata.
        if self._cut_mesh_actor is not None:
            cast(vtkPolyDataMapper, self._cut_mesh_actor.GetMapper()).SetInputData(mesh_to_vtk_polydata(mesh))
            return

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh_to_vtk_polydata(mesh))
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
        if self._point_mode is not None:
            self.set_sphere_mode('')  # both take over left-clicks — only one at a time

    _POINT_COLORS = {'rca': RCA_POINT_COLOR, 'lca': LCA_POINT_COLOR}

    def clear_points(self, category: str) -> None:
        self._points[category].clear()
        self._rebuild_points_actor(category)
        self.outlet_points_changed.emit(category, 0)

    def set_points(self, category: str, voxel_points: list[tuple[int, int, int]]) -> None:
        """Replace all points for a category (used to restore persisted outlet points
        on load) and re-render + notify the panel of the new count."""
        self._points[category] = voxel_points[:]
        self._rebuild_points_actor(category)
        self.outlet_points_changed.emit(category, len(voxel_points))

    def rca_points_voxel(self) -> list[tuple[int, int, int]]:
        return list(self._points['rca'])

    def lca_points_voxel(self) -> list[tuple[int, int, int]]:
        return list(self._points['lca'])

    def rca_points_world(self) -> list[np.ndarray]:
        return [np.array(self.voxel_to_world(*p)) for p in self._points['rca']]

    def lca_points_world(self) -> list[np.ndarray]:
        return [np.array(self.voxel_to_world(*p)) for p in self._points['lca']]

    def _add_point(self, category: str, voxel: tuple[int, int, int]) -> None:
        points = self._points[category]
        points.append(voxel)
        self._rebuild_points_actor(category)
        self.outlet_points_changed.emit(category, len(points))

    def _remove_nearest_point(self, category: str, sx: int, sy: int) -> None:
        points = self._points[category]
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
        points = self._points[category]
        actor = self._points_actors[category]
        if actor is not None:
            self._ren.RemoveActor(actor)
            actor = None

        if points:
            world = np.array([self.voxel_to_world(*p) for p in points])
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(_points_to_polydata(world))
            actor = vtkActor()
            actor.SetMapper(mapper)
            r, g, b = self._POINT_COLORS[category]
            actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
            actor.GetProperty().SetPointSize(12)
            actor.GetProperty().SetRenderPointsAsSpheres(True)
            self._ren.AddActor(actor)

        self._points_actors[category] = actor
        self._vtk_widget.GetRenderWindow().Render()

    # ── sphere brush (local smoothing / remeshing) ──────────────────────────

    @property
    def sphere_mode(self) -> str:
        return self._sphere_mode

    def set_sphere_mode(self, mode: str) -> None:
        """Turn the sphere brush on as 'smooth' or 'remesh' (hover to preview, click to
        emit sphere_clicked) or off (''). Camera rotate/zoom keep working while it's on.
        The page calls this once it has accepted/rejected a toggle; the buttons are
        synced here silently, so only the chosen mode's button stays checked."""
        self._sphere_mode = mode
        if mode:
            self._point_mode = None
        for key, btn in self._sphere_btns.items():
            btn.blockSignals(True)
            btn.setChecked(key == mode)
            btn.blockSignals(False)
        if not mode:
            self._sphere_hover_timer.stop()
            self.hide_sphere_preview()

    def _on_sphere_btn_toggled(self, mode: str, checked: bool) -> None:
        self.sphere_mode_toggled.emit(mode if checked else '')

    def set_sphere_highlight(self, points: np.ndarray) -> None:
        """Show `points` (N, 3) as the vertices the brush would move, and render."""
        self._sphere_brush.set_highlight(points)
        self._vtk_widget.GetRenderWindow().Render()

    def hide_sphere_preview(self) -> None:
        self._sphere_brush.hide()
        self._vtk_widget.GetRenderWindow().Render()

    def refresh_sphere_hover(self) -> None:
        """Re-run the hover pick at the last cursor position (after the mesh or radius
        changed under a stationary cursor)."""
        if self._sphere_mode:
            self._sphere_hover_timer.start()

    def remesh_params(self) -> tuple[float, int]:
        """(target edge length mm, iterations) from the Remesh controls — shared by Remesh and Sphere Remesh."""
        return self._remesh_edge_spin.value(), self._remesh_iter_spin.value()

    def smooth_lambda(self) -> float:
        """Taubin lambda from the Smooth spin box — shared by whole-mesh Smooth and the sphere brush."""
        return self._smooth_lambda_spin.value()

    def set_undo_available(self, available: bool) -> None:
        self._undo_btn.setEnabled(available)

    def _on_sphere_radius_changed(self, radius_mm: float) -> None:
        self._sphere_brush.set_radius(radius_mm)
        self.refresh_sphere_hover()  # highlight must follow the new radius

    def _sphere_pick(self, pos: QPoint) -> tuple[float, float, float] | None:
        if self._cut_mesh_actor is None:
            return None
        return self._sphere_brush.pick(self._cut_mesh_actor, pos.x(), self._vtk_widget.height() - 1 - pos.y())

    def _on_sphere_hover_timer(self) -> None:
        if not self._sphere_mode:
            return
        hit = self._sphere_pick(self._sphere_hover_pos)
        if hit is None:
            self.hide_sphere_preview()
            return
        self._sphere_brush.show_at(hit)
        self.sphere_hovered.emit(*hit)  # the caller answers with set_sphere_highlight (which renders)

    def _sphere_event(self, event) -> bool:
        """Sphere-mode handling for one widget event; True = consumed (VTK never sees it).
        Presses/drags are passed through so the camera still rotates; a click is a press +
        release that moved at most 3 px. Same behavior as Fusion's sphere brush."""
        etype = event.type()
        if etype == QEvent.Type.MouseMove:
            if event.buttons() == Qt.MouseButton.NoButton:
                self._sphere_hover_pos = event.pos()
                self._sphere_hover_timer.start()
            elif self._sphere_brush.visible:
                self.hide_sphere_preview()  # rotating/panning — the preview would lag behind
        elif etype == QEvent.Type.Leave:
            self._sphere_hover_timer.stop()
            self.hide_sphere_preview()
        elif etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._press_qt = event.pos()
        elif etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            dp = event.pos() - self._press_qt
            if abs(dp.x()) <= 3 and abs(dp.y()) <= 3:
                hit = self._sphere_pick(event.pos())
                if hit is not None:
                    self.sphere_clicked.emit(*hit)
        elif etype == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.sphere_radius.stepBy(1 if delta > 0 else -1)
            return True  # Ctrl+wheel resizes the brush instead of zooming
        return False

    # ── mouse handling: point-mode add/remove, sphere brush, else camera passthrough ─

    def eventFilter(self, obj, event) -> bool:
        if obj is self._vtk_widget:
            t = event.type()
            if self._sphere_mode and self._sphere_event(event):
                return True
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
