import numpy as np
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkCoordinate,
    vtkLightKit,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkRenderer,
)
from vtkmodules.util import numpy_support
from matplotlib.path import Path as MplPath
from PyQt6.QtCore import pyqtSignal, QEvent, QPoint, Qt
from PyQt6.QtWidgets import QInputDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QApplication

from domain.ccta_display_types import LABEL_COLORS
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from tools.geometry import SplineGeometry

# ---------------------------------------------------------------------------
# Algorithm selection — prefer fastest available in installed VTK build
# ---------------------------------------------------------------------------
try:
    from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D as _SurfaceNets  # VTK ≥ 9.2
    from vtkmodules.vtkFiltersCore import vtkThreshold
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    _ALGO = 'surface_nets'
except ImportError:
    try:
        from vtkmodules.vtkFiltersCore import vtkDiscreteFlyingEdges3D as _DiscreteFE  # type: ignore[attr-defined]  # VTK ≥ 8.1

        _ALGO = 'flying_edges'
    except ImportError:
        from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes as _DiscreteFE

        _ALGO = 'marching_cubes'


class CctaViewer3D(QWidget):
    cursor_moved = pyqtSignal(int, int, int)  # z, y, x voxel coords
    mask_erased = pyqtSignal()  # 3-D lasso erase modified the mask

    def __init__(self, parent=None):
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)

        self._render_btn = QPushButton('Render 3D')
        self._render_btn.clicked.connect(self._on_render)

        self._lasso_btn = QPushButton('Lasso')
        self._lasso_btn.setCheckable(True)
        self._lasso_btn.setToolTip('Draw a closed lasso; right-click to erase selected label inside it')
        self._lasso_btn.toggled.connect(self._on_lasso_toggled)

        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(4, 2, 4, 2)
        btn_bar.addStretch()
        btn_bar.addWidget(self._lasso_btn)
        btn_bar.addWidget(self._render_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._vtk_widget, 1)
        layout.addLayout(btn_bar)

        self._ren = vtkRenderer()
        self._ren.SetBackground(0.0, 0.0, 0.0)
        self._ren.AutomaticLightCreationOff()  # disable headlight

        light_kit = vtkLightKit()
        light_kit.SetKeyLightElevation(30)  # degrees above horizon
        light_kit.SetKeyLightAzimuth(-60)  # from the left
        light_kit.SetKeyLightIntensity(1.0)
        light_kit.SetFillLightWarmth(0.4)  # cooler fill from the right
        light_kit.SetBackLightWarmth(0.35)
        light_kit.AddLightsToRenderer(self._ren)

        self._vtk_widget.GetRenderWindow().AddRenderer(self._ren)
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()
        trackball = vtkInteractorStyleTrackballCamera()
        trackball.SetMotionFactor(7.5)
        self._vtk_widget.SetInteractorStyle(trackball)

        self._mask: np.ndarray | None = None
        self._labels: list[int] = []
        self._voxel_spacing: tuple[float, float, float] | None = None
        self._actors: dict[int, vtkActor] = {}
        self._hidden_labels: set[int] = set()
        self._custom_colors: list[tuple[int, int, int]] | None = None
        self._crosshair_actor: vtkActor | None = None
        self._press_qt: QPoint = QPoint()

        # Lasso state
        self._lasso_mode: bool = False
        self._lasso_pts: list[tuple[int, int]] = []  # screen coords (VTK, y from bottom)
        self._lasso_overlay: list[vtkActor2D] = []  # actors to clean up

        self._vtk_widget.installEventFilter(self)

    def set_mask(
        self,
        mask: np.ndarray,
        labels: list[int],
        voxel_spacing: tuple[float, float, float],
    ) -> None:
        self._mask = mask
        self._labels = labels
        self._voxel_spacing = voxel_spacing
        self._hidden_labels = set()
        self._custom_colors = None
        self.clear_mesh()

    def set_label_colors(self, colors: list[tuple[int, int, int]]) -> None:
        self._custom_colors = list(colors)
        changed = False
        for i, label in enumerate(self._labels):
            if i < len(colors) and label in self._actors:
                r, g, b = colors[i]
                self._actors[label].GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
                changed = True
        if changed:
            self._vtk_widget.GetRenderWindow().Render()

    def set_label_visible(self, label: int, visible: bool) -> None:
        if visible:
            self._hidden_labels.discard(label)
        else:
            self._hidden_labels.add(label)
        actor = self._actors.get(label)
        if actor is not None:
            actor.SetVisibility(int(visible))
            self._vtk_widget.GetRenderWindow().Render()

    def shutdown(self) -> None:
        """Finalize the VTK OpenGL context while the HWND is still valid."""
        rw = self._vtk_widget.GetRenderWindow()
        rw.Finalize()
        rw.SetInteractor(None)

    def clear_mesh(self) -> None:
        for actor in self._actors.values():
            self._ren.RemoveActor(actor)
        self._actors.clear()
        if self._crosshair_actor is not None:
            self._ren.RemoveActor(self._crosshair_actor)
            self._crosshair_actor = None
        self._vtk_widget.GetRenderWindow().Render()

    # ── screen ↔ world ↔ voxel projection ──────────────────────────────────
    # These three utilities are the shared foundation for both the crosshair
    # pick (single ray) and the future lasso erase (polygon → depth extrusion).

    def voxel_to_world(self, z: int, y_vox: int, x_vox: int) -> tuple[float, float, float]:
        """Voxel index → VTK world coordinate (accounts for the Y-flip in _build_vtk_image)."""
        assert self._voxel_spacing is not None and self._mask is not None
        dz, dy, dx = self._voxel_spacing
        Y = self._mask.shape[1]
        return x_vox * dx, (Y - 1 - y_vox) * dy, z * dz

    def world_to_screen(self, wx: float, wy: float, wz: float) -> tuple[float, float]:
        """VTK world coordinate → screen pixel (sx, sy in VTK bottom-left convention)."""
        self._ren.SetWorldPoint(wx, wy, wz, 1.0)
        self._ren.WorldToDisplay()
        sx, sy, _ = self._ren.GetDisplayPoint()
        return sx, sy

    def screen_to_ray(self, sx: int, sy: int) -> tuple[np.ndarray, np.ndarray]:
        """Screen pixel → (near_world, far_world) defining the camera ray through that pixel."""

        def _display_to_world(depth: float) -> np.ndarray:
            self._ren.SetDisplayPoint(float(sx), float(sy), depth)
            self._ren.DisplayToWorld()
            wp = self._ren.GetWorldPoint()
            w = wp[3] if wp[3] != 0.0 else 1.0
            return np.array([wp[0] / w, wp[1] / w, wp[2] / w])

        return _display_to_world(0.0), _display_to_world(1.0)

    # ── crosshair pick (ray march through mask) ─────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self._vtk_widget:
            t = event.type()
            if self._lasso_mode:
                if t == QEvent.Type.MouseButtonPress:
                    vtk_y = self._vtk_widget.height() - 1 - event.pos().y()
                    if event.button() == Qt.MouseButton.LeftButton:
                        self._lasso_add_point(event.pos().x(), vtk_y)
                        return True  # block VTK camera move
                    if event.button() == Qt.MouseButton.RightButton:
                        self._lasso_execute()
                        return True
            else:
                if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    self._press_qt = event.pos()
                elif t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                    dp = event.pos() - self._press_qt
                    if abs(dp.x()) <= 3 and abs(dp.y()) <= 3:
                        vtk_y = self._vtk_widget.height() - 1 - event.pos().y()
                        self._click_pick(event.pos().x(), vtk_y)
        return super().eventFilter(obj, event)

    def set_cursor(self, z: int, y_vox: int, x_vox: int) -> None:
        """Place the 3-D marker at a voxel position (called from 2-D view crosshair moves)."""
        if self._mask is None or self._voxel_spacing is None:
            return
        wx, wy, wz = self.voxel_to_world(z, y_vox, x_vox)
        self._place_crosshair_marker(wx, wy, wz)

    def _click_pick(self, sx: int, sy: int) -> None:
        if self._mask is None or self._voxel_spacing is None:
            return
        hit = self.pick_nearest_along_ray(sx, sy)
        if hit is None:
            return
        z, y_vox, x_vox = hit
        wx, wy, wz = self.voxel_to_world(z, y_vox, x_vox)
        self._place_crosshair_marker(wx, wy, wz)
        self.cursor_moved.emit(z, y_vox, x_vox)

    def pick_nearest_along_ray(self, sx: int, sy: int) -> tuple[int, int, int] | None:
        """Cast a ray through screen pixel (sx, sy) and return the first non-zero
        mask voxel hit, as (z, y_vox, x_vox).  Returns None if the ray misses.

        This is the building block for the lasso erase: call world_to_screen() on
        every non-zero voxel and check if its projection falls inside the polygon.
        """
        assert self._mask is not None and self._voxel_spacing is not None
        dz, dy, dx = self._voxel_spacing
        Z, Y, X = self._mask.shape

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
            if 0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X:
                label = int(self._mask[zi, yi, xi])
                if label != 0 and label not in self._hidden_labels:
                    return zi, yi, xi

        return None

    def _place_crosshair_marker(self, wx: float, wy: float, wz: float) -> None:
        if self._crosshair_actor is not None:
            self._ren.RemoveActor(self._crosshair_actor)

        sphere = vtkSphereSource()
        sphere.SetCenter(wx, wy, wz)
        sphere.SetRadius(2.0)
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        sphere.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # yellow
        actor.GetProperty().SetOpacity(0.9)

        self._ren.AddActor(actor)
        self._crosshair_actor = actor
        self._vtk_widget.GetRenderWindow().Render()

    # ── lasso erase ────────────────────────────────────────────────────────

    _LASSO_CLOSE_PX = 15  # pixels — clicking within this radius of the first point closes the lasso

    def _on_lasso_toggled(self, checked: bool) -> None:
        if checked and (self._mask is None or self._voxel_spacing is None):
            ErrorMessage(self, 'Load a mask before using the lasso.')
            self._lasso_btn.blockSignals(True)
            self._lasso_btn.setChecked(False)
            self._lasso_btn.blockSignals(False)
            return
        self._lasso_mode = checked
        self._lasso_btn.setText('Cancel Lasso' if checked else 'Lasso')
        if not checked:
            self._lasso_clear()

    def _lasso_add_point(self, sx: int, sy: int) -> None:
        # Close the spline when clicking near the first control point (≥3 pts already placed)
        if len(self._lasso_pts) >= 3:
            fx, fy = self._lasso_pts[0]
            if abs(sx - fx) <= self._LASSO_CLOSE_PX and abs(sy - fy) <= self._LASSO_CLOSE_PX:
                self._lasso_execute()
                return
        self._lasso_pts.append((sx, sy))
        self._lasso_redraw()

    def _lasso_clear(self) -> None:
        self._lasso_pts.clear()
        for a in self._lasso_overlay:
            self._ren.RemoveActor2D(a)
        self._lasso_overlay.clear()
        self._vtk_widget.GetRenderWindow().Render()

    def _lasso_redraw(self) -> None:
        for a in self._lasso_overlay:
            self._ren.RemoveActor2D(a)
        self._lasso_overlay.clear()

        for sx, sy in self._lasso_pts:
            self._lasso_overlay.append(self._make_dot2d(sx, sy))

        if len(self._lasso_pts) >= 2:
            self._lasso_overlay.append(self._make_polyline2d(self._lasso_spline_pts()))

        for a in self._lasso_overlay:
            self._ren.AddActor2D(a)
        self._vtk_widget.GetRenderWindow().Render()

    def _lasso_spline_pts(self) -> list[tuple[float, float]]:
        """Interpolated polygon in screen space (SplineGeometry on control points)."""
        pts = self._lasso_pts
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

    def _lasso_execute(self) -> None:
        if len(self._lasso_pts) < 3:
            ErrorMessage(self, 'Draw at least 3 points to define a lasso.')
            return

        visible = [lbl for lbl in self._labels if lbl not in self._hidden_labels]
        if not visible:
            ErrorMessage(self, 'No visible labels to erase from.')
            self._lasso_btn.setChecked(False)
            return

        if len(visible) == 1:
            label = visible[0]
        else:
            chosen, ok = QInputDialog.getItem(
                self, 'Select label', 'Erase voxels of label:', [str(lbl) for lbl in visible], 0, False
            )
            if not ok:
                return
            label = int(chosen)

        polygon = np.array(self._lasso_spline_pts())
        self._erase_inside_lasso(label, polygon)
        self._lasso_btn.setChecked(False)  # triggers _on_lasso_toggled → _lasso_clear

    def _erase_inside_lasso(self, label: int, polygon: np.ndarray) -> None:
        assert self._mask is not None and self._voxel_spacing is not None
        dz, dy, dx = self._voxel_spacing
        _, Y, _ = self._mask.shape

        z_idx, y_idx, x_idx = np.where(self._mask == label)
        if len(z_idx) == 0:
            return

        wx = x_idx.astype(np.float64) * dx
        wy = (Y - 1 - y_idx).astype(np.float64) * dy  # undo Y-flip from _build_vtk_image
        wz = z_idx.astype(np.float64) * dz

        screen = self._project_world_batch(wx, wy, wz)
        inside = MplPath(polygon).contains_points(screen)
        self._mask[z_idx[inside], y_idx[inside], x_idx[inside]] = 0
        self.mask_erased.emit()
        self._rerender_after_erase()

    def _project_world_batch(self, wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
        """Vectorised world → VTK display-pixel projection. Returns (N, 2) float array."""
        vtk_mat = self._ren.GetActiveCamera().GetCompositeProjectionTransformMatrix(
            self._ren.GetTiledAspectRatio(), -1.0, 1.0
        )
        m = np.array([[vtk_mat.GetElement(r, c) for c in range(4)] for r in range(4)])

        world_h = np.stack([wx, wy, wz, np.ones(len(wx))], axis=1)  # (N, 4)
        clip = (m @ world_h.T).T  # (N, 4)

        w = np.where(np.abs(clip[:, 3]) > 1e-10, clip[:, 3], 1.0)
        ndc_x = clip[:, 0] / w
        ndc_y = clip[:, 1] / w

        vp = self._ren.GetViewport()
        W, H = self._vtk_widget.GetRenderWindow().GetSize()
        sx = (ndc_x + 1.0) * 0.5 * (vp[2] - vp[0]) * W + vp[0] * W
        sy = (ndc_y + 1.0) * 0.5 * (vp[3] - vp[1]) * H + vp[1] * H
        return np.column_stack([sx, sy])

    # ── 2D overlay helpers (VTK Actor2D in display coords) ──────────────────

    def _make_dot2d(self, sx: float, sy: float, size: int = 8) -> vtkActor2D:
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
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetPointSize(size)
        return actor

    def _make_polyline2d(self, pts: list[tuple[float, float]]) -> vtkActor2D:
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
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetLineWidth(2)
        return actor

    def _rerender_after_erase(self) -> None:
        if not self._actors:
            return
        camera = self._ren.GetActiveCamera()
        pos, focal, up = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()

        active = [(i, lbl) for i, lbl in enumerate(self._labels) if lbl not in self._hidden_labels]
        for _, lbl in active:
            actor = self._actors.pop(lbl, None)
            if actor is not None:
                self._ren.RemoveActor(actor)

        if active:
            vtk_img = self._build_vtk_image()
            if _ALGO == 'surface_nets':
                self._render_surface_nets(vtk_img, active)
            else:
                self._render_flying_edges(vtk_img, active)

        camera.SetPosition(pos)
        camera.SetFocalPoint(focal)
        camera.SetViewUp(up)
        self._vtk_widget.GetRenderWindow().Render()

    def _on_render(self) -> None:
        if self._mask is None or not self._labels or self._voxel_spacing is None:
            ErrorMessage(self, 'Load a mask before rendering the 3D view.')
            return

        # (color_index, label_value) for every label that is currently checked
        active = [(i, lbl) for i, lbl in enumerate(self._labels) if lbl not in self._hidden_labels]
        if not active:
            ErrorMessage(self, 'No labels are currently visible. Enable at least one label to render.')
            return

        self._render_btn.setEnabled(False)
        self._render_btn.setText('Rendering…')
        QApplication.processEvents()

        self.clear_mesh()

        vtk_img = self._build_vtk_image()

        if _ALGO == 'surface_nets':
            self._render_surface_nets(vtk_img, active)
        else:
            self._render_flying_edges(vtk_img, active)

        self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

        self._render_btn.setText('Render 3D')
        self._render_btn.setEnabled(True)

    def _build_vtk_image(self) -> vtkImageData:
        """
        Convert the (Z, Y, X) uint8 mask to vtkImageData.

        VTK stores scalars in x-fastest order. A numpy C-order (Z, Y, X)
        array already has X varying fastest in memory, so arr.ravel() is
        correct — do NOT transpose first.
        SetDimensions takes (nx, ny, nz) = (X, Y, Z).
        """
        assert self._mask is not None and self._voxel_spacing is not None
        dz, dy, dx = self._voxel_spacing
        Z, Y, X = self._mask.shape

        vtk_arr = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(self._mask[:, ::-1, :]).ravel(),  # x-fastest ✓; Y flipped to match 2-D views
            deep=True,
            array_type=numpy_support.get_vtk_array_type(np.uint8),
        )

        img = vtkImageData()
        img.SetDimensions(X, Y, Z)  # (nx, ny, nz)
        img.SetSpacing(dx, dy, dz)  # x=col, y=row, z=slice
        img.SetOrigin(0.0, 0.0, 0.0)
        img.GetPointData().SetScalars(vtk_arr)
        return img

    def _render_surface_nets(self, vtk_img: vtkImageData, active: list[tuple[int, int]]) -> None:
        """
        Single-pass extraction with vtkSurfaceNets3D (VTK ≥ 9.2).

        Runs once for all active labels, then splits per-label with vtkThreshold
        on the 2-component BoundaryLabels cell array (component 0 = inside label).
        This is O(volume) once, not O(volume x N_labels).
        """
        sn = _SurfaceNets()
        sn.SetInputData(vtk_img)
        for j, (_, label) in enumerate(active):
            sn.SetValue(j, float(label))
        sn.SetOutputMeshTypeToTriangles()
        sn.SmoothingOff()
        sn.Update()
        combined = sn.GetOutput()

        for color_index, label in active:
            QApplication.processEvents()

            thresh = vtkThreshold()
            thresh.SetInputData(combined)
            thresh.SetInputArrayToProcess(
                0,
                0,
                0,
                vtkImageData.FIELD_ASSOCIATION_CELLS,
                'BoundaryLabels',
            )
            try:
                thresh.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
                thresh.SetLowerThreshold(float(label) - 0.5)
                thresh.SetUpperThreshold(float(label) + 0.5)
            except AttributeError:
                thresh.ThresholdBetween(float(label) - 0.5, float(label) + 0.5)  # type: ignore[attr-defined]
            thresh.SetSelectedComponent(0)
            thresh.Update()

            geom = vtkGeometryFilter()
            geom.SetInputConnection(thresh.GetOutputPort())
            geom.Update()

            if geom.GetOutput().GetNumberOfPoints() == 0:
                continue

            self._add_actor(color_index, label, geom.GetOutputPort())

    def _render_flying_edges(self, vtk_img: vtkImageData, active: list[tuple[int, int]]) -> None:
        """
        Per-label extraction with vtkDiscreteFlyingEdges3D (multi-threaded).
        Runs once per label but each pass is fast (O(volume), multi-threaded).
        """
        for color_index, label in active:
            QApplication.processEvents()

            fe = _DiscreteFE()
            fe.SetInputData(vtk_img)
            fe.SetValue(0, float(label))
            fe.ComputeScalarsOff()
            fe.ComputeNormalsOff()
            fe.ComputeGradientsOff()
            fe.Update()

            if fe.GetOutput().GetNumberOfPoints() == 0:
                continue

            self._add_actor(color_index, label, fe.GetOutputPort())

    def _add_actor(self, color_index: int, label: int, output_port) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(output_port)
        mapper.ScalarVisibilityOff()

        actor = vtkActor()
        actor.SetMapper(mapper)
        if self._custom_colors and color_index < len(self._custom_colors):
            r, g, b = self._custom_colors[color_index]
        else:
            r, g, b = LABEL_COLORS[color_index % len(LABEL_COLORS)]
        actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetInterpolationToFlat()

        self._ren.AddActor(actor)
        self._actors[label] = actor
