import numpy as np
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkActor, vtkLightKit, vtkPolyDataMapper, vtkRenderer
from vtkmodules.util import numpy_support
from PyQt6.QtCore import pyqtSignal, QEvent, QPoint, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QApplication

from domain.ccta_display_types import LABEL_COLORS
from pages.intravascular.popup_windows.message_boxes import ErrorMessage

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

    def __init__(self, parent=None):
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)

        self._render_btn = QPushButton('Render 3D')
        self._render_btn.clicked.connect(self._on_render)

        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(4, 2, 4, 2)
        btn_bar.addStretch()
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
        self._crosshair_actor: vtkActor | None = None
        self._press_qt: QPoint = QPoint()

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
        self.clear_mesh()

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
            if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._press_qt = event.pos()
                print(f'[3D] Qt press at {event.pos().x()}, {event.pos().y()}')
            elif t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                dp = event.pos() - self._press_qt
                print(f'[3D] Qt release at {event.pos().x()}, {event.pos().y()}, delta {dp.x()}, {dp.y()}')
                if abs(dp.x()) <= 3 and abs(dp.y()) <= 3:
                    # Qt y=0 is top; VTK y=0 is bottom
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
        print(f'[3D] _click_pick vtk=({sx},{sy}), mask={self._mask is not None}, spacing={self._voxel_spacing}')
        if self._mask is None or self._voxel_spacing is None:
            return
        hit = self.pick_nearest_along_ray(sx, sy)
        print(f'[3D] ray hit: {hit}')
        if hit is None:
            return
        z, y_vox, x_vox = hit
        wx, wy, wz = self.voxel_to_world(z, y_vox, x_vox)
        print(f'[3D] emitting cursor_moved({z}, {y_vox}, {x_vox}), world=({wx:.1f},{wy:.1f},{wz:.1f})')
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
        print(f'[3D] ray near={near}, far={far}')
        ray = far - near
        length = float(np.linalg.norm(ray))
        print(
            f'[3D] ray length={length:.1f}, step={min(dx,dy,dz)*0.5:.3f}, n_steps={int(length/(min(dx,dy,dz)*0.5))+1}'
        )
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
        r, g, b = LABEL_COLORS[color_index % len(LABEL_COLORS)]
        actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetInterpolationToFlat()

        self._ren.AddActor(actor)
        self._actors[label] = actor
