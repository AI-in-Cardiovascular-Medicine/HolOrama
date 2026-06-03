import numpy as np
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkActor, vtkLightKit, vtkPolyDataMapper, vtkRenderer
from vtkmodules.util import numpy_support
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication

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

_BTN_MARGIN = 8


class CctaViewer3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._vtk_widget)

        self._render_btn = QPushButton('Render 3D', self)
        self._render_btn.adjustSize()
        self._render_btn.clicked.connect(self._on_render)
        self._render_btn.raise_()

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

        self._mask: np.ndarray | None = None
        self._labels: list[int] = []
        self._voxel_spacing: tuple[float, float, float] | None = None
        self._actors: dict[int, vtkActor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mask(
        self,
        mask: np.ndarray,
        labels: list[int],
        voxel_spacing: tuple[float, float, float],
    ) -> None:
        self._mask = mask
        self._labels = labels
        self._voxel_spacing = voxel_spacing
        self.clear_mesh()

    def set_label_visible(self, label: int, visible: bool) -> None:
        actor = self._actors.get(label)
        if actor is not None:
            actor.SetVisibility(int(visible))
            self._vtk_widget.GetRenderWindow().Render()

    def clear_mesh(self) -> None:
        for actor in self._actors.values():
            self._ren.RemoveActor(actor)
        self._actors.clear()
        self._vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    # Render slot
    # ------------------------------------------------------------------

    def _on_render(self) -> None:
        if self._mask is None or not self._labels or self._voxel_spacing is None:
            ErrorMessage(self, 'Load a mask before rendering the 3D view.')
            return

        self._render_btn.setEnabled(False)
        self._render_btn.setText('Rendering…')
        QApplication.processEvents()

        self.clear_mesh()

        vtk_img = self._build_vtk_image()

        if _ALGO == 'surface_nets':
            self._render_surface_nets(vtk_img)
        else:
            self._render_flying_edges(vtk_img)

        self._ren.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

        self._render_btn.setText('Render 3D')
        self._render_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # vtkImageData construction
    # ------------------------------------------------------------------

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
            np.ascontiguousarray(self._mask).ravel(),  # x-fastest ✓
            deep=True,
            array_type=numpy_support.get_vtk_array_type(np.uint8),
        )

        img = vtkImageData()
        img.SetDimensions(X, Y, Z)  # (nx, ny, nz)
        img.SetSpacing(dx, dy, dz)  # x=col, y=row, z=slice
        img.SetOrigin(0.0, 0.0, 0.0)
        img.GetPointData().SetScalars(vtk_arr)
        return img

    # ------------------------------------------------------------------
    # Algorithm implementations
    # ------------------------------------------------------------------

    def _render_surface_nets(self, vtk_img: vtkImageData) -> None:
        """
        Single-pass extraction with vtkSurfaceNets3D (VTK ≥ 9.2).

        Runs once for all labels, then splits per-label with vtkThreshold
        on the 2-component BoundaryLabels cell array (component 0 = inside label).
        This is O(volume) once, not O(volume x N_labels).
        """
        sn = _SurfaceNets()
        sn.SetInputData(vtk_img)
        for i, label in enumerate(self._labels):
            sn.SetValue(i, float(label))
        sn.SetOutputMeshTypeToTriangles()
        sn.SmoothingOff()
        sn.Update()
        combined = sn.GetOutput()

        for i, label in enumerate(self._labels):
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
            # Keep cells whose component-0 (inside label) equals this label
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

            self._add_actor(i, label, geom.GetOutputPort())

    def _render_flying_edges(self, vtk_img: vtkImageData) -> None:
        """
        Per-label extraction with vtkDiscreteFlyingEdges3D (multi-threaded).
        Runs once per label but each pass is fast (O(volume), multi-threaded).
        """
        for i, label in enumerate(self._labels):
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

            self._add_actor(i, label, fe.GetOutputPort())

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

    # ------------------------------------------------------------------
    # Qt event handlers
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_button()

    def _reposition_button(self) -> None:
        btn = self._render_btn
        btn.move(
            self.width() - btn.width() - _BTN_MARGIN,
            self.height() - btn.height() - _BTN_MARGIN,
        )
