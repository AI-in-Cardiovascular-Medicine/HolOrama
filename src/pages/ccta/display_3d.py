import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton

_BTN_MARGIN = 8


class _3DViewerCCTA(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._vtk_widget)

        # Overlay button — parented to self, not added to layout
        self._play_btn = QPushButton('Play', self)
        self._play_btn.adjustSize()
        self._play_btn.clicked.connect(self._on_play)
        self._play_btn.raise_()

        self._colors = vtkNamedColors()
        self._ren = vtkRenderer()
        self._ren.SetBackground(self._colors.GetColor3d("BkgColor"))
        self._vtk_widget.GetRenderWindow().AddRenderer(self._ren)
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()

    def _on_play(self) -> None:
        self._ren.RemoveAllViewProps()

        cylinder = vtkCylinderSource()
        cylinder.SetResolution(8)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self._colors.GetColor3d("Tomato"))
        actor.RotateX(30.0)
        actor.RotateY(-45.0)

        self._ren.AddActor(actor)
        self._ren.ResetCamera()
        self._ren.GetActiveCamera().Zoom(1.5)
        self._vtk_widget.GetRenderWindow().Render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_button()

    def _reposition_button(self) -> None:
        btn = self._play_btn
        btn.move(
            self.width() - btn.width() - _BTN_MARGIN,
            self.height() - btn.height() - _BTN_MARGIN,
        )
