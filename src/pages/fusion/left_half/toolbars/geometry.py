from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout

from domain.fusion.types import FusionScene
from pages.fusion.left_half.toolbars.base import SceneToolbar


class GeometryToolbar(SceneToolbar):
    """Toolbar for the CCTA Geometry scene: layer toggles, lasso and sphere-brush local
    smoothing of the final (remeshed) mesh, with undo."""

    sphere_mode_toggled = pyqtSignal(bool)
    sphere_radius_changed = pyqtSignal(float)
    undo_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        self._sphere_btn = QPushButton('Sphere Smooth')
        self._sphere_btn.setCheckable(True)
        self._sphere_btn.setToolTip(
            'Hover over the final mesh to preview, click to smooth the highlighted patch.\n'
            'Ctrl+wheel changes the radius. Needs Fix & Remesh first.'
        )

        self._sphere_radius = QDoubleSpinBox()
        self._sphere_radius.setRange(0.2, 20.0)
        self._sphere_radius.setSingleStep(0.2)
        self._sphere_radius.setValue(2.0)
        self._sphere_radius.setSuffix(' mm')
        self._sphere_radius.setToolTip('Brush radius, measured along the surface (Ctrl+wheel in the view)')

        self._sphere_iterations = QSpinBox()
        self._sphere_iterations.setRange(1, 100)
        self._sphere_iterations.setValue(10)
        self._sphere_iterations.setToolTip(
            'Taubin iterations per click (uses the Taubin lambda from Remesh and Smooth)'
        )

        self._undo_btn = QPushButton('Undo')
        self._undo_btn.setToolTip('Undo the last smoothing step (Ctrl+Z)')
        self._undo_btn.setEnabled(False)

        smooth_box = QGroupBox('Local Smooth')
        smooth_layout = QVBoxLayout(smooth_box)
        top = QHBoxLayout()
        top.addWidget(self._sphere_btn)
        top.addWidget(self._undo_btn)
        smooth_layout.addLayout(top)
        params = QHBoxLayout()
        params.addWidget(QLabel('Radius:'))
        params.addWidget(self._sphere_radius)
        params.addWidget(QLabel('Iterations:'))
        params.addWidget(self._sphere_iterations)
        smooth_layout.addLayout(params)

        super().__init__(FusionScene.CCTA_GEOMETRY, extra_rows=[smooth_box], show_lasso=True, parent=parent)
        self._sphere_btn.toggled.connect(self._on_sphere_toggled)
        self._sphere_radius.valueChanged.connect(self.sphere_radius_changed.emit)
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        assert self._lasso_btn is not None
        self._lasso_btn.toggled.connect(self._on_lasso_checked)

    def _on_lasso_checked(self, checked: bool) -> None:
        if checked:
            self._sphere_btn.setChecked(False)

    def _on_sphere_toggled(self, checked: bool) -> None:
        if checked:
            self.end_lasso()
        self.sphere_mode_toggled.emit(checked)

    def step_sphere_radius(self, steps: int) -> None:
        self._sphere_radius.stepBy(steps)

    def set_undo_available(self, available: bool) -> None:
        self._undo_btn.setEnabled(available)

    def end_sphere_brush(self) -> None:
        """Uncheck Sphere Smooth, which turns the viewer's sphere mode off."""
        self._sphere_btn.setChecked(False)

    def sphere_radius_mm(self) -> float:
        return self._sphere_radius.value()

    def sphere_iterations(self) -> int:
        return self._sphere_iterations.value()
