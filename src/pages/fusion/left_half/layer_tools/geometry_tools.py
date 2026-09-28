from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout

from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class GeometryToolbar(SceneToolbar):
    """Toolbar for the CCTA Geometry scene: mesh + centerline layer toggles, plus the
    sphere-brush local smoothing of the final (remeshed) mesh.

    Sphere Smooth: hover over the final mesh to preview the brush (translucent sphere +
    the vertices it would move, highlighted orange), click to smooth them. Ctrl+wheel
    resizes the brush; camera rotate/zoom keep working as usual. Undo (Ctrl+Z) steps back
    through the local smoothing strokes (and whole-mesh Smooth runs) since the last
    Fix & Remesh. See FusionPage._on_sphere_* and pipeline.run_local_smooth.
    """

    sphere_mode_toggled = pyqtSignal(bool)
    sphere_radius_changed = pyqtSignal(float)
    undo_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        self.sphere_btn = QPushButton('Sphere Smooth')
        self.sphere_btn.setCheckable(True)
        self.sphere_btn.setToolTip(
            'Hover over the final mesh to preview, click to smooth the highlighted patch.\n'
            'Ctrl+wheel changes the radius. Needs Fix & Remesh first.'
        )

        self.sphere_radius = QDoubleSpinBox()
        self.sphere_radius.setRange(0.2, 20.0)
        self.sphere_radius.setSingleStep(0.2)
        self.sphere_radius.setValue(2.0)
        self.sphere_radius.setSuffix(' mm')
        self.sphere_radius.setToolTip('Brush radius, measured along the surface (Ctrl+wheel in the view)')

        self.sphere_iterations = QSpinBox()
        self.sphere_iterations.setRange(1, 100)
        self.sphere_iterations.setValue(10)
        self.sphere_iterations.setToolTip('Taubin iterations per click (uses the Taubin lambda from Remesh and Smooth)')

        self.undo_btn = QPushButton('Undo')
        self.undo_btn.setToolTip('Undo the last smoothing step (Ctrl+Z)')
        self.undo_btn.setEnabled(False)

        smooth_box = QGroupBox('Local Smooth')
        smooth_layout = QVBoxLayout(smooth_box)
        top = QHBoxLayout()
        top.addWidget(self.sphere_btn)
        top.addWidget(self.undo_btn)
        smooth_layout.addLayout(top)
        params = QHBoxLayout()
        params.addWidget(QLabel('Radius:'))
        params.addWidget(self.sphere_radius)
        params.addWidget(QLabel('Iterations:'))
        params.addWidget(self.sphere_iterations)
        smooth_layout.addLayout(params)

        super().__init__(FusionScene.CCTA_GEOMETRY, extra_rows=[smooth_box], show_lasso=True, parent=parent)
        # Connected only now: a QWidget's own signals don't exist before its __init__.
        self.sphere_btn.toggled.connect(self._on_sphere_toggled)
        self.sphere_radius.valueChanged.connect(self.sphere_radius_changed.emit)
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        # Lasso and sphere brush both take over left-clicks in the view — only one at a time.
        self.lasso_btn.toggled.connect(self._on_lasso_checked)

    def _on_lasso_checked(self, checked: bool) -> None:
        if checked:
            self.sphere_btn.setChecked(False)

    def _on_sphere_toggled(self, checked: bool) -> None:
        if checked:
            self.lasso_btn.setChecked(False)
        self.sphere_mode_toggled.emit(checked)

    def step_sphere_radius(self, steps: int) -> None:
        self.sphere_radius.stepBy(steps)

    def set_undo_available(self, available: bool) -> None:
        self.undo_btn.setEnabled(available)
