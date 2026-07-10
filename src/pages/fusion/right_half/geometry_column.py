from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class GeometryColumn(QWidget):
    """Column 1: load the CCTA mesh + centerlines, run label_geometry, then
    discretize_vessel_tree. See pages/fusion/pipeline.py for the multimodars calls
    each button triggers."""

    run_label_geometry_requested = pyqtSignal()
    run_prepare_centerlines_requested = pyqtSignal()
    run_discretize_tree_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mesh_path: str | None = None
        self.centerline_paths: dict[str, str] = {}
        self._default_dir: str = ''

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        title = QLabel('CCTA Geometry && Centerlines')
        title.setStyleSheet('font-weight: bold;')
        root.addWidget(title)

        root.addWidget(self._build_mesh_group())
        root.addWidget(self._build_centerline_group())
        root.addWidget(self._build_label_geometry_group())
        root.addWidget(self._build_vessel_tree_group())
        root.addStretch(1)

    # ------------------------------------------------------------------

    def _build_mesh_group(self) -> QGroupBox:
        box = QGroupBox('CCTA Mesh')
        layout = QHBoxLayout(box)
        self._mesh_edit = QLineEdit()
        self._mesh_edit.setReadOnly(True)
        self._mesh_edit.setPlaceholderText('No STL loaded')
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self._on_browse_mesh)
        layout.addWidget(self._mesh_edit, 1)
        layout.addWidget(browse_btn)
        return box

    def _build_centerline_group(self) -> QGroupBox:
        box = QGroupBox('Centerlines (.vtp)')
        layout = QVBoxLayout(box)
        self._centerline_edits: dict[str, QLineEdit] = {}
        for key, label in [('aorta', 'Aorta'), ('rca', 'RCA'), ('lca', 'LCA')]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f'{label}:'))
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText('Not loaded')
            browse_btn = QPushButton('Browse…')
            browse_btn.clicked.connect(lambda _, k=key: self._on_browse_centerline(k))
            row.addWidget(edit, 1)
            row.addWidget(browse_btn)
            layout.addLayout(row)
            self._centerline_edits[key] = edit
        return box

    def _build_label_geometry_group(self) -> QGroupBox:
        box = QGroupBox('Label Geometry')
        layout = QVBoxLayout(box)

        self._n_points_intramural = QSpinBox()
        self._n_points_intramural.setRange(1, 1000)
        self._n_points_intramural.setValue(100)
        layout.addLayout(_row('Intramural points:', self._n_points_intramural))

        self._bounding_sphere_radius = QDoubleSpinBox()
        self._bounding_sphere_radius.setRange(0.1, 50.0)
        self._bounding_sphere_radius.setSingleStep(0.5)
        self._bounding_sphere_radius.setValue(3.0)
        layout.addLayout(_row('Bounding sphere (mm):', self._bounding_sphere_radius))

        self._anomalous_rca = QCheckBox('Anomalous RCA')
        self._anomalous_lca = QCheckBox('Anomalous LCA')
        layout.addWidget(self._anomalous_rca)
        layout.addWidget(self._anomalous_lca)

        run_btn = QPushButton('Run Label Geometry')
        run_btn.clicked.connect(self.run_label_geometry_requested.emit)
        layout.addWidget(run_btn)

        prepare_btn = QPushButton('Prepare Centerlines')
        prepare_btn.setToolTip('Compute branches + validate/label both coronary centerlines')
        prepare_btn.clicked.connect(self.run_prepare_centerlines_requested.emit)
        layout.addWidget(prepare_btn)
        return box

    def _build_vessel_tree_group(self) -> QGroupBox:
        box = QGroupBox('Discretize Vessel Tree')
        layout = QVBoxLayout(box)

        self._step_size = QDoubleSpinBox()
        self._step_size.setRange(0.01, 10.0)
        self._step_size.setSingleStep(0.1)
        self._step_size.setValue(1.0)
        layout.addLayout(_row('Step size (mm):', self._step_size))

        self._n_points_tree = QSpinBox()
        self._n_points_tree.setRange(3, 1000)
        self._n_points_tree.setValue(100)
        layout.addLayout(_row('Points per contour:', self._n_points_tree))

        self._b_spline = QCheckBox('B-spline smoothing')
        self._b_spline.setChecked(True)
        layout.addWidget(self._b_spline)

        self._bspline_smoothing = QDoubleSpinBox()
        self._bspline_smoothing.setRange(0.0, 1000.0)
        self._bspline_smoothing.setValue(5.0)
        layout.addLayout(_row('Smoothing factor:', self._bspline_smoothing))

        run_btn = QPushButton('Discretize Vessel Tree')
        run_btn.clicked.connect(self.run_discretize_tree_requested.emit)
        layout.addWidget(run_btn)
        return box

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------

    def set_default_dir(self, path: str) -> None:
        self._default_dir = path

    def _on_browse_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open CCTA Mesh', self._default_dir, 'Mesh files (*.stl *.obj *.ply);;All Files (*)'
        )
        if path:
            self.mesh_path = path
            self._mesh_edit.setText(path)

    def _on_browse_centerline(self, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f'Open {key.upper()} Centerline', self._default_dir, 'VTP files (*.vtp)'
        )
        if path:
            self.centerline_paths[key] = path
            self._centerline_edits[key].setText(path)

    # ------------------------------------------------------------------
    # Param getters — read by FusionPage when handling the *_requested signals
    # ------------------------------------------------------------------

    def label_geometry_kwargs(self) -> dict:
        return {
            'anomalous_rca': self._anomalous_rca.isChecked(),
            'anomalous_lca': self._anomalous_lca.isChecked(),
            'n_points_intramural': self._n_points_intramural.value(),
            'bounding_sphere_radius_mm': self._bounding_sphere_radius.value(),
        }

    def discretize_tree_kwargs(self) -> dict:
        return {
            'step_size': self._step_size.value(),
            'n_points': self._n_points_tree.value(),
            'b_spline': self._b_spline.isChecked(),
            'bspline_smoothing': self._bspline_smoothing.value(),
        }


def _row(label: str, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.addWidget(QLabel(label))
    layout.addWidget(widget, 1)
    return layout
