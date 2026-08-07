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
    """Column 1: load the CCTA mesh + centerlines, prepare the centerlines, run
    label_geometry + label_branches_pair, then discretize_vessel_tree. See
    pages/fusion/pipeline.py for the multimodars calls each button triggers."""

    run_label_geometry_requested = pyqtSignal()
    prepare_centerlines_requested = pyqtSignal()
    run_label_branches_pair_requested = pyqtSignal()
    run_discretize_tree_requested = pyqtSignal()
    geometry_files_changed = pyqtSignal()  # mesh and/or centerline path(s) changed

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
        root.addWidget(self._build_prepare_centerline_group())
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
        for key, label in [('aorta', 'Aorta'), ('lca', 'LCA'), ('rca', 'RCA')]:
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

    def _build_prepare_centerline_group(self) -> QGroupBox:
        box = QGroupBox('Prepare Centerline')
        layout = QVBoxLayout(box)

        resample_row = QHBoxLayout()
        self._pc_resample = QCheckBox('Resample to spacing (mm):')
        self._pc_resample.setChecked(True)
        self._pc_spacing_mm = QDoubleSpinBox()
        self._pc_spacing_mm.setRange(0.01, 10.0)
        self._pc_spacing_mm.setSingleStep(0.1)
        self._pc_spacing_mm.setValue(1.0)
        resample_row.addWidget(self._pc_resample)
        resample_row.addWidget(self._pc_spacing_mm, 1)
        layout.addLayout(resample_row)

        self._pc_branch_tolerance = QDoubleSpinBox()
        self._pc_branch_tolerance.setRange(0.1, 20.0)
        self._pc_branch_tolerance.setSingleStep(0.1)
        self._pc_branch_tolerance.setValue(2.0)
        layout.addLayout(_row('Branch spacing tolerance (mm):', self._pc_branch_tolerance))

        self._pc_rm_start_mm = QDoubleSpinBox()
        self._pc_rm_start_mm.setRange(0.0, 50.0)
        self._pc_rm_start_mm.setSingleStep(0.5)
        self._pc_rm_start_mm.setValue(5.0)
        self._pc_rm_start_mm.setToolTip('Applied to RCA/LCA only — the aorta has no inlet to trim.')
        layout.addLayout(_row('Remove start — RCA/LCA (mm):', self._pc_rm_start_mm))

        self._pc_smooth_sigma = QDoubleSpinBox()
        self._pc_smooth_sigma.setRange(0.0, 20.0)
        self._pc_smooth_sigma.setSingleStep(0.1)
        self._pc_smooth_sigma.setValue(2.5)
        layout.addLayout(_row('Smoothing sigma:', self._pc_smooth_sigma))

        prepare_btn = QPushButton('Prepare Centerlines')
        prepare_btn.setToolTip(
            'Load + branch/order/smooth all three centerlines (aorta first, then RCA/LCA oriented to it)'
        )
        prepare_btn.clicked.connect(self.prepare_centerlines_requested.emit)
        layout.addWidget(prepare_btn)
        return box

    def _build_label_geometry_group(self) -> QGroupBox:
        box = QGroupBox('Label Geometry')
        layout = QVBoxLayout(box)

        self._bounding_sphere_radius_rca = QDoubleSpinBox()
        self._bounding_sphere_radius_rca.setRange(0.1, 50.0)
        self._bounding_sphere_radius_rca.setSingleStep(0.5)
        self._bounding_sphere_radius_rca.setValue(3.0)
        layout.addLayout(_row('RCA bounding sphere (mm):', self._bounding_sphere_radius_rca))

        self._bounding_sphere_radius_lca = QDoubleSpinBox()
        self._bounding_sphere_radius_lca.setRange(0.1, 50.0)
        self._bounding_sphere_radius_lca.setSingleStep(0.5)
        self._bounding_sphere_radius_lca.setValue(3.0)
        layout.addLayout(_row('LCA bounding sphere (mm):', self._bounding_sphere_radius_lca))

        self._step_size_labeling = QDoubleSpinBox()
        self._step_size_labeling.setRange(0.01, 10.0)
        self._step_size_labeling.setSingleStep(0.1)
        self._step_size_labeling.setValue(1.0)
        layout.addLayout(_row('Step size (mm):', self._step_size_labeling))

        self._acute_takeoff_rca = QCheckBox('Acute Takeoff RCA')
        self._acute_takeoff_lca = QCheckBox('Acute Takeoff LCA')
        layout.addWidget(self._acute_takeoff_rca)
        layout.addWidget(self._acute_takeoff_lca)

        self._range_mm_takeoff_rca = QDoubleSpinBox()
        self._range_mm_takeoff_rca.setRange(0.1, 200.0)
        self._range_mm_takeoff_rca.setSingleStep(1.0)
        self._range_mm_takeoff_rca.setValue(60.0)
        layout.addLayout(_row('RCA takeoff range (mm):', self._range_mm_takeoff_rca))

        self._range_mm_takeoff_lca = QDoubleSpinBox()
        self._range_mm_takeoff_lca.setRange(0.1, 200.0)
        self._range_mm_takeoff_lca.setSingleStep(1.0)
        self._range_mm_takeoff_lca.setValue(60.0)
        layout.addLayout(_row('LCA takeoff range (mm):', self._range_mm_takeoff_lca))

        run_btn = QPushButton('Run Label Geometry')
        run_btn.clicked.connect(self.run_label_geometry_requested.emit)
        layout.addWidget(run_btn)

        label_branches_btn = QPushButton('Label Branches (Pair)')
        label_branches_btn.setToolTip(
            'Project the prepared RCA/LCA branch structure onto the labelled surface points. '
            'Re-run after editing branches on the Centerline Branches tab.'
        )
        label_branches_btn.clicked.connect(self.run_label_branches_pair_requested.emit)
        layout.addWidget(label_branches_btn)
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
            self.geometry_files_changed.emit()

    def _on_browse_centerline(self, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f'Open {key.upper()} Centerline', self._default_dir, 'VTP files (*.vtp)'
        )
        if path:
            self.centerline_paths[key] = path
            self._centerline_edits[key].setText(path)
            self.geometry_files_changed.emit()

    # ------------------------------------------------------------------
    # Param getters — read by FusionPage when handling the *_requested signals
    # ------------------------------------------------------------------

    def prepare_centerline_kwargs(self, key: str) -> dict:
        """key is 'aorta'/'rca'/'lca'. spacing_mm/branch_spacing_tolerance/smooth_sigma are
        shared across all three; rm_start_mm trims a coronary's inlet and must never be
        applied to the aortic centerline, which has no such inlet."""
        return {
            'spacing_mm': self._pc_spacing_mm.value() if self._pc_resample.isChecked() else None,
            'branch_spacing_tolerance': self._pc_branch_tolerance.value(),
            'rm_start_mm': 0.0 if key == 'aorta' else self._pc_rm_start_mm.value(),
            'smooth_sigma': self._pc_smooth_sigma.value(),
        }

    def label_geometry_kwargs(self) -> dict:
        return {
            'acute_takeoff_rca': self._acute_takeoff_rca.isChecked(),
            'acute_takeoff_lca': self._acute_takeoff_lca.isChecked(),
            'range_mm_takeoff_rca': self._range_mm_takeoff_rca.value(),
            'range_mm_takeoff_lca': self._range_mm_takeoff_lca.value(),
            'step_size_mm': self._step_size_labeling.value(),
            'bounding_sphere_radius_mm_rca': self._bounding_sphere_radius_rca.value(),
            'bounding_sphere_radius_mm_lca': self._bounding_sphere_radius_lca.value(),
        }

    def discretize_tree_kwargs(self) -> dict:
        return {
            'step_size': self._step_size.value(),
            'n_points': self._n_points_tree.value(),
            'b_spline': self._b_spline.isChecked(),
            'bspline_smoothing': self._bspline_smoothing.value(),
        }

    def has_acute_takeoff(self) -> bool:
        """Whether either coronary was marked with an acute takeoff — drives
        align_wall_anomalous in column 2 automatically instead of a separate manual
        toggle there."""
        return self._acute_takeoff_rca.isChecked() or self._acute_takeoff_lca.isChecked()


def _row(label: str, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.addWidget(QLabel(label))
    layout.addWidget(widget, 1)
    return layout
