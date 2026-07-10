from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FusionColumn(QWidget):
    """Column 3: scale the CCTA geometry to match the intravascular lumen, stitch the two
    together, then remesh/smooth/export. See pages/fusion/pipeline.py."""

    run_label_anomalous_requested = pyqtSignal()
    run_compute_scaling_requested = pyqtSignal()
    run_apply_scaling_requested = pyqtSignal()
    run_remove_points_requested = pyqtSignal()
    run_stitch_requested = pyqtSignal()
    run_remesh_requested = pyqtSignal()
    run_smooth_requested = pyqtSignal()
    export_requested = pyqtSignal(str)  # chosen file path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        title = QLabel('Fusion')
        title.setStyleSheet('font-weight: bold;')
        root.addWidget(title)

        root.addWidget(self._build_anomalous_group())
        root.addWidget(self._build_scaling_group())
        root.addWidget(self._build_cleanup_group())
        root.addWidget(self._build_stitch_group())
        root.addWidget(self._build_remesh_group())
        root.addWidget(self._build_export_group())
        root.addStretch(1)

    # ------------------------------------------------------------------

    def _build_anomalous_group(self) -> QGroupBox:
        box = QGroupBox('Anomalous Region')
        layout = QVBoxLayout(box)
        btn = QPushButton('Label Anomalous Region')
        btn.clicked.connect(self.run_label_anomalous_requested.emit)
        layout.addWidget(btn)
        return box

    def _build_scaling_group(self) -> QGroupBox:
        box = QGroupBox('Scaling Factors')
        layout = QVBoxLayout(box)
        self._scaling_labels = {
            'proximal_scaling': QLabel('Proximal: —'),
            'distal_scaling': QLabel('Distal: —'),
            'aortic_scaling': QLabel('Aortic: —'),
            'aortic_wall_scaling': QLabel('Aortic wall: —'),
        }
        for lbl in self._scaling_labels.values():
            layout.addWidget(lbl)

        compute_btn = QPushButton('Compute Scaling Factors')
        compute_btn.clicked.connect(self.run_compute_scaling_requested.emit)
        layout.addWidget(compute_btn)

        apply_btn = QPushButton('Apply Scaling to Mesh')
        apply_btn.setToolTip('Morphs the distal, aortic, and proximal regions along their centerlines in sequence')
        apply_btn.clicked.connect(self.run_apply_scaling_requested.emit)
        layout.addWidget(apply_btn)
        return box

    def _build_cleanup_group(self) -> QGroupBox:
        box = QGroupBox('Remove Labeled Points')
        layout = QVBoxLayout(box)
        self._remove_anomalous = QCheckBox('anomalous_points')
        self._remove_anomalous.setChecked(True)
        self._remove_proximal = QCheckBox('proximal_points')
        self._remove_proximal.setChecked(True)
        layout.addWidget(self._remove_anomalous)
        layout.addWidget(self._remove_proximal)
        btn = QPushButton('Remove')
        btn.clicked.connect(self.run_remove_points_requested.emit)
        layout.addWidget(btn)
        return box

    def _build_stitch_group(self) -> QGroupBox:
        box = QGroupBox('Stitch')
        layout = QVBoxLayout(box)

        self._prox_start_mode = QComboBox()
        self._prox_start_mode.addItems(['nearest_iv', 'highest_z'])
        layout.addLayout(_row('Proximal start mode:', self._prox_start_mode))

        self._dist_start_mode = QComboBox()
        self._dist_start_mode.addItems(['nearest_iv', 'highest_z'])
        layout.addLayout(_row('Distal start mode:', self._dist_start_mode))

        self._clamp_overshoot = QDoubleSpinBox()
        self._clamp_overshoot.setRange(0.0, 10.0)
        self._clamp_overshoot.setSingleStep(0.1)
        self._clamp_overshoot.setValue(0.5)
        layout.addLayout(_row('Clamp overshoot (mm):', self._clamp_overshoot))

        btn = QPushButton('Stitch')
        btn.clicked.connect(self.run_stitch_requested.emit)
        layout.addWidget(btn)
        return box

    def _build_remesh_group(self) -> QGroupBox:
        box = QGroupBox('Remesh && Smooth')
        layout = QVBoxLayout(box)

        self._target_edge_length = QDoubleSpinBox()
        self._target_edge_length.setRange(0.01, 10.0)
        self._target_edge_length.setSingleStep(0.05)
        self._target_edge_length.setValue(0.5)
        layout.addLayout(_row('Target edge length (mm):', self._target_edge_length))

        self._remesh_iterations = QSpinBox()
        self._remesh_iterations.setRange(1, 100)
        self._remesh_iterations.setValue(10)
        layout.addLayout(_row('Iterations:', self._remesh_iterations))

        self._remesh_verbose = QCheckBox('Verbose')
        layout.addWidget(self._remesh_verbose)

        remesh_btn = QPushButton('Fix && Remesh')
        remesh_btn.clicked.connect(self.run_remesh_requested.emit)
        layout.addWidget(remesh_btn)

        self._taubin_lamb = QDoubleSpinBox()
        self._taubin_lamb.setRange(0.0, 1.0)
        self._taubin_lamb.setSingleStep(0.05)
        self._taubin_lamb.setValue(0.6)
        layout.addLayout(_row('Taubin smoothing lambda:', self._taubin_lamb))

        smooth_btn = QPushButton('Smooth')
        smooth_btn.clicked.connect(self.run_smooth_requested.emit)
        layout.addWidget(smooth_btn)
        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox('Export')
        layout = QHBoxLayout(box)
        btn = QPushButton('Export Final Mesh…')
        btn.clicked.connect(self._on_export)
        layout.addWidget(btn)
        return box

    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Export Final Mesh', '', 'STL (*.stl)')
        if not path:
            return
        if not path.endswith('.stl'):
            path += '.stl'
        self.export_requested.emit(path)

    def set_scaling_results(self, results: dict[str, float]) -> None:
        for key, label in self._scaling_labels.items():
            value = results.get(key)
            prefix = label.text().split(':', 1)[0]
            label.setText(f'{prefix}: {value:.3f} mm' if value is not None else f'{prefix}: —')

    # ------------------------------------------------------------------
    # Param getters
    # ------------------------------------------------------------------

    def remove_point_keys(self) -> list[str]:
        keys = []
        if self._remove_anomalous.isChecked():
            keys.append('anomalous_points')
        if self._remove_proximal.isChecked():
            keys.append('proximal_points')
        return keys

    def stitch_kwargs(self) -> dict:
        return {
            'prox_start_mode': self._prox_start_mode.currentText(),
            'dist_start_mode': self._dist_start_mode.currentText(),
            'clamp_overshoot': self._clamp_overshoot.value(),
        }

    def remesh_kwargs(self) -> dict:
        return {
            'target_edge_length_mm': self._target_edge_length.value(),
            'remesh_iterations': self._remesh_iterations.value(),
            'verbose': self._remesh_verbose.isChecked(),
        }

    def taubin_lamb(self) -> float:
        return self._taubin_lamb.value()


def _row(label: str, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.addWidget(QLabel(label))
    layout.addWidget(widget, 1)
    return layout
