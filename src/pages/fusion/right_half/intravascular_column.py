from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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


class IntravascularColumn(QWidget):
    """Column 2: load an intravascular (IVUS/OCT) pullback pair and align it onto the
    RCA or LCA centerline computed in column 1. See pages/fusion/pipeline.py."""

    run_load_requested = pyqtSignal()
    run_align_requested = pyqtSignal()
    run_align_manual_requested = pyqtSignal()
    reference_vessel_changed = pyqtSignal(str)  # 'rca' or 'lca'
    reference_index_changed = pyqtSignal(int)  # index chosen from the Reference combo
    run_label_anomalous_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._default_dir: str = ''

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        title = QLabel('Intravascular Alignment')
        title.setStyleSheet('font-weight: bold;')
        root.addWidget(title)

        root.addWidget(self._build_load_group())
        root.addWidget(self._build_reference_group())
        root.addWidget(self._build_align_group())
        root.addWidget(self._build_manual_align_group())
        root.addWidget(self._build_overlap_group())
        root.addStretch(1)

    # ------------------------------------------------------------------

    def _build_load_group(self) -> QGroupBox:
        box = QGroupBox('Load Pullback')
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self._input_path_edit = QLineEdit()
        self._input_path_edit.setReadOnly(True)
        self._input_path_edit.setPlaceholderText('Case folder (e.g. ivus_rest)')
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self._on_browse_input_path)
        row.addWidget(self._input_path_edit, 1)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        labels_row = QHBoxLayout()
        self._label_dia_edit = QLineEdit('aligned_dia')
        self._label_sys_edit = QLineEdit('aligned_sys')
        labels_row.addWidget(QLabel('Diastole label:'))
        labels_row.addWidget(self._label_dia_edit)
        labels_row.addWidget(QLabel('Systole label:'))
        labels_row.addWidget(self._label_sys_edit)
        layout.addLayout(labels_row)

        self._step_rotation = QDoubleSpinBox()
        self._step_rotation.setRange(0.01, 45.0)
        self._step_rotation.setSingleStep(0.1)
        self._step_rotation.setValue(0.1)
        self._step_rotation.setToolTip(
            'Rotation step (deg) for the coarse alignment search. Finer = more precise but slower.'
        )
        layout.addLayout(_row('Step rotation (deg):', self._step_rotation))

        self._sample_size = QSpinBox()
        self._sample_size.setRange(1, 1000)
        self._sample_size.setValue(200)
        self._sample_size.setToolTip('Number of points each frame is downsampled to before alignment.')
        layout.addLayout(_row('Sample size:', self._sample_size))

        self._n_points = QSpinBox()
        self._n_points.setRange(0, 1000)
        self._n_points.setValue(20)
        self._n_points.setToolTip('Number of points on the synthetic catheter contour (not the lumen contour).')
        layout.addLayout(_row('Catheter points:', self._n_points))

        load_btn = QPushButton('Load Pullback')
        load_btn.clicked.connect(self.run_load_requested.emit)
        layout.addWidget(load_btn)
        return box

    def _build_reference_group(self) -> QGroupBox:
        box = QGroupBox('Reference Points (from Vessel Tree)')
        layout = QVBoxLayout(box)

        self._ref_vessel_combo = QComboBox()
        self._ref_vessel_combo.addItems(['RCA', 'LCA'])
        self._ref_vessel_combo.setToolTip('Which coronary to align the pullback onto.')
        self._ref_vessel_combo.currentIndexChanged.connect(
            lambda _: self.reference_vessel_changed.emit(self.reference_vessel())
        )
        layout.addLayout(_row('Centerline:', self._ref_vessel_combo))

        self._ref_index_combo = QComboBox()
        self._ref_index_combo.setToolTip('Same reference list as the Vessel Tree tab, for the centerline above.')
        self._ref_index_combo.currentIndexChanged.connect(self._on_ref_index_changed)
        layout.addLayout(_row('Reference:', self._ref_index_combo))

        self._ref_labels = {
            'aortic': QLabel('Aortic: —'),
            'superior': QLabel('Superior: —'),
            'inferior': QLabel('Inferior: —'),
        }
        for lbl in self._ref_labels.values():
            layout.addWidget(lbl)
        self._branch_index = QSpinBox()
        self._branch_index.setRange(0, 20)
        self._branch_index.setToolTip('centerline.get_branch(index) — alignment needs a single-branch centerline')
        layout.addLayout(_row('Branch index:', self._branch_index))
        return box

    def _build_align_group(self) -> QGroupBox:
        box = QGroupBox('Align to Centerline')
        layout = QVBoxLayout(box)

        self._angle_range = QDoubleSpinBox()
        self._angle_range.setRange(1.0, 180.0)
        self._angle_range.setValue(30.0)
        self._angle_range.setToolTip('Total rotation search range (deg) for the Hausdorff refinement.')
        layout.addLayout(_row('Angle range (deg):', self._angle_range))

        self._angle_step = QDoubleSpinBox()
        self._angle_step.setRange(0.01, 45.0)
        self._angle_step.setSingleStep(0.1)
        self._angle_step.setValue(1.0)
        self._angle_step.setToolTip('Step size (deg) for the Hausdorff refinement rotation search.')
        layout.addLayout(_row('Angle step (deg):', self._angle_step))

        self._index_range = QSpinBox()
        self._index_range.setRange(0, 50)
        self._index_range.setValue(2)
        self._index_range.setToolTip(
            'Number of centerline indices considered around each reference point during refinement.'
        )
        layout.addLayout(_row('Index range:', self._index_range))

        # No "write intermediate files" toggle — align never writes them in this app.
        # No "align anomalous wall" toggle either — it's driven automatically by
        # whether Anomalous RCA/LCA is checked in column 1 (see FusionPage._on_run_align).
        self._watertight = QCheckBox('Watertight')
        layout.addWidget(self._watertight)

        align_btn = QPushButton('Align')
        align_btn.clicked.connect(self.run_align_requested.emit)
        layout.addWidget(align_btn)
        return box

    def _build_manual_align_group(self) -> QGroupBox:
        box = QGroupBox('Manual Alignment (Optional)')
        layout = QVBoxLayout(box)

        self._manual_rotation_angle = QDoubleSpinBox()
        self._manual_rotation_angle.setRange(-360.0, 360.0)
        self._manual_rotation_angle.setSingleStep(1.0)
        self._manual_rotation_angle.setToolTip('Rotation to apply, in degrees.')
        layout.addLayout(_row('Rotation angle (deg):', self._manual_rotation_angle))

        self._manual_ref_offset = QSpinBox()
        self._manual_ref_offset.setRange(-100, 100)
        self._manual_ref_offset.setToolTip(
            '0 = the centerline point closest to the reference selected above (same one used\n'
            'for automatic alignment). Positive = N centerline points more distal;\n'
            'negative = N centerline points more proximal, towards the ostium (point index 0).\n'
            'Clamped at the ostium — you cannot go more proximal than that.'
        )
        layout.addLayout(_row('Ref. point offset:', self._manual_ref_offset))

        # No "write intermediate files" toggle, same as the automatic Align group above.
        self._manual_watertight = QCheckBox('Watertight')
        layout.addWidget(self._manual_watertight)

        align_manual_btn = QPushButton('Align (Manual)')
        align_manual_btn.setToolTip('Only works for elliptic vessels (anomalous coronaries).')
        align_manual_btn.clicked.connect(self.run_align_manual_requested.emit)
        layout.addWidget(align_manual_btn)
        return box

    def _build_overlap_group(self) -> QGroupBox:
        box = QGroupBox('Overlap Region')
        layout = QVBoxLayout(box)
        btn = QPushButton('Label Overlap Region')
        btn.setToolTip(
            'Partitions the aligned pullback into proximal/overlap/distal sub-regions, '
            'for whichever centerline is selected above.'
        )
        btn.clicked.connect(self.run_label_anomalous_requested.emit)
        layout.addWidget(btn)
        return box

    # ------------------------------------------------------------------

    def set_default_dir(self, path: str) -> None:
        self._default_dir = path

    def _on_browse_input_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Select Pullback Case Folder', self._default_dir)
        if path:
            self._input_path_edit.setText(path)

    def set_manual_rotation_angle(self, degrees: float) -> None:
        self._manual_rotation_angle.setValue(degrees)

    def set_reference_points(self, aortic, superior, inferior) -> None:
        self._ref_labels['aortic'].setText(f'Aortic: {_fmt_point(aortic)}')
        self._ref_labels['superior'].setText(f'Superior: {_fmt_point(superior)}')
        self._ref_labels['inferior'].setText(f'Inferior: {_fmt_point(inferior)}')

    def reference_vessel(self) -> str:
        return 'rca' if self._ref_vessel_combo.currentText() == 'RCA' else 'lca'

    def set_reference_choices(self, labels: list[str]) -> None:
        """Repopulate the Reference dropdown — same label list as the Vessel Tree tab's
        RCA/LCA dropdown for whichever vessel is currently selected above."""
        self._ref_index_combo.blockSignals(True)
        self._ref_index_combo.clear()
        self._ref_index_combo.addItems(labels)
        self._ref_index_combo.blockSignals(False)

    def set_selected_reference_index(self, index: int) -> None:
        self._ref_index_combo.blockSignals(True)
        self._ref_index_combo.setCurrentIndex(index)
        self._ref_index_combo.blockSignals(False)

    def _on_ref_index_changed(self, index: int) -> None:
        if index >= 0:
            self.reference_index_changed.emit(index)

    # ------------------------------------------------------------------
    # Param getters
    # ------------------------------------------------------------------

    def load_kwargs(self) -> dict:
        return {
            'input_path': self._input_path_edit.text(),
            'labels': [self._label_dia_edit.text(), self._label_sys_edit.text()],
            'step_rotation_deg': self._step_rotation.value(),
            'sample_size': self._sample_size.value(),
            'n_points': self._n_points.value(),
        }

    def branch_index(self) -> int:
        return self._branch_index.value()

    def align_kwargs(self) -> dict:
        return {
            'angle_range_deg': self._angle_range.value(),
            'angle_step_deg': self._angle_step.value(),
            'index_range': self._index_range.value(),
            'watertight': self._watertight.isChecked(),
        }

    def manual_rotation_angle_deg(self) -> float:
        return self._manual_rotation_angle.value()

    def manual_ref_point_offset(self) -> int:
        return self._manual_ref_offset.value()

    def manual_align_kwargs(self) -> dict:
        return {'watertight': self._manual_watertight.isChecked()}


def _row(label: str, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.addWidget(QLabel(label))
    layout.addWidget(widget, 1)
    return layout


def _fmt_point(point) -> str:
    if point is None:
        return '—'
    x, y, z = point
    return f'({x:.2f}, {y:.2f}, {z:.2f})'
