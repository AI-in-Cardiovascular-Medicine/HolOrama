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


class IntravascularColumn(QWidget):
    """Column 2: load an intravascular (IVUS/OCT) pullback pair and align it onto the
    RCA centerline computed in column 1. See pages/fusion/pipeline.py."""

    run_load_requested = pyqtSignal()
    run_align_requested = pyqtSignal()

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

        out_row = QHBoxLayout()
        self._output_path_edit = QLineEdit('output/rest')
        out_row.addWidget(QLabel('Output path:'))
        out_row.addWidget(self._output_path_edit, 1)
        layout.addLayout(out_row)

        self._step_rotation = QDoubleSpinBox()
        self._step_rotation.setRange(0.01, 45.0)
        self._step_rotation.setSingleStep(0.1)
        self._step_rotation.setValue(0.5)
        self._step_rotation.setToolTip(
            'Rotation step (deg) for the coarse alignment search. Finer = more precise but slower.'
        )
        layout.addLayout(_row('Step rotation (deg):', self._step_rotation))

        self._sample_size = QSpinBox()
        self._sample_size.setRange(1, 100000)
        self._sample_size.setValue(500)
        self._sample_size.setToolTip('Number of points each frame is downsampled to before alignment.')
        layout.addLayout(_row('Sample size:', self._sample_size))

        self._n_points = QSpinBox()
        self._n_points.setRange(3, 1000)
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
        self._ref_labels = {
            'aortic': QLabel('Aortic: —'),
            'superior': QLabel('Superior: —'),
            'inferior': QLabel('Inferior: —'),
        }
        for lbl in self._ref_labels.values():
            layout.addWidget(lbl)
        self._branch_index = QSpinBox()
        self._branch_index.setRange(0, 20)
        self._branch_index.setToolTip('rca_cl.get_branch(index) — alignment needs a single-branch centerline')
        layout.addLayout(_row('RCA branch index:', self._branch_index))
        return box

    def _build_align_group(self) -> QGroupBox:
        box = QGroupBox('Align to Centerline')
        layout = QVBoxLayout(box)

        self._angle_range = QDoubleSpinBox()
        self._angle_range.setRange(1.0, 180.0)
        self._angle_range.setValue(30.0)
        layout.addLayout(_row('Angle range (deg):', self._angle_range))

        # No "write intermediate files" toggle — align never writes them in this app.
        # No "align anomalous wall" toggle either — it's driven automatically by
        # whether Anomalous RCA/LCA is checked in column 1 (see FusionPage._on_run_align).
        self._watertight = QCheckBox('Watertight')
        layout.addWidget(self._watertight)

        out_row = QHBoxLayout()
        self._align_output_dir = QLineEdit('output/aligned')
        out_row.addWidget(QLabel('Output dir:'))
        out_row.addWidget(self._align_output_dir, 1)
        layout.addLayout(out_row)

        align_btn = QPushButton('Align')
        align_btn.clicked.connect(self.run_align_requested.emit)
        layout.addWidget(align_btn)
        return box

    # ------------------------------------------------------------------

    def set_default_dir(self, path: str) -> None:
        self._default_dir = path

    def _on_browse_input_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Select Pullback Case Folder', self._default_dir)
        if path:
            self._input_path_edit.setText(path)

    def set_reference_points(self, aortic, superior, inferior) -> None:
        self._ref_labels['aortic'].setText(f'Aortic: {_fmt_point(aortic)}')
        self._ref_labels['superior'].setText(f'Superior: {_fmt_point(superior)}')
        self._ref_labels['inferior'].setText(f'Inferior: {_fmt_point(inferior)}')

    # ------------------------------------------------------------------
    # Param getters
    # ------------------------------------------------------------------

    def load_kwargs(self) -> dict:
        return {
            'input_path': self._input_path_edit.text(),
            'labels': [self._label_dia_edit.text(), self._label_sys_edit.text()],
            'output_path': self._output_path_edit.text(),
            'step_rotation_deg': self._step_rotation.value(),
            'sample_size': self._sample_size.value(),
            'n_points': self._n_points.value(),
        }

    def branch_index(self) -> int:
        return self._branch_index.value()

    def align_kwargs(self) -> dict:
        return {
            'angle_range_deg': self._angle_range.value(),
            'watertight': self._watertight.isChecked(),
            'output_dir': self._align_output_dir.text(),
        }


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
