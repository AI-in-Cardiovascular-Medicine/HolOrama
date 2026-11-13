from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fusion_pipeline.annotations import CaseReference, read_case_references
from pages.fusion.widgets import row


class IntravascularColumn(QWidget):
    """Column 2: load an intravascular (IVUS/OCT) pullback pair and align it onto the
    RCA or LCA centerline computed in column 1. See fusion_pipeline/multimodars_api.py."""

    run_load_requested = pyqtSignal()
    run_align_requested = pyqtSignal()
    run_align_manual_requested = pyqtSignal()
    reference_centerline_changed = pyqtSignal(str, int)  # vessel ('rca'/'lca'), branch (0 = main)
    reference_index_changed = pyqtSignal(int)  # index chosen from the Reference combo
    run_label_anomalous_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._default_dir: str = ''
        self._pullback_anchor: int | None = None  # the loaded pullback's anchor frame

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

        # Modality picks the label fields and the loader: IVUS -> from_file_singlepair (both
        # phases) or from_file_single (one phase), OCT -> from_array.
        modality_row = QHBoxLayout()
        self._ivus_radio = QRadioButton('IVUS')
        self._oct_radio = QRadioButton('OCT')
        self._ivus_radio.setChecked(True)
        self._modality_group = QButtonGroup(self)
        self._modality_group.addButton(self._ivus_radio)
        self._modality_group.addButton(self._oct_radio)
        modality_row.addWidget(QLabel('Modality:'))
        modality_row.addWidget(self._ivus_radio)
        modality_row.addWidget(self._oct_radio)
        modality_row.addStretch(1)
        layout.addLayout(modality_row)

        path_row = QHBoxLayout()
        self._input_path_edit = QLineEdit()
        self._input_path_edit.setReadOnly(True)
        self._input_path_edit.setPlaceholderText('Case folder (e.g. ivus_rest)')
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self._on_browse_input_path)
        path_row.addWidget(self._input_path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # IVUS: one checkbox + label per phase. Both checked loads the pair, one loads that phase.
        self._ivus_phase_widget = QWidget()
        ivus_layout = QVBoxLayout(self._ivus_phase_widget)
        ivus_layout.setContentsMargins(0, 0, 0, 0)
        ivus_layout.setSpacing(4)
        self._dia_check = QCheckBox('Diastole')
        self._dia_check.setChecked(True)
        self._label_dia_edit = QLineEdit('aligned_dia')
        ivus_layout.addLayout(_checkbox_row(self._dia_check, self._label_dia_edit))
        self._sys_check = QCheckBox('Systole')
        self._sys_check.setChecked(True)
        self._label_sys_edit = QLineEdit('aligned_sys')
        ivus_layout.addLayout(_checkbox_row(self._sys_check, self._label_sys_edit))
        # Which reference of each phase anchors the loading (pairing, trimming, orientation).
        self._dia_ref_combo, self._sys_ref_combo, self._oct_ref_combo = QComboBox(), QComboBox(), QComboBox()
        for combo in (self._dia_ref_combo, self._sys_ref_combo, self._oct_ref_combo):
            combo.setToolTip(_ANCHOR_TIP)
        ivus_layout.addLayout(row('Diastole anchor:', self._dia_ref_combo))
        ivus_layout.addLayout(row('Systole anchor:', self._sys_ref_combo))
        layout.addWidget(self._ivus_phase_widget)

        # OCT: a single label field.
        self._oct_label_widget = QWidget()
        oct_layout = QVBoxLayout(self._oct_label_widget)
        oct_layout.setContentsMargins(0, 0, 0, 0)
        self._label_oct_edit = QLineEdit('tagged')
        oct_layout.addLayout(row('Label:', self._label_oct_edit))
        oct_layout.addLayout(row('Anchor:', self._oct_ref_combo))
        layout.addWidget(self._oct_label_widget)

        self._ivus_radio.toggled.connect(self._on_modality_toggled)
        self._input_path_edit.textChanged.connect(self._refresh_load_references)

        self._step_rotation = QDoubleSpinBox()
        self._step_rotation.setRange(0.01, 45.0)
        self._step_rotation.setSingleStep(0.1)
        self._step_rotation.setValue(0.1)
        self._step_rotation.setToolTip(
            'Rotation step (deg) for the coarse alignment search. Finer = more precise but slower.'
        )
        layout.addLayout(row('Step rotation (deg):', self._step_rotation))

        self._sample_size = QSpinBox()
        self._sample_size.setRange(1, 1000)
        self._sample_size.setValue(200)
        self._sample_size.setToolTip('Number of points each frame is downsampled to before alignment.')
        layout.addLayout(row('Sample size:', self._sample_size))

        self._n_points = QSpinBox()
        self._n_points.setRange(0, 1000)
        self._n_points.setValue(20)
        self._n_points.setToolTip('Number of points on the synthetic catheter contour (not the lumen contour).')
        layout.addLayout(row('Catheter points:', self._n_points))

        load_btn = QPushButton('Load Pullback')
        load_btn.clicked.connect(self.run_load_requested.emit)
        layout.addWidget(load_btn)

        self._on_modality_toggled()  # set initial IVUS/OCT field visibility
        return box

    def _on_modality_toggled(self, *_) -> None:
        is_ivus = self._ivus_radio.isChecked()
        self._ivus_phase_widget.setVisible(is_ivus)
        self._oct_label_widget.setVisible(not is_ivus)

    def _build_reference_group(self) -> QGroupBox:
        box = QGroupBox('Reference Points (from Vessel Tree)')
        layout = QVBoxLayout(box)

        self._ref_vessel_combo = QComboBox()
        self._ref_vessel_combo.setToolTip(
            'Which coronary branch to align the pullback onto — the main vessel or one of its '
            'side branches (listed once the vessel tree is discretized).'
        )
        self._with_parents_check = QCheckBox('Include parent vessel')
        self._with_parents_check.setToolTip(
            'Side branches only: align onto the path from the vessel ostium through the parent\n'
            'branches to this branch\'s tip, so frames running proximally past the side\n'
            'branch\'s ostium are placed on the parent instead of being left unaligned.\n'
            'Needs multimodars >= 0.7.8.'
        )
        self.set_centerline_choices([('RCA', 'rca', 0), ('LCA', 'lca', 0)], 'rca', 0)
        self._ref_vessel_combo.currentIndexChanged.connect(self._sync_with_parents_enabled)
        self._ref_vessel_combo.currentIndexChanged.connect(
            lambda _: self.reference_centerline_changed.emit(self.reference_vessel(), self.reference_branch())
        )
        centerline_row = row('Centerline:', self._ref_vessel_combo)
        centerline_row.addWidget(self._with_parents_check)
        layout.addLayout(centerline_row)

        self._ref_index_combo = QComboBox()
        self._ref_index_combo.setToolTip(
            'References reachable on the branch above (its ostium, then each branch leaving it) — '
            'same list as the Vessel Tree tab.'
        )
        self._ref_index_combo.currentIndexChanged.connect(self._on_ref_index_changed)
        layout.addLayout(row('Reference:', self._ref_index_combo))

        self._iv_ref_combo = QComboBox()
        self._iv_ref_combo.setToolTip(
            'Which reference of the loaded pullback (diastole for a pair) is placed on the reference\n'
            'above. Systole follows diastole. Listed once the pullback is loaded.'
        )
        layout.addLayout(row('Pullback reference:', self._iv_ref_combo))

        self._ref_labels = {
            'main': QLabel('Main: —'),
            'rotation_1': QLabel('Rotation 1: —'),
            'rotation_2': QLabel('Rotation 2: —'),
        }
        for lbl in self._ref_labels.values():
            layout.addWidget(lbl)
        return box

    def _build_align_group(self) -> QGroupBox:
        box = QGroupBox('Align to Centerline')
        layout = QVBoxLayout(box)

        self._angle_range = QDoubleSpinBox()
        self._angle_range.setRange(1.0, 180.0)
        self._angle_range.setValue(30.0)
        self._angle_range.setToolTip('Total rotation search range (deg) for the Hausdorff refinement.')
        layout.addLayout(row('Angle range (deg):', self._angle_range))

        self._angle_step = QDoubleSpinBox()
        self._angle_step.setRange(0.01, 45.0)
        self._angle_step.setSingleStep(0.1)
        self._angle_step.setValue(1.0)
        self._angle_step.setToolTip('Step size (deg) for the Hausdorff refinement rotation search.')
        layout.addLayout(row('Angle step (deg):', self._angle_step))

        self._index_range = QSpinBox()
        self._index_range.setRange(0, 50)
        self._index_range.setValue(2)
        self._index_range.setToolTip(
            'Number of centerline indices considered around each reference point during refinement.'
        )
        layout.addLayout(row('Index range:', self._index_range))

        # No "align anomalous wall" toggle: it follows Acute Takeoff in column 1 (main vessel only).
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
        layout.addLayout(row('Rotation angle (deg):', self._manual_rotation_angle))

        self._manual_ref_offset = QSpinBox()
        self._manual_ref_offset.setRange(-100, 100)
        self._manual_ref_offset.setToolTip(
            '0 = the centerline point closest to the reference selected above (same one used\n'
            'for automatic alignment). Positive = N centerline points more distal;\n'
            'negative = N centerline points more proximal, towards the ostium (point index 0).\n'
            'Clamped at the ostium — you cannot go more proximal than that.'
        )
        layout.addLayout(row('Ref. point offset:', self._manual_ref_offset))

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

    def input_path(self) -> str:
        return self._input_path_edit.text()

    def set_input_path(self, path: str) -> None:
        self._input_path_edit.setText(path)

    def set_load_plan(self, plan: dict) -> None:
        """Inverse of load_plan(), for a restored case. Missing keys keep their widget value."""
        if 'mode' in plan:
            (self._oct_radio if plan['mode'] == 'oct' else self._ivus_radio).setChecked(True)
        if plan.get('input_path'):
            self._input_path_edit.setText(plan['input_path'])
        for key, widget in (('step_rotation_deg', self._step_rotation), ('sample_size', self._sample_size)):
            if key in plan:
                widget.setValue(plan[key])
        if 'n_points' in plan:
            self._n_points.setValue(plan['n_points'])
        for key, check in (('diastole', self._dia_check), ('systole', self._sys_check)):
            if key in plan:
                check.setChecked(plan[key])
        for key, edit in (
            ('label', self._label_oct_edit),
            ('dia_label', self._label_dia_edit),
            ('sys_label', self._label_sys_edit),
        ):
            if key in plan:
                edit.setText(plan[key])
        for key, combo in self._anchor_combos().items():
            if plan.get(key) is not None:
                _select_frame(combo, plan[key])

    def _anchor_combos(self) -> dict[str, QComboBox]:
        """load_plan key -> the anchor picker it is read from."""
        return {
            'dia_reference_frame': self._dia_ref_combo,
            'sys_reference_frame': self._sys_ref_combo,
            'reference_frame': self._oct_ref_combo,
        }

    def _refresh_load_references(self, *_) -> None:
        """List each phase's references of the chosen case folder after multimodars' default (the most
        proximal), which is preselected."""
        folder = self._input_path_edit.text()
        for phase, combo in (
            ('diastolic', self._dia_ref_combo),
            ('systolic', self._sys_ref_combo),
            ('tagged', self._oct_ref_combo),
        ):
            references = read_case_references(folder, phase) if folder else []
            _fill_references(combo, references, None, default_item=_MOST_PROXIMAL)

    def set_pullback_references(self, references: list[CaseReference], anchor: int | None) -> None:
        """The loaded pullback's references for Align to Centerline, its load `anchor` (a frame) preselected."""
        self._pullback_anchor = anchor
        _fill_references(self._iv_ref_combo, references, anchor)

    def pullback_reference_frame(self) -> int | None:
        """The reference to place on the centerline instead of the load anchor, None to keep the anchor.

        None rather than the anchor's frame, because multimodars re-anchors on an explicit frame,
        which also moves the three-point rotation's reference to its nearest lumen point.
        """
        frame = self._iv_ref_combo.currentData()
        return None if frame == self._pullback_anchor else frame

    def set_align_settings(self, settings: dict) -> None:
        """Inverse of align_kwargs() / manual_align_kwargs(). Missing keys keep their widget value."""
        for key, widget in (
            ('angle_range_deg', self._angle_range),
            ('angle_step_deg', self._angle_step),
            ('index_range', self._index_range),
        ):
            if key in settings:
                widget.setValue(settings[key])
        if 'watertight' in settings:
            self._watertight.setChecked(settings['watertight'])
            self._manual_watertight.setChecked(settings['watertight'])
        if settings.get('reference_frame') is not None:
            _select_frame(self._iv_ref_combo, settings['reference_frame'])

    def set_align_with_parents(self, checked: bool) -> None:
        self._with_parents_check.setChecked(checked)

    def set_manual_ref_point_offset(self, offset: int) -> None:
        self._manual_ref_offset.setValue(offset)

    def set_manual_rotation_angle(self, degrees: float) -> None:
        self._manual_rotation_angle.setValue(degrees)

    def set_reference_points(self, main, rotation_1, rotation_2) -> None:
        self._ref_labels['main'].setText(f'Main: {_fmt_point(main)}')
        self._ref_labels['rotation_1'].setText(f'Rotation 1: {_fmt_point(rotation_1)}')
        self._ref_labels['rotation_2'].setText(f'Rotation 2: {_fmt_point(rotation_2)}')

    def reference_vessel(self) -> str:
        data = self._ref_vessel_combo.currentData()
        return data.split(':')[0] if data else 'rca'

    def reference_branch(self) -> int:
        """0 for the main vessel, n for side branch n (the centerline.get_branch() id)."""
        data = self._ref_vessel_combo.currentData()
        return int(data.split(':')[1]) if data else 0

    def set_centerline_choices(self, choices: list[tuple[str, str, int]], vessel: str, branch: int) -> None:
        """choices are (label, vessel, branch) triples. Selects (vessel, branch) without emitting."""
        combo = self._ref_vessel_combo
        combo.blockSignals(True)
        combo.clear()
        for label, v, b in choices:
            combo.addItem(label, f'{v}:{b}')
        combo.blockSignals(False)
        self.set_selected_centerline(vessel, branch)

    def set_selected_centerline(self, vessel: str, branch: int) -> None:
        combo = self._ref_vessel_combo
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, combo.findData(f'{vessel}:{branch}')))
        combo.blockSignals(False)
        self._sync_with_parents_enabled()

    def _sync_with_parents_enabled(self, *_) -> None:
        # The main vessel has no parent.
        self._with_parents_check.setEnabled(self.reference_branch() != 0)

    def align_with_parents(self) -> bool:
        """Whether to align onto get_branch(branch, with_parents=True) (side branches only)."""
        return self.reference_branch() != 0 and self._with_parents_check.isChecked()

    def set_reference_choices(self, labels: list[str]) -> None:
        """Repopulate the Reference dropdown without emitting."""
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

    def load_plan(self) -> dict:
        """Loader choice and params for Load Pullback. `mode` is 'oct' or 'ivus'."""
        common = {
            'input_path': self._input_path_edit.text(),
            'step_rotation_deg': self._step_rotation.value(),
            'sample_size': self._sample_size.value(),
            'n_points': self._n_points.value(),
        }
        if self._oct_radio.isChecked():
            return {
                'mode': 'oct',
                'label': self._label_oct_edit.text(),
                'reference_frame': self._oct_ref_combo.currentData(),
                **common,
            }
        return {
            'mode': 'ivus',
            'diastole': self._dia_check.isChecked(),
            'systole': self._sys_check.isChecked(),
            'dia_label': self._label_dia_edit.text(),
            'sys_label': self._label_sys_edit.text(),
            'dia_reference_frame': self._dia_ref_combo.currentData(),
            'sys_reference_frame': self._sys_ref_combo.currentData(),
            **common,
        }

    def align_kwargs(self) -> dict:
        return {
            'angle_range_deg': self._angle_range.value(),
            'angle_step_deg': self._angle_step.value(),
            'index_range': self._index_range.value(),
            'watertight': self._watertight.isChecked(),
            'reference_frame': self.pullback_reference_frame(),
        }

    def manual_rotation_angle_deg(self) -> float:
        return self._manual_rotation_angle.value()

    def manual_ref_point_offset(self) -> int:
        return self._manual_ref_offset.value()

    def manual_align_kwargs(self) -> dict:
        return {'watertight': self._manual_watertight.isChecked(), 'reference_frame': self.pullback_reference_frame()}


_ANCHOR_TIP = (
    'The reference this phase is loaded around: diastole and systole are paired at it, and turned\n'
    'so it points right. Pick the same landmark in both phases. Defaults to the most proximal one.'
)
_MOST_PROXIMAL = 'Most proximal (default)'


def _fill_references(
    combo: QComboBox, references: list[CaseReference], selected: int | None, default_item: str | None = None
) -> None:
    """One item per reference (data: its frame), after `default_item` (data None) if given, `selected`
    preselected (the default item when None), without emitting."""
    combo.blockSignals(True)
    combo.clear()
    if default_item is not None and references:
        combo.addItem(default_item, None)
    for reference in references:
        combo.addItem(f'Frame {reference.frame} ({"auto" if reference.auto else "by hand"})', reference.frame)
    combo.setEnabled(bool(references))
    combo.blockSignals(False)
    if selected is not None:
        _select_frame(combo, selected)


def _select_frame(combo: QComboBox, frame: int) -> None:
    index = combo.findData(frame)
    if index >= 0:
        combo.setCurrentIndex(index)


def _checkbox_row(checkbox: QCheckBox, widget: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.addWidget(checkbox)
    layout.addWidget(widget, 1)
    return layout


def _fmt_point(point) -> str:
    if point is None:
        return '—'
    x, y, z = point
    return f'({x:.2f}, {y:.2f}, {z:.2f})'
