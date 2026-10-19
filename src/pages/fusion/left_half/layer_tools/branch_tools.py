from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from domain.fusion.types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class BranchEditorToolbar(SceneToolbar):
    """Toolbar for the Centerline Branches scene: RCA/LCA colored per branch with sharp
    angles numbered, plus split/merge of branches. Any centerline point can be picked for
    a split, the cos threshold only controls which points get a numbered marker."""

    cos_threshold_changed = pyqtSignal(float)
    split_requested = pyqtSignal()
    merge_requested = pyqtSignal(str, int, int)  # centerline ('rca'/'lca'), branch_id_a, branch_id_b

    def __init__(self, parent=None) -> None:
        self.cos_threshold = QDoubleSpinBox()
        self.cos_threshold.setRange(-1.0, 1.0)
        self.cos_threshold.setSingleStep(0.1)
        self.cos_threshold.setValue(0.0)
        self.cos_threshold.setToolTip(
            'Cosine above which an angle counts as sharp — 0.0 ≈ <90°, 0.5 ≈ <60°.\n'
            'Only controls which points get a numbered marker in the scene — use Pick Point\n'
            'below to select any point on a branch for splitting, marked or not.'
        )

        self._selected_label = QLabel('Split point: none selected')
        self.split_btn = QPushButton('Split Here')
        self.split_btn.setEnabled(False)
        self.split_btn.setToolTip("Split the selected branch at the picked/marked point.")

        # Two rows per group so the controls fit next to the layer list.
        split_box = QGroupBox('Split')
        split_layout = QGridLayout(split_box)
        split_layout.addWidget(QLabel('Sharp-angle threshold (cos):'), 0, 0)
        split_layout.addWidget(self.cos_threshold, 0, 1)
        split_layout.addWidget(self._selected_label, 1, 0)
        split_layout.addWidget(self.split_btn, 1, 1)
        split_layout.setColumnStretch(0, 1)

        self.merge_cl_combo = QComboBox()
        self.merge_cl_combo.addItems(['RCA', 'LCA'])
        self.merge_cl_combo.currentIndexChanged.connect(lambda _: self._refresh_merge_choices())
        self.merge_a_combo = QComboBox()
        self.merge_b_combo = QComboBox()
        merge_btn = QPushButton('Merge')
        merge_btn.setToolTip('Merge branch A and branch B of the chosen centerline into one.')
        merge_btn.clicked.connect(self._on_merge_clicked)

        merge_box = QGroupBox('Merge')
        merge_layout = QGridLayout(merge_box)
        merge_layout.addWidget(QLabel('Centerline:'), 0, 0)
        merge_layout.addWidget(self.merge_cl_combo, 0, 1)
        merge_layout.addWidget(merge_btn, 0, 2, 1, 2)
        merge_layout.addWidget(QLabel('Branch A:'), 1, 0)
        merge_layout.addWidget(self.merge_a_combo, 1, 1)
        merge_layout.addWidget(QLabel('Branch B:'), 1, 2)
        merge_layout.addWidget(self.merge_b_combo, 1, 3)
        merge_layout.setColumnStretch(1, 1)
        merge_layout.setColumnStretch(3, 1)

        # Stack split above merge instead of side by side.
        edit_widget = QWidget()
        edit_layout = QVBoxLayout(edit_widget)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.addWidget(split_box)
        edit_layout.addWidget(merge_box)

        self._branch_choices: dict[str, list[int]] = {'rca': [], 'lca': []}

        super().__init__(
            FusionScene.CENTERLINE_BRANCHES,
            extra_rows=[edit_widget],
            show_layers=True,
            show_pick=True,
            show_color_swatch=True,
            parent=parent,
        )
        self.cos_threshold.valueChanged.connect(self.cos_threshold_changed.emit)
        self.split_btn.clicked.connect(self.split_requested.emit)

    def set_branch_choices(self, rca_branch_ids: list[int], lca_branch_ids: list[int]) -> None:
        """Call after every prepare/split/merge, since branch IDs are reassigned after each
        edit."""
        self._branch_choices = {'rca': rca_branch_ids, 'lca': lca_branch_ids}
        self._refresh_merge_choices()

    def _refresh_merge_choices(self) -> None:
        ids = self._branch_choices['rca' if self.merge_cl_combo.currentText() == 'RCA' else 'lca']
        for combo in (self.merge_a_combo, self.merge_b_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(i) for i in ids])
            combo.blockSignals(False)

    def _on_merge_clicked(self) -> None:
        if self.merge_a_combo.currentText() == '' or self.merge_b_combo.currentText() == '':
            return
        cl_name = 'rca' if self.merge_cl_combo.currentText() == 'RCA' else 'lca'
        self.merge_requested.emit(cl_name, int(self.merge_a_combo.currentText()), int(self.merge_b_combo.currentText()))

    def set_selected_marker(self, description: str | None) -> None:
        """description is e.g. 'RCA branch 0 @ point 123', or None to clear the selection."""
        self._selected_label.setText(f'Split point: {description}' if description else 'Split point: none selected')
        self.split_btn.setEnabled(description is not None)
