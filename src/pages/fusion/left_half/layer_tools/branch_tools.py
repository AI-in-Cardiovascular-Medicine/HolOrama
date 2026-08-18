from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class BranchEditorToolbar(SceneToolbar):
    """Toolbar for the Centerline Branches scene: shows RCA/LCA colored per branch — one
    color per branch, matched by the layer-list swatches (see colors.BRANCH_COLORS_RCA/LCA)
    — with sharp angles additionally numbered (see FusionPage._refresh_branch_scene), and
    lets the user split a branch or merge two branches — mirroring the manual
    split_branch/merge_branches correction step in multimodars' own example workflow.

    Split is on top, Merge below, each in its own labeled group so the two unrelated
    actions don't run together. To split: click 'Pick Point' then click *any* point on an
    RCA/LCA branch in the 3D scene (not just a numbered sharp-angle marker) —
    FusionPage._on_point_picked resolves the click to the nearest actual centerline point
    and calls set_selected_marker() below. The cos threshold only controls which points get
    a numbered marker in the scene; it doesn't limit what you can pick.
    """

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

        split_box = QGroupBox('Split')
        split_layout = QHBoxLayout(split_box)
        split_layout.addWidget(QLabel('Sharp-angle threshold (cos):'))
        split_layout.addWidget(self.cos_threshold)
        split_layout.addWidget(self._selected_label, 1)
        split_layout.addWidget(self.split_btn)

        self.merge_cl_combo = QComboBox()
        self.merge_cl_combo.addItems(['RCA', 'LCA'])
        self.merge_cl_combo.currentIndexChanged.connect(lambda _: self._refresh_merge_choices())
        self.merge_a_combo = QComboBox()
        self.merge_b_combo = QComboBox()
        merge_btn = QPushButton('Merge')
        merge_btn.setToolTip('Merge branch A and branch B of the chosen centerline into one.')
        merge_btn.clicked.connect(self._on_merge_clicked)

        merge_box = QGroupBox('Merge')
        merge_layout = QHBoxLayout(merge_box)
        merge_layout.addWidget(QLabel('Centerline:'))
        merge_layout.addWidget(self.merge_cl_combo)
        merge_layout.addWidget(QLabel('Branch A:'))
        merge_layout.addWidget(self.merge_a_combo)
        merge_layout.addWidget(QLabel('Branch B:'))
        merge_layout.addWidget(self.merge_b_combo)
        merge_layout.addWidget(merge_btn)

        # One widget so split/merge lay out as two stacked rows (split on top, merge below)
        # instead of getting squeezed side by side with the layer list and view buttons.
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
        """Call after every prepare/split/merge — branch IDs are reassigned (by descending
        length) after each edit, so the merge dropdowns must be rebuilt from scratch."""
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
