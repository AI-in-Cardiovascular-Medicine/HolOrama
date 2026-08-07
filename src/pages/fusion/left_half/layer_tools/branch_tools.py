from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QWidget

from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar, labeled


class BranchEditorToolbar(SceneToolbar):
    """Toolbar for the Centerline Branches scene: shows RCA/LCA colored per branch with
    sharp angles numbered (see FusionPage._refresh_branch_scene), and lets the user split
    a branch at a clicked sharp angle or merge two branches — mirroring the manual
    split_branch/merge_branches correction step in multimodars' own example workflow.

    Click a numbered marker in the scene (Pick Point) to select it for splitting;
    FusionPage._on_point_picked resolves the click to the nearest marker and calls
    set_selected_marker() below.
    """

    cos_threshold_changed = pyqtSignal(float)
    split_requested = pyqtSignal()
    merge_requested = pyqtSignal(str, int, int)  # centerline ('rca'/'lca'), branch_id_a, branch_id_b

    def __init__(self, parent=None) -> None:
        self.cos_threshold = QDoubleSpinBox()
        self.cos_threshold.setRange(-1.0, 1.0)
        self.cos_threshold.setSingleStep(0.1)
        self.cos_threshold.setValue(0.0)
        self.cos_threshold.setToolTip('Cosine above which an angle counts as sharp — 0.0 ≈ <90°, 0.5 ≈ <60°.')

        self._selected_label = QLabel('Selected: none')
        self.split_btn = QPushButton('Split Here')
        self.split_btn.setEnabled(False)
        self.split_btn.setToolTip('Split the selected sharp-angle marker\'s branch at that point.')

        self.merge_cl_combo = QComboBox()
        self.merge_cl_combo.addItems(['RCA', 'LCA'])
        self.merge_cl_combo.currentIndexChanged.connect(lambda _: self._refresh_merge_choices())
        self.merge_a_combo = QComboBox()
        self.merge_b_combo = QComboBox()
        merge_btn = QPushButton('Merge')
        merge_btn.clicked.connect(self._on_merge_clicked)

        merge_widget = QWidget()
        merge_layout = QHBoxLayout(merge_widget)
        merge_layout.setContentsMargins(0, 0, 0, 0)
        merge_layout.addWidget(QLabel('Merge:'))
        merge_layout.addWidget(self.merge_cl_combo)
        merge_layout.addWidget(self.merge_a_combo)
        merge_layout.addWidget(self.merge_b_combo)
        merge_layout.addWidget(merge_btn)

        split_widget = QWidget()
        split_layout = QHBoxLayout(split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.addWidget(self._selected_label)
        split_layout.addWidget(self.split_btn)

        self._branch_choices: dict[str, list[int]] = {'rca': [], 'lca': []}

        super().__init__(
            FusionScene.CENTERLINE_BRANCHES,
            extra_rows=[labeled('Sharp angle threshold (cos):', self.cos_threshold), split_widget, merge_widget],
            show_layers=True,
            show_pick=True,
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
        self._selected_label.setText(f'Selected: {description}' if description else 'Selected: none')
        self.split_btn.setEnabled(description is not None)
