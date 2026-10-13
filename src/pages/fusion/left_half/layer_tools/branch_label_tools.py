from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from domain.fusion.types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class BranchLabelsToolbar(SceneToolbar):
    """Toolbar for the Branch Labels scene: the RCA/LCA surface vertices colored by the
    centerline branch label_branches assigned them to (same colors as the Centerline
    Branches tab), one layer per branch. Lasso moves vertices to another branch of the same
    vessel — see FusionPage._on_branch_lasso_closed.

    The Branch group renames a branch (one name for its vertices here and its centerline
    on the Centerline Branches tab, used everywhere the branch is listed) or deletes it
    with multimodars' remove_branch."""

    rename_requested = pyqtSignal(str, int)  # vessel ('rca'/'lca'), branch id
    delete_requested = pyqtSignal(str, int, bool)  # vessel, branch id, fill hole

    def __init__(self, parent=None) -> None:
        self.branch_combo = QComboBox()
        self.branch_combo.setMinimumWidth(140)
        self.rename_btn = QPushButton('Rename…')
        self.delete_btn = QPushButton('Delete')
        self.delete_btn.setToolTip(
            'Cut the branch, and every branch growing out of it, off the mesh and the\n'
            'centerline (multimodars remove_branch). The main vessel cannot be deleted.'
        )
        self.fill_hole_check = QCheckBox('Fill hole')
        self.fill_hole_check.setChecked(True)
        self.fill_hole_check.setToolTip("Close the ostium with a patch following the parent vessel's wall.")

        box = QGroupBox('Branch')
        layout = QVBoxLayout(box)
        layout.addWidget(self.branch_combo)
        buttons = QHBoxLayout()
        buttons.addWidget(self.rename_btn)
        buttons.addWidget(self.delete_btn)
        layout.addLayout(buttons)
        layout.addWidget(self.fill_hole_check)

        hint = QLabel('Lasso: move vertices\nto another branch\nof the same vessel.')
        hint.setStyleSheet('color: gray;')
        super().__init__(
            FusionScene.BRANCH_LABELS,
            extra_rows=[box, hint],
            show_lasso=True,
            show_color_swatch=True,
            parent=parent,
        )
        self.rename_btn.clicked.connect(self._on_rename_clicked)
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        self.branch_combo.currentIndexChanged.connect(self._sync_delete_enabled)
        self._sync_delete_enabled()

    def set_branches(self, choices: list[tuple[str, str, int]]) -> None:
        """choices are (label, vessel, branch id); keeps the current branch selected if it
        is still there."""
        current = self.branch_combo.currentData()
        self.branch_combo.blockSignals(True)
        self.branch_combo.clear()
        for label, vessel, branch in choices:
            self.branch_combo.addItem(label, (vessel, branch))
        index = next((i for i in range(self.branch_combo.count()) if self.branch_combo.itemData(i) == current), 0)
        self.branch_combo.setCurrentIndex(index)
        self.branch_combo.blockSignals(False)
        self._sync_delete_enabled()

    def _sync_delete_enabled(self, *_) -> None:
        data = self.branch_combo.currentData()
        self.rename_btn.setEnabled(data is not None)
        self.delete_btn.setEnabled(data is not None and data[1] != 0)

    def _on_rename_clicked(self) -> None:
        data = self.branch_combo.currentData()
        if data is not None:
            self.rename_requested.emit(*data)

    def _on_delete_clicked(self) -> None:
        data = self.branch_combo.currentData()
        if data is not None:
            self.delete_requested.emit(data[0], data[1], self.fill_hole_check.isChecked())
