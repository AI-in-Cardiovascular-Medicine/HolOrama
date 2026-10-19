from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from domain.fusion.types import FusionScene
from pages.fusion.left_half.toolbars.base import SceneToolbar


class BranchLabelsToolbar(SceneToolbar):
    """Toolbar for the Branch Labels scene: RCA/LCA surface vertices colored by branch, one
    layer per branch. Lasso moves vertices to another branch of the same vessel. The Branch
    group renames or deletes a branch."""

    rename_requested = pyqtSignal(str, int)  # vessel ('rca'/'lca'), branch id
    delete_requested = pyqtSignal(str, int, bool)  # vessel, branch id, fill hole

    def __init__(self, parent=None) -> None:
        self._branch_combo = QComboBox()
        self._branch_combo.setMinimumWidth(140)
        self._rename_btn = QPushButton('Rename…')
        self._delete_btn = QPushButton('Delete')
        self._delete_btn.setToolTip(
            'Cut the branch, and every branch growing out of it, off the mesh and the\n'
            'centerline (multimodars remove_branch). The main vessel cannot be deleted.'
        )
        self._fill_hole_check = QCheckBox('Fill hole')
        self._fill_hole_check.setChecked(True)
        self._fill_hole_check.setToolTip("Close the ostium with a patch following the parent vessel's wall.")

        box = QGroupBox('Branch')
        layout = QVBoxLayout(box)
        layout.addWidget(self._branch_combo)
        buttons = QHBoxLayout()
        buttons.addWidget(self._rename_btn)
        buttons.addWidget(self._delete_btn)
        layout.addLayout(buttons)
        layout.addWidget(self._fill_hole_check)

        hint = QLabel('Lasso: move vertices\nto another branch\nof the same vessel.')
        hint.setStyleSheet('color: gray;')
        super().__init__(
            FusionScene.BRANCH_LABELS,
            extra_rows=[box, hint],
            show_lasso=True,
            show_color_swatch=True,
            parent=parent,
        )
        self._rename_btn.clicked.connect(self._on_rename_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._branch_combo.currentIndexChanged.connect(self._sync_delete_enabled)
        self._sync_delete_enabled()

    def set_branches(self, choices: list[tuple[str, str, int]]) -> None:
        """choices are (label, vessel, branch id). Keeps the current selection if present."""
        current = self._branch_combo.currentData()
        self._branch_combo.blockSignals(True)
        self._branch_combo.clear()
        for label, vessel, branch in choices:
            self._branch_combo.addItem(label, (vessel, branch))
        index = next((i for i in range(self._branch_combo.count()) if self._branch_combo.itemData(i) == current), 0)
        self._branch_combo.setCurrentIndex(index)
        self._branch_combo.blockSignals(False)
        self._sync_delete_enabled()

    def _sync_delete_enabled(self, *_) -> None:
        data = self._branch_combo.currentData()
        self._rename_btn.setEnabled(data is not None)
        self._delete_btn.setEnabled(data is not None and data[1] != 0)

    def _on_rename_clicked(self) -> None:
        data = self._branch_combo.currentData()
        if data is not None:
            self.rename_requested.emit(*data)

    def _on_delete_clicked(self) -> None:
        data = self._branch_combo.currentData()
        if data is not None:
            self.delete_requested.emit(data[0], data[1], self._fill_hole_check.isChecked())
