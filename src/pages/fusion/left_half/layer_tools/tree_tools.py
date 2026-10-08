from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from domain.fusion.types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar, labeled


class TreeToolbar(SceneToolbar):
    """Toolbar for the Vessel Tree scene. There's nothing to show/hide here, the only
    interaction is choosing which reference-point triplet to highlight per vessel (RCA
    row above, LCA row below): first the branch (main or a side branch), then one of the
    references reachable on that branch, or by clicking a reference marker in the scene
    (see FusionPage._on_point_picked), so show_layers is off. Whichever vessel the
    Intravascular Alignment column has chosen follows its row here."""

    branch_selected = pyqtSignal(str, int)  # vessel ('rca'/'lca'), branch (0 = main, n = side branch n)
    reference_selected = pyqtSignal(str, int)  # vessel, index into that branch's reference list

    def __init__(self, parent=None) -> None:
        self._branch_combos: dict[str, QComboBox] = {}
        self._reference_combos: dict[str, QComboBox] = {}

        rows = QWidget()
        rows_layout = QVBoxLayout(rows)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        for vessel in ('rca', 'lca'):
            branch_combo = QComboBox()
            reference_combo = QComboBox()
            self._branch_combos[vessel] = branch_combo
            self._reference_combos[vessel] = reference_combo
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(labeled(f'{vessel.upper()}:', branch_combo))
            row_layout.addWidget(labeled('Reference:', reference_combo))
            rows_layout.addWidget(row)

        super().__init__(
            FusionScene.VESSEL_TREE,
            extra_rows=[rows],
            show_layers=False,
            show_pick=True,
            parent=parent,
        )
        for vessel in ('rca', 'lca'):
            self._branch_combos[vessel].currentIndexChanged.connect(lambda _, v=vessel: self._on_branch_changed(v))
            self._reference_combos[vessel].currentIndexChanged.connect(
                lambda index, v=vessel: self._on_reference_changed(v, index)
            )

    def set_branches(self, vessel: str, choices: list[tuple[str, int]], selected: int) -> None:
        """choices are (label, branch id) pairs; `selected` is the branch id to show."""
        combo = self._branch_combos[vessel]
        combo.blockSignals(True)
        combo.clear()
        for label, branch in choices:
            combo.addItem(label, branch)
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        combo.blockSignals(False)

    def set_selected_branch(self, vessel: str, branch: int) -> None:
        combo = self._branch_combos[vessel]
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, combo.findData(branch)))
        combo.blockSignals(False)

    def set_references(self, vessel: str, labels: list[str], selected: int) -> None:
        combo = self._reference_combos[vessel]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(labels)
        combo.setCurrentIndex(selected)
        combo.blockSignals(False)

    def set_selected_reference(self, vessel: str, index: int) -> None:
        combo = self._reference_combos[vessel]
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _on_reference_changed(self, vessel: str, index: int) -> None:
        if index >= 0:
            self.reference_selected.emit(vessel, index)

    def _on_branch_changed(self, vessel: str) -> None:
        branch = self._branch_combos[vessel].currentData()
        if branch is not None:
            self.branch_selected.emit(vessel, branch)
