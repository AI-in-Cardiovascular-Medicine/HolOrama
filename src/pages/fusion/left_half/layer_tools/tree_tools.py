from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox

from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar, labeled


class TreeToolbar(SceneToolbar):
    """Toolbar for the Vessel Tree scene: branch selector plus the usual layer/pick controls."""

    branch_changed = pyqtSignal(str)  # branch key, e.g. 'rca_main', 'lca_side_1'

    def __init__(self, parent=None) -> None:
        self.branch_combo = QComboBox()
        super().__init__(FusionScene.VESSEL_TREE, extra_rows=[labeled('Branch:', self.branch_combo)], parent=parent)
        self.branch_combo.currentTextChanged.connect(self.branch_changed.emit)

    def set_branches(self, branch_keys: list[str]) -> None:
        self.branch_combo.blockSignals(True)
        self.branch_combo.clear()
        self.branch_combo.addItems(branch_keys)
        self.branch_combo.blockSignals(False)
