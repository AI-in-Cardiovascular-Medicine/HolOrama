from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox

from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar, labeled


class TreeToolbar(SceneToolbar):
    """Toolbar for the Vessel Tree scene. There's nothing to show/hide here — the only
    interaction is choosing which RCA reference-point triplet to align on, either from
    this dropdown or by clicking a reference marker in the scene (see
    FusionPage._on_point_picked), so show_layers is off."""

    reference_selected = pyqtSignal(int)  # index into PyDiscretizedVesselTree.rca_references

    def __init__(self, parent=None) -> None:
        self.reference_combo = QComboBox()
        super().__init__(
            FusionScene.VESSEL_TREE,
            extra_rows=[labeled('RCA reference:', self.reference_combo)],
            show_layers=False,
            parent=parent,
        )
        self.reference_combo.currentIndexChanged.connect(self._on_index_changed)

    def set_references(self, labels: list[str]) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItems(labels)
        self.reference_combo.blockSignals(False)

    def set_selected_index(self, index: int) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.setCurrentIndex(index)
        self.reference_combo.blockSignals(False)

    def _on_index_changed(self, index: int) -> None:
        if index >= 0:
            self.reference_selected.emit(index)
