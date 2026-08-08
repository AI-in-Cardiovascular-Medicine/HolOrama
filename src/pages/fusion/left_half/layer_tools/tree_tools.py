from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar, labeled


class TreeToolbar(SceneToolbar):
    """Toolbar for the Vessel Tree scene. There's nothing to show/hide here, the only
    interaction is choosing which RCA or LCA reference-point triplet to highlight, either
    from these dropdowns (RCA above, LCA below) or by clicking a reference marker in the
    scene (see FusionPage._on_point_picked), so show_layers is off. Only the RCA one feeds
    the alignment pipeline (see FusionPage._on_run_align*) — LCA is inspection-only, since
    intravascular alignment in this app is always against the RCA."""

    reference_selected = pyqtSignal(int)  # index into PyDiscretizedVesselTree.rca_references
    lca_reference_selected = pyqtSignal(int)  # index into PyDiscretizedVesselTree.lca_references

    def __init__(self, parent=None) -> None:
        self.reference_combo = QComboBox()
        self.lca_reference_combo = QComboBox()

        rows = QWidget()
        rows_layout = QVBoxLayout(rows)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.addWidget(labeled('RCA reference:', self.reference_combo))
        rows_layout.addWidget(labeled('LCA reference:', self.lca_reference_combo))

        super().__init__(
            FusionScene.VESSEL_TREE,
            extra_rows=[rows],
            show_layers=False,
            show_pick=True,
            parent=parent,
        )
        self.reference_combo.currentIndexChanged.connect(self._on_index_changed)
        self.lca_reference_combo.currentIndexChanged.connect(self._on_lca_index_changed)

    def set_references(self, labels: list[str]) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItems(labels)
        self.reference_combo.blockSignals(False)

    def set_selected_index(self, index: int) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.setCurrentIndex(index)
        self.reference_combo.blockSignals(False)

    def set_lca_references(self, labels: list[str]) -> None:
        self.lca_reference_combo.blockSignals(True)
        self.lca_reference_combo.clear()
        self.lca_reference_combo.addItems(labels)
        self.lca_reference_combo.blockSignals(False)

    def set_selected_lca_index(self, index: int) -> None:
        self.lca_reference_combo.blockSignals(True)
        self.lca_reference_combo.setCurrentIndex(index)
        self.lca_reference_combo.blockSignals(False)

    def _on_index_changed(self, index: int) -> None:
        if index >= 0:
            self.reference_selected.emit(index)

    def _on_lca_index_changed(self, index: int) -> None:
        if index >= 0:
            self.lca_reference_selected.emit(index)
