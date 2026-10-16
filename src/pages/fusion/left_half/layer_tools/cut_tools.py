from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from domain.fusion.types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class CutToolbar(SceneToolbar):
    """Toolbar for the Inlets & Outlets scene: pick a slice of the vessel tree (Pick Point,
    then step with ◀ ▶), choose which side stays, and cut. The cuts are listed in order and
    can be renamed or removed; every change replays them on the chosen source mesh (see
    pages/fusion/cutting.py). Exporting is in the Fusion column."""

    source_changed = pyqtSignal(str)  # 'final' / 'stitched' / 'ccta'
    step_requested = pyqtSignal(int)  # -1 / +1 slice along the selected branch
    cut_requested = pyqtSignal(str, str)  # keep ('larger'/'proximal'/'distal'), name
    rename_requested = pyqtSignal(int)  # index into the cut list
    remove_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        self._source = QComboBox()
        self._source.setToolTip('Mesh the cuts are made on — any mesh works, only the planes come from the tree')
        self._selection = QLabel('Pick a slice')
        self._selection.setMinimumWidth(160)
        prev_btn = QPushButton('◀')
        next_btn = QPushButton('▶')
        for btn, step, tip in ((prev_btn, -1, 'Previous (proximal) slice'), (next_btn, 1, 'Next (distal) slice')):
            btn.setFixedWidth(28)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _, s=step: self.step_requested.emit(s))
        self._keep = QComboBox()
        for label, keep in (('Larger side', 'larger'), ('Proximal side', 'proximal'), ('Distal side', 'distal')):
            self._keep.addItem(label, keep)
        self._keep.setToolTip(
            'Side of the cut that stays: the one with the larger surface area, or the side\n'
            'against / along the centerline direction.'
        )
        self._name = QLineEdit()
        self._name.setPlaceholderText('Cap name')
        self._name.setToolTip("Names the cap's file on export")
        cut_btn = QPushButton('Cut')
        cut_btn.clicked.connect(self._on_cut)

        slice_row = QHBoxLayout()
        slice_row.addWidget(prev_btn)
        slice_row.addWidget(self._selection, 1)
        slice_row.addWidget(next_btn)
        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel('Source:'), 0, 0)
        grid.addWidget(self._source, 0, 1)
        grid.addWidget(QLabel('Slice:'), 1, 0)
        grid.addLayout(slice_row, 1, 1)
        grid.addWidget(QLabel('Keep:'), 2, 0)
        grid.addWidget(self._keep, 2, 1)
        grid.addWidget(QLabel('Name:'), 3, 0)
        grid.addWidget(self._name, 3, 1)
        grid.addWidget(cut_btn, 4, 0, 1, 2)

        self._cuts = QListWidget()
        self._cuts.setMinimumWidth(240)
        rename_btn = QPushButton('Rename')
        rename_btn.clicked.connect(lambda: self._emit_for_current(self.rename_requested))
        remove_btn = QPushButton('Remove')
        remove_btn.clicked.connect(lambda: self._emit_for_current(self.remove_requested))
        list_buttons = QHBoxLayout()
        list_buttons.addWidget(rename_btn)
        list_buttons.addWidget(remove_btn)
        cut_list = QWidget()
        list_layout = QVBoxLayout(cut_list)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel('Cuts:'))
        list_layout.addWidget(self._cuts, 1)
        list_layout.addLayout(list_buttons)

        super().__init__(
            FusionScene.INLETS_OUTLETS,
            extra_rows=[controls, cut_list],
            show_pick=True,
            parent=parent,
        )
        self._source.currentIndexChanged.connect(self._on_source_changed)

    def set_sources(self, choices: list[tuple[str, str]], selected: str | None) -> None:
        """choices are (label, key) pairs, latest mesh first."""
        self._source.blockSignals(True)
        self._source.clear()
        for label, key in choices:
            self._source.addItem(label, key)
        self._source.setCurrentIndex(max(0, self._source.findData(selected)))
        self._source.blockSignals(False)

    def set_selection(self, text: str | None, name: str = '') -> None:
        self._selection.setText(text or 'Pick a slice')
        self._name.setText(name)

    def set_cuts(self, entries: list[tuple[str, str | None]]) -> None:
        """entries are (text, why it could not be applied or None), in cut order."""
        row = self._cuts.currentRow()
        self._cuts.clear()
        for text, failure in entries:
            item = QListWidgetItem(f'⚠ {text}' if failure else text)
            if failure:
                item.setToolTip(f'Not applied on this mesh: {failure}')
            self._cuts.addItem(item)
        self._cuts.setCurrentRow(min(row, self._cuts.count() - 1))

    def _on_cut(self) -> None:
        self.cut_requested.emit(self._keep.currentData(), self._name.text())

    def _on_source_changed(self, index: int) -> None:
        key = self._source.itemData(index)
        if key is not None:
            self.source_changed.emit(key)

    def _emit_for_current(self, signal) -> None:
        row = self._cuts.currentRow()
        if row >= 0:
            signal.emit(row)
