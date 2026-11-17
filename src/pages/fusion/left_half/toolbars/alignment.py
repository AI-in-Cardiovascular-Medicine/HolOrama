from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from domain.fusion.types import FusionScene
from pages.fusion.left_half.toolbars.base import SceneToolbar


class IntravascularLoadedToolbar(SceneToolbar):
    """Toolbar for the Intravascular Loaded scene: the raw geometry before centerline
    alignment, to check whether a bad result was already wrong before alignment."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.INTRAVASCULAR_LOADED, parent=parent)


class AlignmentToolbar(SceneToolbar):
    """Toolbar for the Intravascular Aligned scene. After Remove Labeled Points it lists where
    the stitch will cut (end rims, side branches): pick a slice per target, like an outlet."""

    target_selected = pyqtSignal(int)  # index into the target list
    step_requested = pyqtSignal(int)  # -1 / +1 slice along the target's branch
    default_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        self._targets = QListWidget()
        self._targets.setMinimumWidth(110)
        self._targets.setToolTip(
            'Where the stitch cuts (needs the vessel tree): the end rims of the opening and the\n'
            'side branches it reconnects. Select one, then Pick Point on one of its slices.'
        )
        self._selection = QLabel('Remove labeled points first')
        self._selection.setMinimumWidth(60)
        prev_btn = QPushButton('◀')
        next_btn = QPushButton('▶')
        for btn, step, tip in ((prev_btn, -1, 'Previous (proximal) slice'), (next_btn, 1, 'Next (distal) slice')):
            btn.setFixedWidth(28)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _, s=step: self.step_requested.emit(s))
        default_btn = QPushButton('Default')
        default_btn.setToolTip("Back to multimodars' choice")
        default_btn.clicked.connect(lambda: self.default_requested.emit())

        slice_row = QHBoxLayout()
        slice_row.addWidget(prev_btn)
        slice_row.addWidget(self._selection, 1)
        slice_row.addWidget(next_btn)
        slice_row.addWidget(default_btn)
        cuts = QWidget()
        layout = QVBoxLayout(cuts)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Stitch cuts:'))
        layout.addWidget(self._targets, 1)
        layout.addLayout(slice_row)

        super().__init__(FusionScene.INTRAVASCULAR_ALIGNED, extra_rows=[cuts], show_pick=True, parent=parent)
        self._targets.currentRowChanged.connect(self._on_row_changed)

    def set_targets(self, entries: list[tuple[str, str | None]], selected: int, selection: str) -> None:
        """entries are (text, warning or None) per target, `selection` the selected one's slice."""
        self._targets.blockSignals(True)
        self._targets.clear()
        for text, warning in entries:
            item = QListWidgetItem(f'⚠ {text}' if warning else text)
            if warning:
                item.setToolTip(warning)
            self._targets.addItem(item)
        self._targets.setCurrentRow(selected if entries else -1)
        self._targets.blockSignals(False)
        self._selection.setText(selection)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.target_selected.emit(row)
