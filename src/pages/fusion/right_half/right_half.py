from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QSplitter, QVBoxLayout

from pages.fusion.right_half.fusion_column import FusionColumn
from pages.fusion.right_half.geometry_column import GeometryColumn
from pages.fusion.right_half.intravascular_column import IntravascularColumn
from pages.intravascular.utils.helpers import SplitterPane


class RightHalf:
    """Three side-by-side columns: CCTA geometry/centerlines, intravascular alignment,
    fusion. Each column is its own widget (see the *_column.py files); this class just
    lays them out and owns the scroll areas, since every column has more controls than
    fit in view at once."""

    def __init__(self, main_window) -> None:
        self.main_window = main_window
        self.widget = SplitterPane()

        self.geometry_column = GeometryColumn()
        self.intravascular_column = IntravascularColumn()
        self.fusion_column = FusionColumn()

        splitter = QSplitter(Qt.Orientation.Horizontal, self.widget)
        for column in (self.geometry_column, self.intravascular_column, self.fusion_column):
            splitter.addWidget(_scrollable(column))
        splitter.setSizes([1, 1, 1])

        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def __call__(self):
        return self.widget


def _scrollable(widget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area
