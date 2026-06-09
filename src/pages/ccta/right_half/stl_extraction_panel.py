# This combines aortic root, coronaries and allows to cut-off LVOT into one new combined mask, which can be exported as a STL for fluid dynamics
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFrame,
    QButtonGroup,
    QRadioButton,
)
from PyQt6.QtCore import pyqtSignal


def _separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


class StlExtractionPanel(QWidget):
    """Panel for extracting and exporting the aortic root with coronaries and LVOT."""

    line_draw_requested = pyqtSignal(int)  # 0 = coronal line, 1 = sagittal line
    extract_requested = pyqtSignal(int, int, int, str)  # coronaries_label, aorta_label, lv_label, format

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        title = QLabel('Aortic Root with Coronaries')
        title.setStyleSheet('font-weight: bold;')
        root.addWidget(title)
        root.addWidget(_separator())

        # Mask selectors
        self._coronaries_combo = QComboBox()
        self._aorta_combo = QComboBox()
        self._lv_combo = QComboBox()

        for text, combo in [
            ('Coronaries:', self._coronaries_combo),
            ('Aorta:', self._aorta_combo),
            ('LV:', self._lv_combo),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(text)
            lbl.setFixedWidth(72)
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            root.addLayout(row)
            combo.currentIndexChanged.connect(self._update_extract_btn)

        root.addWidget(_separator())

        # Cut-line buttons with status indicators
        self._line_status: list[QLabel] = []
        for i, view in enumerate(['coronal', 'sagittal']):
            row = QHBoxLayout()
            btn = QPushButton(f'Draw cut line ({view})')
            btn.clicked.connect(lambda _, idx=i: self.line_draw_requested.emit(idx))
            status = QLabel('○')
            status.setFixedWidth(18)
            status.setStyleSheet('color: #888; font-size: 14px;')
            row.addWidget(btn, 1)
            row.addWidget(status)
            root.addLayout(row)
            self._line_status.append(status)

        root.addWidget(_separator())

        # Export format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel('Export as:'))
        self._fmt_group = QButtonGroup(self)
        for fmt, text in [('nifti', 'NIfTI'), ('stl', 'STL')]:
            rb = QRadioButton(text)
            rb.setProperty('fmt', fmt)
            self._fmt_group.addButton(rb)
            fmt_row.addWidget(rb)
        self._fmt_group.buttons()[0].setChecked(True)
        fmt_row.addStretch()
        root.addLayout(fmt_row)

        self._extract_btn = QPushButton('Extract && Export')
        self._extract_btn.setEnabled(False)
        self._extract_btn.clicked.connect(self._on_extract)
        root.addWidget(self._extract_btn)

        self._lines_drawn = [False, False]

    def set_labels(self, labels: list[int], names: dict[int, str]) -> None:
        """Populate the mask-selector dropdowns."""
        for combo in (self._coronaries_combo, self._aorta_combo, self._lv_combo):
            combo.blockSignals(True)
            combo.clear()
            for lv in labels:
                combo.addItem(names.get(lv, f'Label {lv}'), userData=lv)
            combo.blockSignals(False)
        self._update_extract_btn()

    def update_label_name(self, label: int, name: str) -> None:
        """Sync a renamed label across all dropdowns."""
        for combo in (self._coronaries_combo, self._aorta_combo, self._lv_combo):
            for i in range(combo.count()):
                if combo.itemData(i) == label:
                    combo.setItemText(i, name)
                    break

    def set_line_drawn(self, index: int) -> None:
        """Mark a cut line as complete (called by page.py after line_drawn signal)."""
        self._lines_drawn[index] = True
        self._line_status[index].setText('●')
        self._line_status[index].setStyleSheet('color: #44cc44; font-size: 14px;')
        self._update_extract_btn()

    def _update_extract_btn(self) -> None:
        labels_ok = all(c.count() > 0 for c in (self._coronaries_combo, self._aorta_combo, self._lv_combo))
        self._extract_btn.setEnabled(labels_ok and all(self._lines_drawn))

    def _on_extract(self) -> None:
        cor = self._coronaries_combo.currentData()
        aorta = self._aorta_combo.currentData()
        lv = self._lv_combo.currentData()
        fmt = next(
            (btn.property('fmt') for btn in self._fmt_group.buttons() if btn.isChecked()),
            'nifti',
        )
        self.extract_requested.emit(cor, aorta, lv, fmt)
