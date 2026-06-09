from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QLineEdit,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from domain.ccta_display_types import LABEL_COLORS, DEFAULT_MASK_ALPHA


class _LabelRow(QWidget):
    visibility_changed = pyqtSignal(bool)
    name_changed = pyqtSignal(str)

    def __init__(self, label: int, color: tuple[int, int, int], parent=None) -> None:
        super().__init__(parent)
        self.label_value = label

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self._checkbox = QCheckBox()
        self._checkbox.setChecked(True)
        self._checkbox.setToolTip('Show / hide this label')
        self._checkbox.toggled.connect(self.visibility_changed)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        r, g, b = color
        swatch.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 1px solid #666; border-radius: 2px;')

        self._name_edit = QLineEdit(f'Label {label}')
        self._name_edit.setPlaceholderText(f'Label {label}')
        self._name_edit.textChanged.connect(self._on_name_changed)

        num_lbl = QLabel(str(label))
        num_lbl.setFixedWidth(22)
        num_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        num_lbl.setStyleSheet('color: #888; font-size: 10px;')

        layout.addWidget(self._checkbox)
        layout.addWidget(swatch)
        layout.addWidget(self._name_edit, 1)
        layout.addWidget(num_lbl)

    @property
    def name(self) -> str:
        return self._name_edit.text() or f'Label {self.label_value}'

    @property
    def visible(self) -> bool:
        return self._checkbox.isChecked()

    def _on_name_changed(self, text: str) -> None:
        self.name_changed.emit(text or f'Label {self.label_value}')


class MaskPanel(QWidget):
    """Side panel for controlling mask overlay: opacity and per-label visibility + names."""

    alpha_changed = pyqtSignal(float)  # 0.0–1.0
    label_visibility_changed = pyqtSignal(int, bool)  # label_value, visible
    label_name_changed = pyqtSignal(int, str)  # label_value, name

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(210)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(QLabel('Mask opacity'))

        alpha_row = QHBoxLayout()
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(round(DEFAULT_MASK_ALPHA * 100))
        self._alpha_value_lbl = QLabel(f'{round(DEFAULT_MASK_ALPHA * 100)}%')
        self._alpha_value_lbl.setFixedWidth(34)
        self._alpha_value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        alpha_row.addWidget(self._alpha_slider)
        alpha_row.addWidget(self._alpha_value_lbl)
        root.addLayout(alpha_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel('Labels'))
        self._all_cb = QCheckBox('All')
        self._all_cb.setChecked(True)
        self._all_cb.setToolTip('Show / hide all labels')
        self._all_cb.toggled.connect(self._on_toggle_all)
        header_row.addStretch()
        header_row.addWidget(self._all_cb)
        root.addLayout(header_row)

        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.addStretch()
        root.addWidget(self._rows_widget)

        self._rows: dict[int, _LabelRow] = {}

    def set_labels(self, labels: list[int]) -> None:
        """Populate the label list. Colors are assigned by position in LABEL_COLORS."""
        self._clear_rows()
        for i, label in enumerate(labels):
            color = LABEL_COLORS[i % len(LABEL_COLORS)]
            row = _LabelRow(label, color)
            row.visibility_changed.connect(lambda visible, lbl=label: self.label_visibility_changed.emit(lbl, visible))
            row.name_changed.connect(lambda name, lbl=label: self.label_name_changed.emit(lbl, name))
            # Insert before the trailing stretch
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
            self._rows[label] = row
        self._all_cb.setChecked(True)

    def clear_labels(self) -> None:
        self._clear_rows()

    def label_names(self) -> dict[int, str]:
        """Return the current user-defined name for every label."""
        return {label: row.name for label, row in self._rows.items()}

    def set_brush_panel(self, panel: 'QWidget') -> None:
        """Attach a widget below the label scroll area (called once at setup)."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root: QVBoxLayout = self.layout()  # type: ignore[assignment]
        root.addWidget(sep)
        root.addWidget(panel)
        root.addStretch(1)  # remaining space below the brush panel

    def _on_alpha_changed(self, value: int) -> None:
        self._alpha_value_lbl.setText(f'{value}%')
        self.alpha_changed.emit(value / 100.0)

    def _on_toggle_all(self, checked: bool) -> None:
        for row in self._rows.values():
            row._checkbox.blockSignals(True)
            row._checkbox.setChecked(checked)
            row._checkbox.blockSignals(False)
            self.label_visibility_changed.emit(row.label_value, checked)

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
