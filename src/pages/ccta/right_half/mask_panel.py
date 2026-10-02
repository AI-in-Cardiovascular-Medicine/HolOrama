from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from domain.ccta.display_types import DEFAULT_MASK_ALPHA
from domain.ccta.presets import fallback_color


class _LabelRow(QWidget):
    """One label of the mask: whether it is shown, and the colour and name its preset gives
    it. The name is renamed in CCTA Contour Settings, not here."""

    visibility_changed = pyqtSignal(bool)

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

        self._swatch = QLabel()
        self._swatch.setFixedSize(14, 14)
        swatch = self._swatch
        r, g, b = color
        swatch.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 1px solid #666; border-radius: 2px;')

        self._name_label = QLabel(f'Label {label}')

        num_lbl = QLabel(str(label))
        num_lbl.setFixedWidth(22)
        num_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        num_lbl.setStyleSheet('color: #888; font-size: 10px;')

        layout.addWidget(self._checkbox)
        layout.addWidget(swatch)
        layout.addWidget(self._name_label, 1)
        layout.addWidget(num_lbl)

    @property
    def name(self) -> str:
        return self._name_label.text()

    @property
    def visible(self) -> bool:
        return self._checkbox.isChecked()

    def set_color(self, color: tuple[int, int, int]) -> None:
        r, g, b = color
        self._swatch.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 1px solid #666; border-radius: 2px;')

    def set_label_name(self, name: str) -> None:
        self._name_label.setText(name)


class MaskPanel(QWidget):
    """Side panel for controlling mask overlay: opacity and per-label visibility, with each
    label's name and colour as the active CCTA preset gives them (see set_label_appearance)."""

    alpha_changed = pyqtSignal(float)  # 0.0–1.0
    label_visibility_changed = pyqtSignal(int, bool)  # label_value, visible
    switch_default_requested = pyqtSignal()

    def __init__(
        self,
        initial_alpha: float = DEFAULT_MASK_ALPHA,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(210)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(QLabel('Mask opacity'))

        alpha_row = QHBoxLayout()
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(round(initial_alpha * 100))
        self._alpha_value_lbl = QLabel(f'{round(initial_alpha * 100)}%')
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
        self._btn_switch = QPushButton('Switch default')
        self._btn_switch.setToolTip(
            'Switch between the two built-in presets, Colorful and Publication. Names and colours '
            'are edited in Settings > CCTA Contour Settings'
        )
        self._btn_switch.setFixedHeight(22)
        self._btn_switch.clicked.connect(self.switch_default_requested)
        header_row.addWidget(self._btn_switch)
        header_row.addWidget(self._all_cb)
        root.addLayout(header_row)

        self._preset_lbl = QLabel()  # the preset the labels are named and coloured after
        self._preset_lbl.setStyleSheet('color: #888; font-size: 10px;')
        self._preset_lbl.setWordWrap(True)
        root.addWidget(self._preset_lbl)

        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.addStretch()

        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rows_scroll.setWidget(self._rows_widget)
        root.addWidget(rows_scroll, 1)

        self._rows: dict[int, _LabelRow] = {}

    def set_labels(self, labels: list[int]) -> None:
        """Populate the label list, each row in its palette colour and as 'Label <value>'
        until set_label_appearance names and colours it."""
        self._clear_rows()
        for label in labels:
            row = _LabelRow(label, fallback_color(label))
            row.visibility_changed.connect(lambda visible, lbl=label: self.label_visibility_changed.emit(lbl, visible))
            # Insert before the trailing stretch
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
            self._rows[label] = row
        self._all_cb.setChecked(True)

    def set_label_appearance(self, preset_name: str, names: dict[int, str], colors: dict[int, tuple]) -> None:
        """Show each label with the name and colour the active preset (`preset_name`) gives it."""
        self._preset_lbl.setText(preset_name)
        for label, row in self._rows.items():
            row.set_label_name(names.get(label, f'Label {label}'))
            if label in colors:
                row.set_color(colors[label])

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

    def set_default_alpha(self, alpha: float) -> None:
        """Push a new default mask alpha (e.g. from Settings) into the slider — the
        existing valueChanged -> alpha_changed wiring propagates it live."""
        self._alpha_slider.setValue(round(alpha * 100))

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
