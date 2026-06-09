from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal

from domain.ccta_display_types import LABEL_COLORS
from tools.painting import BrushGeometry

_ERASE_COLOR: tuple[int, int, int] = (160, 160, 160)


class BrushPanel(QWidget):
    """
    Brush controls embedded below the mask label list.

    Signals:
        brush_enabled_changed(bool) -> Enable checkbox toggled
        geometry_changed(BrushGeometry) -> any control changed while enabled
    """

    brush_enabled_changed = pyqtSignal(bool)
    geometry_changed = pyqtSignal(object)  # BrushGeometry

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._labels: list[int] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 8)
        root.setSpacing(6)

        # Enable toggle
        self._enable_cb = QCheckBox('Enable brush')
        self._enable_cb.toggled.connect(self._on_enabled_toggled)
        root.addWidget(self._enable_cb)

        # All controls below; visually disabled when checkbox is off
        self._controls = QWidget()
        ctl = QVBoxLayout(self._controls)
        ctl.setContentsMargins(0, 0, 0, 0)
        ctl.setSpacing(6)

        # Label selector + colour swatch
        ctl.addWidget(QLabel('Label'))
        label_row = QHBoxLayout()
        label_row.setSpacing(6)
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        label_row.addWidget(self._combo, 1)
        self._swatch = QLabel()
        self._swatch.setFixedSize(16, 16)
        self._swatch.setStyleSheet('border: 1px solid #666; border-radius: 2px;')
        label_row.addWidget(self._swatch)
        ctl.addLayout(label_row)

        # Add / Erase radio buttons
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self._add_rb = QRadioButton('Add')
        self._add_rb.setChecked(True)
        self._erase_rb = QRadioButton('Erase')
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._add_rb)
        self._mode_group.addButton(self._erase_rb)
        self._mode_group.buttonToggled.connect(lambda *_: self._emit())
        mode_row.addWidget(self._add_rb)
        mode_row.addWidget(self._erase_rb)
        mode_row.addStretch()
        ctl.addLayout(mode_row)

        # Radius slider
        radius_hdr = QHBoxLayout()
        radius_hdr.addWidget(QLabel('Radius (px)'))
        self._radius_lbl = QLabel('10')
        self._radius_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        radius_hdr.addWidget(self._radius_lbl)
        ctl.addLayout(radius_hdr)

        self._radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._radius_slider.setRange(1, 50)
        self._radius_slider.setValue(10)
        self._radius_slider.valueChanged.connect(self._on_radius_changed)
        ctl.addWidget(self._radius_slider)

        root.addWidget(self._controls)
        self._controls.setEnabled(False)

    def set_labels(self, labels: list[int]) -> None:
        """Populate the label combo. Call whenever a mask is loaded or created."""
        self._labels = labels
        self._combo.blockSignals(True)
        self._combo.clear()
        for label in labels:
            self._combo.addItem(f'Label {label}', userData=label)
        self._combo.blockSignals(False)
        self._combo.setCurrentIndex(0 if labels else -1)
        self._update_swatch()
        if self._enable_cb.isChecked() and labels:
            self._emit()

    def current_geometry(self) -> BrushGeometry | None:
        """Return brush geometry for the current control state, or None if no labels."""
        radius = self._radius_slider.value()
        if self._erase_rb.isChecked():
            return BrushGeometry(label=0, color=_ERASE_COLOR, radius_px=radius)
        idx = self._combo.currentIndex()
        if idx < 0 or idx >= len(self._labels):
            return None
        label = self._labels[idx]
        color = LABEL_COLORS[idx % len(LABEL_COLORS)]
        return BrushGeometry(label=label, color=color, radius_px=radius)

    def _on_enabled_toggled(self, checked: bool) -> None:
        self._controls.setEnabled(checked)
        self.brush_enabled_changed.emit(checked)

    def _on_combo_changed(self, _idx: int) -> None:
        self._update_swatch()
        self._emit()

    def _on_radius_changed(self, value: int) -> None:
        self._radius_lbl.setText(str(value))
        self._emit()

    def _update_swatch(self) -> None:
        idx = self._combo.currentIndex()
        if 0 <= idx < len(self._labels):
            r, g, b = LABEL_COLORS[idx % len(LABEL_COLORS)]
            self._swatch.setStyleSheet(
                f'background-color: rgb({r},{g},{b}); border: 1px solid #666; border-radius: 2px;'
            )
        else:
            self._swatch.setStyleSheet('border: 1px solid #666; border-radius: 2px;')

    def _emit(self) -> None:
        if not self._enable_cb.isChecked():
            return
        geo = self.current_geometry()
        if geo is not None:
            self.geometry_changed.emit(geo)
