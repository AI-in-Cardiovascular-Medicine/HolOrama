from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from domain.colors import CATEGORICAL_PALETTE
from gui import settings_io

DEFAULT_CCTA_SETTINGS: dict[str, Any] = {
    'windowing_sensitivity': 0.03,
    'zoom_sensitivity': 0.005,
    'default_mask_alpha': 0.45,
}

# windowing/zoom sensitivity and the default mask alpha are shared across pages
# (config.common); label_colors is CCTA-only (config.ccta). See gui/settings_io.py.
_KEY_SECTIONS: dict[str, str] = {
    'windowing_sensitivity': 'common',
    'zoom_sensitivity': 'common',
    'default_mask_alpha': 'common',
    'label_colors': 'ccta',
}

# (label, slider_min, slider_max, to_slider, from_slider, value_fmt)
_GATED_SLIDER_SPECS = {
    'windowing_sensitivity': (
        'Windowing Sensitivity',
        1,
        200,
        lambda v: round(v * 1000),
        lambda s: s / 1000.0,
        lambda v: f'{v:.3f}',
    ),
    'zoom_sensitivity': (
        'Zoom Sensitivity',
        1,
        100,
        lambda v: round(v * 2000),
        lambda s: s / 2000.0,
        lambda v: f'{v:.4f}',
    ),
    'default_mask_alpha': (
        'Default Mask Alpha',
        0,
        100,
        lambda v: round(v * 100),
        lambda s: s / 100.0,
        lambda v: f'{v:.2f}',
    ),
}


def _as_color_tuple(value: Any) -> tuple[int, int, int]:
    """label_colors round-trips through YAML as a list of lists — normalize any given
    entry to a plain (r, g, b) int tuple."""
    r, g, b = value
    return int(r), int(g), int(b)


def resolve_label_colors(config) -> tuple[tuple[int, int, int], ...]:
    """The base CCTA label palette per the live config (config.ccta.label_colors),
    falling back to the shared CATEGORICAL_PALETTE if unset. Used both by this dialog
    and by CctaPage to construct its displays with the right starting palette."""
    ccta_cfg = getattr(config, 'ccta', None)
    raw = getattr(ccta_cfg, 'label_colors', None) if ccta_cfg is not None else None
    if not raw:
        return CATEGORICAL_PALETTE
    return tuple(_as_color_tuple(c) for c in raw)


class CctaSettingsDialog(QDialog):
    def __init__(self, ccta_page) -> None:
        super().__init__(ccta_page)
        self.ccta_page = ccta_page
        self.setWindowTitle('CCTA Settings')

        current = settings_io.current_values(ccta_page.config, _KEY_SECTIONS, DEFAULT_CCTA_SETTINGS)
        current_label_colors = resolve_label_colors(ccta_page.config)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self._gated_controls: dict[str, tuple[QCheckBox, QSlider, Callable[[int], Any]]] = {}
        self._to_slider_fns: dict[str, Callable[[Any], int]] = {}
        for key, (label, lo, hi, to_slider, from_slider, fmt) in _GATED_SLIDER_SPECS.items():
            row, checkbox, slider = self._make_gated_slider_row(lo, hi, to_slider(current[key]), from_slider, fmt)
            form.addRow(label, row)
            self._gated_controls[key] = (checkbox, slider, from_slider)
            self._to_slider_fns[key] = to_slider

        self._label_colors_default: tuple[tuple[int, int, int], ...] = CATEGORICAL_PALETTE
        resolved = [_as_color_tuple(c) for c in current_label_colors]
        # Pad out to the full default palette length in case config.yaml was hand-edited
        # with fewer overrides than CATEGORICAL_PALETTE has entries.
        self._label_colors: list[tuple[int, int, int]] = [
            resolved[i] if i < len(resolved) else self._label_colors_default[i]
            for i in range(len(self._label_colors_default))
        ]
        self._color_swatches: list[QPushButton] = []
        for i in range(len(self._label_colors_default)):
            swatch = QPushButton()
            swatch.setFixedSize(28, 20)
            swatch.setStyleSheet(f'background-color: {self._color_hex(i)}; border: 1px solid #666;')
            swatch.clicked.connect(lambda _checked, idx=i: self._pick_color(idx))
            self._color_swatches.append(swatch)
            form.addRow(f'Label color {i + 1}', swatch)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        reset_btn = button_box.addButton('Reset to Defaults', QDialogButtonBox.ButtonRole.ResetRole)
        assert reset_btn is not None
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _color_hex(self, index: int) -> str:
        r, g, b = self._label_colors[index]
        return f'#{r:02x}{g:02x}{b:02x}'

    def _make_gated_slider_row(
        self, slider_min: int, slider_max: int, initial_slider_value: int, from_slider, value_fmt
    ) -> tuple[QWidget, QCheckBox, QSlider]:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        checkbox = QCheckBox()
        checkbox.setToolTip('Check to allow changing this value')
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(slider_min, slider_max)
        slider.setValue(initial_slider_value)
        slider.setEnabled(False)
        value_label = QLabel(value_fmt(from_slider(initial_slider_value)))
        value_label.setMinimumWidth(50)
        checkbox.toggled.connect(slider.setEnabled)
        slider.valueChanged.connect(lambda v: value_label.setText(value_fmt(from_slider(v))))
        h.addWidget(checkbox)
        h.addWidget(slider, 1)
        h.addWidget(value_label)
        return row, checkbox, slider

    def _pick_color(self, index: int) -> None:
        initial = QColor(*self._label_colors[index])
        color = QColorDialog.getColor(initial, self, f'Select Label Color {index + 1}')
        if color.isValid():
            self._label_colors[index] = (color.red(), color.green(), color.blue())
            self._color_swatches[index].setStyleSheet(
                f'background-color: {self._color_hex(index)}; border: 1px solid #666;'
            )

    def _reset_to_defaults(self) -> None:
        for key, (_checkbox, slider, _from_slider) in self._gated_controls.items():
            slider.setValue(self._to_slider_fns[key](DEFAULT_CCTA_SETTINGS[key]))
        for i, swatch in enumerate(self._color_swatches):
            self._label_colors[i] = self._label_colors_default[i]
            swatch.setStyleSheet(f'background-color: {self._color_hex(i)}; border: 1px solid #666;')

    def get_values(self) -> dict:
        values: dict[str, Any] = {}
        for key, (_checkbox, slider, from_slider) in self._gated_controls.items():
            raw = from_slider(slider.value())
            values[key] = raw if isinstance(DEFAULT_CCTA_SETTINGS[key], float) else int(raw)
        values['label_colors'] = [list(c) for c in self._label_colors]
        return values


def apply_and_save(ccta_page, values: dict) -> None:
    """Mutate the shared config, apply live to the CCTA page's displays, then persist."""
    settings_io.apply_values(ccta_page.config, _KEY_SECTIONS, values)
    ccta_page.apply_ccta_settings(values)

    config_path = settings_io.resolve_config_path(ccta_page.config)
    settings_io.save_values(config_path, _KEY_SECTIONS, values)
