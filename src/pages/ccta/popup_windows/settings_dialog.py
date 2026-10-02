from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui import settings_io

DEFAULT_CCTA_SETTINGS: dict[str, Any] = {
    'windowing_sensitivity': 0.03,
    'zoom_sensitivity': 0.005,
    'default_mask_alpha': 0.45,
}

# windowing/zoom sensitivity and the default mask alpha are shared across pages
# (config.common). Each label's name and colour is its CCTA preset's (see
# domain.ccta.presets), edited in CCTA Contour Settings. See gui/settings_io.py.
_KEY_SECTIONS: dict[str, str] = {
    'windowing_sensitivity': 'common',
    'zoom_sensitivity': 'common',
    'default_mask_alpha': 'common',
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


class CctaSettingsDialog(QDialog):
    def __init__(self, ccta_page) -> None:
        super().__init__(ccta_page)
        self.ccta_page = ccta_page
        self.setWindowTitle('CCTA Settings')

        current = settings_io.current_values(ccta_page.config, _KEY_SECTIONS, DEFAULT_CCTA_SETTINGS)

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

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        reset_btn = button_box.addButton('Reset to Defaults', QDialogButtonBox.ButtonRole.ResetRole)
        assert reset_btn is not None
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

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

    def _reset_to_defaults(self) -> None:
        for key, (_checkbox, slider, _from_slider) in self._gated_controls.items():
            slider.setValue(self._to_slider_fns[key](DEFAULT_CCTA_SETTINGS[key]))

    def get_values(self) -> dict:
        values: dict[str, Any] = {}
        for key, (_checkbox, slider, from_slider) in self._gated_controls.items():
            raw = from_slider(slider.value())
            values[key] = raw if isinstance(DEFAULT_CCTA_SETTINGS[key], float) else int(raw)
        return values


def apply_and_save(ccta_page, values: dict) -> None:
    """Mutate the shared config, apply live to the CCTA page's displays, then persist."""
    settings_io.apply_values(ccta_page.config, _KEY_SECTIONS, values)
    ccta_page.apply_ccta_settings(values)

    config_path = settings_io.resolve_config_path(ccta_page.config)
    settings_io.save_values(config_path, _KEY_SECTIONS, values)
