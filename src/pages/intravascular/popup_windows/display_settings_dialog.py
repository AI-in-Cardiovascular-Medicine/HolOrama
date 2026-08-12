from pathlib import Path
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

DEFAULT_DISPLAY_SETTINGS: dict[str, Any] = {
    'windowing_sensitivity': 0.03,
    'zoom_sensitivity': 0.005,
    'alpha_contour': 100,
    'n_points_contour': 200,
    'n_interactive_points': 20,
    'contour_thickness': 2,
    'point_thickness': 1,
    'point_radius': 10,
    'color_contour': 'green',
    'color_eem': '#03b1fc',
    'color_calcium': 'white',
    'color_branch': '#006400',
    'color_start_point': 'yellow',
    'color_end_point': 'red',
    'color_angle': '#ffa500',
}

_COLOR_LABELS = {
    'color_contour': 'Change Color Contour',
    'color_eem': 'Change Color EEM',
    'color_calcium': 'Change Color Calcium',
    'color_branch': 'Change Color Branch',
    'color_start_point': 'Change Color Start Point',
    'color_end_point': 'Change Color End Point',
    'color_angle': 'Change Color Angle',
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
    'alpha_contour': (
        'Alpha Contour',
        0,
        255,
        lambda v: int(v),
        lambda s: int(s),
        lambda v: str(int(v)),
    ),
    'n_points_contour': (
        'Number of Contour Points',
        1,
        10,
        lambda v: round(v / 100),
        lambda s: s * 100,
        lambda v: str(int(v)),
    ),
}


class DisplaySettingsDialog(QDialog):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle('Display Settings')

        cfg = main_window.config.display
        current = {k: getattr(cfg, k, v) for k, v in DEFAULT_DISPLAY_SETTINGS.items()}

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

        self._n_interactive_points_box = QSpinBox()
        self._n_interactive_points_box.setRange(4, 200)
        self._n_interactive_points_box.setValue(int(current['n_interactive_points']))
        form.addRow('Number of Interactive Points', self._n_interactive_points_box)

        self._contour_thickness_box = QSpinBox()
        self._contour_thickness_box.setRange(1, 20)
        self._contour_thickness_box.setValue(int(current['contour_thickness']))
        form.addRow('Contour Thickness', self._contour_thickness_box)

        self._point_thickness_box = QSpinBox()
        self._point_thickness_box.setRange(1, 20)
        self._point_thickness_box.setValue(int(current['point_thickness']))
        form.addRow('Point Thickness', self._point_thickness_box)

        self._point_radius_box = QSpinBox()
        self._point_radius_box.setRange(1, 50)
        self._point_radius_box.setValue(int(current['point_radius']))
        form.addRow('Point Radius', self._point_radius_box)

        self._color_values: dict[str, str] = {k: current[k] for k in _COLOR_LABELS}
        self._color_swatches: dict[str, QPushButton] = {}
        for key, label in _COLOR_LABELS.items():
            swatch = QPushButton()
            swatch.setFixedSize(28, 20)
            swatch.setStyleSheet(f'background-color: {self._color_values[key]}; border: 1px solid #666;')
            swatch.clicked.connect(lambda _checked, k=key: self._pick_color(k))
            self._color_swatches[key] = swatch
            form.addRow(label, swatch)

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

    def _pick_color(self, key: str) -> None:
        initial = QColor(self._color_values[key])
        color = QColorDialog.getColor(initial, self, f'Select {key}')
        if color.isValid():
            hex_value = color.name()
            self._color_values[key] = hex_value
            self._color_swatches[key].setStyleSheet(f'background-color: {hex_value}; border: 1px solid #666;')

    def _reset_to_defaults(self) -> None:
        for key, (_checkbox, slider, _from_slider) in self._gated_controls.items():
            slider.setValue(self._to_slider_fns[key](DEFAULT_DISPLAY_SETTINGS[key]))
        self._n_interactive_points_box.setValue(DEFAULT_DISPLAY_SETTINGS['n_interactive_points'])
        self._contour_thickness_box.setValue(DEFAULT_DISPLAY_SETTINGS['contour_thickness'])
        self._point_thickness_box.setValue(DEFAULT_DISPLAY_SETTINGS['point_thickness'])
        self._point_radius_box.setValue(DEFAULT_DISPLAY_SETTINGS['point_radius'])
        for key, swatch in self._color_swatches.items():
            default = DEFAULT_DISPLAY_SETTINGS[key]
            self._color_values[key] = default
            swatch.setStyleSheet(f'background-color: {default}; border: 1px solid #666;')

    def get_values(self) -> dict:
        values = {}
        for key, (_checkbox, slider, from_slider) in self._gated_controls.items():
            raw = from_slider(slider.value())
            values[key] = raw if isinstance(DEFAULT_DISPLAY_SETTINGS[key], float) else int(raw)
        values['n_interactive_points'] = self._n_interactive_points_box.value()
        values['contour_thickness'] = self._contour_thickness_box.value()
        values['point_thickness'] = self._point_thickness_box.value()
        values['point_radius'] = self._point_radius_box.value()
        values.update(self._color_values)
        return values


def save_display_settings(config_path: Path, values: dict) -> None:
    """Round-trip config.yaml via ruamel.yaml, updating only display.<key> for each key
    in `values` while preserving comments/formatting elsewhere in the file."""
    from ruamel.yaml import YAML

    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']  # type: ignore[attr-defined]  # config.yaml uses capitalized booleans
    with open(config_path, encoding='utf-8') as f:
        data = yaml.load(f)

    display_map = data['display']
    for key, value in values.items():
        display_map[key] = value

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)


def apply_and_save(main_window, values: dict) -> None:
    """Mutate the shared config, apply live to open display-related objects, then persist."""
    display_cfg = main_window.config.display
    for key, value in values.items():
        setattr(display_cfg, key, value)

    main_window.display.apply_display_settings(values)

    longitudinal_view = getattr(main_window, 'longitudinal_view', None)
    if longitudinal_view is not None:
        longitudinal_view.color = values['color_contour']
        longitudinal_view.plot_areas()

    small_display = getattr(main_window, 'small_display', None)
    if small_display is not None:
        small_display.n_points_contour = values['n_points_contour']
        small_display.contour_thickness = values['contour_thickness']
        small_display.point_thickness = values['point_thickness']
        small_display.point_radius = values['point_radius']

    breathing_sort_viewer = getattr(main_window, 'breathing_sort_viewer', None)
    if breathing_sort_viewer is not None:
        breathing_sort_viewer.n_points_contour = values['n_points_contour']
        breathing_sort_viewer.contour_color = values['color_contour']

    config_path = getattr(main_window.config, '_config_path', None)
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / 'config.yaml'
    save_display_settings(config_path, values)
