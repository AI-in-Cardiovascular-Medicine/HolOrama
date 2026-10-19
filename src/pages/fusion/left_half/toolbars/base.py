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

from domain.fusion.display_types import region_label
from domain.fusion.types import FusionScene

# Minimum size of the layer list. The toolbar height itself is set by LeftHalf's splitter.
_LAYERS_MIN_HEIGHT = 60
_LAYERS_MIN_WIDTH = 220


class SceneToolbar(QWidget):
    """Toolbar above the 3D viewer for one FusionScene tab: view buttons, an optional layer
    list (opacity slider or color swatch per layer) and scene-specific `extra_rows`.
    show_pick only works for scenes FusionPage._on_point_picked handles."""

    layer_visibility_changed = pyqtSignal(str, bool)  # layer key, visible
    layer_opacity_changed = pyqtSignal(str, float)  # layer key, opacity [0, 1]
    pick_mode_toggled = pyqtSignal(bool)
    lasso_mode_toggled = pyqtSignal(bool)
    reset_camera_requested = pyqtSignal()
    clear_all_data_requested = pyqtSignal()

    def __init__(
        self,
        scene: FusionScene,
        extra_rows: list[QWidget] | None = None,
        show_layers: bool = True,
        show_pick: bool = False,
        show_lasso: bool = False,
        show_color_swatch: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.scene = scene
        self._show_layers = show_layers
        self._show_color_swatch = show_color_swatch
        self._layer_labels: dict[str, str] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(8)

        self._layer_rows: dict[str, tuple[QCheckBox, QSlider | QLabel]] = {}
        if show_layers:
            self._layers_box = QVBoxLayout()
            layers_widget = QWidget()
            layers_widget.setLayout(self._layers_box)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMinimumHeight(_LAYERS_MIN_HEIGHT)
            scroll.setMinimumWidth(_LAYERS_MIN_WIDTH)
            scroll.setWidget(layers_widget)
            root.addWidget(scroll, 1)
        else:
            root.addStretch(1)

        # Scene controls and view buttons share the other half (50/50 with the layer list).
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        for row in extra_rows or []:
            controls_layout.addWidget(row, 1)
        root.addWidget(controls, 1)

        # Reset View / Pick Point / Lasso / Clear All Data, stacked.
        view_buttons = QVBoxLayout()
        reset_btn = QPushButton('Reset View')
        reset_btn.clicked.connect(self.reset_camera_requested.emit)
        view_buttons.addWidget(reset_btn)

        self._pick_btn: QPushButton | None = None
        self._lasso_btn: QPushButton | None = None
        if show_pick:
            self._pick_btn = QPushButton('Pick Point')
            self._pick_btn.setCheckable(True)
            self._pick_btn.setToolTip('Click a point in the 3D scene')
            self._pick_btn.toggled.connect(self.pick_mode_toggled.emit)
            view_buttons.addWidget(self._pick_btn)

        if show_lasso:
            self._lasso_btn = QPushButton('Lasso')
            self._lasso_btn.setCheckable(True)
            self._lasso_btn.setToolTip(
                'Draw a closed lasso around points (left-click to add points, click near the\n'
                'start or right-click to close), then choose which label to reclassify them as.'
            )
            self._lasso_btn.toggled.connect(self._on_lasso_toggled)
            view_buttons.addWidget(self._lasso_btn)

        clear_btn = QPushButton('Clear All Data')
        clear_btn.setToolTip('Discards every loaded/computed fusion result and clears the 3D viewer.')
        clear_btn.clicked.connect(self.clear_all_data_requested.emit)
        view_buttons.addWidget(clear_btn)
        controls_layout.addLayout(view_buttons)

    def _on_lasso_toggled(self, checked: bool) -> None:
        assert self._lasso_btn is not None
        self._lasso_btn.setText('Cancel Lasso' if checked else 'Lasso')
        self.lasso_mode_toggled.emit(checked)

    def lasso_active(self) -> bool:
        return self._lasso_btn is not None and self._lasso_btn.isChecked()

    def end_lasso(self) -> None:
        """Uncheck Lasso, which ends lasso mode and clears the overlay."""
        if self._lasso_btn is not None:
            self._lasso_btn.setChecked(False)

    def set_layer_labels(self, labels: dict[str, str]) -> None:
        """Display names for layer keys, used by the next refresh() instead of the key."""
        self._layer_labels = dict(labels)

    def refresh(self, layer_states: dict[str, tuple[bool, float, tuple[int, int, int]]]) -> None:
        """Rebuild the layer rows, initialized to each layer's (visible, opacity, color).
        No-op when show_layers=False."""
        if not self._show_layers:
            return
        _clear_layout(self._layers_box)
        self._layer_rows.clear()

        for key, (visible, opacity, color) in layer_states.items():
            row = QHBoxLayout()
            # region_label maps internal region keys to display names.
            box = QCheckBox(self._layer_labels.get(key) or region_label(key).replace('_', ' ').title())
            box.setChecked(visible)
            box.toggled.connect(lambda checked, k=key: self.layer_visibility_changed.emit(k, checked))
            row.addWidget(box)

            if self._show_color_swatch:
                r, g, b = color
                swatch = QLabel()
                swatch.setFixedSize(20, 14)
                swatch.setToolTip(f'rgb({r}, {g}, {b})')
                swatch.setStyleSheet(
                    f'background-color: rgb({r}, {g}, {b}); border: 1px solid #666; border-radius: 2px;'
                )
                row.addWidget(swatch)
                self._layer_rows[key] = (box, swatch)
            else:
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(round(opacity * 100))
                slider.setFixedWidth(80)
                slider.valueChanged.connect(lambda value, k=key: self.layer_opacity_changed.emit(k, value / 100.0))
                row.addWidget(slider)
                self._layer_rows[key] = (box, slider)

            self._layers_box.addLayout(row)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            break
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        else:
            sub_layout = item.layout()
            if sub_layout is not None:
                _clear_layout(sub_layout)
