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

from domain.fusion_types import FusionScene

# Caps the toolbar's height regardless of how many layer rows it holds, so a scene
# with many layers (CCTA_GEOMETRY can have a dozen) never pushes the 3-D viewer down.
# The layer list scrolls internally within this budget instead.
_LAYERS_MAX_HEIGHT = 90
_TOOLBAR_MAX_HEIGHT = 110


class SceneToolbar(QWidget):
    """Toolbar shown above the 3-D viewer for one FusionScene tab.

    Provides the controls every scene needs (reset camera) and, when show_layers=True,
    a per-layer visibility/opacity list capped to a fixed, internally-scrolling height.
    Point picking (show_pick=True) is only wired up end-to-end for the Vessel Tree scene
    (see FusionPage._on_point_picked), so other scenes leave it off. Subclasses add
    scene-specific extras by passing extra_rows — see geometry_tools.py /
    alignment_tools.py / tree_tools.py.
    """

    layer_visibility_changed = pyqtSignal(str, bool)  # layer key, visible
    layer_opacity_changed = pyqtSignal(str, float)  # layer key, opacity [0, 1]
    pick_mode_toggled = pyqtSignal(bool)
    reset_camera_requested = pyqtSignal()

    def __init__(
        self,
        scene: FusionScene,
        extra_rows: list[QWidget] | None = None,
        show_layers: bool = True,
        show_pick: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.scene = scene
        self._show_layers = show_layers
        self.setMaximumHeight(_TOOLBAR_MAX_HEIGHT)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(8)

        self._layer_rows: dict[str, tuple[QCheckBox, QSlider]] = {}
        if show_layers:
            self._layers_box = QVBoxLayout()
            layers_widget = QWidget()
            layers_widget.setLayout(self._layers_box)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMaximumHeight(_LAYERS_MAX_HEIGHT)
            scroll.setWidget(layers_widget)
            root.addWidget(scroll, 1)
        else:
            root.addStretch(1)

        for row in extra_rows or []:
            root.addWidget(row)

        if show_pick:
            self.pick_btn = QPushButton('Pick Point')
            self.pick_btn.setCheckable(True)
            self.pick_btn.setToolTip('Click a point in the 3-D scene')
            self.pick_btn.toggled.connect(self.pick_mode_toggled.emit)
            root.addWidget(self.pick_btn)

        reset_btn = QPushButton('Reset View')
        reset_btn.clicked.connect(self.reset_camera_requested.emit)
        root.addWidget(reset_btn)

    def refresh(self, layer_states: dict[str, tuple[bool, float]]) -> None:
        """Rebuild the layer visibility/opacity rows for the current set of layers,
        initializing each checkbox/slider to that layer's actual (visible, opacity) —
        not a fixed assumed default, since layers can be added at less than 100% opacity
        (e.g. a translucent base mesh). No-op when built with show_layers=False."""
        if not self._show_layers:
            return
        _clear_layout(self._layers_box)
        self._layer_rows.clear()

        for key, (visible, opacity) in layer_states.items():
            row = QHBoxLayout()
            box = QCheckBox(key.replace('_', ' ').title())
            box.setChecked(visible)
            box.toggled.connect(lambda checked, k=key: self.layer_visibility_changed.emit(k, checked))

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(round(opacity * 100))
            slider.setFixedWidth(80)
            slider.valueChanged.connect(lambda value, k=key: self.layer_opacity_changed.emit(k, value / 100.0))

            row.addWidget(box)
            row.addWidget(slider)
            self._layers_box.addLayout(row)
            self._layer_rows[key] = (box, slider)


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


def labeled(text: str, widget: QWidget) -> QWidget:
    """Small helper for scene-specific extra rows: 'Label:' + control, as one widget."""
    wrapper = QWidget()
    box = QHBoxLayout(wrapper)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(QLabel(text))
    box.addWidget(widget)
    return wrapper
