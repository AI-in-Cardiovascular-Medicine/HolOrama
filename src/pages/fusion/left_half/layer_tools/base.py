from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from domain.fusion_types import FusionScene


class SceneToolbar(QWidget):
    """Toolbar shown above the 3-D viewer for one FusionScene tab.

    Provides the controls every scene needs (per-layer visibility + opacity,
    point picking, reset camera). Subclasses add scene-specific extras by
    passing extra_rows — see geometry_tools.py / alignment_tools.py / tree_tools.py.
    """

    layer_visibility_changed = pyqtSignal(str, bool)  # layer key, visible
    layer_opacity_changed = pyqtSignal(str, float)  # layer key, opacity [0, 1]
    pick_mode_toggled = pyqtSignal(bool)
    reset_camera_requested = pyqtSignal()

    def __init__(self, scene: FusionScene, extra_rows: list[QWidget] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(8)

        self._layers_box = QVBoxLayout()
        self._layer_rows: dict[str, tuple[QCheckBox, QSlider]] = {}
        root.addLayout(self._layers_box)
        root.addStretch(1)

        for row in extra_rows or []:
            root.addWidget(row)

        self.pick_btn = QPushButton('Pick Point')
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip('Click a point in the 3-D scene')
        self.pick_btn.toggled.connect(self.pick_mode_toggled.emit)
        root.addWidget(self.pick_btn)

        reset_btn = QPushButton('Reset View')
        reset_btn.clicked.connect(self.reset_camera_requested.emit)
        root.addWidget(reset_btn)

    def refresh(self, layer_keys: list[str]) -> None:
        """Rebuild the layer visibility/opacity rows for the current set of layers."""
        _clear_layout(self._layers_box)
        self._layer_rows.clear()

        for key in layer_keys:
            row = QHBoxLayout()
            box = QCheckBox(key.replace('_', ' ').title())
            box.setChecked(True)
            box.toggled.connect(lambda checked, k=key: self.layer_visibility_changed.emit(k, checked))

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(100)
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
