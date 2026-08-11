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

# Minimum height for the layer list's internal scroll area — below this a couple of rows
# wouldn't even fit. The toolbar itself has no max height: it sits in a QSplitter (see
# LeftHalf) so the user can drag it taller when a scene's layer list or extra controls
# (e.g. BranchEditorToolbar's split/merge groups) need more room than this minimum.
_LAYERS_MIN_HEIGHT = 60
# Minimum width for the layer list, so checkbox labels (e.g. 'Rca Branch 3') and their
# color swatch/opacity slider don't get squeezed down to nothing next to a scene's other
# controls (e.g. BranchEditorToolbar's Split/Merge groups) when the toolbar is narrow.
_LAYERS_MIN_WIDTH = 220


class SceneToolbar(QWidget):
    """Toolbar shown above the 3-D viewer for one FusionScene tab.

    Provides the controls every scene needs (reset camera) and, when show_layers=True,
    a per-layer visibility list with either an opacity slider or (show_color_swatch=True)
    a small color swatch matching that layer's actual render color — more useful than
    opacity for scenes made of colored point clouds (e.g. Centerline Branches) where you
    want to match a legend entry to what's on screen, not fade it. Point picking
    (show_pick=True) is only wired up end-to-end for the Vessel Tree and Centerline
    Branches scenes (see FusionPage._on_point_picked), so other scenes leave it off.
    Subclasses add scene-specific extras by passing extra_rows — see geometry_tools.py /
    alignment_tools.py / tree_tools.py / branch_tools.py.
    """

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
            root.addWidget(scroll, 2)
        else:
            root.addStretch(1)

        for row in extra_rows or []:
            root.addWidget(row)

        # Reset View / Pick Point / Lasso / Clear All Data, stacked in that order — Pick
        # Point and Lasso sit between the two so they read as "how you interact with the
        # view" rather than a stray button off to the side.
        view_buttons = QVBoxLayout()
        reset_btn = QPushButton('Reset View')
        reset_btn.clicked.connect(self.reset_camera_requested.emit)
        view_buttons.addWidget(reset_btn)

        if show_pick:
            self.pick_btn = QPushButton('Pick Point')
            self.pick_btn.setCheckable(True)
            self.pick_btn.setToolTip('Click a point in the 3-D scene')
            self.pick_btn.toggled.connect(self.pick_mode_toggled.emit)
            view_buttons.addWidget(self.pick_btn)

        if show_lasso:
            self.lasso_btn = QPushButton('Lasso')
            self.lasso_btn.setCheckable(True)
            self.lasso_btn.setToolTip(
                'Draw a closed lasso around points (left-click to add points, click near the\n'
                'start or right-click to close), then choose which label to reclassify them as.'
            )
            self.lasso_btn.toggled.connect(self._on_lasso_toggled)
            view_buttons.addWidget(self.lasso_btn)

        clear_btn = QPushButton('Clear All Data')
        clear_btn.setToolTip('Discards every loaded/computed fusion result and clears the 3-D viewer.')
        clear_btn.clicked.connect(self.clear_all_data_requested.emit)
        view_buttons.addWidget(clear_btn)
        root.addLayout(view_buttons)

    def _on_lasso_toggled(self, checked: bool) -> None:
        self.lasso_btn.setText('Cancel Lasso' if checked else 'Lasso')
        self.lasso_mode_toggled.emit(checked)

    def refresh(self, layer_states: dict[str, tuple[bool, float, tuple[int, int, int]]]) -> None:
        """Rebuild the layer visibility rows for the current set of layers, initializing
        each checkbox (and slider/swatch) to that layer's actual (visible, opacity, color)
        — not a fixed assumed default, since layers can be added at less than 100% opacity
        (e.g. a translucent base mesh). No-op when built with show_layers=False."""
        if not self._show_layers:
            return
        _clear_layout(self._layers_box)
        self._layer_rows.clear()

        for key, (visible, opacity, color) in layer_states.items():
            row = QHBoxLayout()
            box = QCheckBox(key.replace('_', ' ').title())
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


def labeled(text: str, widget: QWidget) -> QWidget:
    """Small helper for scene-specific extra rows: 'Label:' + control, as one widget."""
    wrapper = QWidget()
    box = QHBoxLayout(wrapper)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(QLabel(text))
    box.addWidget(widget)
    return wrapper
