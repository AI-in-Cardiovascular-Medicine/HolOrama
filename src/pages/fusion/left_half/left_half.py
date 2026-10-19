from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout

from domain.fusion.types import FusionScene
from pages.fusion.left_half.viewer import FusionViewer3D
from pages.fusion.left_half.toolbars.alignment import AlignmentToolbar, IntravascularLoadedToolbar
from pages.fusion.left_half.toolbars.branch_labels import BranchLabelsToolbar
from pages.fusion.left_half.toolbars.centerline_branches import BranchEditorToolbar
from pages.fusion.left_half.toolbars.cuts import CutToolbar
from pages.fusion.left_half.toolbars.geometry import GeometryToolbar
from pages.fusion.left_half.toolbars.tree import TreeToolbar
from pages.intravascular.utils.helpers import SplitterPane


class LeftHalf:
    """Tabbed 3D scene: one tab per FusionScene with its toolbar above the shared
    FusionViewer3D. Switching tabs switches the viewer's visible scene."""

    def __init__(self, main_window) -> None:
        self.main_window = main_window
        self.widget = SplitterPane()

        self.viewer = FusionViewer3D()

        self.geometry_toolbar = GeometryToolbar()
        self.branch_toolbar = BranchEditorToolbar()
        self.branch_labels_toolbar = BranchLabelsToolbar()
        self.intravascular_loaded_toolbar = IntravascularLoadedToolbar()
        self.alignment_toolbar = AlignmentToolbar()
        self.tree_toolbar = TreeToolbar()
        self.cut_toolbar = CutToolbar()

        self._scene_order = [
            FusionScene.CCTA_GEOMETRY,
            FusionScene.CENTERLINE_BRANCHES,
            FusionScene.BRANCH_LABELS,
            FusionScene.VESSEL_TREE,
            FusionScene.INTRAVASCULAR_LOADED,
            FusionScene.INTRAVASCULAR_ALIGNED,
            FusionScene.INLETS_OUTLETS,
        ]
        self._toolbars = {
            FusionScene.CCTA_GEOMETRY: self.geometry_toolbar,
            FusionScene.CENTERLINE_BRANCHES: self.branch_toolbar,
            FusionScene.BRANCH_LABELS: self.branch_labels_toolbar,
            FusionScene.INTRAVASCULAR_LOADED: self.intravascular_loaded_toolbar,
            FusionScene.INTRAVASCULAR_ALIGNED: self.alignment_toolbar,
            FusionScene.VESSEL_TREE: self.tree_toolbar,
            FusionScene.INLETS_OUTLETS: self.cut_toolbar,
        }
        for scene, toolbar in self._toolbars.items():
            toolbar.layer_visibility_changed.connect(partial(self.viewer.set_layer_visible, scene))
            toolbar.layer_opacity_changed.connect(partial(self.viewer.set_layer_opacity, scene))
            toolbar.pick_mode_toggled.connect(self.viewer.set_pick_mode)
            toolbar.reset_camera_requested.connect(self.viewer.reset_camera)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        for scene in self._scene_order:
            # '&&': a single '&' marks a keyboard shortcut in a tab label (Inlets & Outlets)
            self._tabs.addTab(self._toolbars[scene], FusionScene.label(scene).replace('&', '&&'))
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Splitter so the toolbar strip can be dragged taller when a scene needs more room.
        splitter = QSplitter(Qt.Orientation.Vertical, self.widget)
        splitter.addWidget(self._tabs)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([160, 600])

        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)

        self._on_tab_changed(0)

    def __call__(self):
        return self.widget

    def _on_tab_changed(self, index: int) -> None:
        scene = self._scene_order[index]
        self.viewer.set_scene(scene)
        self.refresh_toolbar(scene)

    def refresh_toolbar(self, scene: FusionScene) -> None:
        """Call after adding/removing layers in a scene so its checkbox list stays in sync."""
        self._toolbars[scene].refresh(self.viewer.layer_states(scene))

    def on_scene_shown(self, callback) -> None:
        """Call `callback(scene)` whenever the user or show_scene switches to a tab."""
        self._tabs.currentChanged.connect(lambda index: callback(self._scene_order[index]))

    def show_scene(self, scene: FusionScene) -> None:
        self._tabs.setCurrentIndex(self._scene_order.index(scene))
