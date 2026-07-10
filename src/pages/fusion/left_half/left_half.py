from functools import partial

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout

from domain.fusion_types import FusionScene
from pages.fusion.left_half.display_results import FusionViewer3D
from pages.fusion.left_half.layer_tools.alignment_tools import AlignmentToolbar, IntravascularLoadedToolbar
from pages.fusion.left_half.layer_tools.geometry_tools import GeometryToolbar
from pages.fusion.left_half.layer_tools.tree_tools import TreeToolbar
from pages.intravascular.utils.helpers import SplitterPane


class LeftHalf:
    """Tabbed 3-D scene: one tab per FusionScene, each showing that scene's toolbar
    above the shared FusionViewer3D. Switching tabs swaps which actor group is visible
    in the viewer — see FusionViewer3D.set_scene()."""

    def __init__(self, main_window) -> None:
        self.main_window = main_window
        self.widget = SplitterPane()

        self.viewer = FusionViewer3D()

        self.geometry_toolbar = GeometryToolbar()
        self.intravascular_loaded_toolbar = IntravascularLoadedToolbar()
        self.alignment_toolbar = AlignmentToolbar()
        self.tree_toolbar = TreeToolbar()

        self._scene_order = [
            FusionScene.CCTA_GEOMETRY,
            FusionScene.VESSEL_TREE,
            FusionScene.INTRAVASCULAR_LOADED,
            FusionScene.INTRAVASCULAR_ALIGNED,
        ]
        self._toolbars = {
            FusionScene.CCTA_GEOMETRY: self.geometry_toolbar,
            FusionScene.INTRAVASCULAR_LOADED: self.intravascular_loaded_toolbar,
            FusionScene.INTRAVASCULAR_ALIGNED: self.alignment_toolbar,
            FusionScene.VESSEL_TREE: self.tree_toolbar,
        }
        for scene, toolbar in self._toolbars.items():
            toolbar.layer_visibility_changed.connect(partial(self.viewer.set_layer_visible, scene))
            toolbar.layer_opacity_changed.connect(partial(self.viewer.set_layer_opacity, scene))
            toolbar.pick_mode_toggled.connect(self.viewer.set_pick_mode)
            toolbar.reset_camera_requested.connect(self.viewer.reset_camera)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        for scene in self._scene_order:
            self.tabs.addTab(self._toolbars[scene], FusionScene.label(scene))
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs, 0)
        layout.addWidget(self.viewer, 1)

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

    def show_scene(self, scene: FusionScene) -> None:
        self.tabs.setCurrentIndex(self._scene_order.index(scene))
