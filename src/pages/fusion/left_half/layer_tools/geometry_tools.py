from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class GeometryToolbar(SceneToolbar):
    """Toolbar for the CCTA Geometry scene: mesh + centerline layer toggles, point picking
    (used to grab manual reference points if the automatic ones need correcting)."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.CCTA_GEOMETRY, parent=parent)
