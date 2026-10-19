from domain.fusion.types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class IntravascularLoadedToolbar(SceneToolbar):
    """Toolbar for the Intravascular Loaded scene: the raw geometry before centerline
    alignment, to check whether a bad result was already wrong before alignment."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.INTRAVASCULAR_LOADED, parent=parent)


class AlignmentToolbar(SceneToolbar):
    """Toolbar for the Intravascular Aligned scene."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.INTRAVASCULAR_ALIGNED, parent=parent)
