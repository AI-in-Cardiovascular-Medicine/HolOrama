from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class IntravascularLoadedToolbar(SceneToolbar):
    """Toolbar for the Intravascular Loaded scene: the raw geometry straight out of
    from_file_singlepair, before centerline alignment -> lets you check whether a bad
    final result already started out wrong here, independent of centerline alignment."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.INTRAVASCULAR_LOADED, parent=parent)


class AlignmentToolbar(SceneToolbar):
    """Toolbar for the Intravascular Aligned scene: aligned-geometry / resampled-centerline
    layer toggles."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.INTRAVASCULAR_ALIGNED, parent=parent)
