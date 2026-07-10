from domain.fusion_types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class AlignmentToolbar(SceneToolbar):
    """Toolbar for the Intravascular Aligned scene: aligned-geometry / resampled-centerline
    layer toggles, point picking (used to inspect reference points used for alignment)."""

    def __init__(self, parent=None) -> None:
        super().__init__(FusionScene.INTRAVASCULAR_ALIGNED, parent=parent)
