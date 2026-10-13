from PyQt6.QtWidgets import QLabel

from domain.fusion.types import FusionScene
from pages.fusion.left_half.layer_tools.base import SceneToolbar


class BranchLabelsToolbar(SceneToolbar):
    """Toolbar for the Branch Labels scene: the RCA/LCA surface vertices colored by the
    centerline branch label_branches assigned them to (same colors as the Centerline
    Branches tab), one layer per branch. Lasso moves vertices to another branch of the
    same vessel — see FusionPage._on_branch_lasso_closed."""

    def __init__(self, parent=None) -> None:
        hint = QLabel('Lasso: move vertices\nto another branch\nof the same vessel.')
        hint.setStyleSheet('color: gray;')
        super().__init__(
            FusionScene.BRANCH_LABELS,
            extra_rows=[hint],
            show_lasso=True,
            show_color_swatch=True,
            parent=parent,
        )
