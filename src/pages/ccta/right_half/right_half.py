from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pages.ccta.right_half.mask_panel import MaskPanel
from pages.ccta.right_half.brush_panel import BrushPanel
from pages.ccta.right_half.stl_extraction_panel import StlExtractionPanel


class RightHalf:
    """Manages the right-side panel with mask controls, brush settings, and STL extraction."""

    def __init__(self, label_colors: tuple[tuple[int, int, int], ...], initial_mask_alpha: float, parent=None) -> None:
        self.widget = QWidget(parent)

        self.mask_panel = MaskPanel(label_colors=label_colors, initial_alpha=initial_mask_alpha)
        self.brush_panel = BrushPanel(label_colors=label_colors)
        self.stl_extraction_panel = StlExtractionPanel()

        # Wire up the panels to work together
        self.mask_panel.set_brush_panel(self.brush_panel)
        self.mask_panel.label_name_changed.connect(self.brush_panel.update_label_name)
        self.mask_panel.label_name_changed.connect(self.stl_extraction_panel.update_label_name)

        # Mask labels (top, stretches) + STL extraction (bottom, fixed)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.mask_panel, 1)
        layout.addWidget(self.stl_extraction_panel, 0)

    def __call__(self):
        return self.widget
