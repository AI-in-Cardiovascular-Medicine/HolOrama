"""Pieces shared by the IVUS and OCT variants of the right half.

Both variants end in the same command-button row and both host the single
LongitudinalView the page owns; everything above that row differs enough that
the two are built separately (see right_half_ivus.py / right_half_oct.py).
"""

from functools import partial

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from pages.intravascular.popup_windows.small_display import SmallDisplay
from segmentation.segment import segment


def build_lower_buttons(main_window, right_button: QPushButton) -> QVBoxLayout:
    """Automatic segmentation next to the modality's own action button."""
    layout = QVBoxLayout()
    segment_button = QPushButton('Automatic Segmentation')
    segment_button.setToolTip('Run deep learning based segmentation of lumen')
    segment_button.clicked.connect(partial(segment, main_window))
    command_buttons = QHBoxLayout()
    command_buttons.addWidget(segment_button)
    command_buttons.addWidget(right_button)
    layout.addLayout(command_buttons)
    layout.addLayout(QHBoxLayout())  # measures placeholder
    return layout


def open_small_display(main_window):
    if main_window.image_displayed:
        main_window.small_display = SmallDisplay(main_window)
        main_window.small_display.move(
            main_window.x() + main_window.width() // 2, main_window.y() + main_window.height() // 2
        )
        next_gated = main_window.display_slider.next_gated_frame(set=False)
        main_window.small_display.update_frame(next_gated, update_image=True, update_contours=True, update_text=True)
        main_window.small_display.show()


class LongitudinalSlot(QWidget):
    """Mount point for the shared LongitudinalView.

    The page owns a single LongitudinalView but each modality's right half shows it
    in its own layout, so the view is moved between the two slots on a modality
    switch (see RightHalf.update_for_modality).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def attach(self, view: QWidget) -> None:
        if view.parent() is not self:  # idempotent: reloading the same modality re-activates
            self._layout.addWidget(view)

    def detach(self, view: QWidget) -> None:
        self._layout.removeWidget(view)
        view.setParent(None)
