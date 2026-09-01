"""Holds one right half per modality and shows whichever the loaded pullback needs.

IVUS and OCT need different controls almost everywhere in the right half, so each
modality gets its own widget (right_half_ivus.py / right_half_oct.py) rather than one
widget branching internally.  Both are built once and kept alive, so everything they
contain has a permanent parent and nothing has to survive a teardown on a modality
switch.  The single exception is the LongitudinalView, which the page owns and both
halves show, so it moves between their slots (see update_for_modality).
"""

from PyQt6.QtWidgets import QLayout, QStackedWidget, QVBoxLayout

from pages.intravascular.right_half.right_half_ivus import RightHalfIvus
from pages.intravascular.right_half.right_half_oct import RightHalfOct
from pages.intravascular.utils.helpers import SplitterPane


class RightHalf:
    def __init__(self, main_window):
        self.main_window = main_window

        # Outer container — stays in the main splitter forever
        self.right_widget = SplitterPane()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self.ivus = RightHalfIvus(main_window)
        self.oct = RightHalfOct(main_window)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.ivus)
        self.stack.addWidget(self.oct)
        self.right_layout.addWidget(self.stack)

        # IVUS is the default until a pullback says otherwise
        self.active: RightHalfIvus | RightHalfOct = self.ivus
        self.ivus.activate()
        self.stack.setCurrentWidget(self.ivus)

    @property
    def is_oct(self) -> bool:
        return self.active is self.oct

    def update_for_modality(self):
        mw = self.main_window
        is_oct = mw.runtime_data.metadata.get('modality') == 'OCT'
        mw.longitudinal_view.set_oct_mode(is_oct)

        half = self.oct if is_oct else self.ivus
        if half is not self.active:
            self.active.deactivate()
            self.active = half
        half.activate()  # also on a reload of the same modality: it sets up for a new pullback
        self.stack.setCurrentWidget(half)

    def __call__(self):
        return self.right_widget
