from PyQt6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QRadioButton,
    QButtonGroup,
    QPushButton,
)
from PyQt6.QtCore import Qt, QTimer


class HoverButton(QPushButton):
    """QPushButton that calls callbacks on mouse enter/leave."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_hover_enter = None
        self._on_hover_leave = None

    def enterEvent(self, event):
        if self._on_hover_enter is not None:
            self._on_hover_enter()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._on_hover_leave is not None:
            self._on_hover_leave()
        super().leaveEvent(event)


class BrushSettingsPopup(QFrame):
    """
    Frameless hover popup showing brush radius and add/erase toggle.

    Show it with show_near(widget); it hides itself via a short timer when
    the mouse leaves both the trigger button and the popup itself.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setAutoFillBackground(True)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(300)
        self._hide_timer.timeout.connect(self.hide)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        layout.addWidget(QLabel('<b>Brush Settings</b>'))

        mode_row = QHBoxLayout()
        self._add_rb = QRadioButton('Add')
        self._add_rb.setChecked(True)
        self._erase_rb = QRadioButton('Erase')
        grp = QButtonGroup(self)
        grp.addButton(self._add_rb)
        grp.addButton(self._erase_rb)
        mode_row.addWidget(self._add_rb)
        mode_row.addWidget(self._erase_rb)
        layout.addLayout(mode_row)

        r_hdr = QHBoxLayout()
        r_hdr.addWidget(QLabel('Radius (px):'))
        self._r_lbl = QLabel('10')
        self._r_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        r_hdr.addWidget(self._r_lbl)
        layout.addLayout(r_hdr)

        self._radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._radius_slider.setRange(1, 50)
        self._radius_slider.setValue(10)
        self._radius_slider.setMinimumWidth(140)
        self._radius_slider.valueChanged.connect(lambda v: self._r_lbl.setText(str(v)))
        layout.addWidget(self._radius_slider)

        self.adjustSize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_erase(self) -> bool:
        return self._erase_rb.isChecked()

    @property
    def radius_px(self) -> int:
        return self._radius_slider.value()

    def show_near(self, widget) -> None:
        """Position popup just below *widget* and show it."""
        from PyQt6.QtCore import QPoint

        global_pos = widget.mapToGlobal(QPoint(0, widget.height()))
        self.move(global_pos)
        self._hide_timer.stop()
        self.show()

    def schedule_hide(self) -> None:
        self._hide_timer.start()

    def cancel_hide(self) -> None:
        self._hide_timer.stop()

    # ------------------------------------------------------------------
    # Qt overrides – keep popup alive while mouse is inside it
    # ------------------------------------------------------------------

    def enterEvent(self, event):
        self._hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hide_timer.start()
        super().leaveEvent(event)
