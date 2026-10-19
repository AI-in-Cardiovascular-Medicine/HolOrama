"""Small layout helpers shared by the fusion columns and toolbars."""

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


def row(label: str, widget: QWidget) -> QHBoxLayout:
    """'Label:' + control as a layout, the control taking the remaining width."""
    layout = QHBoxLayout()
    layout.addWidget(QLabel(label))
    layout.addWidget(widget, 1)
    return layout


def labeled(text: str, widget: QWidget) -> QWidget:
    """'Label:' + control as one widget."""
    wrapper = QWidget()
    box = QHBoxLayout(wrapper)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(QLabel(text))
    box.addWidget(widget)
    return wrapper
