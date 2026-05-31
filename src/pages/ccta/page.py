from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel
from PyQt6.QtCore import Qt


class CctaPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout()
        for row, col, name in [
            (0, 0, 'Axial'), (0, 1, 'Sagittal'),
            (1, 0, 'Coronal'), (1, 1, '3D / CPR'),
        ]:
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, row, col)
        self.setLayout(layout)
