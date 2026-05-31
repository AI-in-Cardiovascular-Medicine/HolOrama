import os

from omegaconf import DictConfig
from PyQt6.QtWidgets import (
    QMainWindow,
    QMenuBar,
    QStatusBar,
    QStackedWidget,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QStylePainter,
    QStyleOptionButton,
    QStyle,
)
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon

from pages.intravascular.page import IntravascularPage
from pages.ccta.page import CctaPage


class _NavButton(QPushButton):
    """Checkable push button with text rotated 90° to fit a narrow sidebar."""

    def sizeHint(self) -> QSize:
        s = super().sizeHint()
        return QSize(s.height(), s.width())

    def minimumSizeHint(self) -> QSize:
        s = super().minimumSizeHint()
        return QSize(s.height(), s.width())

    def paintEvent(self, _event) -> None:
        painter = QStylePainter(self)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.rect = opt.rect.transposed()
        painter.rotate(-90)
        painter.translate(-self.height(), 0)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, opt)


class Master(QMainWindow):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        self.setWindowTitle('AIVUS Software')
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'media', 'desktop_img.ico')
        self.setWindowIcon(QIcon(icon_path))

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.stack = QStackedWidget()
        self.intravascular_page = IntravascularPage(config, self.menu_bar, self.status_bar)
        self.ccta_page = CctaPage()
        self.stack.addWidget(self.intravascular_page)
        self.stack.addWidget(self.ccta_page)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_nav_bar())
        layout.addWidget(self.stack, 1)  # stretch=1: stack always fills all remaining width
        self.setCentralWidget(central)
        self.showMaximized()

    def _build_nav_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedWidth(40)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(2, 8, 2, 8)
        layout.setSpacing(4)

        ivus_btn = _NavButton('Intravascular')
        ivus_btn.setCheckable(True)
        ivus_btn.setChecked(True)

        ccta_btn = _NavButton('CCTA')
        ccta_btn.setCheckable(True)

        ivus_btn.clicked.connect(lambda: self._switch_page(0, ivus_btn, ccta_btn))
        ccta_btn.clicked.connect(lambda: self._switch_page(1, ccta_btn, ivus_btn))

        layout.addWidget(ivus_btn)
        layout.addWidget(ccta_btn)
        layout.addStretch()
        return bar

    def _switch_page(self, index: int, active_btn: QPushButton, other_btn: QPushButton) -> None:
        self.stack.setCurrentIndex(index)
        active_btn.setChecked(True)
        other_btn.setChecked(False)
