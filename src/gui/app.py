import os
from enum import Enum

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
from gui.shortcuts import init_shortcuts, init_ccta_shortcuts, init_menu


class ActivePage(Enum):
    INTRAVASCULAR = 0
    CCTA = 1

    @classmethod
    def from_index(cls, index: int) -> 'ActivePage':
        for page in cls:
            if page.value == index:
                return page
        raise ValueError(f"No ActivePage with index {index}")

    @classmethod
    def from_name(cls, name: str) -> 'ActivePage':
        for page in cls:
            if page.name == name:
                return page
        raise ValueError(f"No ActivePage with name {name}")

    @classmethod
    def value_to_string(cls, value: int) -> str:
        mapping = {
            0: 'Intravascular',
            1: 'CCTA',
        }
        return mapping.get(value, 'Unknown')


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

        self.active_page = ActivePage.INTRAVASCULAR
        self.stack = QStackedWidget()
        self.ccta_page = CctaPage(self.status_bar)
        self.intravascular_page = IntravascularPage(config, self.menu_bar, self.status_bar)
        self.stack.addWidget(self.intravascular_page)
        self.stack.addWidget(self.ccta_page)

        init_shortcuts(self.intravascular_page)
        init_ccta_shortcuts(self.ccta_page)
        init_menu(self.intravascular_page, self.ccta_page)

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

        ivus_btn = _NavButton(ActivePage.value_to_string(ActivePage.INTRAVASCULAR.value))
        ivus_btn.setCheckable(True)
        ivus_btn.setChecked(True)

        ccta_btn = _NavButton(ActivePage.value_to_string(ActivePage.CCTA.value))
        ccta_btn.setCheckable(True)

        ivus_btn.clicked.connect(lambda: self._switch_page(ActivePage.INTRAVASCULAR.value, ivus_btn, ccta_btn))
        ccta_btn.clicked.connect(lambda: self._switch_page(ActivePage.CCTA.value, ccta_btn, ivus_btn))

        layout.addWidget(ivus_btn)
        layout.addWidget(ccta_btn)
        layout.addStretch()
        return bar

    def _switch_page(self, active_page_index: int, active_btn: QPushButton, other_btn: QPushButton) -> None:
        self.stack.setCurrentIndex(active_page_index)
        active_btn.setChecked(True)
        other_btn.setChecked(False)
        self.active_page = ActivePage(active_page_index)
