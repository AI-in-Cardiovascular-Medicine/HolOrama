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
from gui.shortcuts import init_shortcuts, init_ccta_shortcuts, init_menu
from gui.active_page import ActivePage
from domain.io_types import MetaDataCCTA, MetaDataIntravascular


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
    intravascular_metadata: MetaDataIntravascular
    ccta_metadata: MetaDataCCTA

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        for page in ActivePage:
            metadata_name = f"{ActivePage.value_to_string(page.value).lower()}_metadata"
            data_type = ActivePage.metadata_type(page)
            if data_type == 'unknown_metadata':
                raise ValueError(f"Unknown metadata type for page {page}")

            setattr(self, metadata_name, data_type)

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
        self._nav_bar, self._nav_btns = self._build_nav_bar()
        layout.addWidget(self._nav_bar)
        layout.addWidget(self.stack, 1)  # stretch=1: stack always fills all remaining width
        self.setCentralWidget(central)
        self.showMaximized()

    def _build_nav_bar(self) -> tuple[QWidget, dict[ActivePage, _NavButton]]:
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

        btns = {ActivePage.INTRAVASCULAR: ivus_btn, ActivePage.CCTA: ccta_btn}

        ivus_btn.clicked.connect(lambda: self._switch_page(ActivePage.INTRAVASCULAR.value))
        ccta_btn.clicked.connect(lambda: self._switch_page(ActivePage.CCTA.value))

        layout.addWidget(ivus_btn)
        layout.addWidget(ccta_btn)
        layout.addStretch()
        return bar, btns

    def reload_intravascular(self) -> None:
        old = self.intravascular_page
        self.stack.removeWidget(old)
        old.deleteLater()

        new_page = IntravascularPage(self.config, self.menu_bar, self.status_bar)
        self.stack.insertWidget(ActivePage.INTRAVASCULAR.value, new_page)
        self.intravascular_page = new_page
        self.stack.setCurrentIndex(ActivePage.INTRAVASCULAR.value)

        init_shortcuts(new_page)
        self.menu_bar.clear()
        init_menu(new_page, self.ccta_page)

    def reload_ccta(self) -> None:
        old = self.ccta_page
        old.shutdown()  # release VTK OpenGL context before HWND is invalidated
        self.stack.removeWidget(old)
        old.deleteLater()

        new_page = CctaPage(self.status_bar)
        self.stack.insertWidget(ActivePage.CCTA.value, new_page)
        self.ccta_page = new_page
        self.stack.setCurrentIndex(ActivePage.CCTA.value)

        init_ccta_shortcuts(new_page)
        self.menu_bar.clear()
        init_menu(self.intravascular_page, new_page)

    def _switch_page(self, active_page_index: int) -> None:
        self.stack.setCurrentIndex(active_page_index)
        active = ActivePage(active_page_index)
        for page, btn in self._nav_btns.items():
            btn.setChecked(page == active)
        self.active_page = active
