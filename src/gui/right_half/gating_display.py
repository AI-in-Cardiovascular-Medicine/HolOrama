import darkdetect
import matplotlib

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')  # needed to embed matplotlib figure in PyQt5 window


class GatingDisplay(FigureCanvasQTAgg):
    def __init__(self, main_window, parent=None, width=None, height=None, dpi=100):
        # if darkdetect.isDark():
        #     plt.style.use('dark_background')
        plt.style.use('dark_background')

        width = main_window.config.display.image_size if width is None else width
        height = width // 2 if height is None else height
        width /= dpi  # convert pixels to inches
        height /= dpi
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

        self.setParent(parent)
        self.toolbar = NavigationToolbar2QT(self, parent)
