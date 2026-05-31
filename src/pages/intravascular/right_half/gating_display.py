import matplotlib

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

matplotlib.use('Qt5Agg')  # needed to embed matplotlib figure in PyQt5 window


class GatingDisplay(FigureCanvasQTAgg):
    def __init__(self, main_window, parent=None, width: int | None = None, height: int | None = None, dpi: int = 100):
        # if darkdetect.isDark():
        #     plt.style.use('dark_background')
        plt.style.use('dark_background')

        w: int = main_window.config.display.image_size if width is None else width
        h: int = (w // 2) if height is None else height
        width_in: float = w / dpi  # convert pixels to inches
        height_in: float = h / dpi
        self.fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
        super().__init__(self.fig)

        self.setParent(parent)
        self.toolbar = NavigationToolbar2QT(self, parent)
