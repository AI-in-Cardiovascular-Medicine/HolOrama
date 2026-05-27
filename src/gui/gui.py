import os
from functools import partial
from typing import Any

import numpy as np
from omegaconf import DictConfig
from PyQt6.QtWidgets import (
    QMainWindow,
    QMenuBar,
    QSplitter,
    QTableWidget,
    QStatusBar,
    QCheckBox,
    QPushButton,
    QButtonGroup,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QIcon

from gui.left_half.left_half import LeftHalf
from gui.left_half.display import Display
from gui.utils.slider import Slider, Communicate
from gui.right_half.right_half import (
    RightHalf,
    toggle_diastolic_frame,
    toggle_systolic_frame,
    toggle_tagged_frame,
    use_diastolic,
    use_tagged,
    set_oct_quality,
)
from gui.right_half.gating_display import GatingDisplay
from gui.right_half.longitudinal_view import LongitudinalView
from gui.shortcuts import init_shortcuts, init_menu
from input_output.output.contours import write_contours
from gating.contour_based_gating import ContourBasedGating
from segmentation.predict import Predict
from domain.runtime_types import RuntimeData
from domain.all_types import OCT_QUALITY_LABELS


class Master(QMainWindow):
    """Main Window Class"""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config: DictConfig = config
        self.file_name: str | None = None
        self.contour_based_gating: ContourBasedGating = ContourBasedGating(self)
        self.predictor: Predict = Predict(self)
        self.image_displayed: bool = False
        self.segmentation: bool = False
        self.contours_drawn: bool = False
        self.hide_contours: bool = False
        self.hide_special_points: bool = False
        self.colormap_enabled: bool = False
        self.tmp_contours: dict[
            str, tuple[list[float], list[float]]
        ] = {}  # per-contour-type undo storage, e.g. {'lumen': (xlist, ylist)}
        self.gated_frames: list[int] = []
        self.gated_frames_dia: list[int] = []
        self.gated_frames_sys: list[int] = []
        self.data: RuntimeData = RuntimeData()
        self.gating_signal: dict[str, Any] = {}  # global gating signal, saved separately from per-frame data
        self.metadata: dict[str, Any] = {}  # metadata used outside of read_image (not saved to JSON file)
        self.images: np.ndarray | None = None
        self.images_display: int | None = None
        self.images_rgb: np.ndarray | None = None
        self.diastole_color: tuple[int, int, int] = (39, 69, 219)
        self.diastole_color_plt: tuple[float, ...] = tuple(x / 255 for x in self.diastole_color)  # for matplotlib
        self.systole_color: tuple[int, int, int] = (209, 55, 38)
        self.systole_color_plt: tuple[float, ...] = tuple(x / 255 for x in self.systole_color)
        self.waiting_status: str = 'Waiting for user input...'
        self.small_display = None
        self.results_plot = None
        self.init_gui()
        init_shortcuts(self)

    def init_gui(self) -> None:
        self.menu_bar: QMenuBar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.file_name = "default_file_name"  # Initialize file_name with a default value
        init_menu(self)
        self.metadata_table: QTableWidget = QTableWidget()

        self.status_bar: QStatusBar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.waiting_status)

        # Left-half widgets
        self.display: Display = Display(self)
        self.display_frame_comms: Communicate = Communicate()
        self.display_frame_comms.updateBW[int].connect(self.display.set_frame)
        self.display_slider: Slider = Slider(self, Qt.Orientation.Horizontal)
        self.hide_contours_box: QCheckBox = QCheckBox('&Hide Contours')
        self.hide_contours_box.setChecked(False)
        self.hide_special_points_box: QCheckBox = QCheckBox('&Hide Metrics')
        self.hide_special_points_box.setChecked(False)
        self.mask_mode_box: QCheckBox = QCheckBox('&Mask mode')
        self.mask_mode_box.setChecked(False)

        # Right-half widgets
        self.diastolic_frame_box: QCheckBox = QCheckBox('Diastolic Frame')
        self.diastolic_frame_box.setChecked(False)
        self.diastolic_frame_box.stateChanged.connect(partial(toggle_diastolic_frame, self))
        self.systolic_frame_box: QCheckBox = QCheckBox('Systolic Frame')
        self.systolic_frame_box.setChecked(False)
        self.systolic_frame_box.stateChanged.connect(partial(toggle_systolic_frame, self))
        self.use_diastolic_button: QPushButton = QPushButton('Diastolic Frames')
        self.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.diastole_color}')
        self.use_diastolic_button.setCheckable(True)
        self.use_diastolic_button.setChecked(True)
        self.use_diastolic_button.clicked.connect(partial(use_diastolic, self))
        self.use_diastolic_button.setToolTip('Press button to switch between diastolic and systolic frames')
        self.gating_display: GatingDisplay = GatingDisplay(self)
        self.longitudinal_view: LongitudinalView = LongitudinalView(self)

        # OCT-specific widgets
        self.tagged_frame_button: QCheckBox = QCheckBox('Tagged Frame')
        self.tagged_frame_button.setChecked(False)
        self.tagged_frame_button.stateChanged.connect(partial(toggle_tagged_frame, self))
        self.use_tagged_button: QPushButton = QPushButton('Tagged Frames')
        self.use_tagged_button.setStyleSheet('background-color: yellow')
        self.use_tagged_button.clicked.connect(partial(use_tagged, self))
        self.oct_quality_buttons: dict[str, QPushButton] = {}
        self.oct_quality_button_group: QButtonGroup = QButtonGroup(self)
        self.oct_quality_button_group.setExclusive(True)
        for label in OCT_QUALITY_LABELS:
            btn: QPushButton = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(partial(set_oct_quality, self, label))
            self.oct_quality_buttons[label] = btn
            self.oct_quality_button_group.addButton(btn)
        self.oct_quality_buttons[OCT_QUALITY_LABELS[-1]].setChecked(True)
        self.gated_frames_oct: list[int] = []

        main_window_splitter: QSplitter = QSplitter()
        self.left_half: LeftHalf = LeftHalf(self)
        main_window_splitter.addWidget(self.left_half())
        self.right_half: RightHalf = RightHalf(self)
        main_window_splitter.addWidget(self.right_half())

        self.setWindowTitle('AIVUS Software')
        icon_path: str = os.path.join(os.path.dirname(__file__), '..', '..', 'media', 'desktop_img.ico')
        self.setWindowIcon(QIcon(icon_path))
        self.setCentralWidget(main_window_splitter)
        self.showMaximized()

        timer: QTimer = QTimer(self)
        timer.timeout.connect(self.auto_save)
        timer.start(self.config.save.autosave_interval)  # autosave interval in milliseconds

    def auto_save(self) -> None:
        if self.image_displayed:
            write_contours(self)

    def reset_state(self) -> None:
        if self.results_plot is not None:
            self.results_plot.close()
        if self.small_display is not None:
            self.small_display.close()
            self.small_display = None
        self.display.reset()
        self.file_name = None
        self.image_displayed = False
        self.segmentation = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.tmp_contours = {}
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = RuntimeData()
        self.gating_signal = {}
        self.metadata = {}
        self.images = None
        self.images_display = None
        self.images_rgb = None
        self.dicom = None
        self.gated_frames_oct = []
        self.tagged_frame_button.setChecked(False)
        self.oct_quality_buttons[OCT_QUALITY_LABELS[-1]].setChecked(True)
        self.init_gui()
