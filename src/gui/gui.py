import os
from functools import partial
from loguru import logger

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
from input_output.contours_io import write_contours
from gating.contour_based_gating import ContourBasedGating
from segmentation.predict import Predict


class Master(QMainWindow):
    """Main Window Class"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.file_name = None
        self.autosave_interval = config.save.autosave_interval
        self.contour_based_gating = ContourBasedGating(self)
        self.predictor = Predict(self)
        self.image_displayed = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.filter = None
        self.tmp_contours = {}  # per-contour-type undo storage, e.g. {'lumen': (xlist, ylist)}
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {}  # container to be saved in JSON file later, includes contours, etc.
        self.gating_signal = {}  # global gating signal, saved separately from per-frame data
        self.metadata = {}  # metadata used outside of read_image (not saved to JSON file)
        self.images = None
        self.diastole_color = (39, 69, 219)
        self.diastole_color_plt = tuple(x / 255 for x in self.diastole_color)  # for matplotlib
        self.systole_color = (209, 55, 38)
        self.systole_color_plt = tuple(x / 255 for x in self.systole_color)
        self.waiting_status = 'Waiting for user input...'
        self.init_gui()
        init_shortcuts(self)

    def init_gui(self):
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.file_name = "default_file_name"  # Initialize file_name with a default value
        init_menu(self)
        self.metadata_table = QTableWidget()

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.waiting_status)

        # Left-half widgets
        self.display = Display(self)
        self.display_frame_comms = Communicate()
        self.display_frame_comms.updateBW[int].connect(self.display.set_frame)
        self.display_slider = Slider(self, Qt.Orientation.Horizontal)
        self.hide_contours_box = QCheckBox('&Hide Contours')
        self.hide_contours_box.setChecked(False)
        self.hide_special_points_box = QCheckBox('&Hide Metrics')
        self.hide_special_points_box.setChecked(False)
        self.mask_mode_box = QCheckBox('&Mask mode')
        self.mask_mode_box.setChecked(False)

        # Right-half widgets
        self.diastolic_frame_box = QCheckBox('Diastolic Frame')
        self.diastolic_frame_box.setChecked(False)
        self.diastolic_frame_box.stateChanged.connect(partial(toggle_diastolic_frame, self))
        self.systolic_frame_box = QCheckBox('Systolic Frame')
        self.systolic_frame_box.setChecked(False)
        self.systolic_frame_box.stateChanged.connect(partial(toggle_systolic_frame, self))
        self.use_diastolic_button = QPushButton('Diastolic Frames')
        self.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.diastole_color}')
        self.use_diastolic_button.setCheckable(True)
        self.use_diastolic_button.setChecked(True)
        self.use_diastolic_button.clicked.connect(partial(use_diastolic, self))
        self.use_diastolic_button.setToolTip('Press button to switch between diastolic and systolic frames')
        self.gating_display = GatingDisplay(self)
        self.longitudinal_view = LongitudinalView(self)

        # OCT-specific widgets
        self.tagged_frame_button = QCheckBox('Tagged Frame')
        self.tagged_frame_button.setChecked(False)
        self.tagged_frame_button.stateChanged.connect(partial(toggle_tagged_frame, self))
        self.use_tagged_button = QPushButton('Tagged Frames')
        self.use_tagged_button.setStyleSheet('background-color: yellow')
        self.use_tagged_button.clicked.connect(partial(use_tagged, self))
        self.oct_quality_buttons = {}
        self.oct_quality_button_group = QButtonGroup(self)
        self.oct_quality_button_group.setExclusive(True)
        for label in ['Very Bad', 'Bad', 'Ok', 'Good', 'Very Good']:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(partial(set_oct_quality, self, label))
            self.oct_quality_buttons[label] = btn
            self.oct_quality_button_group.addButton(btn)
        self.oct_quality_buttons['Very Good'].setChecked(True)
        self.gated_frames_oct = []

        main_window_splitter = QSplitter()
        self.left_half = LeftHalf(self)
        main_window_splitter.addWidget(self.left_half())
        self.right_half = RightHalf(self)
        main_window_splitter.addWidget(self.right_half())

        self.setWindowTitle('AIVUS Software')
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'media', 'desktop_img.ico')
        self.setWindowIcon(QIcon(icon_path))
        self.setCentralWidget(main_window_splitter)
        self.showMaximized()

        timer = QTimer(self)
        timer.timeout.connect(self.auto_save)
        timer.start(self.autosave_interval)  # autosave interval in milliseconds

    def auto_save(self):
        if self.image_displayed:
            write_contours(self)

    def reset_state(self):
        self.file_name = None
        self.image_displayed = False
        self.segmentation = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.filter = None
        self.tmp_contours = {}
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {}
        self.gating_signal = {}
        self.metadata = {}
        self.images = None
        self.images_display = None
        self.images_rgb = None
        self.gated_frames_oct = []
        self.tagged_frame_button.setChecked(False)
        self.oct_quality_buttons['Very Good'].setChecked(True)
