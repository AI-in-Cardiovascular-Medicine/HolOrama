import bisect

import matplotlib.pyplot as plt
from loguru import logger
from functools import partial
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QSplitter, QPushButton, QCheckBox, QWidget

from gui.right_half.gating_display import GatingDisplay
from gui.right_half.longitudinal_view import LongitudinalView
from gui.popup_windows.small_display import SmallDisplay
from gui.utils.contours_gui import new_measure, new_reference
# from segmentation.segment import segment


class RightHalf:
    def __init__(self, main_window):
        self.main_window = main_window
        self.right_widget = QWidget()
        right_vbox = QVBoxLayout()
        checkboxes = QHBoxLayout()
        
        self.main_window.diastolic_frame_box = QCheckBox('Diastolic Frame')
        self.main_window.diastolic_frame_box.setChecked(False)
        self.main_window.diastolic_frame_box.stateChanged.connect(partial(toggle_diastolic_frame, main_window))
        checkboxes.addWidget(self.main_window.diastolic_frame_box)
        
        self.main_window.systolic_frame_box = QCheckBox('Systolic Frame')
        self.main_window.systolic_frame_box.setChecked(False)
        self.main_window.systolic_frame_box.stateChanged.connect(partial(toggle_systolic_frame, main_window))
        checkboxes.addWidget(self.main_window.systolic_frame_box)
        
        self.main_window.use_diastolic_button = QPushButton('Diastolic Frames')
        self.main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.main_window.diastole_color}')
        self.main_window.use_diastolic_button.setCheckable(True)
        self.main_window.use_diastolic_button.setChecked(True)
        self.main_window.use_diastolic_button.clicked.connect(partial(use_diastolic, main_window))
        self.main_window.use_diastolic_button.setToolTip('Press button to switch between diastolic and systolic frames')
        checkboxes.addWidget(self.main_window.use_diastolic_button)
        
        small_display_button = QPushButton('Compare Frames')
        small_display_button.setToolTip('Open a small display to compare two frames')
        small_display_button.clicked.connect(partial(open_small_display, main_window))
        checkboxes.addWidget(small_display_button)
        
        main_window.gating_display = GatingDisplay(main_window)
        checkboxes.addWidget(main_window.gating_display.toolbar)
        right_vbox.addLayout(checkboxes)
        
        # Change 2: Scoped Orientation Enum
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(main_window.gating_display)
        main_window.longitudinal_view = LongitudinalView(main_window)
        splitter.addWidget(main_window.longitudinal_view)
        
        gating_display_size = main_window.gating_display.sizeHint().height()
        splitter.setSizes([gating_display_size, gating_display_size])
        splitter.setStretchFactor(0, main_window.config.display.gating_display_stretch)
        splitter.setStretchFactor(1, main_window.config.display.lview_display_stretch)
        right_vbox.addWidget(splitter)

        right_lower_vbox = QVBoxLayout()
        segment_button = QPushButton('Automatic Segmentation')
        segment_button.setToolTip('Run deep learning based segmentation of lumen')
        # segment_button.clicked.connect(partial(segment, main_window))
        gating_button = QPushButton('Extract Diastolic and Systolic Frames')
        gating_button.setToolTip('Extract diastolic and systolic images from pullback')
        gating_button.clicked.connect(main_window.contour_based_gating)
        command_buttons = QHBoxLayout()
        command_buttons.addWidget(segment_button)
        command_buttons.addWidget(gating_button)
        right_lower_vbox.addLayout(command_buttons)
        measures = QHBoxLayout()
        right_lower_vbox.addLayout(measures)
        right_vbox.addLayout(right_lower_vbox)
        
        self.right_widget.setLayout(right_vbox)

    def __call__(self):
        return self.right_widget


def open_small_display(main_window):
    if main_window.image_displayed:
        main_window.small_display = SmallDisplay(main_window)
        main_window.small_display.move(
            main_window.x() + main_window.width() // 2, main_window.y() + main_window.height() // 2
        )
        next_gated = main_window.display_slider.next_gated_frame(set=False)
        main_window.small_display.update_frame(
            next_gated, update_image=True, update_contours=True, update_text=True
        )
        main_window.small_display.show()


def toggle_diastolic_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            main_window.use_diastolic_button.setChecked(True)
            use_diastolic(main_window)
            if frame not in main_window.gated_frames_dia:
                bisect.insort_left(main_window.gated_frames_dia, frame)
                main_window.data[frame].phase = 'D'
                main_window.contour_based_gating.update_color(main_window.diastole_color_plt)
                main_window.contour_based_gating.current_phase = 'D'
                plt.draw()
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.systolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.gated_frames_dia.remove(frame)
                main_window.contour_based_gating.current_phase = None
                if (
                    main_window.data[frame].phase == 'D'
                ):  # do not reset when function is called from toggle_systolic_frame
                    main_window.data[frame].phase = '-'
                    if not drag:
                        main_window.contour_based_gating.update_color()
            except ValueError:
                pass

        main_window.display.update_display()


def toggle_systolic_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            main_window.use_diastolic_button.setChecked(False)
            use_diastolic(main_window)
            if frame not in main_window.gated_frames_sys:
                bisect.insort_left(main_window.gated_frames_sys, frame)
                main_window.data[frame].phase = 'S'
                main_window.contour_based_gating.update_color(main_window.systole_color_plt)
                main_window.contour_based_gating.current_phase = 'S'
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.diastolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.gated_frames_sys.remove(frame)
                main_window.contour_based_gating.current_phase = None
                if (
                    main_window.data[frame].phase == 'S'
                ):  # do not reset when function is called from toggle_diastolic_frame
                    main_window.data[frame].phase = '-'
                    if not drag:
                        main_window.contour_based_gating.update_color()
            except ValueError:
                pass

        main_window.display.update_display()


def use_diastolic(main_window):
    if main_window.image_displayed:
        if main_window.use_diastolic_button.isChecked():
            main_window.use_diastolic_button.setText('Diastolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.diastole_color}')
            main_window.gated_frames = main_window.gated_frames_dia
        else:
            main_window.use_diastolic_button.setText('Systolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.systole_color}')
            main_window.gated_frames = main_window.gated_frames_sys

        try:
            next_gated = main_window.display_slider.next_gated_frame(set=False)
            main_window.small_display.update_frame(next_gated, update_image=True, update_contours=True, update_text=True)  # update small display
        except AttributeError:
            pass
