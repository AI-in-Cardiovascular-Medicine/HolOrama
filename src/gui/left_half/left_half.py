import time
import bisect

from loguru import logger
from functools import partial
from PyQt6.QtWidgets import (
    QPushButton,
    QButtonGroup,
    QStyle,
    QApplication,
    QLabel,
    QWidget,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
)
from PyQt6.QtCore import Qt

from gui.left_half.IVUS_display import IVUSDisplay, ContourType, SegmentationTool
from gui.utils.slider import Slider, Communicate
from gui.utils.contours_gui import new_measure, new_reference, new_angle, set_tool


class LeftHalf:
    def __init__(self, main_window):
        self.main_window = main_window
        self.left_widget = QWidget()
        left_vbox = QVBoxLayout()
        main_window.display = IVUSDisplay(main_window)
        main_window.display_frame_comms = Communicate()
        main_window.display_frame_comms.updateBW[int].connect(main_window.display.set_frame)

        display_buttons_hbox = QHBoxLayout()
        self.display_button_group = QButtonGroup()
        self.display_button_group.setExclusive(True)

        self.closed_spline_btn = QPushButton('⭕Closed Spline')
        self.closed_spline_btn.setCheckable(True)
        self.closed_spline_btn.setChecked(True) # this is the default button
        self.closed_spline_btn.setToolTip("Set drawing mode to closed spline")
        self.closed_spline_btn.clicked.connect(partial(set_tool, main_window, SegmentationTool.CLOSED_SPLINE))

        self.open_spline_btn = QPushButton('➰Open Spline')
        self.open_spline_btn.setCheckable(True)
        self.open_spline_btn.setToolTip("Set drawing mode to open spline")
        self.open_spline_btn.clicked.connect(partial(set_tool, main_window, SegmentationTool.OPEN_SPLINE))

        self.brush_btn = QPushButton('🖌️Brush')
        self.brush_btn.setCheckable(True)
        self.brush_btn.setToolTip("Set drawing mode to brush")
        self.brush_btn.clicked.connect(partial(set_tool, main_window, SegmentationTool.BRUSH))

        self.reference_btn = QPushButton('🟡Reference')
        self.reference_btn.setCheckable(True)
        self.reference_btn.setToolTip("Set a reference point")
        self.reference_btn.setStyleSheet(f'border-color: {main_window.reference_color}')
        self.reference_btn.clicked.connect(partial(new_reference, main_window))

        self.measure_btn_1 = QPushButton('📏Measurement 1')
        self.measure_btn_1.setCheckable(True)
        self.measure_btn_1.setToolTip("Measure distance between two points")
        self.measure_btn_1.setStyleSheet(f'border-color: {main_window.measure_colors[0]}')
        self.measure_btn_1.clicked.connect(partial(new_measure, main_window, 0))

        self.measure_btn_2 = QPushButton('📏Measurement 2')
        self.measure_btn_2.setCheckable(True)
        self.measure_btn_2.setToolTip("Measure distance between two points")
        self.measure_btn_2.setStyleSheet(f'border-color: {main_window.measure_colors[1]}')
        self.measure_btn_2.clicked.connect(partial(new_measure, main_window, 1))

        self.angle_btn = QPushButton('📐Angle Wire')
        self.angle_btn.setCheckable(True)
        self.angle_btn.setToolTip("Set angle between two points for wire shadow")
        self.angle_btn.setStyleSheet(f'border-color: {main_window.display.color_angle}')
        self.angle_btn.clicked.connect(partial(new_angle, main_window, ContourType.WIRE))

        self.display_buttons = [
            self.closed_spline_btn, self.open_spline_btn, self.brush_btn,
            self.reference_btn, self.measure_btn_1, self.measure_btn_2, self.angle_btn,
        ]
        for btn in self.display_buttons:
            self.display_button_group.addButton(btn)
            display_buttons_hbox.addWidget(btn)
        left_vbox.addLayout(display_buttons_hbox)

        left_vbox.addWidget(main_window.display)

        left_lower_grid = QGridLayout()
        hide_checkboxes = QHBoxLayout()
        main_window.hide_contours_box = QCheckBox('&Hide Contours')
        main_window.hide_contours_box.setChecked(False)
        main_window.hide_contours_box.stateChanged[int].connect(self.toggle_hide_contours)
        
        main_window.hide_special_points_box = QCheckBox('&Hide Metrics')
        main_window.hide_special_points_box.setChecked(False)
        main_window.hide_special_points_box.stateChanged[int].connect(self.toggle_hide_special_points)

        main_window.mask_mode_box = QCheckBox('&Mask mode')
        main_window.mask_mode_box.setChecked(False)
        main_window.mask_mode_box.stateChanged[int].connect(self.toggle_mask_mode)

        hide_checkboxes.addWidget(main_window.hide_contours_box)
        hide_checkboxes.addWidget(main_window.hide_special_points_box)
        hide_checkboxes.addWidget(main_window.mask_mode_box)
        left_lower_grid.addLayout(hide_checkboxes, 0, 0)

        self.play_button = QPushButton()
        self.play_icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.pause_icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        
        self.play_button.setIcon(self.play_icon)
        self.play_button.setMaximumWidth(30)
        self.play_button.clicked.connect(partial(self.play, main_window))
        self.paused = True
        
        main_window.display_slider = Slider(main_window, Qt.Orientation.Horizontal)
        main_window.display_slider.valueChanged[int].connect(self.change_value)
        
        slider_hbox = QHBoxLayout()
        slider_hbox.addWidget(self.play_button)
        slider_hbox.addWidget(main_window.display_slider)
        left_lower_grid.addLayout(slider_hbox, 0, 1)

        self.frame_number_label = QLabel()
        self.frame_number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_number_label.setText(f'Frame {main_window.display_slider.value() + 1}')
        
        frame_num_hbox = QHBoxLayout()
        frame_num_hbox.addWidget(self.frame_number_label)
        left_lower_grid.addLayout(frame_num_hbox, 1, 1)
        left_vbox.addLayout(left_lower_grid)
        self.left_widget.setLayout(left_vbox)

    def __call__(self):
        return self.left_widget

    def play(self, main_window):
        """Plays all frames until end of pullback starting from currently selected frame"""
        if not main_window.image_displayed:
            return

        start_frame = main_window.display_slider.value()
        if self.paused:
            self.paused = False
            self.play_button.setIcon(self.pause_icon)
        else:
            self.paused = True
            self.play_button.setIcon(self.play_icon)

        for frame in range(start_frame, main_window.metadata['num_frames']):
            if not self.paused:
                main_window.display_slider.set_value(frame)
                QApplication.processEvents()
                time.sleep(0.05)
                self.frame_number_label.setText(f'Frame {frame + 1}')

        self.play_button.setIcon(self.play_icon)

    def change_value(self, value):
        self.main_window.display_frame_comms.updateBW.emit(value)
        self.main_window.display.update_display()
        self.frame_number_label.setText(f'Frame {value + 1}')

        if value in self.main_window.gated_frames_dia:
            self.main_window.diastolic_frame_box.setChecked(True)
        else:
            self.main_window.diastolic_frame_box.setChecked(False)
            if value in self.main_window.gated_frames_sys:
                self.main_window.systolic_frame_box.setChecked(True)
            else:
                self.main_window.systolic_frame_box.setChecked(False)

    def toggle_hide_contours(self, value):
        if self.main_window.image_displayed:
            self.main_window.hide_contours = bool(value) # Cast to bool for safety
            self.main_window.display.update_display()
            if hasattr(self.main_window, "small_display"):
                next_gated = self.main_window.display_slider.next_gated_frame(set=False)
                if next_gated is not None:
                    self.main_window.small_display.update_frame(next_gated, update_contours=True)
            if not value:
                self.main_window.longitudinal_view.show_lview_contours()

    def toggle_hide_special_points(self, value):
        if self.main_window.image_displayed:
            self.main_window.hide_special_points = bool(value)
            self.main_window.display.update_display()

    def toggle_mask_mode(self, _):
        if self.main_window.image_displayed:
            self.main_window.display.update_display()