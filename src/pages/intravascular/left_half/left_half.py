import time

from functools import partial
from PyQt6.QtWidgets import (
    QPushButton,
    QButtonGroup,
    QComboBox,
    QStyle,
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLayout,
)
from PyQt6.QtCore import Qt

from domain.all_types import ContourType, SegmentationTool
from pages.intravascular.utils.contours_gui import (
    new_contour,
    new_contour_append,
    new_measure,
    new_reference,
    new_angle,
    set_tool,
)
from pages.intravascular.utils.helpers import SplitterPane
from pages.intravascular.brush_panel import HoverButton

# (label, ContourType, new-shortcut, append-shortcut or None)
_CONTOUR_TYPE_ITEMS = [
    ('Lumen', ContourType.LUMEN, 'E', None),
    ('EEM', ContourType.EEM, 'Q', None),
    ('Calcium', ContourType.CALCIUM, '7', 'Ctrl+7'),
    ('Branch', ContourType.BRANCH, '8', 'Ctrl+8'),
    ('Lipid', ContourType.LIPID, '9', 'Ctrl+9'),
    ('Macrophage', ContourType.MACROPHAGE, '0', 'Ctrl+0'),
]


class LeftHalf:
    def __init__(self, main_window):
        self.main_window = main_window
        self.left_widget = SplitterPane()
        left_vbox = QVBoxLayout()
        self.measure_colors: list[str] = ['red', 'cyan']
        self.reference_color: str = 'yellow'

        display_buttons_hbox = QHBoxLayout()
        self.display_button_group = QButtonGroup()
        self.display_button_group.setExclusive(True)

        self.closed_spline_btn = QPushButton('⭕ Closed Spline')
        self.closed_spline_btn.setCheckable(True)
        self.closed_spline_btn.setChecked(True)  # this is the default button
        self.closed_spline_btn.setToolTip("Set drawing mode to closed spline")
        self.closed_spline_btn.clicked.connect(partial(set_tool, main_window, SegmentationTool.CLOSED_SPLINE))

        self.open_spline_btn = QPushButton('➰ Open Spline')
        self.open_spline_btn.setCheckable(True)
        self.open_spline_btn.setToolTip("Set drawing mode to open spline")
        self.open_spline_btn.clicked.connect(partial(set_tool, main_window, SegmentationTool.OPEN_SPLINE))

        self.brush_btn = HoverButton('🖌️ Brush')
        self.brush_btn.setCheckable(True)
        self.brush_btn.setToolTip("Set drawing mode to brush (requires Mask Mode)")
        self.brush_btn.clicked.connect(partial(set_tool, main_window, SegmentationTool.BRUSH))
        popup = main_window.brush_settings_popup
        self.brush_btn._on_hover_enter = lambda: popup.show_near(self.brush_btn)
        self.brush_btn._on_hover_leave = popup.schedule_hide

        self.reference_btn = QPushButton('🟡 Reference')
        self.reference_btn.setCheckable(True)
        self.reference_btn.setToolTip("Set a reference point")
        self.reference_btn.setStyleSheet(f'border-color: {self.reference_color}')
        self.reference_btn.clicked.connect(partial(new_reference, main_window))

        self.measure_btn_1 = QPushButton('📏 Measurement 1')
        self.measure_btn_1.setCheckable(True)
        self.measure_btn_1.setToolTip("Measure distance between two points")
        self.measure_btn_1.setStyleSheet(f'border-color: {self.measure_colors[0]}')
        self.measure_btn_1.clicked.connect(partial(new_measure, main_window, 0))

        self.measure_btn_2 = QPushButton('📏 Measurement 2')
        self.measure_btn_2.setCheckable(True)
        self.measure_btn_2.setToolTip("Measure distance between two points")
        self.measure_btn_2.setStyleSheet(f'border-color: {self.measure_colors[1]}')
        self.measure_btn_2.clicked.connect(partial(new_measure, main_window, 1))

        self.angle_btn = QPushButton('📐 Angle Wire')
        self.angle_btn.setCheckable(True)
        self.angle_btn.setToolTip("Set angle between two points for wire shadow")
        self.angle_btn.setStyleSheet(f'border-color: {main_window.display.color_angle}')
        self.angle_btn.clicked.connect(partial(new_angle, main_window, ContourType.WIRE))

        self.display_buttons = [
            self.closed_spline_btn,
            self.open_spline_btn,
            self.brush_btn,
            self.reference_btn,
            self.measure_btn_1,
            self.measure_btn_2,
            self.angle_btn,
        ]
        for btn in self.display_buttons:
            self.display_button_group.addButton(btn)
            display_buttons_hbox.addWidget(btn)
        left_vbox.addLayout(display_buttons_hbox)

        # Second row: contour type selector + new/add buttons
        contour_row_hbox = QHBoxLayout()

        self.contour_type_combo = QComboBox()
        for label, _, _, _ in _CONTOUR_TYPE_ITEMS:
            self.contour_type_combo.addItem(label)
        self.contour_type_combo.setToolTip("Select contour type")
        self.contour_type_combo.currentIndexChanged.connect(self._on_contour_type_changed)

        self.new_contour_btn = QPushButton('New Contour')
        self.new_contour_btn.clicked.connect(self._on_new_contour)

        self.add_contour_btn = QPushButton('+ Add Contour')
        self.add_contour_btn.clicked.connect(self._on_add_contour)

        contour_row_hbox.addWidget(self.contour_type_combo)
        contour_row_hbox.addWidget(self.new_contour_btn)
        contour_row_hbox.addWidget(self.add_contour_btn)
        left_vbox.addLayout(contour_row_hbox)

        self._on_contour_type_changed(0)  # set initial tooltips and button state

        left_vbox.addWidget(main_window.display)

        left_lower_grid = QGridLayout()
        hide_checkboxes = QHBoxLayout()
        main_window.hide_contours_box.stateChanged[int].connect(self.toggle_hide_contours)
        main_window.hide_special_points_box.stateChanged[int].connect(self.toggle_hide_special_points)
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
        left_vbox.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
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

        for frame in range(start_frame, main_window.runtime_data.metadata['num_frames']):
            if not self.paused:
                main_window.display_slider.set_value(frame)
                QApplication.processEvents()
                time.sleep(0.05)
                self.frame_number_label.setText(f'Frame {frame + 1}')

        self.play_button.setIcon(self.play_icon)

    def change_value(self, value: int):
        self.main_window.display_frame_comms.updateBW.emit(value)
        self.main_window.display.update_display()
        self.frame_number_label.setText(f'Frame {value + 1}')

        if value in self.main_window.runtime_data.gated_frames_dia:
            self.main_window.diastolic_frame_box.setChecked(True)
        else:
            self.main_window.diastolic_frame_box.setChecked(False)
            if value in self.main_window.runtime_data.gated_frames_sys:
                self.main_window.systolic_frame_box.setChecked(True)
            else:
                self.main_window.systolic_frame_box.setChecked(False)

    def toggle_hide_contours(self, value: int):
        if self.main_window.image_displayed:
            self.main_window.hide_contours = bool(value)  # Cast to bool for safety
            self.main_window.display.update_display()
            if self.main_window.small_display is not None:
                next_gated = self.main_window.display_slider.next_gated_frame(set=False)
                if next_gated is not None:
                    self.main_window.small_display.update_frame(next_gated, update_contours=True)
            if not value:
                self.main_window.longitudinal_view.show_lview_contours()

    def toggle_hide_special_points(self, value: int):
        if self.main_window.image_displayed:
            self.main_window.hide_special_points = bool(value)
            self.main_window.display.update_display()

    def _on_contour_type_changed(self, index: int):
        _, _, new_key, add_key = _CONTOUR_TYPE_ITEMS[index]
        self.new_contour_btn.setToolTip(f"Draw new contour ({new_key})" if new_key else "Draw new contour")
        self.add_contour_btn.setEnabled(add_key is not None)
        self.add_contour_btn.setToolTip(
            f"Append contour ({add_key})" if add_key else "No append shortcut for this type"
        )

    def _on_new_contour(self):
        _, contour_type, _, _ = _CONTOUR_TYPE_ITEMS[self.contour_type_combo.currentIndex()]
        new_contour(self.main_window, contour_type)

    def _on_add_contour(self):
        _, contour_type, _, _ = _CONTOUR_TYPE_ITEMS[self.contour_type_combo.currentIndex()]
        new_contour_append(self.main_window, contour_type)

    def set_active_contour_type_ui(self, contour_type: ContourType):
        for i, (_, ct, _, _) in enumerate(_CONTOUR_TYPE_ITEMS):
            if ct == contour_type:
                self.contour_type_combo.setCurrentIndex(i)
                break

    def toggle_mask_mode(self, _):
        if self.main_window.image_displayed:
            if not self.main_window.mask_mode_box.isChecked():
                if self.main_window.display._brush_active:
                    self.main_window.display.disable_brush()
                    self.closed_spline_btn.setChecked(True)
            self.main_window.display.update_display()
