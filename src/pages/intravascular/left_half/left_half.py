import time
from functools import partial
from typing import Optional, Tuple

from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
)

from domain.intravascular.types import ContourType, SegmentationTool
from domain.intravascular.contour_presets import ContourPreset, active_preset
from domain.intravascular.knot_resampling import MAX_KNOTS, MIN_KNOTS
from pages.intravascular.brush_panel import HoverButton
from pages.intravascular.utils.contours_gui import (
    delete_active_contour_on_all_frames,
    delete_all_on_frame,
    new_angle,
    new_contour,
    new_contour_append,
    new_measure,
    new_reference,
    selected_contour_knot_count,
    set_selected_contour_knots,
    set_tool,
)
from pages.intravascular.utils.helpers import SplitterPane

_TypeItem = Tuple[str, ContourType, Optional[str], Optional[str]]  # label, type, new/append shortcuts (or None)


def _type_items(preset: ContourPreset, angle: bool) -> list[_TypeItem]:
    """One drop-down entry per spline type (angle=False) or sector type (angle=True) of `preset`,
    in row order. Sector placement is shared (tools.angle), so a new sector type needs only its preset row."""
    rows = preset.angle_types if angle else preset.spline_types
    return [(defn.name, defn.type, *preset.shortcuts(defn.type)) for defn in rows]


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
        self.closed_spline_btn.setChecked(True)
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

        # Filled from the active preset by refresh_contour_types.
        self._contour_type_items: list[_TypeItem] = []
        self._angle_type_items: list[_TypeItem] = []
        # Sector type both angle controls act on (None if the preset has none), synced by set_active_contour_type_ui.
        self.angle_type: ContourType | None = None

        self.angle_type_combo = QComboBox()
        self.angle_type_combo.currentIndexChanged.connect(self._on_angle_type_changed)
        self.angle_type_combo.activated.connect(self._on_angle_type_activated)

        self.add_angle_btn = QPushButton()
        self.add_angle_btn.setCheckable(True)
        self.add_angle_btn.clicked.connect(self._on_add_angle)

        self.display_buttons = [
            self.closed_spline_btn,
            self.open_spline_btn,
            self.brush_btn,
            self.reference_btn,
            self.measure_btn_1,
            self.measure_btn_2,
            self.add_angle_btn,
        ]
        for btn in self.display_buttons:
            self.display_button_group.addButton(btn)

        # Picking a type from the angle drop-down starts a sector of it and points ➕📐 Add at the same type.
        for widget in (
            self.closed_spline_btn,
            self.open_spline_btn,
            self.brush_btn,
            self.reference_btn,
            self.measure_btn_1,
            self.measure_btn_2,
            self.angle_type_combo,
            self.add_angle_btn,
        ):
            display_buttons_hbox.addWidget(widget)
        left_vbox.addLayout(display_buttons_hbox)

        contour_row_hbox = QHBoxLayout()

        self.contour_type_combo = QComboBox()
        self.contour_type_combo.setToolTip("Select contour type")
        self.contour_type_combo.currentIndexChanged.connect(self._on_contour_type_changed)

        self.new_contour_btn = QPushButton('New Contour')
        self.new_contour_btn.clicked.connect(self._on_new_contour)

        self.add_contour_btn = QPushButton('+ Add Contour')
        self.add_contour_btn.clicked.connect(self._on_add_contour)

        self.delete_all_btn = QPushButton('🗑️ Delete All On Frame')
        self.delete_all_btn.clicked.connect(self._on_delete_all)
        self.delete_all_btn.setStyleSheet('background: darkred')
        self.delete_all_btn.setToolTip("Deletes all currently displayed contours on the image")

        contour_row_hbox.addWidget(self.contour_type_combo)
        contour_row_hbox.addWidget(self.new_contour_btn)
        contour_row_hbox.addWidget(self.add_contour_btn)
        self.delete_all_frames_btn = QPushButton('🗑️ Delete Current Contour On All')
        self.delete_all_frames_btn.clicked.connect(self._on_delete_all_frames)
        self.delete_all_frames_btn.setStyleSheet('background: darkred')
        self.delete_all_frames_btn.setToolTip("Deletes the selected contour type on every frame (Ctrl+Z undoes it)")

        # Shows the selected contour's knot count, kept in step by sync_knot_count.
        self.knot_count_box = QSpinBox()
        self.knot_count_box.setPrefix('Points: ')
        self.knot_count_box.setRange(MIN_KNOTS, MAX_KNOTS)
        self.knot_count_box.setKeyboardTracking(False)  # typing 15 is not a stop at 1 first
        self.knot_count_box.setToolTip(
            "Number of points on the selected contour, its shape kept (Shift+Wheel on the image, Ctrl+Z undoes it)"
        )
        self.knot_count_box.valueChanged[int].connect(self._on_knot_count_changed)
        self.knot_count_box.setEnabled(False)

        contour_row_hbox.addWidget(self.delete_all_btn)
        contour_row_hbox.addWidget(self.delete_all_frames_btn)
        contour_row_hbox.addWidget(self.knot_count_box)
        left_vbox.addLayout(contour_row_hbox)

        self.refresh_contour_types()  # fills both drop-downs, tooltips and button state

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

    def refresh_contour_types(self) -> None:
        """(Re)fill both drop-downs from the active preset, each starting on its first type."""
        preset = active_preset()
        self._contour_type_items = _type_items(preset, angle=False)
        self._angle_type_items = _type_items(preset, angle=True)

        for combo, items, prefix in (
            (self.contour_type_combo, self._contour_type_items, ''),
            (self.angle_type_combo, self._angle_type_items, '📐 Angle '),
        ):
            combo.blockSignals(True)
            combo.clear()
            for label, _, _, _ in items:
                combo.addItem(f'{prefix}{label}')
            combo.setCurrentIndex(0)
            combo.blockSignals(False)

        self.angle_type = self._angle_type_items[0][1] if self._angle_type_items else None
        self.angle_type_combo.setVisible(self.angle_type is not None)
        self.add_angle_btn.setVisible(self.angle_type is not None)
        if self.angle_type is not None:
            self._apply_angle_type(self.angle_type)
        self._on_contour_type_changed(0)

    def play(self, main_window):
        """Play (or pause) from the current frame to the pullback's end."""
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
        if sip.isdeleted(self.main_window):
            return
        self.main_window.display_frame_comms.updateBW.emit(value)
        self.main_window.display.update_display()
        self.frame_number_label.setText(f'Frame {value + 1}')

        # OCT pullbacks have no cardiac phase to mirror into the gating checkboxes
        if self.main_window.right_half.is_oct:
            return
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
            self.main_window.hide_contours = bool(value)
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
        _, contour_type, new_key, add_key = self._contour_type_items[index]
        preset = active_preset()
        self.new_contour_btn.setToolTip(f"Draw new contour ({new_key})" if new_key else "Draw new contour")
        appendable = preset[contour_type].appendable
        self.add_contour_btn.setEnabled(appendable)
        if not appendable:
            self.add_contour_btn.setToolTip("A frame holds one contour of this type")
        else:
            self.add_contour_btn.setToolTip(f"Append contour ({add_key})" if add_key else "Append contour")

        allowed = preset.allowed_tools(contour_type)
        tool_btns = [
            (self.closed_spline_btn, SegmentationTool.CLOSED_SPLINE),
            (self.open_spline_btn, SegmentationTool.OPEN_SPLINE),
            (self.brush_btn, SegmentationTool.BRUSH),
        ]
        for btn, tool in tool_btns:
            btn.setEnabled(tool in allowed)

        # The checked tool is no longer allowed: fall back to the first allowed one
        if not any(btn.isChecked() and btn.isEnabled() for btn, _ in tool_btns):
            for btn, tool in tool_btns:
                if tool in allowed:
                    btn.setChecked(True)
                    if self.main_window.image_displayed:
                        set_tool(self.main_window, tool)
                    break

        if self.main_window.image_displayed:
            self.main_window.display.set_active_contour_type(contour_type)

    def _on_new_contour(self):
        _, contour_type, _, _ = self._contour_type_items[self.contour_type_combo.currentIndex()]
        new_contour(self.main_window, contour_type)

    def _on_add_contour(self):
        _, contour_type, _, _ = self._contour_type_items[self.contour_type_combo.currentIndex()]
        new_contour_append(self.main_window, contour_type)

    def _on_delete_all(self):
        delete_all_on_frame(self.main_window)

    def _on_delete_all_frames(self):
        delete_active_contour_on_all_frames(self.main_window)

    def _on_knot_count_changed(self, count: int):
        set_selected_contour_knots(self.main_window, count)

    def sync_knot_count(self) -> None:
        """Show the selected contour's knot count (disabled if none)."""
        count = selected_contour_knot_count(self.main_window)
        self.knot_count_box.blockSignals(True)
        self.knot_count_box.setEnabled(count is not None)
        if count is not None:
            self.knot_count_box.setMaximum(max(MAX_KNOTS, count))  # an imported contour may have more
            self.knot_count_box.setValue(count)
        self.knot_count_box.blockSignals(False)

    def _on_angle_type_changed(self, index: int):
        """Aim the angle controls and display at the chosen sector type."""
        contour_type = self._angle_type_items[index][1]
        self._apply_angle_type(contour_type)
        # As with the contour drop-down, the selection *is* the active type, so Delete and
        # Ctrl+Z act on that sector type before one is placed.
        if self.main_window.image_displayed:
            self.main_window.display.set_active_contour_type(contour_type)

    def _on_angle_type_activated(self, index: int):
        """Picking a type from the drop-down starts a sector of it at once: the entry is the action."""
        new_angle(self.main_window, self._angle_type_items[index][1])

    def _on_add_angle(self):
        if self.angle_type is not None:
            new_angle(self.main_window, self.angle_type, True)

    def _apply_angle_type(self, contour_type: ContourType):
        """Label and colour angle controls for `contour_type`."""
        self.angle_type = contour_type
        label, _, new_key, add_key = next(item for item in self._angle_type_items if item[1] == contour_type)
        color = self.main_window.display.contour_color(contour_type)
        new_hint = f' ({new_key})' if new_key else ''
        add_hint = f' ({add_key})' if add_key else ''

        self.angle_type_combo.setStyleSheet(f'border-color: {color}')
        self.angle_type_combo.setToolTip(
            f'Set the angle of one {label.lower()} sector, replacing the existing ones{new_hint}'
        )
        self.add_angle_btn.setText(f'➕📐 Add {label}')
        self.add_angle_btn.setStyleSheet(f'border-color: {color}')
        self.add_angle_btn.setToolTip(f'Add another {label.lower()} sector, keeping the existing ones{add_hint}')

    def set_active_contour_type_ui(self, contour_type: ContourType):
        """Show the display's active type in the control owning it."""
        if active_preset().is_angle(contour_type):
            for i, (_, ct, _, _) in enumerate(self._angle_type_items):
                if ct == contour_type:
                    self.angle_type_combo.setCurrentIndex(i)
                    break
            self._apply_angle_type(contour_type)  # also covers picking the type already shown
            return
        for i, (_, ct, _, _) in enumerate(self._contour_type_items):
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
