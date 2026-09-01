"""Right half shown for IVUS pullbacks.

IVUS pullbacks are cardiac-gated, so this half is built around that: marking
diastolic/systolic frames at the top, the gating plot above the longitudinal view,
and gating extraction at the bottom.  Its OCT counterpart is right_half_oct.py; the
shell that shows one or the other is right_half.py.
"""

import bisect
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pages.intravascular.right_half.common import (
    LongitudinalSlot,
    build_lower_buttons,
    open_small_display,
)
from pages.intravascular.right_half.gating_display import GatingDisplay


class RightHalfIvus(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        mw = main_window

        self.diastolic_frame_box = QCheckBox('Diastolic Frame')
        self.diastolic_frame_box.setChecked(False)
        self.diastolic_frame_box.stateChanged.connect(partial(toggle_diastolic_frame, mw))
        self.systolic_frame_box = QCheckBox('Systolic Frame')
        self.systolic_frame_box.setChecked(False)
        self.systolic_frame_box.stateChanged.connect(partial(toggle_systolic_frame, mw))
        self.use_diastolic_button = QPushButton('Diastolic Frames')
        self.use_diastolic_button.setStyleSheet(f'background-color: rgb{mw.diastole_color}')
        self.use_diastolic_button.setCheckable(True)
        self.use_diastolic_button.setChecked(True)
        self.use_diastolic_button.clicked.connect(partial(use_diastolic, mw))
        self.use_diastolic_button.setToolTip('Press button to switch between diastolic and systolic frames')

        self.gating_display = GatingDisplay(mw)

        vbox = QVBoxLayout(self)

        checkboxes = QHBoxLayout()
        checkboxes.addWidget(self.diastolic_frame_box)
        checkboxes.addWidget(self.systolic_frame_box)
        checkboxes.addWidget(self.use_diastolic_button)
        compare_btn = QPushButton('Compare Frames')
        compare_btn.setToolTip('Open a small display to compare two frames')
        compare_btn.clicked.connect(partial(open_small_display, mw))
        checkboxes.addWidget(compare_btn)
        checkboxes.addWidget(self.gating_display.toolbar)
        vbox.addLayout(checkboxes)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.gating_display)
        self.splitter.addWidget(self._build_longitudinal_pane())
        self.splitter.setStretchFactor(0, mw.config.intravascular.gating_display_stretch)
        self.splitter.setStretchFactor(1, mw.config.intravascular.lview_display_stretch)
        vbox.addWidget(self.splitter)

        extract_button = QPushButton('Extract Diastolic and Systolic Frames')
        extract_button.setToolTip('Extract diastolic and systolic images from pullback')
        extract_button.clicked.connect(mw.gating_plot)
        vbox.addLayout(build_lower_buttons(mw, extract_button))

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_longitudinal_pane(self):
        """Longitudinal view with a Raw/Filtered mode selector column on its left.

        'Raw'      → images in acquisition order (default).
        'Filtered' → gated frames reordered into breathing-corrected anatomical
                     order (en bloc per cardiac cycle); the left-half slider then
                     scrolls gated frames in this sorted order.
        """
        mw = self.main_window
        pane = QWidget()
        hbox = QHBoxLayout(pane)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(2)

        btn_col = QVBoxLayout()
        btn_col.setContentsMargins(2, 2, 2, 2)

        self.raw_btn = QPushButton('Raw')
        self.raw_btn.setCheckable(True)
        self.raw_btn.setChecked(True)
        self.raw_btn.setFixedWidth(80)
        self.raw_btn.setToolTip('Show frames in acquisition (pullback) order')

        self.filtered_btn = QPushButton('Filtered')
        self.filtered_btn.setCheckable(True)
        self.filtered_btn.setFixedWidth(80)
        self.filtered_btn.setToolTip('Reorder gated frames into breathing-corrected order (en bloc per cardiac cycle)')

        group = QButtonGroup(pane)
        group.setExclusive(True)
        group.addButton(self.raw_btn)
        group.addButton(self.filtered_btn)
        self._lview_mode_group = group  # keep reference

        self.raw_btn.clicked.connect(partial(set_longitudinal_mode, mw, 'raw'))
        self.filtered_btn.clicked.connect(partial(set_longitudinal_mode, mw, 'filtered'))

        hide_label = QLabel('Hide')

        self.hide_phase_lines_cb = QCheckBox('Dia/Sys Lines')
        self.hide_phase_lines_cb.setToolTip(
            'Clear the diastolic/systolic frame marker lines from the longitudinal view'
        )
        self.hide_phase_lines_cb.toggled.connect(partial(toggle_phase_lines_visible, mw))

        self.hide_breathing_cb = QCheckBox('Breathing')
        self.hide_breathing_cb.setToolTip(
            'Clear the breathing curve and peak/valley markers from the longitudinal view'
        )
        self.hide_breathing_cb.toggled.connect(partial(toggle_breathing_visible, mw))

        self.hide_areas_cb = QCheckBox('Areas')
        self.hide_areas_cb.setToolTip('Clear the lumen-area dots from the longitudinal view')
        self.hide_areas_cb.toggled.connect(partial(toggle_areas_visible, mw))

        btn_col.addWidget(self.raw_btn)
        btn_col.addWidget(self.filtered_btn)
        btn_col.addWidget(hide_label)
        btn_col.addWidget(self.hide_phase_lines_cb)
        btn_col.addWidget(self.hide_breathing_cb)
        btn_col.addWidget(self.hide_areas_cb)
        btn_col.addStretch(1)

        self.lview_slot = LongitudinalSlot()

        hbox.addLayout(btn_col)
        hbox.addWidget(self.lview_slot, stretch=1)
        return pane

    # ------------------------------------------------------------------
    # Shell API
    # ------------------------------------------------------------------

    def activate(self):
        self.lview_slot.attach(self.main_window.longitudinal_view)
        # Re-applied on every activation so a newly loaded pullback starts with the
        # gating plot and the longitudinal view sharing the pane, whatever the handle
        # was dragged to for the previous one.
        gating_size = self.gating_display.sizeHint().height()
        self.splitter.setSizes([gating_size, gating_size])

    def deactivate(self):
        self.lview_slot.detach(self.main_window.longitudinal_view)


# ---------------------------------------------------------------------------
# Gating callbacks
# ---------------------------------------------------------------------------


def toggle_diastolic_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            main_window.use_diastolic_button.setChecked(True)
            use_diastolic(main_window)
            if frame not in main_window.runtime_data.gated_frames_dia:
                bisect.insort_left(main_window.runtime_data.gated_frames_dia, frame)
                main_window.runtime_data.frame_data_dct[frame].phase = 'D'
                main_window.gating_plot.update_color(main_window.diastole_color_plt)
                main_window.gating_plot.current_phase = 'D'
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.systolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.runtime_data.gated_frames_dia.remove(frame)
                main_window.gating_plot.current_phase = None
                if (
                    main_window.runtime_data.frame_data_dct[frame].phase == 'D'
                ):  # do not reset when function is called from toggle_systolic_frame
                    main_window.runtime_data.frame_data_dct[frame].phase = '-'
                    if not drag:
                        main_window.gating_plot.update_color()
            except ValueError:
                pass

        main_window.display.update_display()


def toggle_systolic_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            main_window.use_diastolic_button.setChecked(False)
            use_diastolic(main_window)
            if frame not in main_window.runtime_data.gated_frames_sys:
                bisect.insort_left(main_window.runtime_data.gated_frames_sys, frame)
                main_window.runtime_data.frame_data_dct[frame].phase = 'S'
                main_window.gating_plot.update_color(main_window.systole_color_plt)
                main_window.gating_plot.current_phase = 'S'
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.diastolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.runtime_data.gated_frames_sys.remove(frame)
                main_window.gating_plot.current_phase = None
                if (
                    main_window.runtime_data.frame_data_dct[frame].phase == 'S'
                ):  # do not reset when function is called from toggle_diastolic_frame
                    main_window.runtime_data.frame_data_dct[frame].phase = '-'
                    if not drag:
                        main_window.gating_plot.update_color()
            except ValueError:
                pass

        main_window.display.update_display()


def use_diastolic(main_window):
    if main_window.image_displayed:
        if main_window.use_diastolic_button.isChecked():
            main_window.use_diastolic_button.setText('Diastolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.diastole_color}')
            main_window.runtime_data.gated_frames = main_window.runtime_data.gated_frames_dia
        else:
            main_window.use_diastolic_button.setText('Systolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.systole_color}')
            main_window.runtime_data.gated_frames = main_window.runtime_data.gated_frames_sys

        try:
            next_gated = main_window.display_slider.next_gated_frame(set=False)
            main_window.small_display.update_frame(
                next_gated, update_image=True, update_contours=True, update_text=True
            )  # update small display
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Longitudinal view column
# ---------------------------------------------------------------------------


def set_longitudinal_mode(main_window, mode):
    """Raw/Filtered toggle.

    'Filtered' opens the breathing-sorted paired viewer (diastole | systole in
    breathing-corrected order). 'Raw' just clears the status hint.
    """
    if not main_window.image_displayed:
        return
    if mode == 'filtered':
        from pages.intravascular.popup_windows.breathing_sort_viewer import (
            BreathingSortViewer,
        )

        main_window.breathing_sort_viewer = BreathingSortViewer(main_window)
        main_window.breathing_sort_viewer.show()
    else:
        main_window.status_bar.showMessage(main_window.waiting_status)


def toggle_phase_lines_visible(main_window, checked):
    main_window.longitudinal_view.set_phase_lines_visible(not checked)


def toggle_breathing_visible(main_window, checked):
    main_window.longitudinal_view.set_breathing_visible(not checked)


def toggle_areas_visible(main_window, checked):
    if checked:
        main_window.longitudinal_view.hide_lview_contours()
    else:
        main_window.longitudinal_view.show_lview_contours()
