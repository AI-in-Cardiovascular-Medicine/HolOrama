import warnings
import numpy as np
from matplotlib.backend_bases import MouseButton

from gating.signal_processing import prepare_data
from pages.intravascular.utils.helpers import connect_consecutive_frames
from gating.automatic_gating import AutomaticGating
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from pages.intravascular.popup_windows.frame_range_dialog import FrameRangeDialog
from pages.intravascular.right_half.right_half import toggle_diastolic_frame, toggle_systolic_frame
from input_output.output.reports import report


class ContourBasedGating:
    def __init__(self, main_window):
        self.main_window = main_window
        self.vertical_lines = []
        self.selected_line = None
        self.current_phase = None
        self.tmp_phase = None
        self.frame_marker = None
        self.default_line_color = 'grey'
        self.default_linestyle = (0, (1, 3))

    def _draw(self):
        """Redraw the embedded gating canvas without touching any other figure."""
        try:
            self.fig.canvas.draw_idle()
        except AttributeError:
            pass

    @property
    def _ready(self) -> bool:
        """True once plot_data() has been called and axes exist."""
        return hasattr(self, 'ax') and hasattr(self, 'fig')

    def _toolbar_active(self) -> bool:
        """Return True when the matplotlib toolbar has zoom/pan mode active.

        Replaces the old cursor().shape() != 0 check, which broke in PyQt6
        because Qt.CursorShape enums are strict Python enums and never compare
        equal to bare integers.
        """
        toolbar = getattr(self.main_window.gating_display, 'toolbar', None)
        return toolbar is not None and bool(toolbar.mode)

    def __call__(self):
        self.main_window.status_bar.showMessage('Contour-based gating...')
        dialog_success = self.define_roi()
        if not dialog_success:
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return
        (
            image_based_gating,
            contour_based_gating,
            image_based_gating_filtered,
            contour_based_gating_filtered,
        ) = prepare_data(self.main_window, self.frames, self.report_data)
        self.plot_data(
            image_based_gating, contour_based_gating, image_based_gating_filtered, contour_based_gating_filtered
        )
        self.main_window.status_bar.showMessage(self.main_window.waiting_status)

    def define_roi(self):
        dialog = FrameRangeDialog(self.main_window)
        if dialog.exec():
            lower_limit, upper_limit = dialog.getInputs()
            self.report_data = report(self.main_window, lower_limit, upper_limit, suppress_messages=True)
            if self.report_data is None:
                ErrorMessage(self.main_window, 'Please ensure that an input file was read and contours were drawn')
                self.main_window.status_bar.showMessage(self.main_window.waiting_status)
                return False

            if len(self.report_data) != upper_limit - lower_limit:
                missing_frames = [
                    frame
                    for frame in range(lower_limit + 1, upper_limit + 1)
                    if frame not in self.report_data['frame'].values
                ]
                str_missing = connect_consecutive_frames(missing_frames)
                ErrorMessage(self.main_window, f'Please add contours to frames {str_missing}')
                return False
            self.frames = self.main_window.runtime_data.images[lower_limit:upper_limit]
            self.x = self.report_data['frame'].values
            return True
        return False

    def plot_data(
        self, image_based_gating, contour_based_gating, image_based_gating_filtered, contour_based_gating_filtered
    ):
        min_signal_range = min(np.min(image_based_gating), np.min(contour_based_gating))

        shift_amount = min_signal_range - np.max(image_based_gating)
        image_based_gating += shift_amount

        shift_amount = min_signal_range - np.max(contour_based_gating)
        contour_based_gating += shift_amount

        self.fig = self.main_window.gating_display.fig
        self.fig.clear()
        self.ax = self.fig.add_subplot()

        self.ax.plot(self.x, image_based_gating_filtered, color='green', label='Image based gating')
        self.ax.plot(self.x, contour_based_gating_filtered, color='yellow', label='Contour based gating')
        self.ax.plot(
            self.x, image_based_gating, color='green', linestyle='dashed', label='Image based gating (unfiltered)'
        )
        self.ax.plot(
            self.x, contour_based_gating, color='yellow', linestyle='dashed', label='Contour based gating (unfiltered)'
        )

        self.ax.set_xlabel('Frame')
        self.ax.get_yaxis().set_visible(False)
        legend = self.ax.legend(ncol=2, loc='lower right')
        legend.set_draggable(True)

        # Connect mouse events to the embedded canvas only — not to pyplot global state
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.fig.tight_layout()

        self._draw()

        self.draw_existing_lines(self.main_window.runtime_data.gated_frames_dia, self.main_window.diastole_color_plt)
        self.draw_existing_lines(self.main_window.runtime_data.gated_frames_sys, self.main_window.systole_color_plt)

        if not self.main_window.runtime_data.gated_frames_dia and not self.main_window.runtime_data.gated_frames_sys:
            auto_gating = AutomaticGating(self.main_window, self.report_data)
            auto_gating.automatic_gating(image_based_gating_filtered, contour_based_gating_filtered)
            self.draw_existing_lines(
                self.main_window.runtime_data.gated_frames_dia, self.main_window.diastole_color_plt
            )
            self.draw_existing_lines(self.main_window.runtime_data.gated_frames_sys, self.main_window.systole_color_plt)
            self._draw()

        return True

    def on_click(self, event):
        if self._toolbar_active():
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            new_line = True
            set_dia = False
            set_sys = False
            set_slider_to = event.xdata
            if self.selected_line is not None:
                self.selected_line.set_linestyle(self.default_linestyle)
                self.selected_line = None
            if self.vertical_lines:
                distances = [abs(line.get_xdata()[0] - event.xdata) for line in self.vertical_lines]
                if min(distances) < len(self.frames) / 100:
                    self.selected_line = self.vertical_lines[np.argmin(distances)]
                    new_line = False
                    set_slider_to = self.selected_line.get_xdata()[0]
            if new_line:
                if self.current_phase == 'D':
                    color = self.main_window.diastole_color_plt
                    set_dia = True
                elif self.current_phase == 'S':
                    color = self.main_window.systole_color_plt
                    set_sys = True
                else:
                    color = self.default_line_color
                self.selected_line = self.ax.axvline(x=event.xdata, color=color, linestyle=self.default_linestyle)
                self.vertical_lines.append(self.selected_line)

            assert self.selected_line is not None
            self.selected_line.set_linestyle('dashed')
            self._draw()

            set_slider_to = round(set_slider_to - 1)
            self.main_window.display_slider.set_value(set_slider_to, reset_highlights=False)

            if set_slider_to in self.main_window.runtime_data.gated_frames_dia or set_dia:
                self.tmp_phase = 'D'
                toggle_diastolic_frame(self.main_window, False, drag=True)
            elif set_slider_to in self.main_window.runtime_data.gated_frames_sys or set_sys:
                self.tmp_phase = 'S'
                toggle_systolic_frame(self.main_window, False, drag=True)

    def on_release(self, event):
        if self._toolbar_active():
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            if self.tmp_phase == 'D':
                self.main_window.diastolic_frame_box.setChecked(True)
                toggle_diastolic_frame(self.main_window, True, drag=True)
            elif self.tmp_phase == 'S':
                self.main_window.systolic_frame_box.setChecked(True)
                toggle_systolic_frame(self.main_window, True, drag=True)
        self.tmp_phase = None

    def on_motion(self, event):
        if self._toolbar_active():
            return
        if event.button is MouseButton.LEFT and self.selected_line:
            self.selected_line.set_xdata(np.array([event.xdata]))
            if event.xdata is not None:
                self.main_window.display_slider.set_value(round(event.xdata - 1), reset_highlights=False)
                self._draw()
            else:
                self.vertical_lines.remove(self.selected_line)
                self.selected_line = None
                self.tmp_phase = None
                self._draw()

    def set_frame(self, frame):
        if not self._ready:
            return
        self.ax.set_autoscale_on(False)
        if self.frame_marker:
            self.frame_marker[0].remove()
        self.frame_marker = self.ax.plot(frame + 1, self.ax.get_ylim()[0], 'yo', clip_on=False)
        self._draw()

    def draw_existing_lines(self, frames, color):
        if not self._ready:
            return
        frames = [frame for frame in frames if frame in (self.x - 1)]
        for frame in frames:
            self.vertical_lines.append(self.ax.axvline(x=frame + 1, color=color, linestyle=self.default_linestyle))

    def remove_lines(self):
        for line in self.vertical_lines:
            line.remove()
        self.vertical_lines = []
        self._draw()

    def redraw_phase_lines(self, frames_dia, color_dia, frames_sys, color_sys):
        """Remove all phase lines, redraw them, and flush the canvas in one step."""
        self.remove_lines()
        self.draw_existing_lines(frames_dia, color_dia)
        self.draw_existing_lines(frames_sys, color_sys)
        self._draw()

    def update_color(self, color=None):
        color = color or self.default_line_color
        if self.selected_line is not None:
            self.selected_line.set_color(color)
            self._draw()

    def reset_highlights(self):
        if self.selected_line is not None:
            self.selected_line.set_linestyle(self.default_linestyle)
            self.selected_line = None
            self._draw()
