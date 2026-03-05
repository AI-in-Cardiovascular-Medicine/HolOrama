import warnings
import time
import itertools
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from loguru import logger

from gating.signal_processing import *
from gui.utils.helpers import connect_consecutive_frames
from gating.automatic_gating import AutomaticGating
from gui.popup_windows.message_boxes import ErrorMessage
from gui.popup_windows.frame_range_dialog import FrameRangeDialog, StartFramesDialog
from gui.right_half.right_half import toggle_diastolic_frame, toggle_systolic_frame
from report.report import report


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class ContourBasedGating:
    def __init__(self, main_window):
        self.main_window = main_window
        # signals
        self.vertical_lines = []
        self.selected_line = None
        self.current_phase = None
        self.tmp_phase = None
        self.frame_marker = None
        self.default_line_color = 'grey'
        self.default_linestyle = (0, (1, 3))

    def __call__(self):
        self.main_window.status_bar.showMessage('Contour-based gating...')
        dialog_success = self.define_roi()
        if not dialog_success:
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return
        image_based_gating, contour_based_gating, image_based_gating_filtered, contour_based_gating_filtered = (
            prepare_data(self.main_window, self.frames, self.report_data)
        )
        self.plot_data(
            image_based_gating, contour_based_gating, image_based_gating_filtered, contour_based_gating_filtered
        )
        self.main_window.status_bar.showMessage(self.main_window.waiting_status)

    def define_roi(self):
        dialog = FrameRangeDialog(self.main_window)
        if dialog.exec():
            lower_limit, upper_limit = dialog.getInputs()
            self.report_data = report(
                self.main_window, lower_limit, upper_limit, suppress_messages=True
            )  # compute all needed data
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
            self.frames = self.main_window.images[lower_limit:upper_limit]
            self.x = self.report_data['frame'].values  # want 1-based indexing for GUI
            return True
        return False

    def plot_data(
        self, image_based_gating, contour_based_gating, image_based_gating_filtered, contour_based_gating_filtered
    ):
        # Scale `_nor` signals to the same range
        min_signal_range = min(np.min(image_based_gating), np.min(contour_based_gating))
        max_signal_range = max(np.max(image_based_gating), np.max(contour_based_gating))

        # Shift `unfiltered` signals down so their max aligns with the min of the main signals
        shift_amount = min_signal_range - np.max(image_based_gating)
        image_based_gating += shift_amount

        shift_amount = min_signal_range - np.max(contour_based_gating)
        contour_based_gating += shift_amount

        # Plotting
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

        # Interactive event connections
        plt.connect('button_press_event', self.on_click)
        plt.connect('motion_notify_event', self.on_motion)
        plt.connect('button_release_event', self.on_release)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            plt.tight_layout()

            if any("tight_layout" in str(w.message) for w in caught_warnings):
                plt.draw()
            else:
                plt.draw()

        # Draw any existing lines first
        self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
        self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)
        
        # Only run automatic gating if no frames are already gated
        if not self.main_window.gated_frames_dia and not self.main_window.gated_frames_sys:
            # Show method selection dialog after plot is rendered
            auto_gating = AutomaticGating(self.main_window, self.report_data)
            auto_gating.automatic_gating(image_based_gating_filtered, contour_based_gating_filtered)
            
            # Redraw lines with new automatic gating results
            self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
            self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)
            plt.draw()

        return True

    def on_click(self, event):
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
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
                # Check if click is near any existing line
                distances = [abs(line.get_xdata()[0] - event.xdata) for line in self.vertical_lines]
                if min(distances) < len(self.frames) / 100:  # sensitivity for line selection
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
                self.selected_line = plt.axvline(x=event.xdata, color=color, linestyle=self.default_linestyle)
                self.vertical_lines.append(self.selected_line)

            self.selected_line.set_linestyle('dashed')
            plt.draw()

            set_slider_to = round(set_slider_to - 1)  # slider is 0-based
            self.main_window.display_slider.set_value(set_slider_to, reset_highlights=False)

            if set_slider_to in self.main_window.gated_frames_dia or set_dia:
                self.tmp_phase = 'D'
                toggle_diastolic_frame(self.main_window, False, drag=True)
            elif set_slider_to in self.main_window.gated_frames_sys or set_sys:
                self.tmp_phase = 'S'
                toggle_systolic_frame(self.main_window, False, drag=True)

    def on_release(self, event):
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
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
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
            return
        if event.button is MouseButton.LEFT and self.selected_line:
            self.selected_line.set_xdata(np.array([event.xdata]))
            if event.xdata is not None:
                self.main_window.display_slider.set_value(
                    round(event.xdata - 1), reset_highlights=False
                )  # slider is 0-based
                plt.draw()
            else:
                self.vertical_lines.remove(self.selected_line)
                self.selected_line = None
                self.tmp_phase = None
                plt.draw()

    def set_frame(self, frame):
        plt.autoscale(False)
        if self.frame_marker:
            self.frame_marker[0].remove()
        self.frame_marker = self.ax.plot(frame + 1, self.ax.get_ylim()[0], 'yo', clip_on=False)
        plt.draw()

    def draw_existing_lines(self, frames, color):
        frames = [frame for frame in frames if frame in (self.x - 1)]  # remove frames outside of user-defined range
        for frame in frames:
            self.vertical_lines.append(plt.axvline(x=frame + 1, color=color, linestyle=self.default_linestyle))

    def remove_lines(self):
        for line in self.vertical_lines:
            line.remove()
        self.vertical_lines = []
        plt.draw

    def update_color(self, color=None):
        color = color or self.default_line_color
        if self.selected_line is not None:
            self.selected_line.set_color(color)
            plt.draw()

    def reset_highlights(self):
        if self.selected_line is not None:
            self.selected_line.set_linestyle(self.default_linestyle)
            self.selected_line = None
            plt.draw()
