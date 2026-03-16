import os
import time
import cv2
import numpy as np

from loguru import logger
from functools import partial
from PyQt6.QtGui import QKeySequence, QDesktopServices, QShortcut, QAction
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QUrl

from gui.popup_windows.frame_range_dialog import FrameRangeDialog
from gui.popup_windows.message_boxes import ErrorMessage, SuccessMessage
from gui.popup_windows.video_player import VideoPlayer
from gui.utils.contours_gui import new_contour, new_contour_append, new_measure, new_angle, set_tool
from input_output.metadata import MetadataWindow
from input_output.read_image import read_image
from input_output.contours_io import write_contours, save_gated_images
from segmentation.save_as_nifti import save_as_nifti
from segmentation.segment import segment
from report.report import report


from gui.popup_windows.results_plot import ResultsPlot
from gui.left_half.display import ContourType, SegmentationTool


def init_shortcuts(main_window):
    # General
    QShortcut(QKeySequence('J'), main_window, partial(jiggle_frame, main_window))
    QShortcut(QKeySequence('Escape'), main_window, partial(stop_all, main_window))
    QShortcut(QKeySequence('Delete'), main_window, partial(delete_contour, main_window))
    QShortcut(QKeySequence('Ctrl+Z'), main_window, partial(undo_delete, main_window))
    # Gating
    QShortcut(QKeySequence('Alt+P'), main_window, partial(plot_results, main_window))
    QShortcut(QKeySequence('Alt+Delete'), main_window, partial(reset_phases, main_window))
    QShortcut(QKeySequence('Alt+S'), main_window, partial(switch_phases, main_window))
    # Traverse frames
    QShortcut(QKeySequence('W'), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence(Qt.Key.Key_Up), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence('A'), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence(Qt.Key.Key_Left), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence('S'), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence(Qt.Key.Key_Down), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence('D'), main_window, main_window.display_slider.next_frame)
    QShortcut(QKeySequence(Qt.Key.Key_Right), main_window, main_window.display_slider.next_frame)
    # Scroll wheel navigates frames
    main_window.display.wheelEvent = lambda event: (
        main_window.display_slider.next_frame()
        if event.angleDelta().y() > 0
        else main_window.display_slider.last_frame()
    )


def init_menu(main_window):
    file_menu = main_window.menu_bar.addMenu('File')
    open_action = file_menu.addAction('Open File', partial(read_image, main_window))
    open_action.setShortcut('Ctrl+O')
    file_menu.addSeparator()
    save_contours = file_menu.addAction('Save Contours', partial(write_contours, main_window))
    save_contours.setShortcut('Ctrl+S')
    nifti_menu = file_menu.addMenu('Save NIfTis')
    nifti_menu.addAction('Contoured Frames', partial(save_as_nifti, main_window, mode='contoured'))
    nifti_menu.addAction('Gated Frames', partial(save_as_nifti, main_window, mode='gated'))
    nifti_menu.addAction('All Frames', partial(save_as_nifti, main_window, mode='all'))
    save_report = file_menu.addAction('Save Report', partial(report, main_window))
    save_report.setShortcut('Ctrl+R')
    file_menu.addAction('Save Video Pullback', partial(save_video_pullback, main_window))
    file_menu.addAction('Save Gated Images', partial(save_gated_images, main_window))
    file_menu.addSeparator()
    exit_action = file_menu.addAction('Exit', main_window.close)
    exit_action.setShortcut('Ctrl+Q')

    edit_menu = main_window.menu_bar.addMenu('Edit')
    manual_lumen_contour = edit_menu.addAction(
        'Manual Lumen Contour', partial(new_contour, main_window, ContourType.LUMEN)
    )
    manual_lumen_contour.setShortcut('E')
    manual_eem_contour = edit_menu.addAction('Manual EEM Contour', partial(new_contour, main_window, ContourType.EEM))
    manual_eem_contour.setShortcut('Q')
    manual_calc_contour = edit_menu.addAction(
        'Manual Calcium Contour', partial(new_contour, main_window, ContourType.CALCIUM)
    )
    manual_calc_contour.setShortcut('7')
    manual_branch_contour = edit_menu.addAction(
        'Manual Branch Contour', partial(new_contour, main_window, ContourType.BRANCH)
    )
    manual_branch_contour.setShortcut('8')
    manual_lipid_contour = edit_menu.addAction(
        'Manual Lipid Contour', partial(new_contour, main_window, ContourType.LIPID)
    )
    manual_lipid_contour.setShortcut('9')
    manual_macroph_contour = edit_menu.addAction(
        'Manual Macrophage Contour', partial(new_contour, main_window, ContourType.MACROPHAGE)
    )
    manual_macroph_contour.setShortcut('0')
    edit_menu.addSeparator()
    add_calc_contour = edit_menu.addAction(
        'Add Calcium Contour', partial(new_contour_append, main_window, ContourType.CALCIUM)
    )
    add_calc_contour.setShortcut('Ctrl+7')
    add_branch_contour = edit_menu.addAction(
        'Add Branch Contour', partial(new_contour_append, main_window, ContourType.BRANCH)
    )
    add_branch_contour.setShortcut('Ctrl+8')
    add_lipid_contour = edit_menu.addAction(
        'Add Lipid Contour', partial(new_contour_append, main_window, ContourType.LIPID)
    )
    add_lipid_contour.setShortcut('Ctrl+9')
    add_macroph_contour = edit_menu.addAction(
        'Add Macrophage Contour', partial(new_contour_append, main_window, ContourType.MACROPHAGE)
    )
    add_macroph_contour.setShortcut('Ctrl+0')
    edit_menu.addAction('Remove Contours', partial(remove_contours, main_window))
    edit_menu.addSeparator()
    edit_menu.addAction('Reset Phases', partial(reset_phases, main_window))
    edit_menu.addSeparator()
    measure_1 = edit_menu.addAction('Measurement 1', partial(new_measure, main_window, index=0))
    measure_1.setShortcut('1')
    measure_2 = edit_menu.addAction('Measurement 2', partial(new_measure, main_window, index=1))
    measure_2.setShortcut('2')
    angle_wire = edit_menu.addAction('Angle Wire Shadow', partial(new_angle, main_window, ContourType.WIRE))
    angle_wire.setShortcut('3')
    closed_spline = edit_menu.addAction('Closed Spline', partial(set_tool, main_window, SegmentationTool.CLOSED_SPLINE))
    closed_spline.setShortcut('4')
    open_spline = edit_menu.addAction('Open Spline', partial(set_tool, main_window, SegmentationTool.OPEN_SPLINE))
    open_spline.setShortcut('5')
    brush = edit_menu.addAction('Brush', partial(set_tool, main_window, SegmentationTool.BRUSH))
    brush.setShortcut('6')

    view_menu = main_window.menu_bar.addMenu('View')
    hide_contours_action = view_menu.addAction('Hide Contours', partial(hide_contours, main_window))
    hide_contours_action.setShortcut('H')
    hide_special_points_action = view_menu.addAction('Hide Measurements', partial(hide_special_points, main_window))
    hide_special_points_action.setShortcut('G')
    view_menu.addSeparator()
    reset_windowing_action = view_menu.addAction('Reset Windowing', partial(reset_windowing, main_window))
    reset_windowing_action.setShortcut('R')
    reset_zoom_action = view_menu.addAction('Reset Zoom', partial(reset_zoom, main_window))
    reset_zoom_action.setShortcut('F')
    toggle_color_action = view_menu.addAction('Toggle Color', partial(toggle_color, main_window))
    toggle_color_action.setShortcut('C')
    view_menu.addSeparator()

    run_menu = main_window.menu_bar.addMenu('Run')
    run_menu.addAction('Extract Diastolic and Systolic Frames', main_window.contour_based_gating)
    # run_menu.addAction('Automatic Segmentation', partial(segment, main_window))

    metadata_menu = main_window.menu_bar.addMenu('Metadata')
    metadata_menu.addAction('Show Metadata', partial(show_metadata, main_window))

    help_menu = main_window.menu_bar.addMenu('Help')
    help_menu.addAction('GitHub Page', partial(open_url, main_window, description='github'))
    help_menu.addAction('Keyboard Shortcuts', partial(open_url, main_window, description='keyboard_shortcuts'))
    help_menu.addAction('Report a Problem', partial(open_url, main_window, description='issue'))
    help_menu.addAction('Request a Feature', partial(open_url, main_window, description='feature'))
    help_menu.addSeparator()
    help_menu.addAction('About', partial(open_url, main_window))


def is_gating_display_active(main_window):
    """
    Checks if an image is displayed in the gating display box.

    Parameters:
        main_window: The main window containing the gating display.

    Returns:
        bool: True if the gating display contains an image, False otherwise.
    """
    return (
        main_window.gating_display is not None
        and main_window.gating_display.fig.axes  # Check if axes exist
        and any(ax.has_data() for ax in main_window.gating_display.fig.axes)  # Check if any axis has data
    )


def remove_contours(main_window):
    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if dialog.exec():
            main_window.status_bar.showMessage('Removing contours...')
            lower_limit, upper_limit = dialog.getInputs()
            key = main_window.display.contour_key()
            for frame in range(lower_limit, upper_limit):
                fd = main_window.data.get(frame)
                if fd:
                    contour_obj = getattr(fd, key, None)
                    if contour_obj:
                        contour_obj.contours = []
            main_window.longitudinal_view.remove_contours(lower_limit, upper_limit)
            main_window.display.update_display()
            main_window.status_bar.showMessage(main_window.waiting_status)


def reset_phases(main_window):
    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if dialog.exec():
            main_window.status_bar.showMessage('Resetting phases...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                fd = main_window.data.get(frame)
                if fd is None:
                    continue
                if fd.phase == 'D':
                    main_window.gated_frames_dia.remove(frame)
                    main_window.diastolic_frame_box.setChecked(False)
                elif fd.phase == 'S':
                    main_window.gated_frames_sys.remove(frame)
                    main_window.systolic_frame_box.setChecked(False)
                elif fd.phase == 'T':
                    try:
                        main_window.gated_frames_oct.remove(frame)
                    except ValueError:
                        pass
                fd.phase = '-'
            if main_window.metadata.get('modality') == 'OCT':
                main_window.gated_frames = main_window.gated_frames_oct
            else:
                main_window.gated_frames = main_window.gated_frames_dia + main_window.gated_frames_sys
                main_window.gated_frames.sort()
            main_window.gated_frames_dia.sort()
            main_window.gated_frames_sys.sort()
            main_window.status_bar.showMessage(main_window.waiting_status)

            main_window.contour_based_gating.remove_lines()
            main_window.contour_based_gating.draw_existing_lines(
                main_window.gated_frames_dia, main_window.diastole_color_plt
            )
            main_window.contour_based_gating.draw_existing_lines(
                main_window.gated_frames_sys, main_window.systole_color_plt
            )  # somehow only updates after first user input

            main_window.display.update_display()


def switch_phases(main_window):
    # Check if gating display is active; if not, show a message and return
    if not is_gating_display_active(main_window):
        ErrorMessage(main_window, 'Please extract diastolic and systolic frames first.')
        return

    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if dialog.exec():
            main_window.status_bar.showMessage('Switching phases...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                fd = main_window.data.get(frame)
                if fd is None:
                    continue
                if fd.phase == 'D':
                    fd.phase = 'S'
                    main_window.gated_frames_dia.remove(frame)
                    main_window.gated_frames_sys.append(frame)
                    main_window.diastolic_frame_box.setChecked(False)
                    main_window.systolic_frame_box.setChecked(True)
                elif fd.phase == 'S':
                    fd.phase = 'D'
                    main_window.gated_frames_sys.remove(frame)
                    main_window.gated_frames_dia.append(frame)
                    main_window.diastolic_frame_box.setChecked(True)
                    main_window.systolic_frame_box.setChecked(False)

            main_window.gated_frames = main_window.gated_frames_dia + main_window.gated_frames_sys

        # order all gated frames again, important otherwise slider will jump around
        main_window.gated_frames.sort()
        main_window.gated_frames_dia.sort()
        main_window.gated_frames_sys.sort()
        main_window.status_bar.showMessage(main_window.waiting_status)

        # Call draw_existing_lines on the ContourBasedGating instance, but first remove all existing lines to live update plot
        main_window.contour_based_gating.remove_lines()
        main_window.contour_based_gating.draw_existing_lines(
            main_window.gated_frames_dia, main_window.diastole_color_plt
        )
        main_window.contour_based_gating.draw_existing_lines(
            main_window.gated_frames_sys, main_window.systole_color_plt
        )

        main_window.display.update_display()


def show_metadata(main_window):
    if main_window.image_displayed:
        metadata_window = MetadataWindow(main_window)
        metadata_window.show()


def open_url(main_window, description=None):
    if description == 'github':
        url = 'https://github.com/yungselm/AIVUS-OCT'
    elif description == 'keyboard_shortcuts':
        url = 'https://github.com/yungselm/AIVUS-OCT?tab=readme-ov-file#keyboard-shortcuts'
    elif description == 'issue':
        url = 'https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/issues/new?template=bug_report.md'
    elif description == 'feature':
        url = 'https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/issues/new?template=feature_request.md'
    else:
        video_player = VideoPlayer(main_window)
        video_player.play('../media/about.mp4')
        video_player.move(main_window.x() + main_window.width() // 2, main_window.y() + main_window.height() // 2)
        return
    if not QDesktopServices.openUrl(QUrl(url)):
        ErrorMessage(main_window, 'Could not open the browser. Please visit\n' + url)


def hide_contours(main_window):
    if main_window.image_displayed:
        main_window.hide_contours_box.setChecked(not main_window.hide_contours_box.isChecked())


def hide_special_points(main_window):
    if main_window.image_displayed:
        main_window.hide_special_points_box.setChecked(not main_window.hide_special_points_box.isChecked())


def jiggle_frame(main_window):
    if main_window.image_displayed:
        current_frame = main_window.display_slider.value()
        main_window.display_slider.set_value(current_frame + 1)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.set_value(current_frame)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.set_value(current_frame - 1)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.set_value(current_frame)
        QApplication.processEvents()


def stop_all(main_window):
    main_window.display.stop_contour()
    main_window.display.measure_index = None


def delete_contour(main_window):
    if main_window.image_displayed:
        key = main_window.display.contour_key()

        if not hasattr(main_window, 'tmp_contours'):
            main_window.tmp_contours = {}

        frame = main_window.display.frame
        fd = main_window.data.get(frame)
        if fd:
            contour_obj = getattr(fd, key, None)
            if contour_obj and contour_obj.contours and contour_obj.contours[0]:
                xlist = list(contour_obj.contours[0][0]) if contour_obj.contours[0][0] else []
                ylist = list(contour_obj.contours[0][1]) if len(contour_obj.contours[0]) > 1 else []
            else:
                xlist, ylist = [], []
            main_window.tmp_contours[key] = (xlist, ylist)
            if contour_obj:
                contour_obj.contours = []

        main_window.display.display_image(update_contours=True)


def undo_delete(main_window):
    if main_window.image_displayed:
        key = main_window.display.contour_key()
        if hasattr(main_window, 'tmp_contours') and key in main_window.tmp_contours:
            xlist, ylist = main_window.tmp_contours.pop(key)
            frame = main_window.display.frame
            fd = main_window.data.get(frame)
            if fd:
                contour_obj = getattr(fd, key, None)
                if contour_obj is not None:
                    contour_obj.contours = [[xlist, ylist]]

            main_window.display.update_display()


def reset_zoom(main_window):
    if main_window.image_displayed:
        main_window.display.resetTransform()


def reset_windowing(main_window):
    if main_window.image_displayed:
        main_window.display.window_level = main_window.display.initial_window_level
        main_window.display.window_width = main_window.display.initial_window_width
        main_window.display.display_image(update_image=True)


def toggle_color(main_window):
    if main_window.image_displayed:
        main_window.colormap_enabled = not main_window.colormap_enabled
        main_window.display.display_image(update_image=True)


def plot_results(main_window):
    if main_window.image_displayed:
        report_data = report(main_window, suppress_messages=True)
        if report_data is None:
            logger.error('No report data available to plot')
            return
        results_plot = ResultsPlot(main_window, report_data)
        results_plot.show()


def save_video_pullback(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save video pullback before reading the image.')
        return
    main_window.status_bar.showMessage('Saving video pullback...')
    image_stack = main_window.images
    size = (image_stack[0].shape[1], image_stack[0].shape[0])
    fps = main_window.metadata['frame_rate']
    duration = len(image_stack) // fps
    out_path = os.path.splitext(main_window.file_name)[0] + '_pullback.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for frame in range(fps * duration):
        out.write(image_stack[frame, :, :])
    out.release()
    SuccessMessage(main_window, f'Saving video')
    main_window.status_bar.showMessage(main_window.waiting_status)
