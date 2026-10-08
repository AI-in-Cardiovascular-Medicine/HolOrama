from loguru import logger
from PyQt6.QtWidgets import QMessageBox

from domain.intravascular.types import ContourType, SegmentationTool
from domain.intravascular.contour_presets import active_preset
from domain.intravascular.io_types import Contour, clear_frame_annotations, is_contour_key
from domain.intravascular.knot_resampling import KnotHistory, without_closing_repeat
from domain.intravascular.undo import (
    push_contour_snapshot,
    push_frame_annotation_snapshot,
    push_pullback_contours_snapshot,
)
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from pages.intravascular.utils.metrics import clear_lumen_measurements


def delete_all_on_frame(main_window):
    """Clear every contour, measurement, reference and wire angle on the current frame.

    One undo entry restores the whole frame. Phase and OCT label describe the frame, not the
    drawing on it, so they stay.
    """
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot delete contours before reading input file')
        return

    frame = main_window.display.frame
    frame_data = main_window.runtime_data.frame_data_dct.get(frame)
    if frame_data is None:
        return

    push_frame_annotation_snapshot(main_window.runtime_data, frame)
    clear_frame_annotations(frame_data)

    main_window.display.working_spline = None
    main_window.display.active_contour_index = 0
    main_window.save_contours_soon()
    main_window.display.update_display()
    try:  # the pullback overviews read these contours
        main_window.longitudinal_view.plot_areas()
    except Exception as exc:
        logger.debug(f'Could not refresh the pullback overviews after Delete All: {exc}')


def delete_active_contour_on_all_frames(main_window):
    """Clear the active contour type on every frame of the pullback.

    All affected frames share one undo entry.
    """
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot delete contours before reading input file')
        return

    display = main_window.display
    key = display.contour_key()
    if not is_contour_key(key):
        ErrorMessage(main_window, 'Select a contour type to delete it on all frames')
        return

    frame_data_dct = main_window.runtime_data.frame_data_dct or {}
    frames = [index for index, fd in frame_data_dct.items() if fd.contours.get(key) and fd.contours[key].contours]
    defn = active_preset().get(display.active_contour_type)
    name = defn.name if defn else key
    if not frames:
        main_window.status_bar.showMessage(f'No {name} contours to delete')
        return

    reply = QMessageBox.question(
        main_window,
        'Delete On All Frames',
        f'Delete the {name} contour on all {len(frames)} frames that have one?\n(Ctrl+Z undoes it)',
    )
    if reply != QMessageBox.StandardButton.Yes:
        return

    push_pullback_contours_snapshot(main_window.runtime_data, display.frame, frames)
    for index in frames:
        frame_data = frame_data_dct[index]
        if key == ContourType.LUMEN.value:
            clear_lumen_measurements(frame_data)  # derived from the lumen being deleted
        frame_data.contours[key] = Contour()

    display.working_spline = None
    display.active_contour_index = 0
    display.update_display()
    display.refresh_all_frame_metrics()  # also redraws both pullback overviews
    main_window.status_bar.showMessage(f'Deleted {name} on {len(frames)} frames (Ctrl+Z undoes it)')


def _selected_knots(main_window):
    """(contour, index, xs, ys, closed) of the selected contour, or None if it cannot be
    resampled (sector, measurement, mid-draw or drag)."""
    display = main_window.display
    if not main_window.image_displayed or display.drawing_mode or display.active_point_index is not None:
        return None
    if active_preset().is_angle(display.active_contour_type):
        return None  # a sector's points mark angles, not a shape
    contour = display._frame_contour()
    ci = display.active_contour_index
    if contour is None or ci >= len(contour.contours) or not contour.contours[ci] or not contour.contours[ci][0]:
        return None
    closed = contour.closed[ci] if ci < len(contour.closed) else True
    xs, ys = without_closing_repeat(contour.contours[ci][0], contour.contours[ci][1], closed)
    return contour, ci, xs, ys, closed


def selected_contour_knot_count(main_window) -> int | None:
    """Selected contour's knot count (None if not resamplable)."""
    selected = _selected_knots(main_window)
    return len(selected[2]) if selected else None


def set_selected_contour_knots(main_window, count: int):
    """Resample the selected contour to `count` knots, keeping its shape (see knot_resampling).

    Each count reached is remembered per contour, so stepping back restores the earlier knots
    instead of resampling a thinned contour. A run of changes is one Ctrl+Z entry, or a mouse
    wheel turn would fill the small undo stack.
    """
    selected = _selected_knots(main_window)
    if selected is None:
        return
    contour, ci, xs, ys, closed = selected
    display = main_window.display
    key = display.contour_key()
    history_key = (display.frame, key, ci)
    history = display.knot_histories.get(history_key)
    if history is None or history.closed != closed or not history.holds(xs, ys):
        # First change, or edited elsewhere since: start from the current knots
        pinned = [
            point for labels in (contour.start_coords, contour.end_coords) if ci < len(labels) for point in labels[ci]
        ]
        try:
            history = KnotHistory(xs, ys, closed, pinned)
        except Exception as exc:
            logger.debug(f'Cannot resample the knots of {key} #{ci}: {exc}')
            return
        display.knot_histories[history_key] = history

    new_xs, new_ys = history.knots(count)
    if len(new_xs) == len(xs):
        return  # already at the count asked for, or as far as it goes

    if history.is_original(xs, ys):
        push_contour_snapshot(main_window.runtime_data, display.frame, key, ci)
    else:
        main_window.runtime_data.mark_unsaved()
    contour.contours[ci] = (new_xs, new_ys)

    main_window.save_contours_soon()
    display.display_image(update_contours=True)
    try:  # the area of a lumen or EEM shifts a little with its knots
        main_window.longitudinal_view.plot_areas()
    except Exception as exc:
        logger.debug(f'Could not refresh the pullback overviews after resampling: {exc}')
    defn = active_preset().get(display.active_contour_type)
    main_window.status_bar.showMessage(f'{defn.name if defn else key}: {len(new_xs)} points')


def step_selected_contour_knots(main_window, step: int):
    """Move the selected contour's knot count by `step` (Shift+Wheel)."""
    count = selected_contour_knot_count(main_window)
    if count is not None:
        set_selected_contour_knots(main_window, count + step)


def new_contour(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading input file')
        return

    main_window.display.set_active_contour_type(contour_type)

    main_window.display.start_contour(contour_type=contour_type)
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True


def new_measure(main_window, index: int):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual measure before reading input file')
        return

    main_window.display.start_measure(index)
    main_window.hide_contours_box.setChecked(False)


def new_reference(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual reference before reading input file')
        return

    main_window.display.set_active_contour_type(ContourType.REFERENCE)
    main_window.display.start_reference()
    main_window.hide_contours_box.setChecked(False)


def new_angle(main_window, contour_type: ContourType, append: bool = False):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual angle before reading input file')
        return

    main_window.display.set_active_contour_type(contour_type)
    main_window.display.start_angle(append=append)
    main_window.hide_contours_box.setChecked(False)


def set_tool(main_window, segmentation_tool: SegmentationTool):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot set tool before reading input file')
        return

    if segmentation_tool == SegmentationTool.BRUSH:
        if not getattr(main_window, 'mask_mode_box', None) or not main_window.mask_mode_box.isChecked():
            ErrorMessage(main_window, 'Enable Mask Mode to use the brush tool')
            main_window.left_half.closed_spline_btn.setChecked(True)
            return
        active = main_window.display.active_contour_type
        preset = active_preset()
        if SegmentationTool.BRUSH not in preset.allowed_tools(active):
            defn = preset.get(active)
            ErrorMessage(main_window, f'The brush tool cannot be used for {defn.name if defn else active.value}')
            main_window.left_half.closed_spline_btn.setChecked(True)
            return
        main_window.display.active_segmentation_tool = segmentation_tool
        main_window.display.enable_brush()
        return

    main_window.display.disable_brush()
    main_window.display.active_segmentation_tool = segmentation_tool


def new_contour_append(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading input file')
        return

    main_window.display.set_active_contour_type(contour_type)
    main_window.display.start_contour(contour_type=contour_type, append=True)
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True
