from loguru import logger

from domain.all_types import ALLOWED_TOOLS, ContourType, SegmentationTool
from domain.io_types import clear_frame_annotations
from domain.undo import push_frame_annotation_snapshot
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


def delete_all_on_frame(main_window):
    """Clear every contour, measurement, reference and wire angle on the current frame.

    A single undo entry covers the lot, so Ctrl+Z brings the whole frame back in one press.
    The frame's phase and its OCT label describe the frame rather than the drawing on it,
    so they are left alone.
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
    try:  # both pullback overviews read these contours, so they have to follow
        main_window.longitudinal_view.plot_areas()
    except Exception as exc:
        logger.debug(f'Could not refresh the pullback overviews after Delete All: {exc}')


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
        if SegmentationTool.BRUSH not in ALLOWED_TOOLS.get(active, set()):
            ErrorMessage(main_window, f'The brush tool cannot be used for {active.value}')
            main_window.left_half.closed_spline_btn.setChecked(True)
            return
        main_window.display.active_segmentation_tool = segmentation_tool
        main_window.display.enable_brush()
        return

    # Any other tool: deactivate brush if it was on.
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
