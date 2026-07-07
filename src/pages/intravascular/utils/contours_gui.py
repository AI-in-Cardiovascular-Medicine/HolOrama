from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from domain.all_types import ContourType, SegmentationTool


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


def new_angle(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual angle before reading input file')
        return

    main_window.display.set_active_contour_type(contour_type)
    main_window.display.start_angle()
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
