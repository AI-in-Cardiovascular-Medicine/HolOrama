from gui.popup_windows.message_boxes import ErrorMessage
from gui.left_half.IVUS_display import ContourType, SegmentationTool

def new_contour(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading input file')
        return
    
    main_window.display.set_active_contour_type(contour_type)
    key = contour_type.value

    fd = main_window.data.get(main_window.display.frame)
    if fd:
        contour_obj = getattr(fd, key, None)
        if contour_obj and contour_obj.contours and contour_obj.contours[0]:
            xlist = list(contour_obj.contours[0][0]) if contour_obj.contours[0][0] else []
            ylist = list(contour_obj.contours[0][1]) if len(contour_obj.contours[0]) > 1 else []
        else:
            xlist, ylist = [], []
    else:
        xlist, ylist = [], []
    main_window.tmp_contours[key] = (xlist, ylist)

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

    main_window.display.start_reference()

def new_angle(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual angle before reading input file')
        return

    main_window.display.start_angle()

def set_tool(main_window, segmentation_tool: SegmentationTool):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot set tool before reading input file')
        return

    main_window.display.active_segmentation_tool = segmentation_tool

def new_contour_append(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading input file')
        return

    main_window.display.set_active_contour_type(contour_type)
    main_window.display.start_contour(contour_type=contour_type, append=True)
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True
