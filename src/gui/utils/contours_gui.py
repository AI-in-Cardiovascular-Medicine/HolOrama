from gui.popup_windows.message_boxes import ErrorMessage
from gui.left_half.IVUS_display import ContourType

def new_contour(main_window, contour_type: ContourType):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading input file')
        return
    
    main_window.display.set_active_contour_type(contour_type)
    key = contour_type.value

    xlist = main_window.data[key][0][main_window.display.frame] or []
    ylist = main_window.data[key][1][main_window.display.frame] or []
    main_window.tmp_contours[key] = (xlist.copy(), ylist.copy())

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

    main_window.display.start_angle(contour_type)
