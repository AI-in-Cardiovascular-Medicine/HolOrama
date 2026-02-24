import numpy as np
import matplotlib.path as mplPath
from loguru import logger
from skimage import measure

from gui.popup_windows.message_boxes import ErrorMessage, SuccessMessage
from gui.popup_windows.frame_range_dialog import FrameRangeDialog


# def segment(main_window):
#     """Automatic segmentation of IVUS images"""
#     main_window.status_bar.showMessage('Segmenting frames...')
#     if not main_window.image_displayed:
#         ErrorMessage(main_window, 'Cannot perform automatic segmentation before reading input file')
#         main_window.status_bar.showMessage(main_window.waiting_status)
#         return

#     segment_dialog = FrameRangeDialog(main_window)

#     if segment_dialog.exec_():
#         lower_limit, upper_limit = segment_dialog.getInputs()
#         masks = main_window.predictor(main_window.images, lower_limit, upper_limit)
#         if masks is not None:
#             main_window.data['lumen'] = mask_to_contours(main_window, masks, lower_limit, upper_limit)
#             main_window.data['lumen_area'] = [0] * main_window.metadata[
#                 'num_frames'
#             ]  # ensure all metrics are recalculated for the report
#             main_window.contours_drawn = True
#             main_window.display.set_data(main_window.data['lumen'], main_window.images)
#             main_window.hide_contours_box.setChecked(False)

#     SuccessMessage(main_window, 'Automatic segmentation')
#     main_window.status_bar.showMessage(main_window.waiting_status)


def mask_to_contours(main_window, masks, lower_limit, upper_limit, config=None):
    """Extracts contours from masked images. Writes directly to main_window.data[frame].lumen."""
    if main_window is None:
        return
    config = main_window.config
    num_points = config.display.n_interactive_points
    image_shape = masks.shape[1:3]
    counter = 0
    for frame in range(lower_limit, upper_limit):
        fd = main_window.data.get(frame)
        if fd is None:
            continue
        if np.sum(masks[frame, :, :]) > 0:
            counter += 1
            contours_frame = label_contours(masks[frame, :, :])
            keep_lumen_x, keep_lumen_y = downsample(keep_largest_contour(contours_frame, image_shape), num_points)
            fd.lumen.contours = [[keep_lumen_x, keep_lumen_y]]
        else:
            fd.lumen.contours = []
    logger.info(f'Found contours in {counter} frames')


def label_contours(image):
    """generate contours for labels"""
    contours = measure.find_contours(image)
    lumen = []
    for contour in contours:
        lumen.append(np.array((contour[:, 0], contour[:, 1])))

    return lumen


def keep_largest_contour(contours, image_shape):
    max_length = 0
    keep_contour = [[], []]
    for contour in contours:
        if keep_valid_contour(contour, image_shape):
            if len(contour[0]) > max_length:
                keep_contour = [[list(contour[1, :])], [list(contour[0, :])]]  # to match format expected by downsample
                max_length = len(contour[0])

    return keep_contour


def keep_valid_contour(contour, image_shape):
    """Contour is valid if it contains the centroid of the image"""
    bbPath = mplPath.Path(np.transpose(contour))
    centroid = [image_shape[0] // 2, image_shape[1] // 2]
    return bbPath.contains_point(centroid)


def downsample(contours, num_points):
    """Downsamples input contour data by selecting n points from original contour"""
    num_frames = len(contours[0])
    downsampled = [[] for _ in range(num_frames)], [[] for _ in range(num_frames)]

    for frame in range(num_frames):
        if len(contours[0][frame]) > num_points * 1.2:
            points_to_sample = range(0, len(contours[0][frame]), len(contours[0][frame]) // num_points)
            for axis in range(2):
                downsampled[axis][frame] = [contours[axis][frame][point] for point in points_to_sample]

    if num_frames == 1:
        downsampled = [downsampled[0][0], downsampled[1][0]]  # remove unnecessary dimension

    return downsampled
