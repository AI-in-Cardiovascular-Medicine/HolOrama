import numpy as np
import matplotlib.path as mplPath
from loguru import logger
from skimage import measure

from gui.popup_windows.message_boxes import ErrorMessage, SuccessMessage
from gui.popup_windows.frame_range_dialog import FrameRangeDialog


def segment(main_window):
    """Automatic segmentation of IVUS images"""
    main_window.status_bar.showMessage('Segmenting frames...')
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot perform automatic segmentation before reading input file')
        main_window.status_bar.showMessage(main_window.waiting_status)
        return

    segment_dialog = FrameRangeDialog(main_window)

    if segment_dialog.exec():
        lower_limit, upper_limit = segment_dialog.getInputs()
        masks = main_window.predictor(main_window.images, lower_limit, upper_limit)
        if masks is not None:
            mask_to_contours(main_window, masks, lower_limit, upper_limit)
            main_window.contours_drawn = True
            main_window.display.set_data(main_window.images)
            main_window.hide_contours_box.setChecked(False)

    SuccessMessage(main_window, 'Automatic segmentation')
    main_window.status_bar.showMessage(main_window.waiting_status)


def mask_to_contours(main_window, masks, lower_limit, upper_limit, config=None):
    """Extracts contours from masked images.

    When main_window is provided, writes in-place to main_window.data[frame].lumen.
    When main_window is None (headless), returns a Dict[int, FrameData].
    """
    if main_window is not None:
        config = main_window.config
    if config is None:
        logger.error('mask_to_contours: no config available')
        return None

    num_points = config.display.n_interactive_points
    image_shape = masks.shape[1:3]
    counter = 0

    if main_window is None:
        from input_output.contours_io import FrameData
        data = {}
        for frame in range(lower_limit, upper_limit):
            fd = FrameData()
            if np.sum(masks[frame, :, :]) > 0:
                counter += 1
                contours_frame = label_contours(masks[frame, :, :])
                keep_lumen_x, keep_lumen_y = downsample(keep_largest_contour(contours_frame, image_shape), num_points)
                # remove last point after segmentation
                keep_lumen_x, keep_lumen_y = keep_lumen_x[:-1], keep_lumen_y[:-1]
                fd.lumen.contours = [[keep_lumen_x, keep_lumen_y]]
            data[frame] = fd
        logger.info(f'Found contours in {counter} frames')
        return data

    resolution = main_window.metadata.get('resolution', 0.1)  # mm/pixel
    fallback_radius_px = 0.5 / resolution

    for frame in range(lower_limit, upper_limit):
        fd = main_window.data.get(frame)
        if fd is None:
            continue
        keep_lumen_x, keep_lumen_y = [], []
        if np.sum(masks[frame, :, :]) > 0:
            contours_frame = label_contours(masks[frame, :, :])
            keep_lumen_x, keep_lumen_y = downsample(keep_largest_contour(contours_frame, image_shape), num_points)
            keep_lumen_x, keep_lumen_y = keep_lumen_x[:-1], keep_lumen_y[:-1]
        if keep_lumen_x:
            counter += 1
            fd.lumen.contours = [[keep_lumen_x, keep_lumen_y]]
        else:
            fd.lumen.contours = [_catheter_fallback_contour(image_shape, fallback_radius_px, num_points)]
    logger.info(f'Found contours in {counter} frames')


def _catheter_fallback_contour(image_shape, radius_px, num_points):
    """Circle at image centre — used when segmentation fails for a frame."""
    cy, cx = image_shape[0] / 2.0, image_shape[1] / 2.0
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = (cx + radius_px * np.cos(angles)).tolist()
    y = (cy + radius_px * np.sin(angles)).tolist()
    return [x, y]


def label_contours(image):
    """generate contours for labels"""
    contours = measure.find_contours(image)
    lumen = []
    for contour in contours:
        # find_contours closes the contour by duplicating the first point at the end — strip it
        if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]
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
