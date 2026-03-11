import os

import numpy as np
from scipy.interpolate import splprep, splev
import SimpleITK as sitk
from loguru import logger
from PyQt6.QtWidgets import QProgressDialog, QApplication
from PyQt6.QtCore import Qt
from skimage.draw import polygon2mask

from gui.popup_windows.message_boxes import ErrorMessage

import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid


def save_as_nifti(main_window, mode=None):
    main_window.status_bar.showMessage('Saving frames as NIfTi files...')
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save as NIfTi before reading input file')
        return

    out_path = os.path.join(main_window.config.save.nifti_dir, f'{mode}_frames')
    if mode == 'contoured':
        frames_to_save = [
            frame
            for frame in range(main_window.metadata['num_frames'])
            if main_window.data.get(frame) and main_window.data[frame].lumen.contours
        ]
    elif mode == 'gated':
        frames_to_save = [
            frame
            for frame in range(main_window.metadata['num_frames'])
            if main_window.data.get(frame)
            and main_window.data[frame].lumen.contours
            and main_window.data[frame].phase in ['D', 'S']
        ]
    elif mode == 'all':
        frames_to_save = range(main_window.metadata['num_frames'])
    else:
        return  # nothing to save

    if frames_to_save:
        main_window.status_bar.showMessage('Saving frames as NIfTi files...')
        file_name = os.path.splitext(os.path.basename(main_window.file_name))[0]  # remove file extension
        os.makedirs(out_path, exist_ok=True)
        mask = contours_to_mask(
            main_window.images[frames_to_save], frames_to_save, main_window.data, main_window.metadata
        )

        progress = QProgressDialog()
        progress.setWindowFlags(Qt.Dialog)
        progress.setModal(True)
        progress.setMinimum(0)
        progress_max = len(frames_to_save) * main_window.config.save.save_2d + main_window.config.save.save_3d
        progress.setMaximum(progress_max)
        progress.resize(500, 100)
        progress.setWindowTitle('Saving frames as NIfTi files...')
        progress.show()

        if main_window.config.save.save_2d:
            for i, frame in enumerate(frames_to_save):  # save individual frames as NIfTi
                progress.setValue(i)
                QApplication.processEvents()
                if progress.wasCanceled():
                    break
                if (
                    main_window.data.get(frame) and main_window.data[frame].lumen.contours
                ):  # only save mask if contour exists
                    sitk.WriteImage(
                        sitk.GetImageFromArray(mask[i, :, :]),
                        os.path.join(out_path, f'{file_name}_frame_{frame}_seg.nii.gz'),
                    )
                sitk.WriteImage(
                    sitk.GetImageFromArray(main_window.images[frame, :, :]),
                    os.path.join(out_path, f'{file_name}_frame_{frame}_img.nii.gz'),
                )
        if main_window.config.save.save_3d:
            if any(
                main_window.data.get(f) and main_window.data[f].lumen.contours for f in frames_to_save
            ):  # only save mask if any contour exists
                sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(out_path, f'{file_name}_seg.nii.gz'))
            sitk.WriteImage(
                sitk.GetImageFromArray(main_window.images[frames_to_save]),
                os.path.join(out_path, f'{file_name}_img.nii.gz'),
            )
            progress.setValue(len(frames_to_save) * main_window.config.save.save_2d + 1)
            QApplication.processEvents()

        progress.close()
        main_window.status_bar.showMessage(main_window.waiting_status)

        # Call DICOM conversion function
        if main_window.config.save.save_dicom:  # Add a config flag to enable/disable DICOM conversion
            convert_nifti_to_dicom(main_window, out_path, file_name, frames_to_save)


def convert_nifti_to_dicom(main_window, out_path, file_name, frames_to_save):
    pass


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL_BACKGROUND = 0  # catheter zone, wire shadow, outside 4.75 mm
LABEL_LUMEN = 1
LABEL_EEM_WALL = 2  # between lumen and EEM
LABEL_CALCIUM = 3
LABEL_LIPID = 4
LABEL_MACROPHAGE = 5
LABEL_ADVENTITIA = 6  # outside EEM, within 4.75 mm
LABEL_BRANCH = 7  # side-branch lumen

_CATHETER_MM = 0.45
_MAX_VESSEL_MM = 4.75
_N_INTERP = 500  # dense interpolation points for smooth polygon boundaries


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _smooth_contour(xs, ys, is_closed=True):
    """
    Re-interpolate sparse knot points through a B-spline, returning
    _N_INTERP densely-sampled (x, y) arrays for a smooth polygon boundary.
    Falls back to the original arrays on failure.
    """
    xs, ys = list(xs), list(ys)
    n = len(xs)
    if n < 2:
        return np.array(xs), np.array(ys)
    k = min(3, n - 1)
    try:
        tck, _ = splprep(np.array([xs, ys]), s=0.0, k=k, per=int(is_closed))
        x_new, y_new = splev(np.linspace(0.0, 1.0, _N_INTERP), tck)
        return x_new, y_new
    except Exception:
        return np.array(xs), np.array(ys)


def _closed_polygon_mask(xs, ys, image_shape):
    """polygon2mask for a closed contour. xs/ys in original image pixel coords."""
    xs_s, ys_s = _smooth_contour(xs, ys, is_closed=True)
    return polygon2mask(image_shape, np.column_stack([ys_s, xs_s]))


def _open_outer_sector_mask(xs, ys, centroid_x, centroid_y, image_shape):
    """
    Mask for the region on the OUTER side of an open arc (toward EEM/adventitia).

    Computes the angular sector defined by the arc's endpoint rays from the
    centroid, then subtracts the inner polygon (centroid → arc → centroid).
    The caller clips the result to eem_mask & ~lumen_mask.
    """
    xs_s, ys_s = _smooth_contour(xs, ys, is_closed=False)
    H, W = image_shape
    yy, xx = np.mgrid[0:H, 0:W]
    pixel_angles = np.arctan2(yy.astype(float) - centroid_y, xx.astype(float) - centroid_x)

    # Determine which CCW/CW angular direction contains the arc midpoint
    x0, y0 = xs_s[0], ys_s[0]
    xN, yN = xs_s[-1], ys_s[-1]
    xm, ym = xs_s[len(xs_s) // 2], ys_s[len(ys_s) // 2]
    a_start = np.arctan2(y0 - centroid_y, x0 - centroid_x)
    a_end = np.arctan2(yN - centroid_y, xN - centroid_x)
    a_mid = np.arctan2(ym - centroid_y, xm - centroid_x)

    ccw_size = (a_end - a_start) % (2 * np.pi)
    mid_in_ccw = ((a_mid - a_start) % (2 * np.pi)) <= ccw_size

    if mid_in_ccw:
        full_sector = ((pixel_angles - a_start) % (2 * np.pi)) <= ccw_size
    else:
        cw_size = (2 * np.pi) - ccw_size
        full_sector = ((pixel_angles - a_end) % (2 * np.pi)) <= cw_size

    # Inner polygon: centroid → arc → centroid (the lumen-side region to subtract)
    inner_poly_yx = np.empty((len(xs_s) + 2, 2))
    inner_poly_yx[0] = (centroid_y, centroid_x)
    inner_poly_yx[1:-1] = np.column_stack([ys_s, xs_s])
    inner_poly_yx[-1] = (centroid_y, centroid_x)
    inner_mask = polygon2mask(image_shape, inner_poly_yx)

    return full_sector & ~inner_mask


def _contour_obj_to_mask(contour_obj, centroid_x, centroid_y, image_shape):
    """
    Convert a Contour dataclass to a boolean mask.
    Handles multiple sub-contours (OR-combined) and open/closed flag per entry.
    """
    if not contour_obj.contours:
        return np.zeros(image_shape, dtype=bool)

    combined = np.zeros(image_shape, dtype=bool)
    for idx, entry in enumerate(contour_obj.contours):
        try:
            xs, ys = entry[0], entry[1]
            if not xs or not ys:
                continue
            is_closed = contour_obj.closed[idx] if idx < len(contour_obj.closed) else True
            if is_closed:
                combined |= _closed_polygon_mask(xs, ys, image_shape)
            else:
                combined |= _open_outer_sector_mask(xs, ys, centroid_x, centroid_y, image_shape)
        except Exception:
            continue
    return combined


def _wire_shadow_mask(wire, image_shape, center_y, center_x):
    """
    Boolean mask for the guide-wire angular shadow (smaller sector between the
    two radial wire lines).

    wire: None | tuple of 1-2 (x, y) points in original image pixel coords.
    """
    if wire is None or len(wire) < 2:
        return np.zeros(image_shape, dtype=bool)

    H, W = image_shape
    yy, xx = np.mgrid[0:H, 0:W]

    p1x, p1y = wire[0]
    p2x, p2y = wire[1]

    a1 = np.arctan2(p1y - center_y, p1x - center_x)
    a2 = np.arctan2(p2y - center_y, p2x - center_x)

    pixel_angles = np.arctan2(yy.astype(float) - center_y, xx.astype(float) - center_x)

    # CCW arc from a1 → a2; pick the smaller of the two arcs
    ccw_size = (a2 - a1) % (2 * np.pi)
    if ccw_size <= np.pi:
        # CCW a1→a2 is the smaller sector
        return ((pixel_angles - a1) % (2 * np.pi)) <= ccw_size
    else:
        # CCW a2→a1 is the smaller sector
        cw_size = (a1 - a2) % (2 * np.pi)
        return ((pixel_angles - a2) % (2 * np.pi)) <= cw_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def contours_to_mask(images, contoured_frames, data, metadata):
    """
    Convert IVUS contours to a multi-label numpy mask.

    Labels
    ------
    0  background  - catheter zone (< 0.45 mm from centre),
                     guide-wire shadow, outside 4.75 mm
    1  lumen
    2  EEM wall    - inside EEM contour, outside lumen
    3  calcium     - within EEM (open or closed spline)
    4  lipid       - within EEM (open or closed spline)
    5  macrophage  - within EEM (open or closed spline)
    6  adventitia  - outside EEM, within 4.75 mm
    7  branch      - side-branch lumen (closed spline, not EEM-clipped)

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
    contoured_frames : list[int]
        Frame indices in the original timeline; mask[i] is built from
        data[contoured_frames[i]].
    data : Dict[int, FrameData]
    metadata : dict
        Must contain 'resolution' (mm / original-image-pixel).
    """
    image_shape = images.shape[1:3]
    H, W = image_shape
    mask = np.zeros((len(contoured_frames), H, W), dtype=np.uint8)

    resolution = metadata['resolution']  # mm / pixel
    catheter_r = _CATHETER_MM / resolution  # pixels
    outer_r = _MAX_VESSEL_MM / resolution  # pixels

    center_y, center_x = H / 2.0, W / 2.0

    yy, xx = np.mgrid[0:H, 0:W]
    dist_sq = (xx.astype(float) - center_x) ** 2 + (yy.astype(float) - center_y) ** 2
    catheter_zone = dist_sq <= catheter_r**2
    outside_vessel = dist_sq > outer_r**2

    for i, frame in enumerate(contoured_frames):
        fd = data.get(frame)
        if fd is None:
            continue

        # Lumen centroid for open-spline wedge direction (stored unscaled)
        cx, cy = fd.centroid if fd.centroid is not None else (center_x, center_y)

        eem_mask = _contour_obj_to_mask(fd.eem, cx, cy, image_shape)
        lumen_mask = _contour_obj_to_mask(fd.lumen, cx, cy, image_shape)

        fm = np.zeros(image_shape, dtype=np.uint8)

        # Layer bottom-up; later layers overwrite earlier ones
        if fd.eem.contours:
            # Adventitia: inside 4.75 mm, outside EEM
            fm[~outside_vessel & ~eem_mask] = LABEL_ADVENTITIA
            # EEM wall: inside EEM, outside lumen
            fm[eem_mask & ~lumen_mask] = LABEL_EEM_WALL

        if fd.lumen.contours:
            fm[lumen_mask] = LABEL_LUMEN

        # Plaques: clipped to EEM when EEM exists
        for label, contour_obj in (
            (LABEL_CALCIUM, fd.calcium),
            (LABEL_LIPID, fd.lipid),
            (LABEL_MACROPHAGE, fd.macrophage),
        ):
            if not contour_obj.contours:
                continue
            plaque = _contour_obj_to_mask(contour_obj, cx, cy, image_shape)
            if fd.eem.contours:
                plaque &= eem_mask
            plaque &= ~lumen_mask  # never inside lumen
            fm[plaque] = label

        # Branch: side-branch lumen, not clipped to EEM
        if fd.branch.contours:
            branch_mask = _contour_obj_to_mask(fd.branch, cx, cy, image_shape)
            fm[branch_mask] = LABEL_BRANCH

        # Final exclusion: catheter zone, wire shadow, outside 4.75 mm → 0
        wire_shadow = _wire_shadow_mask(fd.wire, image_shape, center_y, center_x)
        fm[catheter_zone | wire_shadow | outside_vessel] = LABEL_BACKGROUND

        mask[i] = fm

    return mask
