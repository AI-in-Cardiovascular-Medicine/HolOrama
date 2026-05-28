import os

import numpy as np
from scipy.interpolate import splprep, splev
import SimpleITK as sitk
from PyQt6.QtWidgets import QProgressDialog, QApplication
from skimage.draw import polygon2mask

from domain.all_types import ContourType
from domain.mask_types import MASK_SPECS
from gui.popup_windows.message_boxes import ErrorMessage


def save_as_nifti(main_window, mode=None):
    main_window.status_bar.showMessage('Saving frames as NIfTi files...')
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save as NIfTi before reading input file')
        return

    out_path = os.path.join(main_window.config.save.nifti_dir, f'{mode}_frames')
    if mode == 'contoured':
        frames_to_save = [
            frame
            for frame in range(main_window.runtime_data.metadata['num_frames'])
            if main_window.runtime_data.frame_data_dct.get(frame)
            and main_window.runtime_data.frame_data_dct[frame].lumen.contours
        ]
    elif mode == 'gated':
        frames_to_save = [
            frame
            for frame in range(main_window.runtime_data.metadata['num_frames'])
            if main_window.runtime_data.frame_data_dct.get(frame)
            and main_window.runtime_data.frame_data_dct[frame].lumen.contours
            and main_window.runtime_data.frame_data_dct[frame].phase in ['D', 'S']
        ]
    elif mode == 'all':
        frames_to_save = list(range(main_window.runtime_data.metadata['num_frames']))
    else:
        return  # nothing to save

    if frames_to_save:
        main_window.status_bar.showMessage('Saving frames as NIfTi files...')
        file_name = os.path.splitext(os.path.basename(main_window.file_name))[0]  # remove file extension
        os.makedirs(out_path, exist_ok=True)

        images = (
            main_window.runtime_data.images_rgb
            if main_window.runtime_data.metadata['modality'] == 'OCT'
            else main_window.runtime_data.images
        )

        progress_max = len(frames_to_save) + int(bool(main_window.config.save.save_3d))
        progress = QProgressDialog('Saving frames as NIfTi files...', 'Cancel', 0, progress_max, main_window)
        progress.setWindowTitle('Saving NIfTi files')
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.setValue(0)
        QApplication.processEvents()
        QApplication.processEvents()  # second flush processes the paint event queued by show

        frame_masks: list[np.ndarray] = []
        for i, frame in enumerate(frames_to_save):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            single_mask = contours_to_mask(
                main_window.runtime_data.images[frame : frame + 1], [frame], main_window.runtime_data.frame_data_dct
            )[0]
            if main_window.config.save.save_3d:
                frame_masks.append(single_mask)
            if main_window.config.save.save_2d:
                if (
                    main_window.runtime_data.frame_data_dct.get(frame)
                    and main_window.runtime_data.frame_data_dct[frame].lumen.contours
                ):
                    sitk.WriteImage(
                        sitk.GetImageFromArray(single_mask),
                        os.path.join(out_path, f'{file_name}_frame_{frame}_seg.nii.gz'),
                    )
                sitk.WriteImage(
                    sitk.GetImageFromArray(images[frame, :, :]),
                    os.path.join(out_path, f'{file_name}_frame_{frame}_img.nii.gz'),
                )

        if main_window.config.save.save_3d and not progress.wasCanceled() and frame_masks:
            full_mask = np.stack(frame_masks, axis=0)
            if any(
                main_window.runtime_data.frame_data_dct.get(f)
                and main_window.runtime_data.frame_data_dct[f].lumen.contours
                for f in frames_to_save
            ):
                sitk.WriteImage(sitk.GetImageFromArray(full_mask), os.path.join(out_path, f'{file_name}_seg.nii.gz'))
            sitk.WriteImage(
                sitk.GetImageFromArray(images[frames_to_save]),
                os.path.join(out_path, f'{file_name}_img.nii.gz'),
            )
            progress.setValue(progress_max)
            QApplication.processEvents()

        progress.close()
        main_window.status_bar.showMessage(main_window.waiting_status)


_N_INTERP = 500  # dense interpolation points for smooth polygon boundaries


def _smooth_contour(xs, ys, is_closed=True):
    """
    Re-interpolate sparse knot points through a B-spline, returning
    _N_INTERP densely-sampled (x, y) arrays for a smooth polygon boundary.
    Falls back to the original arrays on failure.
    """
    xs, ys = list(xs), list(ys)
    # Mirror SplineGeometry._ensure_closed(): add closing duplicate only when absent,
    # so the mask spline is computed identically to the interactive display spline.
    if is_closed and len(xs) > 1 and (xs[0] != xs[-1] or ys[0] != ys[-1]):
        xs = xs + [xs[0]]
        ys = ys + [ys[0]]
    n = len(xs)
    if n < 2:
        return np.array(xs), np.array(ys)
    k = min(3, n - 1)
    try:
        tck, u = splprep(np.array([xs, ys]), s=0.0, k=k, per=int(is_closed))
        x_new, y_new = splev(np.linspace(u.min(), u.max(), _N_INTERP), tck)
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


def contours_to_mask(images, contoured_frames, data):
    """
    Convert IVUS contours to a multi-label numpy mask.

    Labels
    ------
    0  background  - everything not covered by another label
    1  lumen
    2  EEM wall    - inside EEM contour, outside lumen
    3  calcium     - within EEM (open or closed spline)
    4  lipid       - within EEM (open or closed spline)
    5  macrophage  - within EEM (open or closed spline)
    7  branch      - side-branch lumen (closed spline, not EEM-clipped)
    9  wire shadow - guide-wire angular shadow

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
    contoured_frames : list[int]
        Frame indices in the original timeline; mask[i] is built from
        data[contoured_frames[i]].
    data : Dict[int, FrameData]
    """
    image_shape = images.shape[1:3]
    H, W = image_shape
    mask = np.zeros((len(contoured_frames), H, W), dtype=np.uint8)

    center_y, center_x = H / 2.0, W / 2.0

    _eem = MASK_SPECS[ContourType.EEM]
    _lumen = MASK_SPECS[ContourType.LUMEN]
    _branch = MASK_SPECS[ContourType.BRANCH]
    _wire = MASK_SPECS[ContourType.WIRE]
    _plaques = [
        MASK_SPECS[ContourType.CALCIUM],
        MASK_SPECS[ContourType.LIPID],
        MASK_SPECS[ContourType.MACROPHAGE],
    ]

    for i, frame in enumerate(contoured_frames):
        fd = data.get(frame)
        if fd is None:
            continue

        # Lumen centroid for open-spline wedge direction (stored unscaled)
        cx, cy = fd.centroid if fd.centroid is not None else (center_x, center_y)

        eem_mask = _contour_obj_to_mask(fd.eem, cx, cy, image_shape)
        lumen_mask = _contour_obj_to_mask(fd.lumen, cx, cy, image_shape)

        fm = np.zeros(image_shape, dtype=np.uint8)

        # Layer bottom-up by paint_order; later layers overwrite earlier ones.
        # Wire is lowest priority — painted first so all other structures appear on top.
        wire_shadow = _wire_shadow_mask(fd.wire, image_shape, center_y, center_x)
        fm[wire_shadow] = _wire.label

        if fd.eem.contours:
            fm[eem_mask & ~lumen_mask] = _eem.label

        # Branch before lumen so branch pixels inside lumen are overwritten by lumen.
        if fd.branch.contours:
            branch_mask = _contour_obj_to_mask(fd.branch, cx, cy, image_shape)
            fm[branch_mask] = _branch.label

        if fd.lumen.contours:
            fm[lumen_mask] = _lumen.label

        # Plaques: clipped to EEM when EEM exists, never inside lumen.
        for spec in _plaques:
            contour_obj = getattr(fd, spec.contour_type.value)
            if not contour_obj.contours:
                continue
            plaque = _contour_obj_to_mask(contour_obj, cx, cy, image_shape)
            if fd.eem.contours:
                plaque &= eem_mask
            plaque &= ~lumen_mask
            fm[plaque] = spec.label

        mask[i] = fm

    return mask
