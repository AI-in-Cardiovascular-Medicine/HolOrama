import os

import numpy as np
import SimpleITK as sitk
from PyQt6.QtWidgets import QApplication, QProgressDialog
from scipy.interpolate import splev, splprep
from skimage.draw import polygon2mask

from domain.all_types import ContourType
from domain.io_types import iter_wires
from domain.mask_types import MASK_SPECS
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


def save_as_nifti(main_window, mode=None):
    main_window.status_bar.showMessage('Saving frames as NIfTi files...')
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save as NIfTi before reading input file')
        return

    out_path = os.path.join(os.path.dirname(main_window.file_name), f'{mode}_frames')
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


def _region_mean_distance(mask: np.ndarray, cx: float, cy: float) -> float | None:
    """Mean distance of mask's True pixels from (cx, cy); None if mask is empty.

    Used to rank overlapping structures by how far they sit from the lumen
    centroid (see contours_to_mask's onion-layering)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(np.hypot(xs - cx, ys - cy).mean())


def _wire_shadow_mask(wire, image_shape, center_y, center_x):
    """
    Boolean mask for the guide-wire angular shadow(s): per wire, the smaller
    sector between its two radial lines, unioned over every wire on the frame.

    wire: Contour holding one entry of 1-2 (x, y) points per wire, in original
    image pixel coords. Wires with fewer than two points are still being drawn
    and contribute no shadow.
    """
    shadow = np.zeros(image_shape, dtype=bool)
    wires = [pts for pts in iter_wires(wire) if len(pts) >= 2]
    if not wires:
        return shadow

    H, W = image_shape
    yy, xx = np.mgrid[0:H, 0:W]
    pixel_angles = np.arctan2(yy.astype(float) - center_y, xx.astype(float) - center_x)

    for pts in wires:
        (p1x, p1y), (p2x, p2y) = pts[0], pts[1]

        a1 = np.arctan2(p1y - center_y, p1x - center_x)
        a2 = np.arctan2(p2y - center_y, p2x - center_x)

        # CCW arc from a1 → a2; pick the smaller of the two arcs
        ccw_size = (a2 - a1) % (2 * np.pi)
        if ccw_size <= np.pi:
            # CCW a1→a2 is the smaller sector
            shadow |= ((pixel_angles - a1) % (2 * np.pi)) <= ccw_size
        else:
            # CCW a2→a1 is the smaller sector
            cw_size = (a1 - a2) % (2 * np.pi)
            shadow |= ((pixel_angles - a2) % (2 * np.pi)) <= cw_size

    return shadow


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

    Where structures overlap, priority follows an "onion" rule: the structure
    whose pixels sit farther (on average) from the lumen centroid displaces
    the one closer to it. Three exceptions override that rule: the wire shadow
    is always the bottom-most layer, the EEM wall is always painted right on
    top of it (a backdrop that never hides lumen/branch/plaques — since a
    plaque's pixels are a subset of the EEM annulus, its mean distance can
    lose to the annulus average even though it must stay visible), and the
    lumen always displaces an overlapping side branch.

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

        # Wire is always the bottom-most layer — painted first so every other
        # structure sits on top of it, regardless of the onion order below.
        wire_shadow = _wire_shadow_mask(fd.wire, image_shape, center_y, center_x)
        fm[wire_shadow] = _wire.label

        # EEM is always the backdrop, painted right on top of the wire and below
        # everything else: it must never hide the lumen, a side branch, or a
        # plaque, even a small one whose own mean distance loses to the wall
        # annulus's average (a plaque's pixels are always a subset of this
        # annulus, so the two masks unavoidably overlap wherever a plaque exists).
        if fd.eem.contours:
            fm[eem_mask & ~lumen_mask] = _eem.label

        # Onion layering: the remaining structures are painted nearest-centroid
        # first, farthest-centroid last, so a structure farther from the lumen
        # centroid always displaces one that's closer wherever they overlap.
        regions: list[tuple] = [(_lumen, lumen_mask)]

        branch_mask = _contour_obj_to_mask(fd.branch, cx, cy, image_shape) if fd.branch.contours else None
        if branch_mask is not None:
            regions.append((_branch, branch_mask))

        for spec in _plaques:
            contour_obj = getattr(fd, spec.contour_type.value)
            if not contour_obj.contours:
                continue
            plaque = _contour_obj_to_mask(contour_obj, cx, cy, image_shape)
            if fd.eem.contours:
                plaque &= eem_mask
            plaque &= ~lumen_mask
            regions.append((spec, plaque))

        scored: list[tuple] = []
        for spec, region in regions:
            dist = _region_mean_distance(region, cx, cy)
            if dist is not None:
                scored.append((spec, region, dist))
        scored.sort(key=lambda entry: entry[2])

        # Exception to the onion order: the lumen is real anatomy and must never
        # be hidden by an overlapping side branch, so force it after branch here.
        lumen_pos = next((i for i, e in enumerate(scored) if e[0] is _lumen), None)
        branch_pos = next((i for i, e in enumerate(scored) if e[0] is _branch), None)
        if lumen_pos is not None and branch_pos is not None and lumen_pos < branch_pos:
            lumen_entry = scored.pop(lumen_pos)
            scored.insert(scored.index(next(e for e in scored if e[0] is _branch)) + 1, lumen_entry)

        for spec, region, _dist in scored:
            fm[region] = spec.label

        mask[i] = fm

    return mask
