import os

import numpy as np
import SimpleITK as sitk
from PyQt6.QtWidgets import QApplication, QProgressDialog
from scipy.interpolate import splev, splprep
from skimage.draw import polygon2mask

from domain.all_types import ANGLE_TYPES, ContourType
from domain.io_types import iter_sectors
from domain.mask_types import MASK_SPECS
from tools.angle import contains_angle, sector_from_points
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
            and main_window.runtime_data.frame_data_dct[frame].phase in ['D', 'S', 'T']
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


def _angle_sector_mask(contour, image_shape, center_y, center_x):
    """
    Boolean mask for one contour type's angular sectors (the guide-wire shadow, the
    blood artefact): the wedge each one covers, unioned over every sector on the frame.

    contour: Contour holding one entry of 1-3 (x, y) points per sector, in original
    image pixel coords — see tools.angle for how those points describe the wedge,
    including the sectors wider than 180 degrees that a stored interior marker allows.
    Sectors with fewer than two points are still being drawn and cover nothing.
    """
    covered = np.zeros(image_shape, dtype=bool)
    sectors = [pts for pts in iter_sectors(contour) if len(pts) >= 2]
    if not sectors:
        return covered

    H, W = image_shape
    yy, xx = np.mgrid[0:H, 0:W]
    pixel_angles = np.arctan2(yy.astype(float) - center_y, xx.astype(float) - center_x)
    centre = (float(center_x), float(center_y))

    for pts in sectors:
        geometry = sector_from_points(pts, centre)
        if geometry is None:
            continue
        covered |= contains_angle(pixel_angles, *geometry)

    return covered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _scaled_frame_view(frame_data, factor: float):
    """A stand-in FrameData with every contour this module rasterizes scaled by *factor*.

    Only the fields _contour_obj_to_mask / _open_outer_sector_mask read are filled in;
    the original frame is left untouched.
    """
    from domain.io_types import Contour, FrameData

    def scaled(contour_obj):
        entries = []
        for entry in contour_obj.contours:
            xs = entry[0] if entry else []
            ys = entry[1] if len(entry) > 1 else []
            entries.append(([x * factor for x in xs], [y * factor for y in ys]))
        return Contour(contours=entries, closed=list(contour_obj.closed))

    centroid = frame_data.centroid
    return FrameData(
        lumen=scaled(frame_data.lumen),
        eem=scaled(frame_data.eem),
        calcium=scaled(frame_data.calcium),
        lipid=scaled(frame_data.lipid),
        macrophage=scaled(frame_data.macrophage),
        centroid=(centroid[0] * factor, centroid[1] * factor) if centroid is not None else None,
    )


def frame_region_areas(frame_data, image_shape, resolution: float, downsample: int = 1) -> dict[str, float]:
    """Areas (mm²) of one frame's lumen, wall and plaque regions, rasterized.

    Plaques are clipped to the wall (inside EEM, outside lumen) exactly the way
    contours_to_mask clips them, so every plaque area is a subset of `wall` and a
    caller's plaque/wall fraction stays in 0..1. Rasterizing rather than taking a
    polygon area is what makes the plaques measurable at all: calcium/lipid are
    usually drawn as open arcs, which only enclose a region once closed against
    the EEM boundary (see _open_outer_sector_mask).

    Unlike contours_to_mask this does not apply the onion-layering priority — each
    region is measured on its own, so overlapping plaques both count in full.

    `downsample` > 1 rasterizes on a grid that many times smaller in each direction
    (cost drops with its square). Boundary pixels then carry more area each, so use
    it where a fraction of a region is wanted rather than an exact mm² — measuring a
    whole pullback, say — and leave it at 1 when the absolute area matters.
    """
    if downsample > 1:
        factor = 1.0 / downsample
        frame_data = _scaled_frame_view(frame_data, factor)
        image_shape = (max(int(image_shape[0] * factor), 1), max(int(image_shape[1] * factor), 1))
        resolution = float(resolution) * downsample

    px_area = float(resolution) ** 2
    cx, cy = frame_data.centroid if frame_data.centroid is not None else (image_shape[1] / 2.0, image_shape[0] / 2.0)

    lumen_mask = _contour_obj_to_mask(frame_data.lumen, cx, cy, image_shape)
    has_eem = bool(frame_data.eem.contours)
    eem_mask = _contour_obj_to_mask(frame_data.eem, cx, cy, image_shape) if has_eem else None
    wall = (eem_mask & ~lumen_mask) if eem_mask is not None else np.zeros(image_shape, dtype=bool)

    areas = {
        'lumen': float(lumen_mask.sum()) * px_area,
        'eem': float(eem_mask.sum()) * px_area if eem_mask is not None else 0.0,
        'wall': float(wall.sum()) * px_area,
    }
    for contour_type in (ContourType.CALCIUM, ContourType.LIPID, ContourType.MACROPHAGE):
        contour_obj = getattr(frame_data, contour_type.value)
        if not contour_obj.contours:
            areas[contour_type.value] = 0.0
            continue
        plaque = _contour_obj_to_mask(contour_obj, cx, cy, image_shape)
        if eem_mask is not None:
            plaque &= eem_mask
        plaque &= ~lumen_mask
        areas[contour_type.value] = float(plaque.sum()) * px_area

    return areas


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
    10 blood       - blood artefact angular sector

    Where structures overlap, priority follows an "onion" rule: the structure
    whose pixels sit farther (on average) from the lumen centroid displaces
    the one closer to it. Three exceptions override that rule: the angular
    sectors are always the bottom-most layers, the EEM wall is always painted right on
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
    _sectors = [MASK_SPECS[contour_type] for contour_type in ANGLE_TYPES]
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

        # The angular sectors are always the bottom-most layers — painted first so every
        # other structure sits on top of them, regardless of the onion order below.
        for sector_spec in _sectors:
            sector = _angle_sector_mask(getattr(fd, sector_spec.contour_type.value), image_shape, center_y, center_x)
            fm[sector] = sector_spec.label

        # EEM is always the backdrop, painted right on top of them and below
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
