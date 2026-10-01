import math
import os
from functools import lru_cache

import numpy as np
import SimpleITK as sitk
from PyQt6.QtWidgets import QApplication, QProgressDialog
from scipy.interpolate import splev, splprep

from domain.all_types import ContourType
from domain.contour_presets import ContourPreset, ContourTypeDef, active_preset
from domain.io_types import iter_sectors
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


def _polygon_mask(polygon_yx, image_shape):
    """Rasterize a closed polygon: every pixel whose centre falls inside it, even-odd.

    A scanline fill rather than skimage's polygon2mask, which tests each pixel of the
    bounding box against every vertex of the polygon — with the ~500-vertex smoothed
    contours this module builds, one lumen costs ~95 ms and a report rasterizes several
    contours per frame over hundreds of frames. Walking one row at a time costs what the
    polygon's height costs instead, some 17x less, and agrees with polygon2mask to a
    pixel or two of boundary tangency out of tens of thousands (pinned in
    tests/test_mask_rasterization.py).

    `polygon_yx` is (row, col); the polygon closes from its last point back to its first.
    """
    mask = np.zeros(image_shape, dtype=bool)
    if len(polygon_yx) < 3:
        return mask

    rows = np.asarray(polygon_yx[:, 0], dtype=float)
    cols = np.asarray(polygon_yx[:, 1], dtype=float)
    next_rows = np.roll(rows, -1)
    next_cols = np.roll(cols, -1)

    height, width = image_shape
    first_row = max(int(np.ceil(rows.min())), 0)
    last_row = min(int(np.floor(rows.max())), height - 1)

    for row in range(first_row, last_row + 1):
        # Half-open in row, so a vertex sitting exactly on this scanline is counted by one
        # of its two edges rather than by both or neither.
        straddling = ((rows <= row) & (next_rows > row)) | ((next_rows <= row) & (rows > row))
        if not straddling.any():
            continue
        row_from = rows[straddling]
        row_to = next_rows[straddling]
        col_from = cols[straddling]
        col_to = next_cols[straddling]
        crossings = np.sort(col_from + (row - row_from) / (row_to - row_from) * (col_to - col_from))
        for left, right in zip(crossings[0::2], crossings[1::2]):
            start = max(int(math.ceil(left)), 0)
            end = min(int(math.floor(right)), width - 1)
            if end >= start:
                mask[row, start : end + 1] = True
    return mask


@lru_cache(maxsize=4)
def _pixel_angles(image_shape, centre_x: float, centre_y: float):
    """Every pixel's angle about (centre_x, centre_y), for one frame geometry.

    Cached because a frame asks for the same grid once per open contour and once per
    angular sector, and building it costs a couple of 4 MB float arrays and an arctan2
    over the whole frame each time. Returned read-only, so a caller cannot poison the
    cache for the next one.
    """
    height, width = image_shape
    yy, xx = np.mgrid[0:height, 0:width]
    angles = np.arctan2(yy.astype(float) - centre_y, xx.astype(float) - centre_x)
    angles.flags.writeable = False
    return angles


def _closed_polygon_mask(xs, ys, image_shape):
    """Rasterize a closed contour. xs/ys in original image pixel coords."""
    xs_s, ys_s = _smooth_contour(xs, ys, is_closed=True)
    return _polygon_mask(np.column_stack([ys_s, xs_s]), image_shape)


def _open_outer_sector_mask(xs, ys, centroid_x, centroid_y, image_shape):
    """
    Mask for the region on the OUTER side of an open arc (toward EEM/adventitia).

    Computes the angular sector defined by the arc's endpoint rays from the
    centroid, then subtracts the inner polygon (centroid → arc → centroid).
    The caller clips the result to eem_mask & ~lumen_mask.
    """
    xs_s, ys_s = _smooth_contour(xs, ys, is_closed=False)
    pixel_angles = _pixel_angles(tuple(image_shape), float(centroid_x), float(centroid_y))

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
    inner_mask = _polygon_mask(inner_poly_yx, image_shape)

    return full_sector & ~inner_mask


# How much of the lumen may fall outside a closed plaque contour for it to still count
# as drawn around the lumen. Generous on purpose: a circumferential plaque is drawn along
# the lumen border by hand and will cut inside it here and there, while a plaque drawn as
# an arc in the wall overlaps a sliver of the lumen at most.
_ENCIRCLES_LUMEN_FRACTION = 0.5


def _encircles_lumen(polygon_mask, lumen_mask):
    """Whether a closed contour was drawn around the lumen rather than inside the wall.

    Nothing that lies within a container other than the lumen contains the lumen, so a
    contour that swallows most of it cannot be the region itself — see _contained_mask.
    """
    lumen_area = lumen_mask.sum()
    if not lumen_area:
        return False
    return (polygon_mask & lumen_mask).sum() / lumen_area >= _ENCIRCLES_LUMEN_FRACTION


def _contained_mask(contour_obj, centroid_x, centroid_y, image_shape, lumen_mask, container_mask, in_lumen):
    """
    Boolean mask of one type that lies inside another (a preset's `inside`: calcium,
    lipid and macrophage in the EEM, say), clipped to its container — and kept out of
    the lumen, unless the lumen is the container (`in_lumen`).

    Every such contour marks the *luminal* side of the region, which then extends
    outwards to the container's boundary. For an open arc that is all it can mean (see
    _open_outer_sector_mask). A closed contour is normally the whole region and is simply
    filled in — but one drawn right around the lumen, as a circumferential calcification
    is, cannot be: nothing inside the wall contains the lumen. That ring is a luminal
    boundary too, so the container *outside* it is filled rather than the disc inside it,
    which is otherwise read as the region from the lumen out to the ring — the opposite
    of what was drawn.

    `container_mask` is None when the container is not drawn on this frame: there is
    then nothing to fill outwards to, so the disc is used either way.
    """
    combined = np.zeros(image_shape, dtype=bool)
    for idx, entry in enumerate(contour_obj.contours):
        try:
            xs, ys = entry[0], entry[1]
            if not xs or not ys:
                continue
            is_closed = contour_obj.closed[idx] if idx < len(contour_obj.closed) else True
            if not is_closed:
                combined |= _open_outer_sector_mask(xs, ys, centroid_x, centroid_y, image_shape)
                continue
            polygon = _closed_polygon_mask(xs, ys, image_shape)
            if container_mask is not None and not in_lumen and _encircles_lumen(polygon, lumen_mask):
                combined |= container_mask & ~polygon
            else:
                combined |= polygon
        except Exception:
            continue

    if container_mask is not None:
        combined &= container_mask
    return combined if in_lumen else combined & ~lumen_mask


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


_ANGLE_BIN_DEG = 1.0  # resolution of a region's reported angular extent


def _angular_extent_deg(mask: np.ndarray, cx: float, cy: float) -> float:
    """How much of the circle around (cx, cy) `mask` covers, in degrees.

    Measured off the rasterized region rather than its contour, so it holds for any shape
    and for several contours of the same type at once: two calcifications that overlap
    angularly count their shared degrees once, and a plaque drawn as an open arc is
    measured over the wedge it actually fills (see _contained_mask). Counted in one-degree
    bins, which is finer than a plaque angle is ever read off an image.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    angles = np.arctan2(ys - cy, xs - cx)
    bins = np.floor(np.degrees(angles) / _ANGLE_BIN_DEG).astype(int) % int(360 / _ANGLE_BIN_DEG)
    return float(np.unique(bins).size) * _ANGLE_BIN_DEG


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

    pixel_angles = _pixel_angles(tuple(image_shape), float(center_x), float(center_y))
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
    """A stand-in FrameData with every contour scaled by *factor*.

    Only what the region masks read is filled in; the original frame is left untouched.
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
        contours={key: scaled(contour_obj) for key, contour_obj in frame_data.contours.items()},
        centroid=(centroid[0] * factor, centroid[1] * factor) if centroid is not None else None,
    )


def _frame_regions(frame_data, image_shape, preset: ContourPreset) -> dict[str, np.ndarray]:
    """Every contour type's region on one frame, by id, each on its own (not layered).

    Angular sectors are wedges about the image centre (the catheter); every other open
    contour fills outwards about the lumen centroid. A type lying inside another is
    clipped to that one's region (see _contained_mask), so the containers are worked out
    first.
    """
    height, width = image_shape
    cx, cy = frame_data.centroid if frame_data.centroid is not None else (width / 2.0, height / 2.0)
    lumen_mask = _contour_obj_to_mask(frame_data.lumen, cx, cy, image_shape)
    regions: dict[str, np.ndarray] = {ContourType.LUMEN.value: lumen_mask}

    def region(defn: ContourTypeDef) -> np.ndarray:
        key = defn.type.value
        if key in regions:
            return regions[key]
        contour_obj = frame_data.contour(key)
        if defn.is_angle:
            mask = _angle_sector_mask(contour_obj, image_shape, height / 2.0, width / 2.0)
        elif defn.inside is None:
            mask = _contour_obj_to_mask(contour_obj, cx, cy, image_shape)
        elif not contour_obj.contours:
            mask = np.zeros(image_shape, dtype=bool)
        else:
            container = preset[defn.inside]
            drawn = bool(frame_data.contour(container.type).contours)
            mask = _contained_mask(
                contour_obj,
                cx,
                cy,
                image_shape,
                lumen_mask,
                region(container) if drawn else None,
                in_lumen=container.type == ContourType.LUMEN,
            )
        regions[key] = mask
        return mask

    for defn in preset.types:
        region(defn)
    return regions


def measured_types(preset: ContourPreset) -> tuple[ContourTypeDef, ...]:
    """The spline types measured as a region of their own, beyond the lumen and the EEM."""
    return tuple(defn for defn in preset.spline_types if defn.type not in (ContourType.LUMEN, ContourType.EEM))


def frame_region_metrics(
    frame_data, image_shape, resolution: float, downsample: int = 1, preset: ContourPreset | None = None
) -> dict[str, float]:
    """Areas (mm²) of one frame's lumen, EEM and wall, and of every other spline type's
    region (see measured_types), rasterized, plus how much of the circle each of those
    covers (`<id>_angle`, in degrees about the lumen centroid — the angle a calcification
    or lipid pool is read off an image as).

    Each region is read and clipped to its container by the same _frame_regions as
    contours_to_mask, so the area of anything lying in the EEM is a subset of `wall` and
    a caller's plaque/wall fraction stays in 0..1. Rasterizing rather than taking a
    polygon area is what makes those measurable at all: calcium/lipid are usually drawn
    as open arcs, which only enclose a region once closed against the EEM boundary (see
    _open_outer_sector_mask).

    Unlike contours_to_mask this does not apply the layering — each region is measured on
    its own, so overlapping regions both count in full.

    `downsample` > 1 rasterizes on a grid that many times smaller in each direction
    (cost drops with its square). Boundary pixels then carry more area each, so use
    it where a fraction of a region is wanted rather than an exact mm² — measuring a
    whole pullback, say — and leave it at 1 when the absolute area matters.
    """
    preset = preset or active_preset()
    if downsample > 1:
        factor = 1.0 / downsample
        frame_data = _scaled_frame_view(frame_data, factor)
        image_shape = (max(int(image_shape[0] * factor), 1), max(int(image_shape[1] * factor), 1))
        resolution = float(resolution) * downsample

    px_area = float(resolution) ** 2
    cx, cy = frame_data.centroid if frame_data.centroid is not None else (image_shape[1] / 2.0, image_shape[0] / 2.0)

    regions = _frame_regions(frame_data, image_shape, preset)
    lumen_mask = regions[ContourType.LUMEN.value]
    has_eem = bool(frame_data.eem.contours)
    eem_mask = regions[ContourType.EEM.value]
    wall = (eem_mask & ~lumen_mask) if has_eem else np.zeros(image_shape, dtype=bool)

    areas = {
        'lumen': float(lumen_mask.sum()) * px_area,
        'eem': float(eem_mask.sum()) * px_area if has_eem else 0.0,
        'wall': float(wall.sum()) * px_area,
    }
    for defn in measured_types(preset):
        key = defn.type.value
        areas[key] = float(regions[key].sum()) * px_area
        areas[f'{key}_angle'] = _angular_extent_deg(regions[key], cx, cy)

    return areas


def contours_to_mask(images, contoured_frames, data, preset: ContourPreset | None = None):
    """
    Convert IVUS contours to a multi-label numpy mask.

    Every contour type is painted as its preset `label` (0 is the background), bottom to
    top in the preset's paint order (see ContourPreset.paint_order): by layer, so where
    two regions overlap the higher layer is what the mask shows, with the lumen on top of
    everything except what lies inside it. Each region is first clipped to the type it
    lies in (see _frame_regions) — plaques to the vessel wall, say. With the default
    preset:

    1  lumen
    2  EEM         - shows as the vessel wall: the lumen is painted over it
    3  calcium     - within EEM (open or closed spline, see _contained_mask)
    4  lipid       - within EEM (open or closed spline, see _contained_mask)
    5  macrophage  - within EEM (open or closed spline, see _contained_mask)
    7  branch      - side-branch lumen (closed spline, not EEM-clipped)
    9  wire shadow - guide-wire angular shadow
    10 blood       - blood artefact angular sector, the bottom-most layer of all

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
    contoured_frames : list[int]
        Frame indices in the original timeline; mask[i] is built from
        data[contoured_frames[i]].
    data : Dict[int, FrameData]
    preset : the contour types to paint; the active preset by default
    """
    preset = preset or active_preset()
    image_shape = images.shape[1:3]
    H, W = image_shape
    mask = np.zeros((len(contoured_frames), H, W), dtype=np.uint8)
    paint_order = preset.paint_order()

    for i, frame in enumerate(contoured_frames):
        fd = data.get(frame)
        if fd is None:
            continue

        regions = _frame_regions(fd, image_shape, preset)
        fm = np.zeros(image_shape, dtype=np.uint8)
        for defn in paint_order:
            fm[regions[defn.type.value]] = defn.label
        mask[i] = fm

    return mask
