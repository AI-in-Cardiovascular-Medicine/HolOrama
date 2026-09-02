import csv
import math
import os
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger
from PyQt6.QtWidgets import QApplication, QProgressDialog
from shapely.errors import TopologicalError
from shapely.geometry import Polygon

from domain.all_types import PLAQUE_TYPES, ContourType
from domain.io_types import iter_sectors
from input_output.output.imgs_masks import frame_region_metrics
from pages.intravascular.popup_windows.message_boxes import ErrorMessage, SuccessMessage
from tools.angle import combined_sweep


def _pullback_positions(main_window) -> np.ndarray:
    """Per-frame axial position (mm), indexable by 0-based frame index.

    metadata['pullback_length'] is intentionally dual-typed (see input/metadata.py): IVUS
    and NIfTI store a per-frame array, whereas OCT stores a single scalar (the total
    pullback length). The report code indexes it per frame, so a scalar OCT value crashed
    here with "'float' object is not subscriptable". Normalize the scalar/missing case into
    the same per-frame array the array case already provides, derived from pullback_speed /
    frame_rate — the mm-per-frame the longitudinal and breathing views already use, and the
    same np.arange(1, N+1) * step convention as the IVUS/NIfTI per-frame arrays."""
    md = main_window.runtime_data.metadata
    pl = md.get('pullback_length')
    if isinstance(pl, np.ndarray):
        return pl
    if isinstance(pl, (list, tuple)):
        return np.asarray(pl, dtype=float)

    num_frames = md.get('num_frames')
    if not num_frames:
        fdd = main_window.runtime_data.frame_data_dct
        num_frames = (max(fdd) + 1) if fdd else 0
    speed = md.get('pullback_speed')
    frame_rate = md.get('frame_rate')
    mm_per_frame = (speed / frame_rate) if speed and frame_rate else 0.0
    return np.arange(1, num_frames + 1) * mm_per_frame


def report(main_window, lower_limit=None, upper_limit=None, suppress_messages=False, write_files=True):
    """Computes the report DataFrame; writes it (plus CSVs/plots per config) to disk unless write_files=False."""

    if not main_window.image_displayed:
        if not suppress_messages:
            ErrorMessage(main_window, 'Cannot write report before reading input file')
        return None

    if lower_limit is not None and upper_limit is not None:
        frame_range = range(lower_limit, upper_limit)
    else:
        frame_range = range(main_window.runtime_data.metadata['num_frames'])
    contoured_frames = [
        frame
        for frame in frame_range
        if frame in main_window.runtime_data.frame_data_dct
        and main_window.runtime_data.frame_data_dct[frame].lumen.contours
    ]
    if not contoured_frames:
        if not suppress_messages:
            ErrorMessage(main_window, 'Cannot write report before drawing contours')
        return None

    report_data = compute_all(
        main_window,
        contoured_frames,
        suppress_messages,
        save_as_csv=main_window.config.report.save_as_csv if write_files else False,
    )
    if report_data is not None:  # else user cancelled progress bar
        # The pullback's own parameters go last, on every row rather than only the first:
        # one row then carries everything needed to interpret it, which is what a reader
        # filtering or concatenating these files gets served by.
        for column, value in _metadata_columns(main_window).items():
            report_data[column] = value

        if write_files:
            report_data.to_csv(
                main_window.file_name + '_report.csv',  # already extension-free; see write_contours
                float_format='%.2f',
                index=False,
                header=True,
            )
            save_combined_sorted_manual(main_window, report_data)

        if not suppress_messages:
            SuccessMessage(main_window, 'Write report')

    return report_data


def _metadata_columns(main_window) -> dict:
    """The pullback's acquisition parameters, one value per column for the whole file.

    Read with .get: a NIfTI or a hand-entered pullback can be missing any of them, and a
    report without a frame rate is still worth writing.
    """
    metadata = main_window.runtime_data.metadata
    return {
        'modality': metadata.get('modality'),
        'pullback_speed': metadata.get('pullback_speed'),
        'pullback_start_frame': metadata.get('pullback_start_frame'),
        'frame_rate': metadata.get('frame_rate'),
        'resolution': metadata.get('resolution'),
    }


def _image_shape(main_window) -> tuple:
    """(H, W) of one frame, for the rasterized plaque metrics."""
    images = getattr(main_window.runtime_data, 'images', None)
    if images is not None and getattr(images, 'ndim', 0) >= 3:
        return int(images.shape[1]), int(images.shape[2])
    dimension = int(main_window.runtime_data.metadata.get('dimension') or 0)
    return dimension, dimension


def _plaque_and_blood_columns(main_window, contoured_frames) -> dict:
    """Per-frame plaque area (mm²) and angle (degrees), plus the combined blood angle.

    Both come off the rasterized mask rather than the contour polygons, because a plaque
    is usually drawn as an open arc that encloses nothing until it is closed against the
    EEM (see imgs_masks._plaque_mask) — and because that way the report says exactly what
    the exported mask holds. A frame with no plaque contour at all skips the rasterization,
    which is every frame while gating runs.

    Blood is reported as one angle per frame, several sectors counted once over any
    overlap. The guide wire is left out: its shadow says where the image cannot be read,
    not what is in the vessel.
    """
    columns: dict = {}
    for contour_type in PLAQUE_TYPES:
        columns[f'{contour_type.value}_area'] = []
        columns[f'{contour_type.value}_angle'] = []
    columns['blood_angle'] = []

    image_shape = _image_shape(main_window)
    resolution = main_window.runtime_data.metadata.get('resolution') or 0.0
    # Sectors are measured from the image centre (the catheter), as the mask measures them.
    sector_centre = (image_shape[1] / 2, image_shape[0] / 2)

    for frame in contoured_frames:
        frame_data = main_window.runtime_data.frame_data_dct.get(frame)
        has_plaque = frame_data is not None and any(
            getattr(frame_data, contour_type.value).contours for contour_type in PLAQUE_TYPES
        )
        metrics = frame_region_metrics(frame_data, image_shape, resolution) if has_plaque else {}
        for contour_type in PLAQUE_TYPES:
            columns[f'{contour_type.value}_area'].append(metrics.get(contour_type.value, 0.0))
            columns[f'{contour_type.value}_angle'].append(metrics.get(f'{contour_type.value}_angle', 0.0))
        blood = iter_sectors(frame_data.blood) if frame_data is not None else []
        columns['blood_angle'].append(math.degrees(combined_sweep(blood, sector_centre)))

    return columns


def save_combined_sorted_manual(main_window, report_data) -> None:
    """Always write combined_sorted_manual.csv alongside the report: the gated/tagged
    frames as one CSV, with no breathing rearrangement.

    This is the un-sorted baseline of the file the breathing-sort viewer produces. For IVUS
    it's the diastolic then systolic gated frames, each in acquisition order; for OCT there
    is no dia/sys gating, so it's the tagged frames instead. The breathing-sort viewer
    overwrites this same file with its hand-adjusted ordering (and a breathing-corrected
    position/distance_from_ostium_mm) when that tool is used — see
    breathing_sort_viewer._write_combined_report."""
    if report_data is None or report_data.empty:
        return
    rt = main_window.runtime_data
    if rt.metadata.get('modality') == 'OCT':
        ordered_frames = sorted(rt.tagged_frames)
    else:
        # Diastole block then systole block — the same combined order the breathing viewer
        # concatenates, just without reordering within each block.
        ordered_frames = sorted(rt.gated_frames_dia) + sorted(rt.gated_frames_sys)
    if not ordered_frames:
        return

    # report_data['frame'] is 1-based (frame + 1); our frame indices are 0-based.
    by_frame = report_data.set_index('frame')
    wanted = [f + 1 for f in ordered_frames if (f + 1) in by_frame.index]
    if not wanted:
        return
    combined = by_frame.loc[wanted].reset_index()

    csv_out_dir = main_window.file_name + '_csv_files'
    os.makedirs(csv_out_dir, exist_ok=True)
    out_path = os.path.join(csv_out_dir, 'combined_sorted_manual.csv')
    combined.to_csv(out_path, index=False)
    logger.info(f'Saved combined_sorted_manual.csv ({len(combined)} frames) to {csv_out_dir}')


def _safe_polygon_area(x_coords, y_coords, frame, contour_name, main_window):
    """Build polygon from coordinate lists and return area in mm².
    On revocerable errors return 0 and log full exception + context."""
    if x_coords is None or y_coords is None or len(x_coords) == 0 or len(y_coords) == 0:
        logger.warning(f'Empty coordinates for {contour_name} contour at frame {frame}, returning area 0.')
        return 0

    try:
        poly = Polygon([(x, y) for x, y in zip(x_coords, y_coords)])
        return poly.area * main_window.runtime_data.metadata['resolution'] ** 2
    except (ValueError, TypeError, TopologicalError) as e:
        logger.bind(frame=frame, contour=contour_name, file=main_window.file_name).exception(
            f'Error computing area for {contour_name} contour at frame {frame}: {e}'
        )
        return 0
    except Exception:
        logger.bind(frame=frame, contour=contour_name, file=main_window.file_name).exception(
            f'Unexpected error computing area for {contour_name} contour at frame {frame}'
        )
        raise


def compute_all(main_window, contoured_frames, suppress_messages, save_as_csv=True):
    """compute all metrics"""
    if not suppress_messages:
        progress = QProgressDialog('Writing report...', 'Cancel', 0, len(contoured_frames), main_window)
        progress.setWindowTitle('Writing report')
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()
        QApplication.processEvents()

    (
        longest_distance,
        farthest_x,
        farthest_y,
        shortest_distance,
        nearest_x,
        nearest_y,
        lumen_area,
        lumen_circumf,
        centroid_x,
        centroid_y,
        elliptic_ratio,
    ) = _prefill_values(main_window, contoured_frames)

    lumen_full_list = _full_list("lumen", main_window, ContourType.LUMEN)
    eem_full_list = _full_list("eem", main_window, ContourType.EEM)
    calc_full_list = _full_list("calcium", main_window, ContourType.CALCIUM)
    branch_full_list = _full_list("branch", main_window, ContourType.BRANCH)

    def build_xy_lists(full_list):
        if full_list is None:
            nframes = main_window.runtime_data.metadata.get("num_frames", 0)
            return [None] * nframes, [None] * nframes
        x_list = [contour[0] if (contour is not None and len(contour) >= 2) else None for contour in full_list]
        y_list = [contour[1] if (contour is not None and len(contour) >= 2) else None for contour in full_list]
        return x_list, y_list

    lumen_x, lumen_y = build_xy_lists(lumen_full_list)
    eem_x, eem_y = build_xy_lists(eem_full_list)
    calc_x, calc_y = build_xy_lists(calc_full_list)
    branch_x, branch_y = build_xy_lists(branch_full_list)

    for i, frame in enumerate(contoured_frames):
        if not suppress_messages:
            progress.setValue(i + 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                progress.close()
                return None

        # skip frames already computed (defensive check)
        if lumen_area[frame] and elliptic_ratio[frame] is not None and elliptic_ratio[frame] != 0:
            fd = main_window.runtime_data.frame_data_dct.get(frame)
            # compute EEM area if not present
            if eem_x and eem_x[frame] is not None and fd and not fd.eem.measurements.area:
                area = _safe_polygon_area(
                    eem_x[frame], eem_y[frame], frame=frame, contour_name="eem", main_window=main_window
                )
                fd.eem.measurements.area = area
            # compute centroid and vector metrics if not already available
            # (these are not persisted to disk, so must be re-derived on load)
            if centroid_x[frame] is None and lumen_x[frame] is not None:
                try:
                    polygon = Polygon([(x, y) for x, y in zip(lumen_x[frame], lumen_y[frame])])
                    _, _, centroid_x[frame], centroid_y[frame] = compute_polygon_metrics(main_window, polygon, frame)
                except Exception:
                    pass
            continue

        # dmake sure lumen contour exists
        if lumen_x[frame] is None or lumen_y[frame] is None:
            continue

        polygon = Polygon([(x, y) for x, y in zip(lumen_x[frame], lumen_y[frame])])
        exterior_coords = polygon.exterior.coords

        lumen_area[frame], lumen_circumf[frame], centroid_x[frame], centroid_y[frame] = compute_polygon_metrics(
            main_window, polygon, frame
        )
        longest_distance[frame], farthest_x[frame], farthest_y[frame] = farthest_points(
            main_window, exterior_coords, frame
        )
        shortest_distance[frame], nearest_x[frame], nearest_y[frame] = closest_points(main_window, polygon, frame)
        if shortest_distance[frame] != 0:
            elliptic_ratio[frame] = longest_distance[frame] / shortest_distance[frame]
        # Compute EEM area for this frame if EEM contour exists
        if eem_x and eem_x[frame] is not None:
            area = _safe_polygon_area(
                eem_x[frame], eem_y[frame], frame=frame, contour_name="eem", main_window=main_window
            )
            fd_eem = main_window.runtime_data.frame_data_dct.get(frame)
            if fd_eem:
                fd_eem.eem.measurements.area = area

    report_data = pd.DataFrame()
    report_data['frame'] = [frame + 1 for frame in contoured_frames]
    report_data['position'] = 0
    n_frames = main_window.runtime_data.metadata.get('num_frames', len(contoured_frames))
    start_frame = main_window.runtime_data.metadata['pullback_start_frame']
    positions = _pullback_positions(main_window)
    if start_frame <= 0.25 * n_frames:
        # start_frame == 0 has no prior frame to reference — measure from frame 0 (offset 0)
        # rather than letting [start_frame - 1] wrap to positions[-1] (the total length),
        # which would collapse the whole column to zero after the max(x, 0) clamp below.
        offset = positions[start_frame - 1] if start_frame >= 1 else 0.0
        report_data['position'] = [positions[frame] for frame in contoured_frames]
        report_data['position'] = report_data['position'] - offset
    else:
        report_data['position'] = [positions[frame] for frame in contoured_frames]
    report_data['position'] = report_data['position'].apply(lambda x: max(x, 0))
    report_data['phase'] = [main_window.runtime_data.frame_data_dct[frame].phase for frame in contoured_frames]
    report_data['lumen_area'] = [lumen_area[frame] for frame in contoured_frames]
    report_data['lumen_circumf'] = [lumen_circumf[frame] for frame in contoured_frames]
    report_data['longest_distance'] = [longest_distance[frame] for frame in contoured_frames]
    report_data['shortest_distance'] = [shortest_distance[frame] for frame in contoured_frames]
    report_data['elliptic_ratio'] = [elliptic_ratio[frame] for frame in contoured_frames]
    report_data['eem_area'] = [
        main_window.runtime_data.frame_data_dct[frame].eem.measurements.area or 0 for frame in contoured_frames
    ]

    # Each plaque as the area it covers and the angle it spans, then blood as one angle.
    for column, values in _plaque_and_blood_columns(main_window, contoured_frames).items():
        report_data[column] = values

    # The hand measurements come last of the per-frame columns, ahead of the pullback's own
    # parameters that report() appends after them.
    for index in (1, 2):
        measurements = [
            getattr(main_window.runtime_data.frame_data_dct[frame], f'measurement_{index}')
            for frame in contoured_frames
        ]
        report_data[f'measurement_{index}'] = [
            measure.length if measure is not None else None for measure in measurements
        ]

    # Write computed metrics back into per-frame measurements
    for frame in contoured_frames:
        fd = main_window.runtime_data.frame_data_dct.get(frame)
        if fd is None:
            continue
        if elliptic_ratio[frame] is not None:
            fd.lumen.measurements.elliptic_ratio = elliptic_ratio[frame]

    # Save CSVs for lumen and for other contours if present. Uses tagged/dia/sys.
    if save_as_csv:
        _save_as_csv(main_window, lumen_x, lumen_y, eem_x, eem_y, calc_x, calc_y, branch_x, branch_y)

    if not suppress_messages:
        progress.close()
        QApplication.processEvents()

    return report_data


def _prefill_values(main_window, contoured_frames):
    n_frames = main_window.runtime_data.metadata['num_frames']
    longest_distance = [None] * n_frames
    farthest_x = [None] * n_frames
    farthest_y = [None] * n_frames
    shortest_distance = [None] * n_frames
    nearest_x = [None] * n_frames
    nearest_y = [None] * n_frames
    lumen_area = [None] * n_frames
    lumen_circumf = [None] * n_frames
    centroid_x = [None] * n_frames
    centroid_y = [None] * n_frames
    elliptic_ratio = [None] * n_frames

    # Pre-fill from stored per-frame measurements
    for frame in contoured_frames:
        fd = main_window.runtime_data.frame_data_dct.get(frame)
        if fd is None:
            continue
        m = fd.lumen.measurements
        if m.area is not None:
            lumen_area[frame] = m.area
        if m.circumference is not None:
            lumen_circumf[frame] = m.circumference
        if fd.centroid:
            centroid_x[frame] = fd.centroid[0]
            centroid_y[frame] = fd.centroid[1]
        if m.major_axis is not None:
            longest_distance[frame] = m.major_axis
        if fd.farthest_points:
            farthest_x[frame] = [fd.farthest_points[0][0], fd.farthest_points[1][0]]
            farthest_y[frame] = [fd.farthest_points[0][1], fd.farthest_points[1][1]]
        if m.minor_axis is not None:
            shortest_distance[frame] = m.minor_axis
        if fd.closest_points:
            nearest_x[frame] = [fd.closest_points[0][0], fd.closest_points[1][0]]
            nearest_y[frame] = [fd.closest_points[0][1], fd.closest_points[1][1]]
        if m.elliptic_ratio is not None:
            elliptic_ratio[frame] = longest_distance[frame] / shortest_distance[frame]

    return (
        longest_distance,
        farthest_x,
        farthest_y,
        shortest_distance,
        nearest_x,
        nearest_y,
        lumen_area,
        lumen_circumf,
        centroid_x,
        centroid_y,
        elliptic_ratio,
    )


def _full_list(name, main_window, contour_type):
    """Per-type interpolated contour lists. Prefer a display.full_contours dict if one is
    ever present (backward compat), otherwise read from frame_data_dct via
    get_full_contour_list, the modality-agnostic source."""
    full_contours = getattr(main_window.display, "full_contours", None)
    if isinstance(full_contours, dict):
        lst = full_contours.get(name)
    elif isinstance(full_contours, list):
        lst = full_contours
    else:
        lst = None
    if lst is not None:
        return lst
    try:
        return main_window.display.get_full_contour_list(contour_type)
    except AttributeError as e:
        logger.bind(file=main_window.file_name).warning(
            f'Could not fetch {name} full contour list via get_full_contour_list: {e}'
        )
        return getattr(main_window.display, "full_contours", None)


def _save_as_csv(main_window, lumen_x, lumen_y, eem_x, eem_y, calc_x, calc_y, branch_x, branch_y):
    rt = main_window.runtime_data
    frame_groups = [
        ('diastolic', rt.gated_frames_dia),
        ('systolic', rt.gated_frames_sys),
        ('tagged', rt.tagged_frames),
    ]
    frame_groups = [(suffix, frames) for suffix, frames in frame_groups if frames]

    for suffix, frames in frame_groups:
        save_csv_files(main_window, lumen_x, lumen_y, name=suffix, frames=frames)

        # save EEM/Calcium/Branch CSVs if contours exist for any frame
    if eem_x is not None and any(elem is not None for elem in eem_x):
        for suffix, frames in frame_groups:
            save_csv_files(main_window, eem_x, eem_y, name=f'eem_{suffix}', frames=frames)
    if calc_x is not None and any(elem is not None for elem in calc_x):
        for suffix, frames in frame_groups:
            save_csv_files(main_window, calc_x, calc_y, name=f'calcium_{suffix}', frames=frames)
    if branch_x is not None and any(elem is not None for elem in branch_x):
        for suffix, frames in frame_groups:
            save_csv_files(main_window, branch_x, branch_y, name=f'branch_{suffix}', frames=frames)


def compute_polygon_metrics(main_window, polygon, frame):
    """Computes lumen area and centroid from contour"""
    lumen_area = polygon.area * main_window.runtime_data.metadata['resolution'] ** 2
    lumen_circumf = polygon.length * main_window.runtime_data.metadata['resolution']
    centroid_x = polygon.centroid.x
    centroid_y = polygon.centroid.y
    fd = main_window.runtime_data.frame_data_dct.get(frame)
    if fd:
        fd.lumen.measurements.area = lumen_area
        fd.lumen.measurements.circumference = lumen_circumf
        fd.centroid = (centroid_x, centroid_y)

    return lumen_area, lumen_circumf, centroid_x, centroid_y


def farthest_points(main_window, exterior_coords, frame):
    max_distance: float = 0
    farthest_points = None

    for point1, point2 in combinations(exterior_coords, 2):
        distance = math.dist(point1, point2)
        if distance > max_distance:
            max_distance = distance
            farthest_points = (point1, point2)

    longest_distance = max_distance * main_window.runtime_data.metadata['resolution']

    if farthest_points is None:
        logger.warning('No farthest points found, probably due to polygon shape')
        farthest_point_x = [0, 0]
        farthest_point_y = [0, 0]
        longest_distance = 0
    else:
        x1, y1 = farthest_points[0]
        x2, y2 = farthest_points[1]
        farthest_point_x = [x1, x2]
        farthest_point_y = [y1, y2]

    fd = main_window.runtime_data.frame_data_dct.get(frame)
    if fd:
        fd.lumen.measurements.major_axis = longest_distance
        fd.farthest_points = (
            (farthest_point_x[0], farthest_point_y[0]),
            (farthest_point_x[1], farthest_point_y[1]),
        )

    return longest_distance, farthest_point_x, farthest_point_y


def closest_points(main_window, polygon, frame):
    contour = polygon.exterior.coords
    num_points = len(contour)
    min_distance = math.inf
    closest_points = None

    index_1 = 0
    index_2 = num_points // 2

    while True:
        distance = math.dist(contour[index_1], contour[index_2])
        if distance < min_distance:
            min_distance = distance
            closest_points = (contour[index_1], contour[index_2])

        index_1 += 1
        index_2 += 1

        if index_1 >= num_points // 2:
            break

    shortest_distance = min_distance * main_window.runtime_data.metadata['resolution']

    if closest_points is None:
        logger.warning('No closest points found, probably due to polygon shape')
        closest_point_x = [0, 0]
        closest_point_y = [0, 0]
        shortest_distance = 0
    else:
        x1, y1 = closest_points[0]
        x2, y2 = closest_points[1]
        closest_point_x = [x1, x2]
        closest_point_y = [y1, y2]

    fd = main_window.runtime_data.frame_data_dct.get(frame)
    if fd:
        fd.lumen.measurements.minor_axis = shortest_distance
        fd.closest_points = (
            (closest_point_x[0], closest_point_y[0]),
            (closest_point_x[1], closest_point_y[1]),
        )

    return shortest_distance, closest_point_x, closest_point_y


def save_csv_files(main_window, lumen_x, lumen_y, name, frames):
    if not frames:
        logger.warning(f'No frames available for {name} contours, skipping CSV saving.')
        return
    csv_out_dir = os.path.join(main_window.file_name + '_csv_files')
    logger.info(f'Saving {name} contours to {csv_out_dir}')
    os.makedirs(csv_out_dir, exist_ok=True)
    img_dim_mm = main_window.runtime_data.metadata['dimension'] * main_window.runtime_data.metadata['resolution']
    positions = _pullback_positions(main_window)

    with open(os.path.join(csv_out_dir, f'{name}_contours.csv'), 'w', newline='') as contours_file:
        contours_writer = csv.writer(contours_file, delimiter='\t')
        distance_offset = positions[frames[0]]
        for frame in frames:
            if lumen_x[frame] is None:
                continue
            rows = zip(
                [x * main_window.runtime_data.metadata['resolution'] for x in lumen_x[frame]],
                [abs(y * main_window.runtime_data.metadata['resolution'] - img_dim_mm) for y in lumen_y[frame]],
            )
            for row in rows:
                csv_row = [frame + 1] + list(row) + [positions[frame] - distance_offset]
                contours_writer.writerow(csv_row)

    if name in ('diastolic', 'systolic', 'tagged'):
        ref_file_name = f'{name}_reference_points.csv'
        with open(os.path.join(csv_out_dir, ref_file_name), 'w', newline='') as reference_file:
            reference_writer = csv.writer(reference_file, delimiter='\t')
            for frame in frames:
                fd = main_window.runtime_data.frame_data_dct.get(frame)
                ref = fd.reference if fd else None
                if ref is not None:
                    reference_writer.writerow(
                        [
                            frame + 1,
                            ref[0] * main_window.runtime_data.metadata['resolution'],
                            abs(ref[1] * main_window.runtime_data.metadata['resolution'] - img_dim_mm),
                            positions[frame] - distance_offset,
                        ]
                    )
