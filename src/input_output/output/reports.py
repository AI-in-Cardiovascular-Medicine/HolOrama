import os
import math
import csv

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from PyQt6.QtWidgets import QProgressDialog, QApplication
from shapely.geometry import Polygon
from shapely.errors import TopologicalError
from itertools import combinations

from pages.intravascular.popup_windows.message_boxes import ErrorMessage, SuccessMessage


def report(main_window, lower_limit=None, upper_limit=None, suppress_messages=False):
    """Writes a report file containing lumen area, etc."""

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
        plot=main_window.config.report.plot,
        save_as_csv=main_window.config.report.save_as_csv,
    )
    if report_data is not None:  # else user cancelled progress bar
        # Add metadata information as columns to the first row
        report_data.loc[0, 'pullback_speed'] = main_window.runtime_data.metadata['pullback_speed']
        report_data.loc[0, 'pullback_start_frame'] = main_window.runtime_data.metadata['pullback_start_frame']
        report_data.loc[0, 'frame_rate'] = main_window.runtime_data.metadata['frame_rate']

        report_data.to_csv(
            os.path.splitext(main_window.file_name)[0] + '_report.txt',
            sep='\t',
            float_format='%.2f',
            index=False,
            header=True,
        )

        if not suppress_messages:
            SuccessMessage(main_window, 'Write report')

    return report_data


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


def compute_all(main_window, contoured_frames, suppress_messages, plot=True, save_as_csv=True):
    """compute all metrics and plot if desired"""
    if not suppress_messages:
        progress = QProgressDialog('Writing report...', 'Cancel', 0, len(contoured_frames), main_window)
        progress.setWindowTitle('Writing report')
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()
        QApplication.processEvents()
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

    # helper to fetch per-type full_contours defensively
    def _get_full_list_by_name(name):
        fc = getattr(main_window.display, "full_contours", None)
        if fc is None:
            return None
        if isinstance(fc, dict):
            return fc.get(name, None)
        if isinstance(fc, list):
            return fc
        return None

    # Lumen full contours (defensive)
    lumen_full_list = _get_full_list_by_name("lumen")
    # Fallback: try display.get_full_contour_list for backward compatibility
    if lumen_full_list is None:
        try:
            from domain.all_types import ContourType

            lumen_full_list = main_window.display.get_full_contour_list(ContourType.LUMEN)
        except (ImportError, AttributeError) as e:
            logger.bind(file=main_window.file_name).warning(
                f'Could not import ContourType/get_full_contour_list; using display.full_contours fallback. Reason: {e}'
            )
            lumen_full_list = getattr(main_window.display, "full_contours", None)

    # !! Build other contour full lists (eem, calcium, branch) for CSV saving / optional metrics, careful if new added !!
    eem_full_list = _get_full_list_by_name("eem")
    calc_full_list = _get_full_list_by_name("calcium")
    branch_full_list = _get_full_list_by_name("branch")

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
    if start_frame <= 0.25 * n_frames:
        offset = main_window.runtime_data.metadata['pullback_length'][start_frame - 1]
        report_data['position'] = [
            main_window.runtime_data.metadata['pullback_length'][frame] for frame in contoured_frames
        ]
        report_data['position'] = report_data['position'] - offset
    else:
        report_data['position'] = [
            main_window.runtime_data.metadata['pullback_length'][frame] for frame in contoured_frames
        ]
    report_data['position'] = report_data['position'].apply(lambda x: max(x, 0))
    report_data['phase'] = [main_window.runtime_data.frame_data_dct[frame].phase for frame in contoured_frames]
    report_data['lumen_area'] = [lumen_area[frame] for frame in contoured_frames]
    report_data['lumen_circumf'] = [lumen_circumf[frame] for frame in contoured_frames]
    report_data['longest_distance'] = [longest_distance[frame] for frame in contoured_frames]
    report_data['shortest_distance'] = [shortest_distance[frame] for frame in contoured_frames]
    report_data['elliptic_ratio'] = [elliptic_ratio[frame] for frame in contoured_frames]
    report_data['measurement_1'] = [
        main_window.runtime_data.frame_data_dct[frame].measurement_1.length
        if main_window.runtime_data.frame_data_dct[frame].measurement_1
        else None
        for frame in contoured_frames
    ]
    report_data['measurement_2'] = [
        main_window.runtime_data.frame_data_dct[frame].measurement_2.length
        if main_window.runtime_data.frame_data_dct[frame].measurement_2
        else None
        for frame in contoured_frames
    ]

    report_data['eem_area'] = [
        main_window.runtime_data.frame_data_dct[frame].eem.measurements.area or 0 for frame in contoured_frames
    ]

    # Write computed metrics back into per-frame measurements
    for frame in contoured_frames:
        fd = main_window.runtime_data.frame_data_dct.get(frame)
        if fd is None:
            continue
        if elliptic_ratio[frame] is not None:
            fd.lumen.measurements.elliptic_ratio = elliptic_ratio[frame]

    # Save CSVs for lumen (diastolic/systolic) and for other contours if present
    if save_as_csv:
        save_csv_files(
            main_window, lumen_x, lumen_y, name='diastolic', frames=main_window.runtime_data.gated_frames_dia
        )
        save_csv_files(main_window, lumen_x, lumen_y, name='systolic', frames=main_window.runtime_data.gated_frames_sys)

        # save EEM/Calcium/Branch CSVs if contours exist for any frame
        if eem_x is not None and any(elem is not None for elem in eem_x):
            save_csv_files(
                main_window, eem_x, eem_y, name='eem_diastolic', frames=main_window.runtime_data.gated_frames_dia
            )
            save_csv_files(
                main_window, eem_x, eem_y, name='eem_systolic', frames=main_window.runtime_data.gated_frames_sys
            )
        if calc_x is not None and any(elem is not None for elem in calc_x):
            save_csv_files(
                main_window, calc_x, calc_y, name='calcium_diastolic', frames=main_window.runtime_data.gated_frames_dia
            )
            save_csv_files(
                main_window, calc_x, calc_y, name='calcium_systolic', frames=main_window.runtime_data.gated_frames_sys
            )
        if branch_x is not None and any(elem is not None for elem in branch_x):
            save_csv_files(
                main_window,
                branch_x,
                branch_y,
                name='branch_diastolic',
                frames=main_window.runtime_data.gated_frames_dia,
            )
            save_csv_files(
                main_window,
                branch_x,
                branch_y,
                name='branch_systolic',
                frames=main_window.runtime_data.gated_frames_sys,
            )

        if plot:
            index_1 = int(len(contoured_frames) * 0.2)
            index_2 = int(len(contoured_frames) * 0.4)
            index_3 = int(len(contoured_frames) * 0.6)
            index_4 = int(len(contoured_frames) * 0.8)
            indices_to_plot = [index_1, index_2, index_3, index_4]
            frames_to_plot = [contoured_frames[frame] for frame in indices_to_plot]
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            for index, frame in enumerate(frames_to_plot):
                ax = axes[index // 2, index % 2]
                ax.plot(
                    lumen_x[frame],
                    lumen_y[frame],
                    '-g',
                    linewidth=2,
                    label='Contour',
                )
                ax.plot(centroid_x[frame], centroid_y[frame], 'ro', markersize=8, label='Centroid')
                ax.plot(farthest_x[frame][0], farthest_y[frame][0], 'bo', markersize=8, label='Farthest Point 1')
                ax.plot(farthest_x[frame][1], farthest_y[frame][1], 'bo', markersize=8, label='Farthest Point 2')
                ax.plot(nearest_x[frame][0], nearest_y[frame][0], 'yo', markersize=8, label='Nearest Point 1')
                ax.plot(nearest_x[frame][1], nearest_y[frame][1], 'yo', markersize=8, label='Nearest Point 2')

                ax.annotate(
                    f'Shortest Distance: {shortest_distance[frame]:.2f} mm',
                    xy=(centroid_x[frame], centroid_y[frame]),
                    xycoords='data',
                    xytext=(10, 30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                )
                ax.annotate(
                    f'Longest Distance: {longest_distance[frame]:.2f} mm',
                    xy=(centroid_x[frame], centroid_y[frame]),
                    xycoords='data',
                    xytext=(10, -30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2'),
                )
                ax.annotate(
                    f'Lumen Area: {lumen_area[frame]:.2f} mm\N{SUPERSCRIPT TWO}\nElliptic Ratio: {longest_distance[frame]/shortest_distance[frame]:.2f}',
                    xy=(centroid_x[frame], centroid_y[frame]),
                    xycoords='data',
                    xytext=(10, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                )
                ax.legend(loc='upper right')
                ax.invert_yaxis()
                ax.grid()
                ax.set_title(f'Frame {frame + 1}')

            fig.tight_layout()
            fig.show()

    if not suppress_messages:
        progress.close()
        QApplication.processEvents()

    return report_data


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

    with open(os.path.join(csv_out_dir, f'{name}_contours.csv'), 'w', newline='') as contours_file:
        contours_writer = csv.writer(contours_file, delimiter='\t')
        distance_offset = main_window.runtime_data.metadata['pullback_length'][frames[0]]
        for frame in frames:
            if lumen_x[frame] is None:
                continue
            rows = zip(
                [x * main_window.runtime_data.metadata['resolution'] for x in lumen_x[frame]],
                [abs(y * main_window.runtime_data.metadata['resolution'] - img_dim_mm) for y in lumen_y[frame]],
            )
            for row in rows:
                csv_row = (
                    [frame + 1]
                    + list(row)
                    + [main_window.runtime_data.metadata['pullback_length'][frame] - distance_offset]
                )
                contours_writer.writerow(csv_row)

    if name in ('diastolic', 'systolic'):
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
                            main_window.runtime_data.metadata['pullback_length'][frame] - distance_offset,
                        ]
                    )
