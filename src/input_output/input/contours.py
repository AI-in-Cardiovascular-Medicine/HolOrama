import os
import re
import json
import glob
import math

from loguru import logger
from typing import List, Tuple, Optional, Dict

from version import version_file_str
from domain.io_types import Measure, Measurements, Contour, FrameData
from domain.all_types import OCT_QUALITY_LABELS
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


_CONTOUR_FILENAME_RE = re.compile(r'_contours_(ho_)?(\d+)_(\d+)_(\d+)\.json$')


def read_contours(main_window, file_name=None) -> bool:
    """Read contours from the most recent JSON file and populate
    main_window.runtime_data.frame_data_dct as Dict[int, FrameData]. Returns True on success."""
    json_files = glob.glob(f'{file_name}_contours*.json')
    if not json_files:
        logger.info('No contour JSON files found.')
        return False

    newest = max(json_files, key=_contour_file_sort_key)
    logger.info(f'Current version: {version_file_str} | Loading: {newest}')

    try:
        with open(newest, 'r') as f:
            raw = json.load(f)
    except Exception as e:
        logger.exception(f'Failed to read {newest}: {e}')
        ErrorMessage(
            main_window,
            f'Contour file is corrupted and could not be loaded:\n{os.path.basename(newest)}\n\nDelete or repair the file and try again.',
        )
        return False

    num_frames = main_window.runtime_data.metadata['num_frames']

    if _is_legacy(raw):
        scaling_factor = main_window.display.image_size / main_window.runtime_data.images.shape[1]
        main_window.runtime_data.frame_data_dct = _build_frame_data_legacy(raw, num_frames, scaling_factor)
        main_window.runtime_data.gating_signal = raw.get('gating_signal', {})
    else:
        main_window.runtime_data.frame_data_dct = _build_frame_data(raw)
        main_window.runtime_data.gating_signal = raw.get('gating_signal', {})

    main_window.contours_drawn = True
    main_window.hide_contours_box.setChecked(False)
    logger.info(f'Loaded {len(main_window.runtime_data.frame_data_dct)} frames from {newest}')
    return True


def _contour_file_sort_key(path: str) -> Tuple[bool, Tuple[int, int, int]]:
    """Sort key for candidate contour files: HolOrama-tagged ('_ho_') files always
    outrank pre-rename AIVUS-CAA files, since HolOrama's version numbering restarted
    at 0.1.0 after the rename and would otherwise lose to old 1.x.y AIVUS-CAA filenames.
    Ties within a group are broken by numeric version rather than string order, so
    e.g. 0_10_0 correctly sorts after 0_9_0."""
    match = _CONTOUR_FILENAME_RE.search(os.path.basename(path))
    if not match:
        return (False, (0, 0, 0))
    is_holorama = match.group(1) is not None
    major, minor, patch = match.groups()[1:]
    return (is_holorama, (int(major), int(minor), int(patch)))


def _is_legacy(raw: dict) -> bool:
    """Detect legacy flat format by checking if 'lumen' is a list/tuple
    rather than a dict (current format produced by asdict)."""
    lumen = raw.get('lumen')
    return isinstance(lumen, (list, tuple))


def _build_frame_data_legacy(raw: dict, num_frames: int, scaling_factor: float = 1.0) -> Dict[int, FrameData]:
    """Convert legacy flat JSON format into Dict[int, FrameData].

    Legacy keys handled:
        phases, lumen, eem, calcium, branch, lipid, macrophage,
        measures, reference, gating_signal
    """
    logger.info('Detected legacy JSON format, converting to FrameData...')

    phases = raw.get('phases', ['-'] * num_frames)
    reference = raw.get('reference', [None] * num_frames)
    # legacy: [[m1, m2], ...] per frame, where m1/m2 were dicts or None
    measures = raw.get('measures', [[None, None]] * num_frames)
    # legacy: [[len1, len2], ...] per frame (may be nan for absent measurements)
    measure_lengths = raw.get('measure_lengths', [[None, None]] * num_frames)

    frames = {}
    for i in range(num_frames):
        m1_raw, m2_raw = measures[i] if i < len(measures) else (None, None)
        ml1, ml2 = measure_lengths[i] if i < len(measure_lengths) else (None, None)
        ml1 = None if ml1 is None or (isinstance(ml1, float) and math.isnan(ml1)) else ml1
        ml2 = None if ml2 is None or (isinstance(ml2, float) and math.isnan(ml2)) else ml2
        frames[i] = FrameData(
            phase=phases[i] if i < len(phases) else '-',
            lumen=_build_contour_legacy(raw, 'lumen', i),
            eem=_build_contour_legacy(raw, 'eem', i),
            calcium=_build_contour_legacy(raw, 'calcium', i),
            branch=_build_contour_legacy(raw, 'branch', i),
            lipid=_build_contour_legacy(raw, 'lipid', i),
            macrophage=_build_contour_legacy(raw, 'macrophage', i),
            measurement_1=_build_measure(m1_raw, scaling_factor, ml1),
            measurement_2=_build_measure(m2_raw, scaling_factor, ml2),
            reference=reference[i] if i < len(reference) else None,
        )
    return frames


def _build_frame_data(raw: dict) -> Dict[int, FrameData]:
    """Convert current JSON format (produced by asdict) into Dict[int, FrameData].
    Top-level non-integer keys (e.g. 'gating_signal') are skipped here."""
    frames = {}
    for key, frame_raw in raw.items():
        if not key.lstrip('-').isdigit():
            continue
        i = int(key)
        frames[i] = FrameData(
            phase=frame_raw.get('phase', '-'),
            quality=frame_raw.get('quality', OCT_QUALITY_LABELS[-1]),
            lumen=_build_contour(frame_raw.get('lumen')),
            eem=_build_contour(frame_raw.get('eem')),
            calcium=_build_contour(frame_raw.get('calcium')),
            branch=_build_contour(frame_raw.get('branch')),
            lipid=_build_contour(frame_raw.get('lipid')),
            macrophage=_build_contour(frame_raw.get('macrophage')),
            measurement_1=_build_measure(frame_raw.get('measurement_1')),
            measurement_2=_build_measure(frame_raw.get('measurement_2')),
            reference=frame_raw.get('reference'),
            wire=frame_raw.get('wire'),
            centroid=frame_raw.get('centroid'),
            closest_points=frame_raw.get('closest_points'),
            farthest_points=frame_raw.get('farthest_points'),
        )
    return frames


def _build_contour_legacy(raw: dict, key: str, i: int) -> Contour:
    """Reconstruct a Contour from the old flat tuple-of-lists format.

    Legacy format stored contours as:
        data['lumen'] = ([x0, x1, ...], [y0, y1, ...])
    where each xN/yN is a list of floats for frame N.
    Closed splines in legacy data have a duplicate closing point at the end — strip it.
    """
    raw_contour = raw.get(key)
    if not raw_contour:
        return Contour()

    if isinstance(raw_contour, (list, tuple)) and len(raw_contour) == 2:
        x_frames, y_frames = raw_contour
        x = x_frames[i] if i < len(x_frames) else []
        y = y_frames[i] if i < len(y_frames) else []
        # Legacy closed splines stored with a duplicate closing point — remove it
        if x and y:
            x = x[:-1]
            y = y[:-1]
        contours = [(x, y)] if (x or y) else []
        return Contour(contours=contours)

    return Contour()


def _build_measure(raw, scaling_factor: float = 1.0, length: Optional[float] = None) -> Optional[Measure]:
    if not raw:
        return None
    # Oldest legacy format: [x1, y1, x2, y2] stored in display coordinates — unscale
    if isinstance(raw, list):
        if len(raw) == 4:
            pts = (
                (raw[0] / scaling_factor, raw[1] / scaling_factor),
                (raw[2] / scaling_factor, raw[3] / scaling_factor),
            )
        else:
            pts = None
        return Measure(points=pts, length=length)
    return Measure(
        points=raw.get('points'),
        length=raw.get('length', length),
    )


def _build_contour(raw: Optional[dict]) -> Contour:
    """Reconstruct a Contour from the current nested dict format (produced by asdict)."""
    if not raw:
        return Contour()
    # Strip duplicate closing points that may have been persisted by SplineGeometry._ensure_closed
    stripped = []
    for entry in raw.get('contours', []):
        x = list(entry[0]) if entry else []
        y = list(entry[1]) if len(entry) > 1 else []
        if x and y and x[0] == x[-1] and y[0] == y[-1]:
            x, y = x[:-1], y[:-1]
        stripped.append((x, y))
    start_coords = [_normalize_coord_entry(e) for e in raw.get('start_coords', [])]
    end_coords = [_normalize_coord_entry(e) for e in raw.get('end_coords', [])]
    return Contour(
        contours=stripped,
        measurements=Measurements(**raw.get('measurements', {})),
        closed=raw.get('closed', []),
        start_coords=start_coords,
        end_coords=end_coords,
    )


def _normalize_coord_entry(item) -> List[Tuple[float, float]]:
    """Normalize a persisted start/end entry to the new list-of-tuples format.

    Handles three legacy shapes:
      None                          → []
      (x, y)  / [x, y]             → [(x, y)]   (old single-point format)
      [(x, y), ...]                 → [(x, y), ...]  (already new format)
    """
    if item is None:
        return []
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return []
        first = item[0]
        # Single (x, y) pair stored directly
        if isinstance(first, (int, float)):
            return [(float(item[0]), float(item[1]))]
        # List of points
        result = []
        for pt in item:
            if pt is not None and isinstance(pt, (list, tuple)) and len(pt) >= 2:
                result.append((float(pt[0]), float(pt[1])))
        return result
    return []
