import os
import re
import json
import glob

from loguru import logger
from typing import List, Tuple, Optional, Dict

from version import version_file_str
from domain.io_types import Measure, Measurements, Contour, FrameData
from domain.all_types import OCT_QUALITY_LABELS
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


_CONTOUR_FILENAME_RE = re.compile(r'_contours_(ho_)?(\d+)_(\d+)_(\d+)\.json$')


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

    main_window.runtime_data.frame_data_dct = _build_frame_data(raw)
    main_window.runtime_data.gating_signal = raw.get('gating_signal', {})

    main_window.contours_drawn = True
    main_window.hide_contours_box.setChecked(False)
    logger.info(f'Loaded {len(main_window.runtime_data.frame_data_dct)} frames from {newest}')
    return True


def _normalize_coord_entry(item) -> List[Tuple[float, float]]:
    """Normalize a persisted start/end entry to a list of (x, y) tuples."""
    if not item:
        return []
    return [
        (float(pt[0]), float(pt[1])) for pt in item if pt is not None and isinstance(pt, (list, tuple)) and len(pt) >= 2
    ]


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


def _build_measure(raw) -> Optional[Measure]:
    if not raw:
        return None
    return Measure(
        points=raw.get('points'),
        length=raw.get('length'),
    )


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
