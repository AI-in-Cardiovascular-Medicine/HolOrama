"""Persist the CCTA cut-geometry inputs (LVOT/aorta-top cut lines, the label
choices they were cut with, and the RCA/LCA outlet points) as a JSON sidecar next
to the case's other files — mirrors input_output/{input,output}/contours.py's
versioned-sidecar convention: `{base}_..._{version}.json`, atomic write via a temp
file + move, and picking the newest file by *parsing the version out of the
filename* (input/contours.py's `_contour_file_sort_key`) rather than by mtime,
which isn't preserved across a git checkout, backup restore, or plain file copy
and could silently pick a stale sidecar.

This only stores the *inputs* (lines/labels/points) — the cut mesh itself is cheap
to rebuild from them (that's exactly what Build Cut Geometry already does), so
there's no mesh data to serialize here. Loading this file is what lets CctaPage
automatically rebuild the cut geometry right after a case's mask loads.
"""

import glob
import json
import os
import re
import shutil
import tempfile

from loguru import logger

from version import version_file_str

_CutLine = tuple[tuple[int, int, int], tuple[int, int, int]]

_CUTSTATE_FILENAME_RE = re.compile(r'_cutstate_(\d+)_(\d+)_(\d+)\.json$')


def _cut_state_sort_key(path: str) -> tuple[int, int, int]:
    """Sort key for candidate cut-state files: the (major, minor, patch) version
    parsed from the filename, so e.g. 0_10_0 correctly sorts after 0_9_0 (a plain
    string/mtime comparison would not)."""
    match = _CUTSTATE_FILENAME_RE.search(os.path.basename(path))
    if not match:
        return (0, 0, 0)
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def save_cut_state(
    source_path: str,
    cor_label: int | None,
    aorta_label: int | None,
    lv_label: int | None,
    cut_line_0: _CutLine | None,
    cut_line_1: _CutLine | None,
    aorta_cut_line: _CutLine | None,
    rca_points: list[tuple[int, int, int]],
    lca_points: list[tuple[int, int, int]],
    label_names: dict[int, str],
) -> None:
    out_path = f'{source_path}_cutstate_{version_file_str}.json'
    state = {
        'version': version_file_str,
        'cor_label': cor_label,
        'aorta_label': aorta_label,
        'lv_label': lv_label,
        'cut_line_0': cut_line_0,
        'cut_line_1': cut_line_1,
        'aorta_cut_line': aorta_cut_line,
        'rca_points': rca_points,
        'lca_points': lca_points,
        'label_names': {str(label): name for label, name in label_names.items()},  # JSON keys must be strings
    }

    out_dir = os.path.dirname(out_path) or '.'
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(state, f, indent=2)
        shutil.move(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        logger.exception('Failed to write cut state to disk')
    else:
        logger.info(f'Wrote cut state to {out_path}')


def load_cut_state(source_path: str) -> dict | None:
    """Load the most recently written cut-state JSON for this case, if any."""
    matches = glob.glob(f'{source_path}_cutstate_*.json')
    if not matches:
        return None

    newest = max(matches, key=_cut_state_sort_key)
    try:
        with open(newest) as f:
            state = json.load(f)
    except Exception:
        logger.exception(f'Failed to read cut state: {newest}')
        return None

    def _as_line(raw) -> _CutLine | None:
        if raw is None:
            return None
        return (tuple(raw[0]), tuple(raw[1]))

    return {
        'cor_label': state.get('cor_label'),
        'aorta_label': state.get('aorta_label'),
        'lv_label': state.get('lv_label'),
        'cut_line_0': _as_line(state.get('cut_line_0')),
        'cut_line_1': _as_line(state.get('cut_line_1')),
        'aorta_cut_line': _as_line(state.get('aorta_cut_line')),
        'rca_points': [tuple(p) for p in state.get('rca_points', [])],
        'lca_points': [tuple(p) for p in state.get('lca_points', [])],
        'label_names': {int(label): name for label, name in state.get('label_names', {}).items()},
    }
