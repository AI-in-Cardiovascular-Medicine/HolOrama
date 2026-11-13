"""The non-pipe parts of a loaded pullback, for the fusion viewer, and the case files they come from.

multimodars keeps every reference on its frame (`frame.reference_point`, the geometry's
`master_reference` says which one aligns it) and turns every structure HolOrama exports (side
branches, calcium, lipid, ...) into a point cloud on its frame (`frame.structures`). Both follow
their frame through every alignment, so here they are only collected.

Only reads attributes, none of it needs multimodars.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDEBRANCH = 'sidebranch'
# Contour files that are pipes, not structures (lumen is `{phase}_contours.csv` itself).
_PIPE_PREFIXES = ('eem', 'wall', 'catheter')


@dataclass(frozen=True)
class CaseReference:
    """One reference of an exported phase: its frame (as in the CSV files) and whether it was set from a side branch."""

    frame: int
    auto: bool


def structure_points(geometry) -> dict[str, np.ndarray]:
    """Every structure's points (N, 3) by kind, over all frames. Empty for a multimodars without structures."""
    points: dict[str, list[tuple[float, float, float]]] = {}
    for frame in geometry.frames:
        for structure in getattr(frame, 'structures', None) or []:
            points.setdefault(str(structure.kind), []).extend((p.x, p.y, p.z) for p in structure.points)
    return {kind: np.array(rows, dtype=float).reshape(-1, 3) for kind, rows in points.items() if rows}


def reference_points(geometry, auto_frames: set[int]) -> tuple[np.ndarray, list[str]]:
    """Every reference point (N, 3) of `geometry` and what each is: 'master' for the geometry's master
    reference, else 'auto' if its frame (as in the CSV files) is in `auto_frames`, else 'hand'."""
    master = getattr(geometry, 'master_reference', None)
    points, status = [], []
    for frame in geometry.frames:
        point = getattr(frame, 'reference_point', None)
        if point is None:
            continue
        original = int(frame.lumen.original_frame)
        points.append((point.x, point.y, point.z))
        status.append('master' if original == master else 'auto' if original in auto_frames else 'hand')
    return np.array(points, dtype=float).reshape(-1, 3), status


def geometry_references(geometry, auto_frames: set[int]) -> list[CaseReference]:
    """The references a loaded geometry carries, by frame as in the CSV files, in frame order."""
    return [CaseReference(frame, frame in auto_frames) for frame in sorted(geometry.reference_frames())]


def contour_points(geometry, contour_type: str) -> np.ndarray:
    """Every point (N, 3) of the frames' `contour_type` extra (e.g. 'Eem')."""
    points = [
        (p.x, p.y, p.z)
        for frame in geometry.frames
        if contour_type in (getattr(frame, 'extras', None) or {})
        for p in frame.extras[contour_type].points
    ]
    return np.array(points, dtype=float).reshape(-1, 3)


def read_case_references(folder: str, phase: str) -> list[CaseReference]:
    """The references of `{phase}_reference_points.csv` (frame, x, y, z[, auto]), empty if there is none."""
    path = Path(folder) / f'{phase}_reference_points.csv'
    if not path.is_file():
        return []
    with open(path, newline='') as file:
        rows = [row for row in csv.reader(file, delimiter='\t') if row]
    return [CaseReference(int(float(row[0])), len(row) > 4 and row[4].strip() == '1') for row in rows]


def default_reference(references: list[CaseReference]) -> int | None:
    """The reference preselected as master: the first placed by hand, else the first."""
    for reference in references:
        if not reference.auto:
            return reference.frame
    return references[0].frame if references else None


def read_case_structures(folder: str, phase: str) -> dict[str, np.ndarray] | None:
    """Every `{type}_{phase}_contours.csv` that is not a pipe, as numpy_to_inputdata takes it:
    type -> rows [frame, x, y, z, contour, closed]. None if the phase has none.

    The file loaders read these themselves, this is for the array path (OCT).
    """
    suffix = f'_{phase}_contours.csv'
    structures = {}
    for path in sorted(Path(folder).glob(f'*{suffix}')):
        kind = path.name[: -len(suffix)]
        if not kind or kind in _PIPE_PREFIXES:
            continue
        with open(path, newline='') as file:
            rows = [[float(value) for value in row[:6]] for row in csv.reader(file, delimiter='\t') if row]
        if rows:
            structures[kind] = np.asarray(rows, dtype=float)
    return structures or None
