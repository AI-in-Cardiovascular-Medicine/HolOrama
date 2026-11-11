"""Intravascular annotations in 3D, after multimodars moved them with their frames.

The intravascular export (input_output.output.annotations) writes side branches, plaques and
reference points per frame. multimodars carries them through every alignment as
`frame.annotations`, so their points are already where their frame went. This turns them into
what the fusion viewer shows:

- a side branch: one tube per run, at its reference frame, leaving the vessel perpendicular to
  it (along that frame's lumen centroid to arc middle), from the lumen border to the arc's far
  end, with the diameter measured there. The run's regions on every frame as points.
- a plaque: its region on every frame filled with random points, spread along the frame normal
  over the frame spacing, so consecutive frames merge into a blob.
- reference points, hand-placed and automatic.

Only reads attributes, so any geometry with `frames`, each with `lumen.points` and
`annotations` (kind, role, group, points with x, y, z), works. None of it needs multimodars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from tools.branch_arc import inside_polygon

SIDEBRANCH = 'sidebranch'
REFERENCE = 'reference'
REFERENCE_AUTO = 1  # group of a reference set from a side branch (input_output.output.annotations)


@dataclass(frozen=True)
class Placed:
    """One annotation where its frame went: `frame` is the frame's position in the geometry."""

    frame: int
    kind: str
    role: str
    group: int
    points: np.ndarray  # (N, 3)


def placed_annotations(geometry) -> list[Placed]:
    """Every annotation of `geometry`'s frames. Empty for a multimodars without annotations."""
    out = []
    for index, frame in enumerate(geometry.frames):
        for annotation in getattr(frame, 'annotations', None) or []:
            points = np.array([(p.x, p.y, p.z) for p in annotation.points], dtype=float).reshape(-1, 3)
            if len(points):
                out.append(Placed(index, str(annotation.kind), str(annotation.role), int(annotation.group), points))
    return out


def frame_normals(geometry) -> dict[int, np.ndarray]:
    """Unit normal of each frame's plane, fitted to its lumen points (sign arbitrary)."""
    normals = {}
    for index, frame in enumerate(geometry.frames):
        points = np.array([(p.x, p.y, p.z) for p in frame.lumen.points], dtype=float).reshape(-1, 3)
        if len(points) >= 3:
            normals[index] = np.linalg.svd(points - points.mean(axis=0))[2][-1]
    return normals


def frame_spacing(geometry) -> dict[int, float]:
    """Each frame's distance to its neighbours (the mean of both where it has two), from the frame centroids."""
    centroids = np.array([frame.centroid for frame in geometry.frames], dtype=float).reshape(-1, 3)
    gaps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    spacing = {}
    for index in range(len(centroids)):
        near = [gaps[i] for i in (index - 1, index) if 0 <= i < len(gaps)]
        spacing[index] = float(np.mean(near)) if near else 0.0
    return spacing


def plaque_blob(
    placed: list[Placed],
    kind: str,
    normals: dict[int, np.ndarray],
    spacing: dict[int, float],
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Points (N, 3) filling every `kind` region, `density` per mm² of region.

    Each frame's points lie inside its outline and are spread along the frame normal by up to
    half the frame spacing either way, so a plaque on neighbouring frames reads as one blob.
    """
    clouds = []
    for annotation in placed:
        if annotation.kind != kind or annotation.role != 'region' or annotation.frame not in normals:
            continue
        clouds.append(
            _fill(annotation.points, normals[annotation.frame], spacing.get(annotation.frame, 0.0), density, rng)
        )
    return np.vstack(clouds) if clouds else np.zeros((0, 3))


def _fill(outline: np.ndarray, normal: np.ndarray, spacing: float, density: float, rng) -> np.ndarray:
    """Random points inside the planar `outline` (N, 3), jittered along `normal` by up to spacing / 2."""
    centre = outline.mean(axis=0)
    rel = outline - centre
    u = rel[np.argmax(np.linalg.norm(rel, axis=1))]
    u = u - (u @ normal) * normal
    if np.linalg.norm(u) == 0:
        return np.zeros((0, 3))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    polygon = np.column_stack([rel @ u, rel @ v])

    x, y = polygon[:, 0], polygon[:, 1]
    area = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0
    count = max(1, round(area * density))
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    box = float(np.prod(high - low)) or 1.0
    inside = np.zeros((0, 2))
    for _ in range(4):  # rejection sampling, the box holds at least the region
        tries = rng.uniform(low, high, size=(int(count * box / max(area, 1e-9) * 1.5) + 8, 2))
        inside = np.vstack([inside, tries[inside_polygon(tries, polygon)]])
        if len(inside) >= count:
            break
    inside = inside[:count]
    offsets = rng.uniform(-spacing / 2, spacing / 2, size=len(inside))
    return centre + inside[:, :1] * u + inside[:, 1:] * v + offsets[:, None] * normal


def branch_tubes(placed: list[Placed], normals: dict[int, np.ndarray], sections: int) -> list[trimesh.Trimesh]:
    """One tube per side-branch run, swept round its axis on the run's reference frame.

    The axis runs from the landmark towards the axis tip (lumen centroid to arc middle), kept in
    the frame's plane, from the region's point nearest the lumen to its far end. Its radius is
    half the distance between the two diameter ends.
    """
    by_group: dict[int, dict[str, Placed]] = {}
    regions: dict[tuple[int, int], Placed] = {}
    for annotation in placed:
        if annotation.kind != SIDEBRANCH:
            continue
        if annotation.role == 'region':
            regions[(annotation.frame, annotation.group)] = annotation
        else:
            by_group.setdefault(annotation.group, {})[annotation.role] = annotation

    tubes = []
    for group, helpers in by_group.items():
        if not {'landmark', 'axis_tip', 'diameter_end'} <= set(helpers):
            continue
        frame = helpers['landmark'].frame
        region = regions.get((frame, group))
        ends = helpers['diameter_end'].points
        if region is None or frame not in normals or len(ends) < 2:
            continue
        landmark = helpers['landmark'].points[0]
        normal = normals[frame]
        axis = helpers['axis_tip'].points[0] - landmark
        axis = axis - (axis @ normal) * normal
        if np.linalg.norm(axis) == 0:
            continue
        axis /= np.linalg.norm(axis)
        along = (region.points - landmark) @ axis
        start, end = landmark + along.min() * axis, landmark + along.max() * axis
        radius = float(np.linalg.norm(ends[0] - ends[1])) / 2
        if radius <= 0 or np.linalg.norm(end - start) <= 0:
            continue
        tubes.append(trimesh.creation.cylinder(radius=radius, segment=np.array([start, end]), sections=sections))
    return tubes


def region_points(placed: list[Placed], kind: str) -> np.ndarray:
    """Every `kind` region outline point (N, 3)."""
    points = [a.points for a in placed if a.kind == kind and a.role == 'region']
    return np.vstack(points) if points else np.zeros((0, 3))


def reference_points(placed: list[Placed]) -> tuple[np.ndarray, list[bool]]:
    """Every reference point (N, 3) and whether each was set from a side branch."""
    references = [a for a in placed if a.kind == REFERENCE and a.role == 'point']
    if not references:
        return np.zeros((0, 3)), []
    return np.vstack([a.points[:1] for a in references]), [a.group == REFERENCE_AUTO for a in references]


def contour_points(geometry, contour_type: str) -> np.ndarray:
    """Every point (N, 3) of the frames' `contour_type` extra (e.g. 'Eem')."""
    points = [
        (p.x, p.y, p.z)
        for frame in geometry.frames
        for p in (getattr(frame, 'extras', {}) or {}).get(contour_type, _NO_CONTOUR).points
    ]
    return np.array(points, dtype=float).reshape(-1, 3)


class _NoContour:
    points: tuple = ()


_NO_CONTOUR = _NoContour()
