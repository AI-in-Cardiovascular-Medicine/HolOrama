"""A closed mesh of an in-wall structure (calcium, lipid, ...) from multimodars' point clouds.

multimodars turns each drawn contour into a jittered ring of points on its frame, moved with the
frame through every alignment. A mesh straight from those points cannot tell inside from outside,
so each frame's ring is filled first, in the frame's own plane (fitted to its lumen):

- per frame, the ring is closed and filled into a region, kept as a signed distance field in mm
- a voxel between two neighbouring frames that both hold the structure takes the signed distance
  interpolated between them, by where it sits between their planes, so the shape changes smoothly
  from frame to frame and follows frames tilted by the alignment
- the first and last frame of a run reach half a frame spacing beyond their plane
- the zero level of that field is the surface (marching cubes, then Taubin smoothing)

Only reads attributes (frames, lumen points, centroids, structures), none of it needs multimodars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

_CLOSING_MM = 0.2  # gaps in a jittered ring closed before filling it
_MAX_GAP_SPACINGS = 2.0  # neighbouring frames further apart (in frame spacings) are not bridged
_OUTSIDE = 1e3  # signed distance where no frame says anything (mm)


@dataclass
class _Slice:
    """One frame's filled region: its plane and the region's signed distance field (mm, negative inside)."""

    centre: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    origin: np.ndarray  # (u, v) of sdf[0, 0]
    sdf: np.ndarray

    def distance(self, points: np.ndarray, pitch: float) -> np.ndarray:
        """Signed distance of the region at `points` projected onto the plane, _OUTSIDE off the grid."""
        rel = points - self.centre
        coords = np.stack([(rel @ self.u - self.origin[0]) / pitch, (rel @ self.v - self.origin[1]) / pitch])
        return ndimage.map_coordinates(self.sdf, coords, order=1, mode='constant', cval=_OUTSIDE)

    def height(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of `points` from the plane along its normal."""
        return (points - self.centre) @ self.normal


def structure_mesh(geometry, kind: str, pitch: float = 0.1, smoothing: int = 20) -> trimesh.Trimesh | None:
    """The closed mesh of every `kind` structure of `geometry`, None if it has none.

    `pitch` is the voxel size in mm, `smoothing` the Taubin iterations.
    """
    frames = list(geometry.frames)
    slices: dict[int, _Slice] = {}
    normals = _frame_normals(frames)
    for index, frame in enumerate(frames):
        points = _points(frame, kind)
        if len(points) >= 3 and index in normals:
            slices[index] = _fill(points, np.asarray(frame.centroid, dtype=float), normals[index], pitch)
    if not slices:
        return None
    spacing = _spacing(frames, normals)

    cloud = np.vstack([_points(frames[i], kind) for i in slices])
    low = cloud.min(axis=0) - spacing - 2 * pitch
    shape = np.ceil((cloud.max(axis=0) + spacing + 2 * pitch - low) / pitch).astype(int) + 1
    grid = np.stack(np.meshgrid(*(low[k] + pitch * np.arange(shape[k]) for k in range(3)), indexing='ij'), axis=-1)
    voxels = grid.reshape(-1, 3)

    # Neighbours further apart than this are not bridged (e.g. a frame the alignment left in place).
    bridge = _MAX_GAP_SPACINGS * spacing
    linked = {i: i + 1 in slices and np.linalg.norm(slices[i + 1].centre - slices[i].centre) <= bridge for i in slices}

    field = np.full(len(voxels), _OUTSIDE)
    for index, current in slices.items():
        here = current.height(voxels)
        following = slices.get(index + 1) if linked[index] else None
        if following is not None:
            # between this plane and the next: interpolate the two regions by position between them
            there = following.height(voxels)
            between = (here >= 0) & (there <= 0) & (here - there > 0)
            t = here[between] / (here[between] - there[between])
            value = (1 - t) * current.distance(voxels[between], pitch) + t * following.distance(voxels[between], pitch)
            field[between] = np.minimum(field[between], value)
        else:
            _cap(field, voxels, current, here, 0 <= here, spacing, pitch)
        if not linked.get(index - 1, False):
            _cap(field, voxels, current, here, here < 0, spacing, pitch)

    volume = field.reshape(shape)
    if volume.min() >= 0:
        return None
    vertices, faces, _, _ = marching_cubes(volume, 0.0, spacing=(pitch,) * 3)
    mesh = trimesh.Trimesh(vertices + low, faces)
    if smoothing:
        trimesh.smoothing.filter_taubin(mesh, iterations=smoothing)
    mesh.fix_normals()
    return mesh


def _cap(field, voxels, current: _Slice, here, side, spacing: float, pitch: float) -> None:
    """The end of a run: the region extends half a frame spacing beyond its plane on `side`."""
    reach = side & (np.abs(here) <= spacing / 2)
    field[reach] = np.minimum(field[reach], current.distance(voxels[reach], pitch))


def _fill(points: np.ndarray, centre: np.ndarray, normal: np.ndarray, pitch: float) -> _Slice:
    """The region a frame's ring of points encloses, in the frame's plane."""
    u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = points - centre
    flat = np.column_stack([rel @ u, rel @ v])
    margin = 2 * _CLOSING_MM + 2 * pitch
    origin = flat.min(axis=0) - margin
    shape = np.ceil((flat.max(axis=0) + margin - origin) / pitch).astype(int) + 1
    mask = np.zeros(shape, dtype=bool)
    ij = np.round((flat - origin) / pitch).astype(int)
    mask[ij[:, 0], ij[:, 1]] = True
    mask = ndimage.binary_closing(mask, iterations=max(1, round(_CLOSING_MM / pitch)))
    mask = ndimage.binary_fill_holes(mask)
    sdf = (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)) * pitch
    return _Slice(centre, normal, u, v, origin, sdf)


def _points(frame, kind: str) -> np.ndarray:
    pts = [(p.x, p.y, p.z) for s in getattr(frame, 'structures', None) or [] if s.kind == kind for p in s.points]
    return np.array(pts, dtype=float).reshape(-1, 3)


def _frame_normals(frames) -> dict[int, np.ndarray]:
    """Each frame's plane normal from its lumen, all pointing the same way along the pullback."""
    normals = {}
    for index, frame in enumerate(frames):
        lumen = np.array([(p.x, p.y, p.z) for p in frame.lumen.points], dtype=float).reshape(-1, 3)
        if len(lumen) >= 3:
            normals[index] = np.linalg.svd(lumen - lumen.mean(axis=0))[2][-1]
    centroids = np.array([f.centroid for f in frames], dtype=float)
    for index, normal in normals.items():
        step = centroids[min(index + 1, len(frames) - 1)] - centroids[max(index - 1, 0)]
        if normal @ step < 0:
            normals[index] = -normal
    return normals


def _spacing(frames, normals: dict[int, np.ndarray]) -> float:
    """The median distance between neighbouring frame planes, along their normals (mm).

    Along the normal rather than between centroids, so a frame moved aside (e.g. one the alignment
    left in place) does not inflate it.
    """
    centroids = np.array([f.centroid for f in frames], dtype=float).reshape(-1, 3)
    gaps = [abs((centroids[i + 1] - centroids[i]) @ normals[i]) for i in range(len(frames) - 1) if i in normals]
    return float(np.median(gaps)) if gaps else 1.0
