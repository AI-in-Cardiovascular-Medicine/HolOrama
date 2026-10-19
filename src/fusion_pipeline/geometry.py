"""Mesh and pullback geometry helpers of the fusion pipeline that don't need multimodars."""

from __future__ import annotations

import numpy as np
import trimesh


def phases(geometry) -> list[tuple[str, object]]:
    """(phase key, PyGeometry) rows of a pullback: a PyGeometryPair gives diastole 'a' and
    systole 'b', a single PyGeometry (OCT or one IVUS phase) is one 'a' row."""
    geom_a = getattr(geometry, 'geom_a', None)
    geom_b = getattr(geometry, 'geom_b', None)
    if geom_a is None and geom_b is None:
        return [('a', geometry)]
    return [(key, geom) for key, geom in (('a', geom_a), ('b', geom_b)) if geom is not None]


def primary_geometry(geometry):
    """The single PyGeometry to read frames from or stitch: geom_a of a pair, else the geometry itself."""
    if geometry is None:
        return None
    geom_a = getattr(geometry, 'geom_a', None)
    return geometry if geom_a is None else geom_a


def load_ccta_mesh(path: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(path)


def export_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    mesh.export(path)


def transfer_labels(results: dict, mesh: trimesh.Trimesh) -> dict:
    """Copy region and branch labels to another mesh of the same surface (nearest vertex).
    Boundary rings are left out, they describe holes of the old mesh."""
    from scipy.spatial import cKDTree

    source = np.asarray(results['mesh'].vertices, dtype=np.float64)
    _, nearest = cKDTree(source).query(np.asarray(mesh.vertices, dtype=np.float64))
    index = {tuple(v): i for i, v in enumerate(source.tolist())}
    targets = [tuple(v) for v in np.asarray(mesh.vertices).tolist()]

    def remap(points: list) -> list:
        members = np.zeros(len(source), dtype=bool)
        members[[index[p] for p in map(tuple, points) if p in index]] = True
        return [targets[i] for i in np.flatnonzero(members[nearest])]

    labelled: dict = {'mesh': mesh}
    for key, value in results.items():
        if key == 'mesh' or 'boundary' in key:
            continue
        if key.endswith('_branches') and isinstance(value, dict):
            labelled[key] = {k: remap(pts) for k, pts in value.items()}
        elif isinstance(value, list) and value and len(value[0]) == 3:
            labelled[key] = remap(value)
    return labelled


def frames_to_mesh(geometry, contour_type: str | None = None) -> trimesh.Trimesh:
    """Loft a tube mesh through a PyGeometry's contours of `contour_type` (lumen if None).

    All contours need the same point count (use geometry.downsample(n) first).
    Visualization:
    ```
    ring i+1:  k ──── k+1
               │   ╲   │
    ring i:    k ──── k+1
    ```
    """
    contours = geometry.get_lumen_contours() if contour_type is None else geometry.get_contours_by_type(contour_type)
    if len(contours) < 2:
        raise ValueError('Need at least 2 contours to loft a mesh.')

    rings = [np.array(c.points_as_tuples(), dtype=np.float64) for c in contours]
    n = len(rings[0])
    if any(len(r) != n for r in rings):
        raise ValueError('All contours must have the same point count — call geometry.downsample(n) first.')

    vertices = np.concatenate(rings, axis=0)
    faces = []
    for i in range(len(rings) - 1):
        base_i, base_j = i * n, (i + 1) * n
        for k in range(n):
            k1 = (k + 1) % n
            faces.append((base_i + k, base_i + k1, base_j + k1))
            faces.append((base_i + k, base_j + k1, base_j + k))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    # Flip all faces if the first one points toward contour 0's centroid.
    face0_center = vertices[np.asarray(faces[0])].mean(axis=0)
    outward = face0_center - np.array(contours[0].centroid, dtype=np.float64)
    if np.dot(mesh.face_normals[0], outward) < 0:
        mesh.invert()
    return mesh


def taubin_smooth(mesh: trimesh.Trimesh, lamb: float = 0.6) -> trimesh.Trimesh:
    """Mutates and returns ``mesh`` (trimesh.smoothing operates in place)."""
    trimesh.smoothing.filter_taubin(mesh, lamb=lamb)
    return mesh


def max_seam_points(results: dict) -> int:
    """Largest seam_points_a/_b the stitch can use for the boundary rings in `results`.

    A ring of n points allows (n/2 - 2) // 2 per seam. Bounded by the smallest ring, since
    which ring is proximal is only decided inside the stitch.
    """
    ring_sizes = [
        len(v)
        for k, v in results.items()
        if k.startswith('boundary_points_') and k.removeprefix('boundary_points_').isdigit()
    ]
    if not ring_sizes:
        return 0
    return max((min(ring_sizes) // 2 - 2) // 2, 0)


def shifted_nearest_index(points, target, offset: int) -> tuple[int, int]:
    """Index of the point closest to `target`, moved by `offset` and clamped to the ends.
    Returns (index, unclamped index), so a caller can tell the shift was cut short."""
    distances = np.linalg.norm(np.asarray(points, dtype=float) - np.asarray(target, dtype=float), axis=1)
    raw_index = int(np.argmin(distances)) + offset
    return max(0, min(len(points) - 1, raw_index)), raw_index
