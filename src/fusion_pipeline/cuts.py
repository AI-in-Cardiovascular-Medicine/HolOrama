"""Flat cuts of a mesh at slices of the discretized vessel tree.

A cut is kept as a CutSpec (which slice, which side stays), never as a mesh. Only the slice's
plane comes from the tree, so the cut geometry is always the specs replayed in order on a
source mesh. A changed mesh only needs a replay; a new tree or new centerlines invalidate the
specs themselves (slice indices and branch ids change).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re

import numpy as np
import trimesh
from loguru import logger

from domain.fusion.runtime_types import CutResult, CutSpec
from fusion_pipeline import geometry as geo
from fusion_pipeline import multimodars_api as mm_api

SliceKey = tuple[str, int, int]  # vessel, branch id, index into that branch's contours


def tree_slices(tree) -> dict[tuple[str, int], list]:
    """(vessel, branch id) -> that branch's contours, the aorta as ('aorta', 0)."""
    slices = {('aorta', 0): list(tree.discretized_aorta)}
    for vessel in ('rca', 'lca'):
        for branch, node in dict(getattr(tree, vessel)).items():
            slices[(vessel, branch)] = list(node.contours)
    return slices


def slice_points(slices: dict[tuple[str, int], list]) -> tuple[list[SliceKey], np.ndarray, np.ndarray]:
    """Every slice key, every contour point (N, 3), and for each point the index of its key."""
    keys: list[SliceKey] = []
    points: list = []
    owners: list[int] = []
    for (vessel, branch), contours in slices.items():
        for j, contour in enumerate(contours):
            pts = contour.points_as_tuples()
            points.extend(pts)
            owners.extend([len(keys)] * len(pts))
            keys.append((vessel, branch, j))
    return keys, np.asarray(points, dtype=float).reshape(-1, 3), np.asarray(owners, dtype=int)


def dropped_slices(slices: dict, spec: CutSpec, cap: trimesh.Trimesh, centerline, cut_centerline) -> set[SliceKey]:
    """Slices a cut took away: the cut slice and the rest of its branch on the dropped side,
    plus every slice of the branches it removed from the centerline. Which side was dropped
    comes from the cap, whose normals point out of the kept mesh."""
    contours = slices[(spec.vessel, spec.branch)]
    centre = np.asarray(cap.vertices[-1])  # the cap's vertices are the ring, then its centre
    normal = np.asarray(cap.face_normals).sum(axis=0)

    def beyond(j: int) -> bool:
        return float(np.dot(np.asarray(contours[j].centroid) - centre, normal)) > 0

    idx = spec.slice_idx
    if idx + 1 < len(contours):
        towards_end = beyond(idx + 1)
    else:
        towards_end = idx == 0 or not beyond(idx - 1)
    span = range(idx, len(contours)) if towards_end else range(idx + 1)
    dropped = {(spec.vessel, spec.branch, j) for j in span}
    for branch in set(centerline.branch_ids()) - set(cut_centerline.branch_ids()):
        dropped |= {(spec.vessel, branch, j) for j in range(len(slices.get((spec.vessel, branch), [])))}
    return dropped


def apply_cuts(start: CutResult, slices: dict, tree, specs: list[CutSpec], first: int = 0) -> CutResult:
    """Replay specs[first:] on a copy of `start`. A spec that fails (slice already cut off,
    or no clean ring on this mesh) is recorded in `failed` and skipped, the others still apply."""
    out = dataclasses.replace(
        start,
        centerlines=dict(start.centerlines),
        caps=dict(start.caps),
        dropped=set(start.dropped),
        failed=dict(start.failed),
    )
    results = {'mesh': start.mesh}  # no labels: the cut only needs the geometry
    for i, spec in enumerate(specs[first:], start=first):
        if (spec.vessel, spec.branch, spec.slice_idx) in out.dropped:
            out.failed[i] = 'its slice was cut off by an earlier cut'
            continue
        centerline = out.centerlines.get(spec.vessel)
        if centerline is None:
            out.failed[i] = f'no {spec.vessel.upper()} centerline'
            continue
        try:
            results, cut_centerline, cap = mm_api.run_cut_at_slice(
                results, tree, centerline, spec.vessel, spec.slice_idx, branch_id=spec.branch, keep=spec.keep
            )
        except Exception as e:
            logger.warning(f'Cut {spec.name!r} not applied: {e}')
            out.failed[i] = str(e)
            continue
        out.dropped |= dropped_slices(slices, spec, cap, centerline, cut_centerline)
        out.centerlines[spec.vessel] = cut_centerline
        out.caps[i] = cap
    out.mesh = results['mesh']
    return out


def mesh_fingerprint(mesh: trimesh.Trimesh) -> tuple:
    """Changes whenever the mesh does, also when smoothing moved its vertices in place."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(mesh.vertices).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces).tobytes())
    return len(mesh.vertices), len(mesh.faces), digest.hexdigest()


def file_slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_').lower() or 'cap'


def export_cut_geometry(result: CutResult, specs: list[CutSpec], path: str) -> list[str]:
    """`path` names the cut mesh (`<stem>.stl`, watertight with its caps). Next to it go one
    `<stem>_<cap name>.stl` per cap and `<stem>_centerline_<vessel>.vtp` per centerline,
    cut where a cut hit that vessel. Returns the files written."""
    stem = os.path.splitext(path)[0]
    written = [f'{stem}.stl']
    geo.export_mesh(result.mesh, written[0])
    used: set[str] = set()
    for i, cap in sorted(result.caps.items()):
        slug = base = file_slug(specs[i].name)
        n = 2
        while slug in used:
            slug, n = f'{base}_{n}', n + 1
        used.add(slug)
        written.append(f'{stem}_{slug}.stl')
        geo.export_mesh(cap, written[-1])
    for vessel, centerline in result.centerlines.items():
        written.append(f'{stem}_centerline_{vessel}.vtp')
        mm_api.write_centerline_vtp(centerline, written[-1])
    return written
