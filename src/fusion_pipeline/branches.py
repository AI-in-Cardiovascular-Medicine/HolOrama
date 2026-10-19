"""Branch bookkeeping across the discretized vessel tree, the labelled results and the centerlines."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from domain.fusion.runtime_types import BranchRemoval
from fusion_pipeline import geometry as geo
from fusion_pipeline import multimodars_api as mm_api

VESSEL_POINT_KEYS = {'rca_points': 'rca', 'lca_points': 'lca'}


def tree_branches(tree, vessel: str) -> dict:
    """Branch id (0 = main vessel) -> PyDiscretizedBranch. Branches that failed to discretize are missing."""
    if tree is None:
        return {}
    return dict(tree.rca if vessel == 'rca' else tree.lca)


def reference_entries(tree, vessel: str, branch: int) -> list[tuple[int | None, tuple]]:
    """(child id, triplet) for every reference on one branch: its own ostium first (child None),
    then each child's takeoff, proximal to distal."""
    branches = tree_branches(tree, vessel)
    node = branches.get(branch)
    if node is None:
        return []
    entries: list[tuple[int | None, tuple]] = [] if node.ostium is None else [(None, node.ostium)]
    bifurcations = node.bifurcations

    def start_contour(child: int) -> int:
        parent = branches[child].parent if child in branches else None
        return parent[1] if parent is not None else 0

    entries += [(child, bifurcations[child]) for child in sorted(bifurcations, key=start_contour)]
    return entries


def references(tree, vessel: str, branch: int) -> list:
    """The reference triplets of one branch, see reference_entries."""
    return [triplet for _child, triplet in reference_entries(tree, vessel, branch)]


def ids_with_references(tree, vessel: str) -> list[int]:
    """Every discretized branch that has references to align onto ([0] without a tree)."""
    if tree is None:
        return [0]
    return [b for b in sorted(tree_branches(tree, vessel)) if reference_entries(tree, vessel, b)]


def branch_name(names: dict[int, str], vessel: str, branch: int) -> str:
    """The user-given name, else 'RCA main' / 'RCA branch 2'."""
    default = f'{vessel.upper()} main' if branch == 0 else f'{vessel.upper()} branch {branch}'
    return names.get(branch) or default


def move_points(groups: dict, source, dest, inside: np.ndarray) -> list[tuple]:
    """Move the points of groups[source] selected by `inside` to groups[dest], in place.
    Returns the moved points."""
    points = np.array(groups[source])
    moved = [tuple(p) for p in points[inside]]
    groups[source] = [tuple(p) for p in points[~inside]]
    groups[dest] = list(groups.get(dest) or []) + moved
    return moved


def sync_branch_labels(results: dict, moved: list, source: str, dest: str, centerline_of: Callable[[str], Any]) -> None:
    """Keep {vessel}_branches partitioning {vessel}_points after `moved` went from region `source` to
    `dest`: drop them from the source vessel's branches, give each the branch of its nearest point
    on the destination vessel's centerline. In place, so manual branch label edits survive."""
    moved_set = set(moved)
    if source in VESSEL_POINT_KEYS and f'{VESSEL_POINT_KEYS[source]}_branches' in results:
        key = f'{VESSEL_POINT_KEYS[source]}_branches'
        results[key] = {b: [p for p in pts if p not in moved_set] for b, pts in results[key].items()}
    if dest not in VESSEL_POINT_KEYS or f'{VESSEL_POINT_KEYS[dest]}_branches' not in results:
        return
    vessel = VESSEL_POINT_KEYS[dest]
    centerline = centerline_of(vessel)
    if centerline is None:
        return
    cl_points = np.array(centerline.points_as_tuples())
    cl_branch = np.array([p.branch_id for p in centerline.points])
    branches = {b: list(pts) for b, pts in results[f'{vessel}_branches'].items()}
    for point in moved:
        nearest = int(np.argmin(np.linalg.norm(cl_points - np.array(point), axis=1)))
        branches.setdefault(int(cl_branch[nearest]), []).append(point)
    results[f'{vessel}_branches'] = branches


def remove_branch(
    results: dict, centerline, branch: int, *, stitched: dict | None, final_mesh, fill_hole: bool
) -> BranchRemoval:
    """Cut a side branch and its descendants off the labelled mesh and the centerline, and off the
    stitched and final meshes when they exist (the final mesh gets its labels from the stitched one)."""
    new_results, new_centerline = mm_api.run_remove_branch(results, centerline, branch, fill_hole=fill_hole)
    new_stitched = new_final = None
    if final_mesh is not None and stitched is not None:
        labelled = geo.transfer_labels(stitched, final_mesh)
        new_final = mm_api.run_remove_branch(labelled, centerline, branch, fill_hole=fill_hole)[0]['mesh']
    if stitched is not None:
        new_stitched = mm_api.run_remove_branch(stitched, centerline, branch, fill_hole=fill_hole)[0]
    removed = sorted(set(centerline.branch_ids()) - set(new_centerline.branch_ids()))
    return BranchRemoval(new_results, new_centerline, new_stitched, new_final, removed)


def prune_vessel_tree(tree: Any, vessel: str, branch_ids) -> None:
    """Drop deleted branches from a discretized tree in place and recompute its reference
    triplets. Other branches keep their ids and contours."""
    gone = set(branch_ids)
    setattr(tree, vessel, {k: b for k, b in getattr(tree, vessel).items() if k not in gone})
    tree.calculate_ref_pts()


def nearest_reference(tree, point) -> tuple[str, int, int] | None:
    """(vessel, branch, reference index) of the reference triplet with a point closest to `point`."""
    target = np.asarray(point, dtype=float)
    best, best_dist = None, float('inf')
    for vessel in ('rca', 'lca'):
        for branch in ids_with_references(tree, vessel):
            for i, triplet in enumerate(references(tree, vessel, branch)):
                dist = float(np.linalg.norm(np.asarray(triplet, dtype=float) - target, axis=1).min())
                if dist < best_dist:
                    best, best_dist = (vessel, branch, i), dist
    return best
