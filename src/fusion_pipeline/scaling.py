"""Morphing the labelled CCTA mesh by the diameter adjustments found from the intravascular data."""

from __future__ import annotations

from typing import Any

from fusion_pipeline import multimodars_api as mm_api

OPPOSITE_VESSEL = {'rca': 'lca', 'lca': 'rca'}


def apply_scaling(
    results: dict,
    vessel: str,
    scaling: dict[str, float],
    *,
    centerline,
    centerline_aorta,
    opposite_centerline: Any | None,
) -> dict:
    """Scale the distal, aortic and proximal regions of `vessel`, in that order, then the remaining
    vessels (no intravascular data, only when remaining_vessels_scaling is set).

    `centerline` is the whole vessel, not the aligned branch: the regions include side branches,
    each vertex morphed around the branch it is labelled with ({vessel}_branches, lasso edits
    included). The remaining vessels are the opposite coronary and this vessel's off_path_points,
    each around its own centerline."""
    branches = f'{vessel}_branches'
    results = scale_region(results, results['distal_points'], centerline, scaling['distal_scaling'], branches)
    aortic_region = results['aorta_points'] + results[f'{vessel}_removed_points']
    results = scale_region(results, aortic_region, centerline_aorta, scaling['aortic_scaling'])
    results = scale_region(results, results['proximal_points'], centerline, scaling['proximal_scaling'], branches)

    remaining = scaling['remaining_vessels_scaling']
    if remaining == 0.0:
        return results
    opposite = OPPOSITE_VESSEL[vessel]
    if opposite_centerline is None:
        raise ValueError(f'Remaining-vessels scaling needs the {opposite.upper()} centerline.')
    for key, region_centerline, branch_key in (
        (f'{opposite}_points', opposite_centerline, f'{opposite}_branches'),
        ('off_path_points', centerline, branches),
    ):
        region = results.get(key) or []
        if region:
            results = scale_region(results, region, region_centerline, remaining, branch_key)
    return results


def scale_region(results: dict, region: list, centerline, adjustment_mm: float, branch_key: str | None = None) -> dict:
    """Morph `region` of the mesh around `centerline` and move every label onto the new mesh.

    With `branch_key` ('rca_branches' / 'lca_branches'), each labelled vertex is morphed around
    its own branch of `centerline` instead of the nearest point of any branch."""
    mesh = results['mesh']
    labels = results.get(branch_key) if branch_key else None
    scaled = mesh
    for part, axis in branch_parts(region, centerline, labels):
        scaled = mm_api.run_scale_region(scaled, part, axis, adjustment_mm)
    return mm_api.run_sync_results_to_mesh(results, mesh, scaled)


def branch_parts(region: list, centerline, labels: dict | None) -> list[tuple[list, Any]]:
    """`region` split by branch label, each part with the centerline of its branch. Vertices
    without a label, or labelled with a branch the centerline lacks, keep the whole centerline."""
    if not labels:
        return [(region, centerline)] if region else []
    left = set(region)
    present = set(centerline.branch_ids())
    parts = []
    for branch in sorted(labels):
        if branch not in present:
            continue
        part = [p for p in labels[branch] if p in left]
        if part:
            parts.append((part, centerline.get_branch(branch)))
            left.difference_update(part)
    rest = [p for p in region if p in left]
    if rest:
        parts.append((rest, centerline))
    return parts
