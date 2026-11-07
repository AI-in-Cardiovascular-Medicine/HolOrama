"""Morphing the labelled CCTA mesh by the radius adjustments found from the intravascular data.

The adjustments are radial: every point moves away from (or towards) its centerline by the given
millimetres, so a vessel's diameter changes by twice the value."""

from __future__ import annotations

from typing import Any

from fusion_pipeline import multimodars_api as mm_api

OPPOSITE_VESSEL = {'rca': 'lca', 'lca': 'rca'}


def scale_region(results: dict, region: list, centerline, adjustment_mm: float) -> dict:
    """Morph `region` of the mesh around `centerline` and move every label onto the new mesh."""
    mesh = results['mesh']
    scaled = mm_api.run_scale_region(mesh, region, centerline, adjustment_mm)
    return mm_api.run_sync_results_to_mesh(results, mesh, scaled)


def apply_scaling(
    results: dict,
    vessel: str,
    scaling: dict[str, float],
    *,
    centerline,
    centerline_aorta,
    opposite_centerline: Any | None,
    tree: Any | None = None,
) -> dict:
    """Scale the distal, aortic and proximal regions of `vessel`, in that order, then the remaining
    vessels (no intravascular data, only when remaining_vessels_scaling is set).

    `centerline` is the whole vessel, not the aligned branch: the regions include side branches,
    each morphed around its own branch. The remaining vessels are the opposite coronary and this
    vessel's off_path_points, each around its own centerline.

    `tree`, the discretized vessel tree, is moved onto the scaled mesh in place once every region
    is done (only then, so a failed scaling leaves it untouched). Otherwise the stitch would cut
    and reconnect side branches at their unscaled size, and the inlet/outlet cuts would use
    unscaled slices."""
    original_mesh = results['mesh']
    results = scale_region(results, results['distal_points'], centerline, scaling['distal_scaling'])
    aortic_region = results['aorta_points'] + results[f'{vessel}_removed_points']
    results = scale_region(results, aortic_region, centerline_aorta, scaling['aortic_scaling'])
    results = scale_region(results, results['proximal_points'], centerline, scaling['proximal_scaling'])

    remaining = scaling['remaining_vessels_scaling']
    if remaining != 0.0:
        if opposite_centerline is None:
            raise ValueError(f'Remaining-vessels scaling needs the {OPPOSITE_VESSEL[vessel].upper()} centerline.')
        for key, region_centerline in (
            (f'{OPPOSITE_VESSEL[vessel]}_points', opposite_centerline),
            ('off_path_points', centerline),
        ):
            region = results.get(key) or []
            if region:
                results = scale_region(results, region, region_centerline, remaining)

    if tree is not None:
        # Same vertices in the same order, only moved: one sync from the original mesh covers all regions.
        mm_api.run_sync_tree_to_mesh(tree, original_mesh, results['mesh'])
    return results
