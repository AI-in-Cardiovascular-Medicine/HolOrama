"""Thin wrappers around module-level ``multimodars`` functions for the fusion pipeline.

Keeps multimodars API changes out of the GUI code. PyCenterline methods are called
directly in page.py. ``control_plot``/``debug_plot`` are always False: their own
matplotlib/trimesh windows would clash with the VTK viewer, which draws the same layers.
"""

from __future__ import annotations

import os

from typing import Any

import numpy as np
import trimesh
import multimodars as mm
from loguru import logger

from fusion_pipeline.annotations import read_case_structures


def load_centerline(path: str, name: str) -> Any:
    return mm.load_centerline(path, name)  # type: ignore[attr-defined]


def prepare_centerline(
    centerline: Any,
    *,
    ref_centerline: Any | None = None,
    spacing_mm: float | None = None,
    branch_spacing_tolerance: float = 2.0,
    rm_start_mm: float = 0.0,
    smooth_sigma: float = 2.5,
) -> Any:
    """Prepare one centerline (branch, trim, resample, orient, smooth).

    Pass the prepared aorta as ``ref_centerline`` for RCA/LCA, ``None`` for the aorta itself.
    """
    return mm.prepare_centerline(  # type: ignore[attr-defined]
        centerline,
        ref_centerline=ref_centerline,
        spacing_mm=spacing_mm,
        branch_spacing_tolerance=branch_spacing_tolerance,
        rm_start_mm=rm_start_mm,
        smooth_sigma=smooth_sigma,
    )


def run_label_geometry(
    path_ccta_geometry: str | trimesh.Trimesh,
    centerline_aorta,
    centerline_rca,
    centerline_lca,
    *,
    acute_takeoff_rca: bool = False,
    acute_takeoff_lca: bool = False,
    range_mm_takeoff_rca: float = 60.0,
    range_mm_takeoff_lca: float = 60.0,
    step_size_mm: float = 1.0,
    bounding_sphere_radius_mm_rca: float = 3.0,
    bounding_sphere_radius_mm_lca: float = 3.0,
) -> dict:
    """Centerlines must already be prepared (see ``prepare_centerline``)."""
    return mm.label_geometry(  # type: ignore[call-arg, return-value]
        path_ccta_geometry=path_ccta_geometry,
        centerline_aorta=centerline_aorta,
        centerline_rca=centerline_rca,
        centerline_lca=centerline_lca,
        acute_takeoff_rca=acute_takeoff_rca,
        acute_takeoff_lca=acute_takeoff_lca,
        range_mm_takeoff_rca=range_mm_takeoff_rca,
        range_mm_takeoff_lca=range_mm_takeoff_lca,
        step_size_mm=step_size_mm,
        bounding_sphere_radius_mm_rca=bounding_sphere_radius_mm_rca,
        bounding_sphere_radius_mm_lca=bounding_sphere_radius_mm_lca,
        control_plot=False,
    )


def run_label_branches_pair(
    rca_cl,
    lca_cl,
    results: dict,
    *,
    bounding_sphere_radius_mm_rca: float = 3.0,
    bounding_sphere_radius_mm_lca: float = 3.0,
) -> dict:
    """Set results['rca_branches'] / ['lca_branches'] (branch id -> vertices).

    Calls mm.label_branches per vessel to expose the sphere radius. Rerun after any
    split_branch/merge_branches edit, since those reassign branch ids."""
    results = mm.label_branches(rca_cl, results, vessel='rca', bounding_sphere_radius_mm=bounding_sphere_radius_mm_rca)
    return mm.label_branches(lca_cl, results, vessel='lca', bounding_sphere_radius_mm=bounding_sphere_radius_mm_lca)


def run_remove_branch(results: dict, centerline, branch_id: int, *, fill_hole: bool = True) -> tuple[dict, Any]:
    """Cut branch `branch_id` and its sub-branches off the mesh and `centerline`, optionally
    closing the ostium. Returns (results, centerline). Other branches keep their ids."""
    return mm.remove_branch(results, centerline, branch_id, fill_hole=fill_hole)


def run_discretize_vessel_tree(
    ao_cl,
    rca_cl,
    lca_cl,
    results: dict,
    *,
    step_size: float = 1.0,
    n_points: int = 100,
) -> Any:
    return mm.discretize_vessel_tree(
        ao_cl,
        rca_cl,
        lca_cl,
        results,
        step_size=step_size,
        n_points=n_points,
        control_plot=False,
    )


def run_from_array(
    input_path: str,
    labels: str,
    *,
    step_rotation_deg: float = 0.1,
    sample_size: int = 200,
    n_points: int = 20,
    output_path: str = 'output/single',
    watertight: bool = False,
    write_obj: bool = False,
    smooth: bool = True,
    reference_frame: int | None = None,
) -> tuple[Any, Any]:
    """OCT: the tagged frames. `reference_frame` is the master reference (frame as in the CSV
    files), None for multimodars' default (the most proximal)."""
    # Tab-delimited, as written by the report exporter. Contours [frame, x, y, z], references
    # [frame, x, y, z, auto], structures [frame, x, y, z, contour, closed].
    oct_raw = np.genfromtxt(os.path.join(input_path, "tagged_contours.csv"), delimiter='\t')
    oct_ref = np.atleast_2d(np.genfromtxt(os.path.join(input_path, "tagged_reference_points.csv"), delimiter='\t'))
    eem_path = os.path.join(input_path, "eem_tagged_contours.csv")
    oct_eem = np.genfromtxt(eem_path, delimiter='\t') if os.path.isfile(eem_path) else None

    oct_input = mm.numpy_to_inputdata(
        lumen_arr=oct_raw,
        reference_points=oct_ref,
        diastole=True,
        eem_arr=oct_eem,
        label=labels,
        structures=read_case_structures(input_path, 'tagged'),
        master_reference=reference_frame,
    )

    return mm.from_array_single(
        input_data=oct_input,
        step_rotation_deg=step_rotation_deg,
        sample_size=sample_size,
        image_center=(5.0, 5.0),
        n_points=n_points,
        output_path=output_path,
        watertight=watertight,
        write_obj=write_obj,
        smooth=smooth,
    )


def run_from_file_single(
    input_path: str,
    label: str,
    *,
    diastole: bool = True,
    step_rotation_deg: float = 0.1,
    sample_size: int = 200,
    n_points: int = 20,
    output_path: str = 'output/single',
    watertight: bool = False,
    write_obj: bool = False,
    smooth: bool = True,
    reference_frame: int | None = None,
) -> tuple[Any, Any]:
    """One IVUS phase of a case folder (`diastole` picks which). `reference_frame` is its master
    reference, None for the most proximal."""
    return mm.from_file_single(
        input_path=input_path,
        labels=[label],
        diastole=diastole,
        step_rotation_deg=step_rotation_deg,
        sample_size=sample_size,
        n_points=n_points,
        output_path=output_path,
        watertight=watertight,
        write_obj=write_obj,
        smooth=smooth,
        master_references=[reference_frame],
    )


def run_from_file_singlepair(
    input_path: str,
    labels: list[str],
    *,
    step_rotation_deg: float = 0.1,
    sample_size: int = 200,
    n_points: int = 20,
    output_path: str = 'output/singlepair',
    watertight: bool = False,
    write_obj: bool = False,
    smooth: bool = True,
    reference_frames: tuple[int | None, int | None] = (None, None),
) -> tuple[Any, tuple[Any, Any]]:
    """Diastole and systole of a case folder as a pair, aligned to each other around their master
    references `reference_frames` (diastole, systole). Pick the same landmark in both."""
    return mm.from_file_singlepair(
        input_path=input_path,
        labels=labels,
        step_rotation_deg=step_rotation_deg,
        sample_size=sample_size,
        n_points=n_points,
        output_path=output_path,
        watertight=watertight,
        write_obj=write_obj,
        smooth=smooth,
        master_references=list(reference_frames),
    )


def run_align_combined(
    centerline,
    geometry,
    main_ref_pt: tuple[float, float, float],
    counterclockwise_ref_pt: tuple[float, float, float],
    clockwise_ref_pt: tuple[float, float, float],
    points: list[tuple[float, float, float]],
    *,
    angle_step_deg: float = 1.0,
    angle_range_deg: float = 15.0,
    index_range: int = 2,
    write: bool = False,
    watertight: bool = True,
    output_dir: str = 'output/aligned',
    align_wall_anomalous: bool = False,
    reference_frame: int | None = None,
) -> tuple[Any, float, float]:
    """Returns (aligned_geometry, spacing_mm, total_rotation_deg).

    `reference_frame` is the reference of the primary geometry placed on main_ref_pt (frame as in
    the CSV files), None for the one the pullback was loaded with.

    Use `centerline` as-is afterwards. Resampling it to spacing_mm anchors at index 0, not
    at main_ref_pt, so it would not match the grid the frames were placed on."""
    return mm.align_combined(
        centerline,
        _reanchored(geometry, reference_frame),
        main_ref_pt,
        counterclockwise_ref_pt,
        clockwise_ref_pt,
        points,
        angle_step_deg=angle_step_deg,
        angle_range_deg=angle_range_deg,
        index_range=index_range,
        write=write,
        watertight=watertight,
        output_dir=output_dir,
        align_wall_anomalous=align_wall_anomalous,
    )


def run_align_manual(
    centerline,
    geometry,
    rotation_angle_deg: float,
    ref_point: tuple[float, float, float],
    *,
    watertight: bool = True,
    align_wall_anomalous: bool = False,
    reference_frame: int | None = None,
) -> tuple[Any, float, float]:
    """Only works for elliptic vessels (anomalous coronaries).

    Returns (aligned_geometry, spacing_mm, total_rotation_deg), see run_align_combined."""
    return mm.align_manual(
        centerline,
        _reanchored(geometry, reference_frame),
        rotation_angle_deg,
        ref_point,
        write=False,
        watertight=watertight,
        align_wall_anomalous=align_wall_anomalous,
    )


def _reanchored(geometry, reference_frame: int | None):
    """`geometry` with the reference at `reference_frame` as its master, the one placed on the
    centerline, unchanged for None. A pair gets it on diastole and is realigned to systole around it."""
    if reference_frame is None:
        return geometry
    if hasattr(geometry, 'set_master_references'):
        geometry.set_master_references(master_a=reference_frame)
    else:
        geometry.master_reference = reference_frame
    return geometry


def run_label_anomalous_region(centerline, frames, results: dict, *, branch_id: int = 0) -> dict:
    """Split the pullback path of `branch_id` (the branch the frames were aligned onto)
    into proximal/anomalous/distal. The vessel's other vertices go to 'off_path_points'."""
    return mm.label_anomalous_region(
        centerline=centerline, frames=frames, results=results, branch_id=branch_id, debug_plot=False
    )


def run_find_scalings(
    frames, centerline_vessel, centerline_aorta, results: dict, *, vessel: str = 'rca'
) -> dict[str, float]:
    """Return proximal, distal and aortic scaling for `vessel` ('rca' or 'lca').

    mm.find_aorta_scaling always reads results['rca_removed_points'], so for LCA it gets a
    copy with lca_removed_points under that key.
    """
    prox, distal = mm.find_distal_and_proximal_scaling(frames=frames, centerline=centerline_vessel, results=results)
    aortic_results = (
        results if vessel == 'rca' else {**results, 'rca_removed_points': results.get('lca_removed_points', [])}
    )
    if not aortic_results.get('rca_removed_points'):
        # No removed points to scale (e.g. no acute takeoff).
        logger.warning(f'No {vessel.upper()} removed points — defaulting aortic_scaling to 0.0')
        return {'proximal_scaling': prox, 'distal_scaling': distal, 'aortic_scaling': 0.0}
    try:
        # ValueError when no frame has aortic wall data (e.g. not an anomalous case).
        aortic = mm.find_aorta_scaling(frames=frames, cl_aorta=centerline_aorta, results=aortic_results)
    except ValueError as e:
        logger.warning(f'No aortic wall reference in frames — defaulting aortic_scaling to 0.0 ({e})')
        aortic = 0.0
    return {
        'proximal_scaling': prox,
        'distal_scaling': distal,
        'aortic_scaling': aortic,
    }


def run_scale_region(
    mesh: trimesh.Trimesh,
    region_points: list,
    centerline,
    radius_adjustment_mm: float,
    branch_points: dict | None = None,
) -> trimesh.Trimesh:
    """Points move radially by radius_adjustment_mm, so the diameter changes by twice that.
    With `branch_points` (branch id -> vertices) each labelled vertex follows its own branch."""
    return mm.scale_region_centerline_morphing(
        mesh=mesh,
        region_points=region_points,
        centerline=centerline,
        radius_adjustment_mm=radius_adjustment_mm,
        branch_points=branch_points,
    )


def run_sync_results_to_mesh(results: dict, old_mesh: trimesh.Trimesh, new_mesh: trimesh.Trimesh) -> dict:
    return mm.sync_results_to_mesh(results, old_mesh, new_mesh)


def run_sync_tree_to_mesh(tree: Any, old_mesh: trimesh.Trimesh, new_mesh: trimesh.Trimesh) -> Any:
    """Move a discretized tree's contours, ostium and reference points with the mesh (in place)."""
    return mm.sync_tree_to_mesh(tree, old_mesh, new_mesh)


def run_remove_labeled_points(results: dict, region_keys: list[str] | str) -> dict:
    return mm.remove_labeled_points_from_mesh(results, region_keys, target_boundaries=2)


def run_stitch(
    iv_mesh,
    mesh: trimesh.Trimesh,
    results: dict,
    *,
    prox_start_mode: str = 'nearest_iv',
    dist_start_mode: str = 'nearest_iv',
    clamp_overshoot: float = 0.5,
    fillet_bulge: float = 1.0,
    fillet_layers: int = 2,
    seam_points_a: int = 2,
    seam_points_b: int = 4,
    tree: Any | None = None,
    centerlines: dict[str, Any] | None = None,
    branch_cut_slices: dict[tuple[str, int], int] | None = None,
    proximal_cut_slice: int | None = None,
    distal_cut_slice: int | None = None,
) -> dict:
    """With `tree` (the vessel tree before removal) and the vessel `centerlines`, the end rims
    are cut flat at tree slices and side branches leaving the removed segment are reconnected
    to the intravascular wall. The *_cut_slice(s) choose those slices (see run_stitch_cut_targets)."""
    return mm.stitch_ccta_to_intravascular(
        iv_mesh,
        mesh,
        results,
        prox_start_mode=prox_start_mode,
        dist_start_mode=dist_start_mode,
        clamp_overshoot=clamp_overshoot,
        fillet_bulge=fillet_bulge,
        fillet_layers=fillet_layers,
        seam_points_a=seam_points_a,
        seam_points_b=seam_points_b,
        tree=tree,
        centerlines=centerlines,
        branch_cut_slices=branch_cut_slices,
        proximal_cut_slice=proximal_cut_slice,
        distal_cut_slice=distal_cut_slice,
    )


def run_stitch_cut_targets(iv_geometry, results: dict, tree: Any) -> dict:
    """What run_stitch with `tree` would cut: 'distal' and 'proximal' (None for the ostium) as
    (vessel, branch, default slice), 'side_branches' as [(vessel, branch)] (default slice 0)."""
    return mm.stitch_cut_targets(iv_geometry, results, tree)


def run_remesh(
    mesh: trimesh.Trimesh,
    *,
    target_edge_length_mm: float | None = None,
    remesh_iterations: int = 10,
    verbose: bool = False,
) -> trimesh.Trimesh:
    return mm.fix_and_remesh_stitched_mesh(
        mesh, target_edge_length_mm=target_edge_length_mm, remesh_iterations=remesh_iterations, verbose=verbose
    )


def run_cut_at_slice(
    results: dict, tree: Any, centerline: Any, vessel: str, slice_idx: int, *, branch_id: int = 0, keep: str = 'larger'
) -> tuple[dict, Any, trimesh.Trimesh]:
    """Cut results['mesh'] (any mesh, labels not needed) at a tree slice's plane.
    Returns (results, cut centerline, cap)."""
    return mm.cut_at_slice(results, tree, centerline, vessel, slice_idx, branch_id=branch_id, keep=keep)


def write_centerline_vtp(centerline: Any, path: str) -> None:
    mm.write_centerline_vtp(centerline, path)
