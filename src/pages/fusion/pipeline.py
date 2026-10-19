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


def load_ccta_mesh(path: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(path)


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
) -> tuple[Any, Any]:
    # Tab-delimited, as written by the report exporter. Rows: [frame, x, y, z].
    oct_raw = np.genfromtxt(os.path.join(input_path, "tagged_contours.csv"), delimiter='\t')
    oct_ref = np.genfromtxt(os.path.join(input_path, "tagged_reference_points.csv"), delimiter='\t')

    oct_input = mm.numpy_to_inputdata(
        lumen_arr=oct_raw,
        ref_point=oct_ref,
        record=None,
        diastole=True,
        label=labels,
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
) -> tuple[Any, Any]:
    # `diastole` picks which phase of the IVUS CSV folder to read.
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
) -> tuple[Any, tuple[Any, Any]]:
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
) -> tuple[Any, float, float]:
    """Returns (aligned_geometry, spacing_mm, total_rotation_deg).

    Use `centerline` as-is afterwards. Resampling it to spacing_mm anchors at index 0, not
    at main_ref_pt, so it would not match the grid the frames were placed on."""
    return mm.align_combined(
        centerline,
        geometry,
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
) -> tuple[Any, float, float]:
    """Only works for elliptic vessels (anomalous coronaries).

    Returns (aligned_geometry, spacing_mm, total_rotation_deg), see run_align_combined."""
    return mm.align_manual(
        centerline,
        geometry,
        rotation_angle_deg,
        ref_point,
        write=False,
        watertight=watertight,
        align_wall_anomalous=align_wall_anomalous,
    )


def frames_to_mesh(geometry, contour_type: str | None = None) -> trimesh.Trimesh:
    """Loft a tube mesh through a PyGeometry's contours of `contour_type` (lumen if None).

    All contours need the same point count (use geometry.downsample(n) first).
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
    mesh: trimesh.Trimesh, region_points: list, centerline, diameter_adjustment_mm: float
) -> trimesh.Trimesh:
    return mm.scale_region_centerline_morphing(
        mesh=mesh, region_points=region_points, centerline=centerline, diameter_adjustment_mm=diameter_adjustment_mm
    )


def run_sync_results_to_mesh(results: dict, old_mesh: trimesh.Trimesh, new_mesh: trimesh.Trimesh) -> dict:
    return mm.sync_results_to_mesh(results, old_mesh, new_mesh)


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
) -> dict:
    """With `tree` (the vessel tree before removal), side branches leaving the removed
    segment are reconnected to the intravascular wall."""
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
    )


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


def run_taubin_smooth(mesh: trimesh.Trimesh, lamb: float = 0.6) -> trimesh.Trimesh:
    """Mutates and returns ``mesh`` (trimesh.smoothing operates in place)."""
    trimesh.smoothing.filter_taubin(mesh, lamb=lamb)
    return mesh


def export_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    mesh.export(path)


def run_cut_at_slice(
    results: dict, tree: Any, centerline: Any, vessel: str, slice_idx: int, *, branch_id: int = 0, keep: str = 'larger'
) -> tuple[dict, Any, trimesh.Trimesh]:
    """Cut results['mesh'] (any mesh, labels not needed) at a tree slice's plane.
    Returns (results, cut centerline, cap)."""
    return mm.cut_at_slice(results, tree, centerline, vessel, slice_idx, branch_id=branch_id, keep=keep)


def prune_vessel_tree(tree: Any, vessel: str, branch_ids) -> None:
    """Drop deleted branches from a discretized tree in place and recompute its reference
    triplets. Other branches keep their ids and contours."""
    gone = set(branch_ids)
    setattr(tree, vessel, {k: b for k, b in getattr(tree, vessel).items() if k not in gone})
    tree.calculate_ref_pts()


def write_centerline_vtp(centerline: Any, path: str) -> None:
    mm.write_centerline_vtp(centerline, path)
