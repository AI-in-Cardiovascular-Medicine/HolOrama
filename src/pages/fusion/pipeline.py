"""Thin wrappers around the ``multimodars`` package for the fusion pipeline.

Each function below corresponds to one button in a right_half column and mirrors
the call signature of the underlying multimodars function as installed
(multimodars>=0.6.0 — see pyproject.toml). Keeping this as a separate module (rather
than calling multimodars directly from the column widgets or page.py) means the GUI
code never has to change if a multimodars upgrade renames or reshapes an argument —
only this file does. Centerline methods (get_branch, split_branch, merge_branches,
find_sharp_angles, orient_by_max_z, ...) are called directly on the PyCenterline
objects in page.py instead, matching how get_branch is already used there — only
module-level ``mm.*`` functions get a wrapper here.

This application only ever loads centerlines from .vtp files (never CSV/array), via
``load_centerline`` below.

None of the wrappers here expose multimodars' own ``control_plot``/``debug_plot``
parameters — they are always passed as False. Those flags pop up the package's own
matplotlib/trimesh scenes, which would fight with our VTK viewer; we recreate the same
visualizations as native VTK layers instead (see page.py's ``_refresh_*_scene`` methods
and left_half/colors.py for the color legend, ported from multimodars/ccta/debug_plots.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
import multimodars as mm


def load_centerline(path: str, name: str) -> Any:
    # multimodars>=0.6.0 API (unreleased — see multimoda-rs branch fix/centerline-workflow).
    # pyproject.toml is still pinned to the last PyPI release (0.5.8), which mypy resolves
    # against and which lacks this function — bump the pin once 0.6.0 ships and drop these.
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
    """Run the load->branch->order->smooth prep pipeline on one centerline.

    ``ref_centerline`` doubles as the "is this a coronary?" signal — pass the
    prepared aorta centerline for RCA/LCA, leave it ``None`` for the aorta itself.
    See ``multimodars.prepare_centerline`` for the full step-by-step docstring.
    """
    # multimodars>=0.6.0 API — see the note in load_centerline() above.
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
    """Centerlines must already be prepared (see ``prepare_centerline``) — label_geometry
    no longer loads or orients them itself."""
    # multimodars>=0.6.0 API — see the note in load_centerline() above. The pinned 0.5.8
    # stub still has the old path_centerline_*/n_points_takeoff_*/(dict, centerlines)-tuple
    # signature, hence the call-arg + return-value mismatches silenced below.
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


def run_label_branches_pair(rca_cl, lca_cl, results: dict) -> dict:
    """Project rca_cl/lca_cl's branch structure onto results' labelled surface points.

    Call after any manual split_branch/merge_branches edit to the centerlines so the
    surface-point branch labels (consumed by discretize_vessel_tree) reflect the edit."""
    # multimodars>=0.6.0 API — see the note in load_centerline() above.
    return mm.label_branches_pair(rca_cl, lca_cl, results, control_plot=False)  # type: ignore[attr-defined]


def run_discretize_vessel_tree(
    ao_cl,
    rca_cl,
    lca_cl,
    results: dict,
    *,
    branch_id_rca: int = 0,
    branch_id_lca: int = 0,
    step_size: float = 1.0,
    n_points: int = 100,
    b_spline: bool = False,
    bspline_smoothing: float = 100.0,
    bspline_degree: int = 3,
) -> Any:
    return mm.discretize_vessel_tree(
        ao_cl,
        rca_cl,
        lca_cl,
        results,
        branch_id_rca=branch_id_rca,
        branch_id_lca=branch_id_lca,
        step_size=step_size,
        n_points=n_points,
        b_spline=b_spline,
        bspline_smoothing=bspline_smoothing,
        bspline_degree=bspline_degree,
        control_plot=False,
    )


def run_from_file_singlepair(
    input_path: str,
    labels: list[str],
    *,
    step_rotation_deg: float = 0.5,
    sample_size: int = 500,
    n_points: int = 20,
    output_path: str = 'output/singlepair',
    watertight: bool = True,
    write_obj: bool = False,
    smooth: bool = True,
) -> tuple[Any, tuple[Any, Any]]:
    # image_center isn't exposed in the UI — always the library default (4.5, 4.5) mm.
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
    angle_range_deg: float = 15.0,
    write: bool = False,
    watertight: bool = True,
    output_dir: str = 'output/aligned',
    align_wall_anomalous: bool = False,
) -> tuple[Any, float, float]:
    """Returns (aligned_geometry, spacing_mm, total_rotation_deg) — multimodars>=0.6.0 no
    longer returns a resampled centerline here. spacing_mm is the arc-length spacing it
    derived from `geometry` internally; call centerline.resample(spacing_mm) yourself if
    you need a centerline at that same spacing (see FusionPage._apply_align_result)."""
    return mm.align_combined(
        centerline,
        geometry,
        main_ref_pt,
        counterclockwise_ref_pt,
        clockwise_ref_pt,
        points,
        angle_range_deg=angle_range_deg,
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
    """Only works for elliptic vessels (anomalous coronaries) — see mm.align_manual.

    Returns (aligned_geometry, spacing_mm, total_rotation_deg) — see run_align_combined's
    docstring above for why there's no resampled centerline in this tuple anymore."""
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
    """Loft a tube mesh through a PyGeometry's contours — a VTK/trimesh port of
    multimodars' own ``_converters.geometry_to_trimesh`` (used internally by
    stitch_ccta_to_intravascular to turn the aligned intravascular geometry into a
    real mesh). contour_type=None uses the lumen; otherwise pass one of the PyContour
    'kind' values ('Eem', 'Calcification', 'Sidebranch', 'Catheter', 'Wall').

    Every contour must have the same point count (call geometry.downsample(n) first
    if they don't) — this is a hard requirement of the quad-strip lofting below, not
    a limitation we added.
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

    # Same inward/outward check multimodars' own geometry_to_trimesh performs: flip
    # every face if the first one points toward contour 0's centroid instead of away.
    face0_center = vertices[np.asarray(faces[0])].mean(axis=0)
    outward = face0_center - np.array(contours[0].centroid, dtype=np.float64)
    if np.dot(mesh.face_normals[0], outward) < 0:
        mesh.invert()
    return mesh


def run_label_anomalous_region(centerline, frames, results: dict, *, results_key: str = 'rca_points') -> dict:
    return mm.label_anomalous_region(
        centerline=centerline, frames=frames, results=results, results_key=results_key, debug_plot=False
    )


def run_find_scalings(frames, centerline_vessel, centerline_aorta, results: dict) -> dict[str, float]:
    """Run all four multimodars scaling lookups and return them by name."""
    prox, distal = mm.find_distal_and_proximal_scaling(frames=frames, centerline=centerline_vessel, results=results)
    aortic = mm.find_aorta_scaling(frames=frames, cl_aorta=centerline_aorta, results=results)
    aortic_wall = mm.find_aortic_wall_scaling(frames=frames, cl_aorta=centerline_aorta, results=results)
    return {
        'proximal_scaling': prox,
        'distal_scaling': distal,
        'aortic_scaling': aortic,
        'aortic_wall_scaling': aortic_wall,
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
    return mm.remove_labeled_points_from_mesh(results, region_keys)


def run_stitch(
    iv_mesh,
    mesh: trimesh.Trimesh,
    results: dict,
    *,
    prox_start_mode: str = 'nearest_iv',
    dist_start_mode: str = 'nearest_iv',
    clamp_overshoot: float = 0.5,
) -> dict:
    return mm.stitch_ccta_to_intravascular(
        iv_mesh,
        mesh,
        results,
        prox_start_mode=prox_start_mode,
        dist_start_mode=dist_start_mode,
        clamp_overshoot=clamp_overshoot,
    )


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
