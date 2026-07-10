"""Thin wrappers around the ``multimodars`` package for the fusion pipeline.

Each function below corresponds to one button in a right_half column and mirrors
the call signature of the underlying multimodars function as installed
(multimodars>=0.5.2 — see pyproject.toml). Keeping this as a separate module (rather
than calling multimodars directly from the column widgets or page.py) means the GUI
code never has to change if a multimodars upgrade renames or reshapes an argument —
only this file does.

This application only ever loads centerlines from .vtp files (never CSV/array), via
``read_centerline_vtp`` below.

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


def read_centerline_vtp(path: str, *, rm_start_mm: float = 5.0, smooth: bool = True, smooth_sigma: float = 2.5) -> Any:
    """Load a centerline .vtp and clean it up (trim side-branch prefixes overlapping the
    main branch, optionally trim the branch-0 inlet, optionally smooth)."""
    cl = mm.read_centerline_vtp(path)
    return cl.cleanup_vtp_data(rm_start_mm=rm_start_mm, smooth=smooth, smooth_sigma=smooth_sigma)


def load_ccta_mesh(path: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(path)


def run_label_geometry(
    path_ccta_geometry: str,
    centerline_aorta,
    centerline_rca,
    centerline_lca,
    *,
    anomalous_rca: bool = False,
    anomalous_lca: bool = False,
    n_points_intramural: int = 120,
    bounding_sphere_radius_mm: float = 3.0,
) -> tuple[dict, tuple[Any, Any, Any]]:
    return mm.label_geometry(
        path_ccta_geometry=path_ccta_geometry,
        path_centerline_aorta=centerline_aorta,
        path_centerline_rca=centerline_rca,
        path_centerline_lca=centerline_lca,
        anomalous_rca=anomalous_rca,
        anomalous_lca=anomalous_lca,
        n_points_intramural=n_points_intramural,
        bounding_sphere_radius_mm=bounding_sphere_radius_mm,
        control_plot=False,
    )


def run_prepare_centerlines(rca_cl, lca_cl, results: dict, *, branch_sigma: float = 2.0) -> tuple[Any, Any, dict]:
    """vtp_data is always True here — branch indices are already set from the .vtp file,
    since that's the only centerline source this app uses, so calculate_branches is skipped."""
    return mm.prepare_centerlines(rca_cl, lca_cl, results, branch_sigma=branch_sigma, vtp_data=True, control_plot=False)


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
    output_path: str = 'output/singlepair',
    watertight: bool = True,
    write_obj: bool = True,
    smooth: bool = True,
) -> tuple[Any, tuple[Any, Any]]:
    return mm.from_file_singlepair(
        input_path=input_path,
        labels=labels,
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
) -> tuple[Any, Any]:
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
