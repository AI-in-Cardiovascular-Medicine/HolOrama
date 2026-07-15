"""Post-cut geometry: turn the combined LVOT/aorta-top-cut mask into an in-memory
mesh, smooth it, and locate the inlet/outlet cut-plane centroids.

Kept separate from stl_export.py (which only ever writes straight to disk) because
this module keeps the mesh in memory so it can be added as a 3-D layer, smoothed,
and re-inspected before anything is exported.
"""

import numpy as np
import trimesh
from skimage.measure import marching_cubes
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkQuadricDecimation, vtkTriangleFilter


def build_cut_mesh(mask: np.ndarray, voxel_spacing: tuple[float, float, float]) -> trimesh.Trimesh:
    """Run marching cubes on a binary mask and return an in-memory mesh in the same
    world-coordinate convention as CctaViewer3D.voxel_to_world() (x*dx, (Y-1-y)*dy, z*dz).

    Mirrors input_output/output/stl_export.export_stl's vertex transform exactly, so
    this layer lines up with the existing label actors/crosshair with no extra
    registration step. Marching cubes always produces a fully closed (watertight)
    surface — the two cut planes (LVOT, aorta-top) show up as flat capped ends, not
    actual holes, so locating them (find_inlet_outlet_centroids) can't rely on open
    mesh boundaries and instead uses the same plane equations the cut itself used.
    """
    _, dy, _ = voxel_spacing
    padded = np.pad(mask > 0, pad_width=1, mode='constant', constant_values=0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=voxel_spacing)
    verts -= np.array(voxel_spacing)  # undo 1-voxel padding offset

    y_ext = (mask.shape[1] - 1) * dy
    verts = np.column_stack(
        [
            verts[:, 2],
            y_ext - verts[:, 1],
            verts[:, 0],
        ]
    )
    faces = faces[:, [0, 2, 1]]
    # process=False: skimage's marching_cubes already returns consistent, de-duplicated
    # connectivity for a single call (same raw verts/faces export_stl writes straight to
    # disk, unprocessed). trimesh's default process=True runs merge_vertices()/repair
    # passes that are both slow on a mesh this size and, worse, can weld across the two
    # open cut-plane boundaries — leaving find_inlet_outlet_centroids with a mesh that
    # looks fully closed (mesh.outline() finds zero boundary edges).
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def signed_distance_to_plane(points: np.ndarray, anchor: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Signed distance from each point to the plane through `anchor` with the given
    `normal`, in whatever units `points`/`anchor` are in. `normal` need not be unit
    length — divide the result by `np.linalg.norm(normal)` for true distance; the
    raw dot product is enough to test which side of the plane a point falls on.
    Broadcasts over any leading shape: `points` can be a flat (N, 3) vertex list or
    a full (Z, Y, X, 3) voxel-index grid. Shared by page.py's mask-cutting (needs
    only the side/sign) and find_inlet_outlet_centroids below (needs true distance)."""
    return ((points - anchor) * normal).sum(axis=-1)


def smooth_mesh(mesh: trimesh.Trimesh, lamb: float = 0.6) -> trimesh.Trimesh:
    """Mutates and returns ``mesh`` (trimesh.smoothing operates in place)."""
    trimesh.smoothing.filter_taubin(mesh, lamb=lamb)
    return mesh


def reduce_mesh(mesh: trimesh.Trimesh, target_reduction: float) -> trimesh.Trimesh:
    """Decimate ``mesh`` down to roughly ``1 - target_reduction`` of its original face
    count (e.g. target_reduction=0.5 -> ~half the triangles), via VTK's quadric-error
    decimation (vtkQuadricDecimation). VTK is already a hard dependency of this app —
    trimesh's own simplify_quadric_decimation needs the optional fast_simplification
    package (not installed), and pymeshlab isn't installed either, so this avoids
    adding a new dependency just for this one operation.

    Vastly fewer surface triangles is the main lever on vmtkcenterlines' runtime (its
    Voronoi-diagram step scales with surface complexity), so this exists to trade
    surface detail for centerline-computation speed on large cut geometries.
    """
    # triangulate=False: vtkQuadricDecimation accepts general polygons directly, and
    # vtk_polydata_to_mesh triangulates the *output* anyway (decimation can itself
    # leave non-triangular faces), so a pre-pass here would be wasted work.
    poly = mesh_to_vtk_polydata(mesh, triangulate=False)

    decimate = vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.SetTargetReduction(max(0.0, min(0.99, target_reduction)))
    decimate.Update()

    return vtk_polydata_to_mesh(decimate.GetOutput())


def mesh_to_vtk_polydata(mesh: trimesh.Trimesh, triangulate: bool = True) -> vtkPolyData:
    """trimesh.Trimesh -> vtkPolyData (points + polygon cells). Shared by this module
    (mesh reduction) and left_half/cut_geometry_viewer.py (rendering) so the
    conversion only exists in one place. Set triangulate=False to skip the
    vtkTriangleFilter pass when the caller doesn't need strictly-triangular output
    (e.g. decimation input, which accepts general polygons and re-triangulates the
    result on the way back out via vtk_polydata_to_mesh anyway)."""
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64)))

    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    id_array = numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True)
    cell_array = vtkCellArray()
    cell_array.SetCells(len(faces), id_array)

    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cell_array)

    if not triangulate:
        return poly
    tri = vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    return tri.GetOutput()


def vtk_polydata_to_mesh(poly: vtkPolyData) -> trimesh.Trimesh:
    """vtkPolyData -> trimesh.Trimesh. Always triangulates first since sources like
    vtkQuadricDecimation can produce non-triangular polygons."""
    tri = vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    triangulated = tri.GetOutput()

    verts = numpy_support.vtk_to_numpy(triangulated.GetPoints().GetData())
    faces = numpy_support.vtk_to_numpy(triangulated.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def find_inlet_outlet_centroids(
    mesh: trimesh.Trimesh,
    voxel_spacing: tuple[float, float, float],
    mask_shape: tuple[int, int, int],
    lvot_anchor: np.ndarray,
    lvot_normal: np.ndarray,
    aorta_anchor: np.ndarray,
    aorta_normal: np.ndarray,
    tol_voxels: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the inlet (LVOT cut plane) and outlet (aorta-top cut plane) centroids
    directly from the same plane equations _compute_combined_mask used to cut the
    mask, rather than trying to detect an open mesh boundary — marching cubes always
    produces a closed, watertight surface (the cut shows up as a flat cap, not a
    hole), so there's no boundary to find at all.

    ``lvot_anchor``/``lvot_normal``/``aorta_anchor``/``aorta_normal`` are in the same
    voxel-index (z, y, x) space as page.py's cut-plane math. Mesh vertices (world mm,
    from build_cut_mesh) are converted back to that voxel-index space — the exact
    algebraic inverse of the point transform build_cut_mesh applies — then the
    identical plane-distance formula from _compute_combined_mask picks out whichever
    vertices sit within ``tol_voxels`` of each plane. Works the same right after
    cutting or after smoothing (vertex positions shift slightly, so re-running this
    after smoothing naturally tracks that).
    """
    dz, dy, dx = voxel_spacing
    _, Y, _ = mask_shape
    verts = mesh.vertices

    # world -> voxel index (inverse of build_cut_mesh's vertex transform)
    vx = verts[:, 0] / dx
    vy = (Y - 1) - verts[:, 1] / dy
    vz = verts[:, 2] / dz
    coords = np.stack([vz, vy, vx], axis=-1)  # (N, 3) in (z, y, x) voxel-index units

    def _plane_centroid(anchor: np.ndarray, normal: np.ndarray, name: str) -> np.ndarray:
        dist = signed_distance_to_plane(coords, anchor, normal) / np.linalg.norm(normal)
        near = np.abs(dist) < tol_voxels
        if not near.any():
            raise ValueError(f'No mesh vertices found near the {name} cut plane. Check the cut lines.')
        return verts[near].mean(axis=0)

    inlet = _plane_centroid(lvot_anchor, lvot_normal, 'LVOT')
    outlet = _plane_centroid(aorta_anchor, aorta_normal, 'aorta-top')
    return inlet, outlet
