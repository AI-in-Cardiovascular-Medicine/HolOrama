"""Sphere-brush local smoothing / remeshing of a triangle mesh, for VTK 3D views.

Used by Fusion's final-mesh brush (pages/fusion/left_half/display_results.py +
FusionPage._on_sphere_*), and meant to be reusable wherever a surface mesh needs
touching up locally (e.g. CCTA's cut geometry before centerline extraction). Same split
as tools.lasso: this module owns the mesh math and the brush's preview actors/picking,
while the Qt mouse-event wiring (hover, click, Ctrl+wheel) and what a click does
(undo stack, which mesh gets replaced) stay with the caller.
"""

import numpy as np
import pymeshlab
import scipy.sparse
import scipy.sparse.csgraph
import trimesh
from scipy.spatial import cKDTree
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkCellPicker, vtkPolyDataMapper, vtkRenderer

# ----------------------------------------------------------------------
# Mesh math (no VTK/Qt)
# ----------------------------------------------------------------------


def local_smooth_region(mesh: trimesh.Trimesh, center, radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertices a sphere brush at `center` acts on, and their falloff weights.

    Not simply every vertex within `radius_mm` (straight-line): on a vessel no wider than
    the brush that would also grab the opposite wall — and so would "in-sphere and
    edge-connected", since the whole ring around a thin tube is both. Instead distance is
    measured along the surface (shortest edge path from the vertex nearest `center`), so
    the brush is the sphere projected onto the clicked surface patch: the opposite wall
    of a radius-r vessel is ~pi*r away along the surface, not 2r.

    Weights fall off smoothly as (1 - (d/r)^2)^2 with that surface distance d, so the
    smoothed patch blends into the untouched surface instead of leaving a step at its rim.
    Returns (vertex_indices, weights); both empty if no vertex lies inside the sphere.
    """
    empty = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if radius_mm <= 0.0 or len(mesh.vertices) == 0:
        return empty
    seed_dist, seed = mesh.kdtree.query(center)
    if seed_dist > radius_mm:
        return empty
    # Surface distance >= straight-line distance, so the ball is a safe superset to run
    # the shortest-path search on, keeps it local instead of over the whole mesh.
    inside = np.asarray(mesh.kdtree.query_ball_point(center, radius_mm), dtype=np.int64)
    local = np.full(len(mesh.vertices), -1, dtype=np.int64)
    local[inside] = np.arange(len(inside))
    edge_mask = (local[mesh.edges_unique] >= 0).all(axis=1)
    edges = local[mesh.edges_unique[edge_mask]]
    graph = scipy.sparse.coo_matrix(
        (mesh.edges_unique_length[edge_mask], (edges[:, 0], edges[:, 1])), shape=(len(inside), len(inside))
    )
    dist = seed_dist + scipy.sparse.csgraph.dijkstra(
        graph, directed=False, indices=local[seed], limit=radius_mm - seed_dist
    )
    keep = dist <= radius_mm
    weights = (1.0 - (dist[keep] / radius_mm) ** 2) ** 2
    return inside[keep], weights


def local_smooth(
    mesh: trimesh.Trimesh, center, radius_mm: float, *, iterations: int = 10, lamb: float = 0.6
) -> trimesh.Trimesh:
    """Taubin-smooth only the sphere-brush region around `center` (see
    local_smooth_region). Returns a new mesh — `mesh` itself is left untouched, so the
    caller can keep it for undo. Faces never change, only vertex positions.

    Each iteration is one shrink (lamb) + one inflate (-nu) step. Unlike trimesh's
    filter_taubin default (nu=0.5 regardless of lamb), nu is derived from lamb to meet
    Taubin's no-shrink condition 0 < 1/lamb - 1/nu < 0.1, so repeated brushing doesn't
    slowly narrow the lumen.
    """
    region, weights = local_smooth_region(mesh, center, radius_mm)
    out = mesh.copy()
    if len(region) == 0 or iterations <= 0 or lamb <= 0.0:
        return out
    nu = 1.0 / (1.0 / lamb - 0.05)
    # Only the region's rows are needed: L[region] @ v averages each region vertex's
    # neighbors, which may lie outside the region (those stay fixed and anchor the patch).
    laplacian = trimesh.smoothing.laplacian_calculation(mesh).tocsr()[region]
    w = weights[:, None]
    vertices = mesh.vertices.copy().view(np.ndarray)
    for _ in range(iterations):
        for step in (lamb, -nu):
            vertices[region] += step * w * (laplacian @ vertices - vertices[region])
    out.vertices = vertices
    return out


def local_remesh(
    mesh: trimesh.Trimesh, center, radius_mm: float, target_edge_length_mm: float, *, iterations: int = 5
) -> trimesh.Trimesh:
    """Isotropically remesh only the sphere-brush region around `center` (see
    local_smooth_region) to roughly `target_edge_length_mm` edges: vertices bunched
    closer than that (the dense clusters behind spikes and pinched holes) are collapsed
    and the survivors spread evenly, too-long edges are split. Returns a new mesh with
    new topology — `mesh` itself is left untouched.

    Only faces with all three vertices inside the region are rebuilt (pymeshlab's
    selected-only remesh); the patch rim stays fixed, so the patch joins the untouched
    surface seamlessly and a closed mesh stays closed. Unlike local_smooth there's no
    falloff — the rim itself is the transition.

    pymeshlab only ever sees the patch plus the one ring of faces around it (which pins
    the rim), not the whole mesh — its per-call cost scales with the mesh it's handed,
    so remeshing a 100k-face surface for a 100-face patch took ~3 s instead of ~0.05 s.
    """
    region, _ = local_smooth_region(mesh, center, radius_mm)
    in_region = np.zeros(len(mesh.vertices), dtype=bool)
    in_region[region] = True
    faces = np.asarray(mesh.faces)
    selected = in_region[faces].all(axis=1)
    if not selected.any() or target_edge_length_mm <= 0.0 or iterations <= 0:
        return mesh.copy()

    # Sub-mesh: the selected patch + every face sharing a vertex with it (the pinning ring).
    sub_face_idx = np.flatnonzero(np.isin(faces, faces[selected]).any(axis=1))
    sub_vertex_idx, sub_faces = np.unique(faces[sub_face_idx], return_inverse=True)
    sub_vertices = np.asarray(mesh.vertices, dtype=np.float64)[sub_vertex_idx]

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=sub_vertices,
            face_matrix=sub_faces.reshape(-1, 3).astype(np.int32),
            f_scalar_array=selected[sub_face_idx].astype(np.float64),  # face quality carries the selection in
        )
    )
    ms.compute_selection_by_condition_per_face(condselect='fq > 0')
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations, selectedonly=True, targetlen=pymeshlab.PureValue(target_edge_length_mm)
    )
    out = ms.current_mesh()
    out_vertices = out.vertex_matrix()
    patch_faces = out.face_matrix()[out.face_selection_array()]  # the rebuilt patch; the ring comes back as-is

    # Stitch back: output vertices sitting exactly on an input vertex (the fixed rim, plus
    # any interior vertex that didn't move) reuse its index; the rest are appended.
    dist, nearest = cKDTree(sub_vertices).query(out_vertices)
    reused = dist < 1e-9
    index_map = np.empty(len(out_vertices), dtype=np.int64)
    index_map[reused] = sub_vertex_idx[nearest[reused]]
    index_map[~reused] = len(mesh.vertices) + np.arange(np.count_nonzero(~reused))

    result = trimesh.Trimesh(
        vertices=np.vstack([mesh.vertices, out_vertices[~reused]]),
        faces=np.vstack([faces[~selected], index_map[patch_faces]]),
        process=False,
    )
    result.remove_unreferenced_vertices()  # interior vertices the remesh collapsed away
    return result


# ----------------------------------------------------------------------
# VTK preview + picking
# ----------------------------------------------------------------------


class SphereBrush:
    """The brush's on-screen parts in one renderer: a translucent sphere at the hovered
    surface point, the vertices that would move (highlighted), and a picker that only
    hits a given target actor — so point clouds or other layers drawn on top of the mesh
    never catch the brush. Owns no mouse-event wiring (see module docstring)."""

    def __init__(
        self,
        renderer: vtkRenderer,
        sphere_color: tuple[float, float, float] = (0.3, 0.7, 1.0),
        highlight_color: tuple[float, float, float] = (1.0, 0.55, 0.0),
    ) -> None:
        self._ren = renderer
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._picker.PickFromListOn()

        self._source = vtkSphereSource()
        self._source.SetThetaResolution(32)
        self._source.SetPhiResolution(16)
        sphere_mapper = vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(self._source.GetOutputPort())
        self._sphere_actor = vtkActor()
        self._sphere_actor.SetMapper(sphere_mapper)
        self._sphere_actor.GetProperty().SetColor(*sphere_color)
        self._sphere_actor.GetProperty().SetOpacity(0.2)

        self._highlight_mapper = vtkPolyDataMapper()
        self._highlight_actor = vtkActor()
        self._highlight_actor.SetMapper(self._highlight_mapper)
        self._highlight_actor.GetProperty().SetColor(*highlight_color)
        self._highlight_actor.GetProperty().SetPointSize(4.0)

        for actor in (self._sphere_actor, self._highlight_actor):
            actor.PickableOff()
            actor.VisibilityOff()
            renderer.AddActor(actor)

    @property
    def visible(self) -> bool:
        return bool(self._sphere_actor.GetVisibility())

    def set_radius(self, radius_mm: float) -> None:
        self._source.SetRadius(radius_mm)

    def pick(self, target: vtkActor, sx: int, sy: int) -> tuple[float, float, float] | None:
        """World point on `target` under display pixel (sx, sy) — VTK display coords,
        y from the bottom — or None on a miss. `target` is passed per call rather than
        stored, since re-adding a layer (after each smoothing step) swaps its actor."""
        self._picker.InitializePickList()
        self._picker.AddPickList(target)
        if not self._picker.Pick(sx, sy, 0, self._ren):
            return None
        x, y, z = self._picker.GetPickPosition()
        return x, y, z

    def show_at(self, center) -> None:
        self._source.SetCenter(*center)
        self._sphere_actor.VisibilityOn()

    def set_highlight(self, points: np.ndarray) -> None:
        """Show `points` (N, 3) as the vertices the brush would move; empty hides them."""
        if len(points) == 0:
            self._highlight_actor.VisibilityOff()
            return
        self._highlight_mapper.SetInputData(_vertices_polydata(points))
        self._highlight_actor.VisibilityOn()

    def hide(self) -> None:
        self._sphere_actor.VisibilityOff()
        self._highlight_actor.VisibilityOff()


def _vertices_polydata(points: np.ndarray) -> vtkPolyData:
    n = len(points)
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64)))
    cells = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).ravel()
    verts = vtkCellArray()
    verts.SetCells(n, numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(verts)
    return poly
