from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from fusion_pipeline import geometry as geo


def _ring(z: float, n: int = 8, radius: float = 1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = [(radius * np.cos(a), radius * np.sin(a), z) for a in angles]
    return SimpleNamespace(points_as_tuples=lambda: points, centroid=(0.0, 0.0, z))


def _pullback(rings, wall=None):
    return SimpleNamespace(
        get_lumen_contours=lambda: rings,
        get_contours_by_type=lambda kind: wall if kind == 'Wall' else [],
    )


def test_phases_of_a_pair_and_a_single_geometry():
    pair = SimpleNamespace(geom_a='dia', geom_b='sys')
    assert geo.phases(pair) == [('a', 'dia'), ('b', 'sys')]
    assert geo.phases(SimpleNamespace(geom_a='dia', geom_b=None)) == [('a', 'dia')]
    single = SimpleNamespace()
    assert geo.phases(single) == [('a', single)]


def test_primary_geometry():
    assert geo.primary_geometry(None) is None
    assert geo.primary_geometry(SimpleNamespace(geom_a='dia', geom_b='sys')) == 'dia'
    single = SimpleNamespace()
    assert geo.primary_geometry(single) is single


def test_frames_to_mesh_lofts_an_outward_tube():
    mesh = geo.frames_to_mesh(_pullback([_ring(0), _ring(1), _ring(2)]))
    assert len(mesh.vertices) == 3 * 8
    assert len(mesh.faces) == 2 * 8 * 2
    centres = mesh.triangles_center
    radial = centres - np.column_stack([np.zeros(len(centres)), np.zeros(len(centres)), centres[:, 2]])
    assert (np.einsum('ij,ij->i', mesh.face_normals, radial) > 0).all()


def test_frames_to_mesh_uses_the_requested_contour_type():
    mesh = geo.frames_to_mesh(_pullback([_ring(0), _ring(1)], wall=[_ring(0, radius=2), _ring(1, radius=2)]), 'Wall')
    assert np.isclose(np.linalg.norm(mesh.vertices[0, :2]), 2.0)


@pytest.mark.parametrize(
    'rings, message',
    [([_ring(0)], 'at least 2'), ([_ring(0, n=8), _ring(1, n=6)], 'same point count')],
)
def test_frames_to_mesh_rejects_unloftable_input(rings, message):
    with pytest.raises(ValueError, match=message):
        geo.frames_to_mesh(_pullback(rings))


def test_max_seam_points_uses_the_smallest_ring():
    results = {'boundary_points_1': [0] * 40, 'boundary_points_2': [0] * 24, 'boundary_points_x': [0] * 4}
    assert geo.max_seam_points(results) == (24 // 2 - 2) // 2
    assert geo.max_seam_points({}) == 0
    assert geo.max_seam_points({'boundary_points_1': [0] * 4}) == 0


def test_transfer_labels_maps_to_the_nearest_vertex_and_drops_boundaries():
    old = trimesh.Trimesh(vertices=[(0, 0, 0), (10, 0, 0), (0, 10, 0)], faces=[(0, 1, 2)], process=False)
    new = trimesh.Trimesh(vertices=[(0.1, 0, 0), (9.9, 0, 0), (0, 9.9, 0)], faces=[(0, 1, 2)], process=False)
    results = {
        'mesh': old,
        'rca_points': [(0.0, 0.0, 0.0)],
        'rca_branches': {0: [(10.0, 0.0, 0.0)]},
        'boundary_points_1': [(0.0, 10.0, 0.0)],
    }
    labelled = geo.transfer_labels(results, new)
    assert labelled['mesh'] is new
    assert labelled['rca_points'] == [(0.1, 0.0, 0.0)]
    assert labelled['rca_branches'] == {0: [(9.9, 0.0, 0.0)]}
    assert 'boundary_points_1' not in labelled


def test_taubin_smooth_works_in_place():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    assert geo.taubin_smooth(mesh) is mesh


@pytest.mark.parametrize('offset, expected', [(0, (2, 2)), (1, (3, 3)), (-5, (0, -3)), (9, (4, 11))])
def test_shifted_nearest_index_clamps_to_the_ends(offset, expected):
    points = [(float(i), 0.0, 0.0) for i in range(5)]
    assert geo.shifted_nearest_index(points, (2.2, 0.0, 0.0), offset) == expected
