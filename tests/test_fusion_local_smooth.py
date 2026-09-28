import numpy as np
import pytest
import trimesh

from pages.fusion.pipeline import local_smooth_region, run_local_smooth


@pytest.fixture
def noisy_sphere():
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
    rng = np.random.default_rng(0)
    mesh.vertices = mesh.vertices + rng.normal(0.0, 0.05, mesh.vertices.shape)
    return mesh


def test_region_stays_on_clicked_wall_of_thin_tube():
    # 3 mm brush on a 1.5 mm-radius tube: the opposite wall is only 3 mm away in a
    # straight line, but ~4.7 mm along the surface, so it must not be selected.
    tube = trimesh.creation.cylinder(radius=1.5, height=20.0, sections=64).subdivide().subdivide().subdivide()
    center = np.array([1.5, 0.0, 0.0])
    region, weights = local_smooth_region(tube, center, 3.0)
    assert len(region) > 0
    assert tube.vertices[region][:, 0].min() > -1.0  # opposite wall sits at x = -1.5
    assert len(region) < len(tube.kdtree.query_ball_point(center, 3.0))
    assert np.all((weights >= 0.0) & (weights <= 1.0))


def test_region_empty_when_sphere_misses_surface(noisy_sphere):
    region, weights = local_smooth_region(noisy_sphere, [100.0, 0.0, 0.0], 3.0)
    assert region.size == 0 and weights.size == 0


def test_local_smooth_only_moves_region_and_reduces_noise(noisy_sphere):
    center = noisy_sphere.vertices[0]
    region, _ = local_smooth_region(noisy_sphere, center, 3.0)
    original = noisy_sphere.vertices.copy()

    out = run_local_smooth(noisy_sphere, center, 3.0, iterations=10, lamb=0.6)

    assert np.array_equal(noisy_sphere.vertices, original)  # input left untouched (undo)
    assert np.array_equal(out.faces, noisy_sphere.faces)
    outside = np.ones(len(original), dtype=bool)
    outside[region] = False
    assert np.array_equal(out.vertices[outside], original[outside])

    radii_before = np.linalg.norm(original[region], axis=1)
    radii_after = np.linalg.norm(out.vertices[region], axis=1)
    assert radii_after.std() < radii_before.std()  # smoother
    assert abs(radii_after.mean() - radii_before.mean()) < 0.01  # Taubin: no shrinking
