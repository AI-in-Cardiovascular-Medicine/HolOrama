"""An in-wall structure's cloud (a jittered ring per frame) becomes one closed mesh following its frames."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fusion_pipeline.structure_mesh import structure_mesh

SPACING = 0.5  # mm between frames
RADIUS = 1.0  # of the drawn calcium contour


def _ring(cx, cy, r, z, n=200, jitter=0.03, seed=0):
    """A contour of radius `r` plus jittered copies, the way multimodars builds a structure cloud."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)])
    copies = ring + rng.normal(0, jitter, ring.shape) * [1, 1, 0]
    return np.vstack([ring, copies])


def _point(p):
    return SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]))


def _geometry(n_frames=5, calcium=(1, 2, 3), transform=lambda p: p, displaced=None):
    """Frames along z with a lumen of radius 3 round the axis and a calcium ring at (5, 0) on `calcium`.
    `displaced` is a frame shifted 20 mm aside, as a frame the alignment left in place."""
    frames = []
    for i in range(n_frames):
        z = i * SPACING
        shift = np.array([20.0, 0, 0]) if i == displaced else np.zeros(3)
        a = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        lumen = np.column_stack([3 * np.cos(a), 3 * np.sin(a), np.full(60, z)]) + shift
        structures = []
        if i in calcium:
            cloud = _ring(5, 0, RADIUS, z, seed=i) + shift
            structures.append(SimpleNamespace(kind='calcification', points=[_point(p) for p in transform(cloud)]))
        frames.append(
            SimpleNamespace(
                lumen=SimpleNamespace(points=[_point(p) for p in transform(lumen)]),
                centroid=tuple(transform((np.array([[0, 0, z]]) + shift))[0]),
                structures=structures,
            )
        )
    return SimpleNamespace(frames=frames)


def test_a_structure_on_neighbouring_frames_is_one_closed_mesh_of_its_volume():
    mesh = structure_mesh(_geometry(), 'calcification')
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1
    # a disc of radius 1 over three frames, each reaching half a spacing beyond its plane at the ends
    expected = np.pi * RADIUS**2 * 3 * SPACING
    assert mesh.volume == pytest.approx(expected, rel=0.15)
    assert mesh.bounds[:, 2] == pytest.approx([SPACING / 2, 3.5 * SPACING], abs=0.15)


def test_the_mesh_follows_its_frames_rigidly():
    rotation = Rotation.from_euler('xyz', [35, -20, 60], degrees=True)
    shift = np.array([12.0, -3.0, 40.0])
    still = structure_mesh(_geometry(), 'calcification')
    moved = structure_mesh(_geometry(transform=lambda p: rotation.apply(p) + shift), 'calcification')
    assert moved.volume == pytest.approx(still.volume, rel=0.05)
    assert moved.centroid == pytest.approx(rotation.apply(np.array(still.centroid)) + shift, abs=0.05)


def test_frames_far_apart_are_not_bridged():
    mesh = structure_mesh(_geometry(displaced=2), 'calcification')
    assert len(mesh.split(only_watertight=False)) == 3  # frame 2 sits 20 mm away, no sheet towards it


def test_no_structure_no_mesh():
    assert structure_mesh(_geometry(calcium=()), 'calcification') is None
