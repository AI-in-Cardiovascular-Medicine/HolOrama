"""Side branches, plaques and references shown in 3D where multimodars moved their frames."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fusion_pipeline import annotations as iv

SPACING = 0.5  # mm between frames
N_FRAMES = 5


def _points(xyz):
    return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in xyz]


def _circle(r, cx, cy, z, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)])


def _annotation(kind, role, group, xyz):
    return SimpleNamespace(kind=kind, role=role, group=group, points=_points(xyz))


def _geometry(transform=lambda p: p):
    """Lumen radius 2 round (0, 0) on every frame, a branch run on frames 1-3 leaving along +x with
    its reference frame 2 (diameter 1.6), calcium on frame 4, a hand-placed and an automatic reference."""
    frames = []
    for i in range(N_FRAMES):
        z = i * SPACING
        annotations = []
        if i in (1, 2, 3):
            # the branch region: lumen border at x = 2 out to the arc at x = 3.5, 1.6 wide
            region = np.array([[2, -0.8, z], [3.5, -0.8, z], [3.5, 0.8, z], [2, 0.8, z]], dtype=float)
            annotations.append(_annotation('sidebranch', 'region', 1, transform(region)))
        if i == 2:
            annotations += [
                _annotation('sidebranch', 'landmark', 1, transform(np.array([[2.75, 0, z]]))),
                _annotation('sidebranch', 'axis_tip', 1, transform(np.array([[3.5, 0, z]]))),
                _annotation('sidebranch', 'diameter_end', 1, transform(np.array([[2.5, -0.8, z], [2.5, 0.8, z]]))),
                _annotation('reference', 'point', 1, transform(np.array([[2.75, 0, z]]))),
            ]
        if i == 0:
            annotations.append(_annotation('reference', 'point', 0, transform(np.array([[0, 1, z]]))))
        if i == 4:
            annotations.append(_annotation('calcification', 'region', 0, transform(_circle(0.5, -1, 0, z))))
        lumen = transform(_circle(2, 0, 0, z))
        frames.append(
            SimpleNamespace(
                lumen=SimpleNamespace(points=_points(lumen)),
                centroid=tuple(transform(np.array([[0, 0, z]]))[0]),
                annotations=annotations,
                extras={'Eem': SimpleNamespace(points=_points(transform(_circle(3, 0, 0, z, n=10))))},
            )
        )
    return SimpleNamespace(frames=frames)


def _tube(geometry):
    placed = iv.placed_annotations(geometry)
    (tube,) = iv.branch_tubes(placed, iv.frame_normals(geometry), sections=32)
    return tube


def test_a_branch_run_becomes_one_tube_across_its_region_on_the_reference_frame():
    tube = _tube(_geometry())
    low, high = tube.bounds
    assert low == pytest.approx([2.0, -0.8, 2 * SPACING - 0.8], abs=1e-6)  # from the lumen border
    assert high == pytest.approx([3.5, 0.8, 2 * SPACING + 0.8], abs=1e-6)  # to the arc, 1.6 wide, in plane


def test_the_tube_moves_rigidly_with_its_frames():
    rotation = Rotation.from_euler('xyz', [30, -50, 70], degrees=True)
    shift = np.array([10.0, -4.0, 7.0])
    still, moved = _tube(_geometry()), _tube(_geometry(lambda p: rotation.apply(p) + shift))
    assert moved.volume == pytest.approx(still.volume, rel=1e-6)
    assert moved.centroid == pytest.approx(rotation.apply(np.array(still.centroid)) + shift, abs=1e-6)


def test_a_plaque_blob_fills_its_region_within_half_a_frame_spacing():
    geometry = _geometry()
    placed = iv.placed_annotations(geometry)
    density = 200.0
    blob = iv.plaque_blob(
        placed,
        'calcification',
        iv.frame_normals(geometry),
        iv.frame_spacing(geometry),
        density,
        np.random.default_rng(0),
    )
    outline_area = 0.5 * 40 * 0.5**2 * np.sin(2 * np.pi / 40)  # the 40-gon the region is drawn as
    assert len(blob) == round(outline_area * density)
    assert np.all(np.hypot(blob[:, 0] + 1, blob[:, 1]) <= 0.5 + 1e-9)  # inside the circle
    assert np.all(np.abs(blob[:, 2] - 4 * SPACING) <= SPACING / 2 + 1e-9)  # around its frame


def test_references_tell_automatic_from_hand_placed_and_eem_comes_from_its_extra():
    geometry = _geometry()
    points, automatic = iv.reference_points(iv.placed_annotations(geometry))
    assert points.tolist() == [[0.0, 1.0, 0.0], [2.75, 0.0, 2 * SPACING]]
    assert automatic == [False, True]
    assert iv.contour_points(geometry, 'Eem').shape == (N_FRAMES * 10, 3)


def test_a_multimodars_without_annotations_shows_none():
    frame = SimpleNamespace(lumen=SimpleNamespace(points=[]), centroid=(0, 0, 0), extras={})
    geometry = SimpleNamespace(frames=[frame])
    assert iv.placed_annotations(geometry) == []
    assert iv.contour_points(geometry, 'Eem').shape == (0, 3)
