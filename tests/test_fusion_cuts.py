from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from domain.fusion.runtime_types import CutResult, CutSpec
from fusion_pipeline import cuts


def _contour(z: float):
    ring = [(1.0, 0.0, z), (0.0, 1.0, z), (-1.0, 0.0, z)]
    return SimpleNamespace(centroid=(0.0, 0.0, z), points_as_tuples=lambda: ring)


def _tree():
    """Aorta with 2 slices, RCA main (5 slices along +z) and RCA side branch 1 (2 slices)."""
    return SimpleNamespace(
        discretized_aorta=[_contour(-2), _contour(-1)],
        rca={0: SimpleNamespace(contours=[_contour(z) for z in range(5)]), 1: SimpleNamespace(contours=[_contour(9)])},
        lca={},
    )


def _cap(z: float, normal_up: bool) -> trimesh.Trimesh:
    """A flat cap at height z: square ring, then its centre (as multimodars builds them)."""
    ring = [(1, 0, z), (0, 1, z), (-1, 0, z), (0, -1, z)]
    centre = len(ring)
    faces = [(i, (i + 1) % len(ring), centre) for i in range(len(ring))]
    if not normal_up:
        faces = [f[::-1] for f in faces]
    return trimesh.Trimesh(vertices=ring + [(0, 0, z)], faces=faces, process=False)


def _centerline(branch_ids):
    return SimpleNamespace(branch_ids=lambda: list(branch_ids))


def _spec(slice_idx=2, branch=0, vessel='rca', name='cut'):
    return CutSpec(vessel, branch, slice_idx, keep='larger', name=name)


def test_tree_slices_keys_aorta_and_every_branch():
    slices = cuts.tree_slices(_tree())
    assert list(slices) == [('aorta', 0), ('rca', 0), ('rca', 1)]
    assert len(slices[('rca', 0)]) == 5


def test_slice_points_maps_every_point_to_its_slice():
    keys, points, owners = cuts.slice_points(cuts.tree_slices(_tree()))
    assert len(keys) == 2 + 5 + 1
    assert points.shape == (len(keys) * 3, 3)
    assert keys[owners[-1]] == ('rca', 1, 0)


def test_slice_points_empty():
    keys, points, owners = cuts.slice_points({})
    assert keys == [] and points.shape == (0, 3) and owners.dtype.kind == 'i'


@pytest.mark.parametrize('normal_up, expected', [(True, {2, 3, 4}), (False, {0, 1, 2})])
def test_dropped_slices_follows_the_cap_normal(normal_up, expected):
    slices = cuts.tree_slices(_tree())
    dropped = cuts.dropped_slices(slices, _spec(2), _cap(2, normal_up), _centerline([0, 1]), _centerline([0, 1]))
    assert dropped == {('rca', 0, j) for j in expected}


def test_dropped_slices_includes_branches_cut_off_the_centerline():
    slices = cuts.tree_slices(_tree())
    dropped = cuts.dropped_slices(slices, _spec(2), _cap(2, True), _centerline([0, 1]), _centerline([0]))
    assert ('rca', 1, 0) in dropped


def test_apply_cuts_records_failures_and_keeps_going(monkeypatch):
    calls = []

    def fake_cut(results, tree, centerline, vessel, slice_idx, branch_id, keep):
        calls.append(slice_idx)
        if slice_idx == 1:
            raise RuntimeError('no clean ring')
        return results, _centerline([0]), _cap(slice_idx, True)

    monkeypatch.setattr(cuts.mm_api, 'run_cut_at_slice', fake_cut)
    start = CutResult(mesh='mesh', centerlines={'rca': _centerline([0, 1])})
    specs = [_spec(1, name='a'), _spec(2, name='b'), _spec(3, name='c'), _spec(0, vessel='lca', name='d')]

    out = cuts.apply_cuts(start, cuts.tree_slices(_tree()), tree=None, specs=specs)

    assert calls == [1, 2]  # slice 3 was dropped by the cut at 2, lca has no centerline
    assert set(out.failed) == {0, 2, 3}
    assert out.failed[0] == 'no clean ring'
    assert 'cut off' in out.failed[2]
    assert list(out.caps) == [1]
    assert start.caps == {} and start.failed == {}  # start is left as is


def test_mesh_fingerprint_sees_vertex_moves():
    mesh = _cap(0, True)
    before = cuts.mesh_fingerprint(mesh)
    mesh.vertices[0] = mesh.vertices[0] + np.array([0.0, 0.0, 0.1])
    assert cuts.mesh_fingerprint(mesh) != before


def test_export_cut_geometry_gives_caps_unique_file_names(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(cuts.geo, 'export_mesh', lambda mesh, path: written.append(path))
    monkeypatch.setattr(cuts.mm_api, 'write_centerline_vtp', lambda cl, path: written.append(path))
    result = CutResult(mesh='m', centerlines={'rca': 'cl'}, caps={0: 'c0', 1: 'c1', 2: 'c2'})
    specs = [_spec(name='RCA outlet'), _spec(name='rca outlet'), _spec(name='???')]

    paths = cuts.export_cut_geometry(result, specs, str(tmp_path / 'case.stl'))

    names = [p.rsplit('case', 1)[1] for p in paths]
    assert names == ['.stl', '_rca_outlet.stl', '_rca_outlet_2.stl', '_cap.stl', '_centerline_rca.vtp']
    assert paths == written
