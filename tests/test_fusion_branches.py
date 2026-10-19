from types import SimpleNamespace

import numpy as np

from fusion_pipeline import branches


def _node(ostium=None, bifurcations=None, parent=None):
    return SimpleNamespace(ostium=ostium, bifurcations=bifurcations or {}, parent=parent)


def _tree():
    """RCA main with two children leaving at contours 7 and 3, a child without references, no LCA."""
    rca = {
        0: _node(ostium='ostium0', bifurcations={1: 'takeoff1', 2: 'takeoff2'}),
        1: _node(ostium='ostium1', parent=(0, 7)),
        2: _node(ostium='ostium2', parent=(0, 3)),
        3: _node(parent=(1, 0)),
    }
    return SimpleNamespace(rca=rca, lca={})


def test_reference_entries_ostium_first_then_takeoffs_proximal_to_distal():
    assert branches.reference_entries(_tree(), 'rca', 0) == [(None, 'ostium0'), (2, 'takeoff2'), (1, 'takeoff1')]
    assert branches.references(_tree(), 'rca', 1) == ['ostium1']
    assert branches.reference_entries(_tree(), 'rca', 9) == []
    assert branches.reference_entries(None, 'rca', 0) == []


def test_ids_with_references_skips_branches_without_any():
    assert branches.ids_with_references(_tree(), 'rca') == [0, 1, 2]
    assert branches.ids_with_references(_tree(), 'lca') == []
    assert branches.ids_with_references(None, 'rca') == [0]


def test_branch_name():
    assert branches.branch_name({}, 'rca', 0) == 'RCA main'
    assert branches.branch_name({}, 'lca', 2) == 'LCA branch 2'
    assert branches.branch_name({2: 'LAD'}, 'lca', 2) == 'LAD'


def test_move_points_in_place():
    groups = {'a': [(0, 0, 0), (1, 1, 1)], 'b': [(2, 2, 2)]}
    moved = branches.move_points(groups, 'a', 'b', np.array([False, True]))
    assert moved == [(1, 1, 1)]
    assert groups == {'a': [(0, 0, 0)], 'b': [(2, 2, 2), (1, 1, 1)]}
    branches.move_points(groups, 'a', 'c', np.array([True]))
    assert groups['a'] == [] and groups['c'] == [(0, 0, 0)]


def _centerline(points_and_branches):
    points = [
        SimpleNamespace(branch_id=b, contour_point=SimpleNamespace(x=p[0], y=p[1], z=p[2]))
        for p, b in points_and_branches
    ]
    return SimpleNamespace(points=points, points_as_tuples=lambda: [p for p, _b in points_and_branches])


def test_sync_branch_labels_follows_a_region_move():
    rca_cl = _centerline([((0, 0, 0), 0), ((10, 0, 0), 1)])
    results = {
        'aorta_points': [],
        'rca_points': [(9.0, 0.0, 0.0)],
        'rca_branches': {0: [(0.5, 0.0, 0.0)], 1: []},
        'lca_branches': {0: [(5.0, 5.0, 5.0)]},
    }
    # aorta -> rca: the joined point goes to the branch of its nearest centerline point.
    branches.sync_branch_labels(results, [(9.0, 0.0, 0.0)], 'aorta_points', 'rca_points', {'rca': rca_cl}.get)
    assert results['rca_branches'] == {0: [(0.5, 0.0, 0.0)], 1: [(9.0, 0.0, 0.0)]}
    # rca -> aorta: it leaves the rca branches again, lca is untouched.
    branches.sync_branch_labels(results, [(9.0, 0.0, 0.0)], 'rca_points', 'aorta_points', {'rca': rca_cl}.get)
    assert results['rca_branches'] == {0: [(0.5, 0.0, 0.0)], 1: []}
    assert results['lca_branches'] == {0: [(5.0, 5.0, 5.0)]}


def test_sync_branch_labels_without_centerline_only_removes():
    results = {'rca_branches': {0: [(1.0, 0.0, 0.0)]}, 'lca_branches': {0: []}}
    branches.sync_branch_labels(results, [(1.0, 0.0, 0.0)], 'rca_points', 'lca_points', lambda vessel: None)
    assert results == {'rca_branches': {0: []}, 'lca_branches': {0: []}}


def test_remove_branch_applies_to_every_mesh_there_is(monkeypatch):
    calls = []

    def fake_remove(results, centerline, branch, fill_hole):
        calls.append(results['name'])
        return {'name': results['name'] + '-cut', 'mesh': results['name'] + '-mesh'}, SimpleNamespace(
            branch_ids=lambda: [0]
        )

    monkeypatch.setattr(branches.mm_api, 'run_remove_branch', fake_remove)
    monkeypatch.setattr(branches.geo, 'transfer_labels', lambda stitched, mesh: {'name': 'final'})
    centerline = SimpleNamespace(branch_ids=lambda: [0, 2, 3])

    out = branches.remove_branch({'name': 'ccta'}, centerline, 2, stitched=None, final_mesh=None, fill_hole=True)
    assert calls == ['ccta'] and out.stitched is None and out.final_mesh is None
    assert out.removed == [2, 3]

    calls.clear()
    out = branches.remove_branch(
        {'name': 'ccta'}, centerline, 2, stitched={'name': 'stitched'}, final_mesh='mesh', fill_hole=True
    )
    assert calls == ['ccta', 'final', 'stitched']
    assert out.final_mesh == 'final-mesh' and out.stitched['name'] == 'stitched-cut'


def test_prune_vessel_tree_keeps_other_ids():
    recalculated = []
    tree = SimpleNamespace(rca={0: 'main', 2: 'b2', 3: 'b3', 4: 'b4'}, calculate_ref_pts=lambda: recalculated.append(1))
    branches.prune_vessel_tree(tree, 'rca', [2, 3])
    assert tree.rca == {0: 'main', 4: 'b4'} and recalculated == [1]


def test_nearest_reference_picks_the_closest_triplet_point():
    tree = SimpleNamespace(
        rca={
            0: _node(ostium=((0, 0, 0), (1, 0, 0), (0, 1, 0)), bifurcations={1: ((10, 0, 0), (11, 0, 0), (10, 1, 0))})
        },
        lca={0: _node(ostium=((0, 0, 20), (1, 0, 20), (0, 1, 20)))},
    )
    assert branches.nearest_reference(tree, (10.9, 0, 0)) == ('rca', 0, 1)
    assert branches.nearest_reference(tree, (0, 1, 19)) == ('lca', 0, 0)
    assert branches.nearest_reference(SimpleNamespace(rca={}, lca={}), (0, 0, 0)) is None
