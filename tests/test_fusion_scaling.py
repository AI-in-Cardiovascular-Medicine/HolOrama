import pytest

from fusion_pipeline import scaling


@pytest.fixture
def passed_labels() -> list:
    """The branch_points of every morph, in order."""
    return []


@pytest.fixture
def morphs(monkeypatch, passed_labels):
    """Record every morph as (region, centerline, adjustment); each sync bumps a mesh version."""
    calls = []

    def fake_scale(mesh, region, centerline, adjustment, branch_points=None):
        calls.append((list(region), centerline, adjustment))
        passed_labels.append(branch_points)
        return mesh + 1

    def fake_sync(results, old_mesh, new_mesh):
        return {**results, 'mesh': new_mesh}

    monkeypatch.setattr(scaling.mm_api, 'run_scale_region', fake_scale)
    monkeypatch.setattr(scaling.mm_api, 'run_sync_results_to_mesh', fake_sync)
    return calls


def _results():
    return {
        'mesh': 0,
        'distal_points': ['d'],
        'aorta_points': ['a'],
        'rca_removed_points': ['r'],
        'proximal_points': ['p'],
        'lca_points': ['l'],
        'off_path_points': ['o'],
    }


def _scaling(remaining=0.0):
    return {
        'distal_scaling': 1.0,
        'aortic_scaling': 2.0,
        'proximal_scaling': 3.0,
        'remaining_vessels_scaling': remaining,
    }


def test_distal_aortic_proximal_in_order(morphs):
    out = scaling.apply_scaling(
        _results(), 'rca', _scaling(), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline=None
    )
    assert morphs == [(['d'], 'rca_cl', 1.0), (['a', 'r'], 'aorta_cl', 2.0), (['p'], 'rca_cl', 3.0)]
    assert out['mesh'] == 3


def test_remaining_vessels_each_around_their_own_centerline(morphs):
    out = scaling.apply_scaling(
        _results(), 'rca', _scaling(0.5), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline='lca_cl'
    )
    assert morphs[3:] == [(['l'], 'lca_cl', 0.5), (['o'], 'rca_cl', 0.5)]
    assert out['mesh'] == 5


def test_remaining_vessels_skip_empty_regions(morphs):
    results = {**_results(), 'off_path_points': []}
    scaling.apply_scaling(
        results, 'rca', _scaling(0.5), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline='lca_cl'
    )
    assert len(morphs) == 4


def test_remaining_vessels_need_the_opposite_centerline(morphs):
    with pytest.raises(ValueError, match='LCA centerline'):
        scaling.apply_scaling(
            _results(), 'rca', _scaling(0.5), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline=None
        )


def test_coronary_regions_pass_their_vessels_branch_labels(morphs, passed_labels):
    # Lasso edits live in {vessel}_branches, multimodars morphs each vertex around its labelled branch.
    results = {**_results(), 'rca_branches': {0: ['d', 'p', 'o'], 1: []}, 'lca_branches': {0: ['l']}}
    scaling.apply_scaling(
        results, 'rca', _scaling(0.5), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline='lca_cl'
    )
    rca, lca = results['rca_branches'], results['lca_branches']
    # distal, aortic, proximal, opposite vessel, off-path
    assert passed_labels == [rca, None, rca, lca, rca]


def test_without_branch_labels_nothing_is_passed(morphs, passed_labels):
    scaling.apply_scaling(
        _results(), 'rca', _scaling(), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline=None
    )
    assert passed_labels == [None, None, None]


@pytest.fixture
def tree_syncs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        scaling.mm_api, 'run_sync_tree_to_mesh', lambda tree, old, new: calls.append((tree, old, new)) or tree
    )
    return calls


def test_tree_follows_the_whole_scaling_once(morphs, tree_syncs):
    scaling.apply_scaling(
        _results(),
        'rca',
        _scaling(0.5),
        centerline='rca_cl',
        centerline_aorta='aorta_cl',
        opposite_centerline='lca_cl',
        tree='tree',
    )
    # Original mesh (0) to the final one (5 morphs): one sync covers every region.
    assert tree_syncs == [('tree', 0, 5)]


def test_no_tree_no_sync(morphs, tree_syncs):
    scaling.apply_scaling(
        _results(), 'rca', _scaling(), centerline='rca_cl', centerline_aorta='aorta_cl', opposite_centerline=None
    )
    assert tree_syncs == []


def test_failed_scaling_leaves_the_tree_alone(morphs, tree_syncs):
    with pytest.raises(ValueError):
        scaling.apply_scaling(
            _results(),
            'rca',
            _scaling(0.5),
            centerline='rca_cl',
            centerline_aorta='aorta_cl',
            opposite_centerline=None,
            tree='tree',
        )
    assert tree_syncs == []
