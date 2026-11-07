import pytest

from fusion_pipeline import scaling


@pytest.fixture
def morphs(monkeypatch):
    """Record every morph as (region, centerline, adjustment); each sync bumps a mesh version."""
    calls = []

    def fake_scale(mesh, region, centerline, adjustment):
        calls.append((list(region), centerline, adjustment))
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
