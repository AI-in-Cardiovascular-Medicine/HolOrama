"""The stitch cuts the end rims and the reconnected side branches at tree slices the user picks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from domain.fusion.runtime_types import FusionRuntimeData, StitchCuts, StitchCutTarget
from fusion_pipeline import multimodars_api as mm_api
from fusion_pipeline import stitch_cuts
from pages.fusion.controllers.stitch_cuts import StitchCutController
from pages.fusion.left_half.toolbars.alignment import AlignmentToolbar


def _contour(z: float):
    ring = [(1.0, 0.0, z), (0.0, 1.0, z), (-1.0, 0.0, z)]
    return SimpleNamespace(centroid=(0.0, 0.0, z), points_as_tuples=lambda: ring)


def _tree():
    """RCA main with 10 slices along z, side branch 3 with 4 slices at x=20."""
    side = [SimpleNamespace(centroid=(20.0, 0.0, z), points_as_tuples=lambda z=z: [(20.0, 0.0, z)]) for z in range(4)]
    return SimpleNamespace(
        rca={0: SimpleNamespace(contours=[_contour(z) for z in range(10)]), 3: SimpleNamespace(contours=side)},
        lca={},
    )


FOUND = {'distal': ('rca', 0, 7), 'proximal': ('rca', 0, 2), 'side_branches': [('rca', 3)]}


@pytest.fixture
def found(monkeypatch):
    monkeypatch.setattr(mm_api, 'run_stitch_cut_targets', lambda geometry, results, tree: dict(FOUND))


def test_targets_are_the_rims_proximal_first_then_the_side_branches_at_slice_0(found):
    targets = stitch_cuts.find_targets(None, {}, None)
    assert [t.key for t in targets] == ['proximal', 'distal', ('rca', 3)]
    assert [t.default for t in targets] == [2, 7, 0]


def test_an_ostium_is_not_cut(monkeypatch):
    monkeypatch.setattr(mm_api, 'run_stitch_cut_targets', lambda *a: {**FOUND, 'proximal': None})
    assert [t.key for t in stitch_cuts.find_targets(None, {}, None)] == ['distal', ('rca', 3)]


def test_only_chosen_slices_become_stitch_arguments():
    cuts = StitchCuts([], (None, None, None), chosen={'distal': 9, ('rca', 3): 2})
    assert stitch_cuts.stitch_kwargs(cuts) == {
        'proximal_cut_slice': None,
        'distal_cut_slice': 9,
        'branch_cut_slices': {('rca', 3): 2},
    }
    assert stitch_cuts.stitch_kwargs(None) == {}


def test_choices_stay_only_for_targets_on_the_same_branch():
    old = StitchCuts(
        [StitchCutTarget('distal', 'rca', 0, 7), StitchCutTarget(('rca', 3), 'rca', 3, 0)],
        (None, None, None),
        chosen={'distal': 9, ('rca', 3): 2},
    )
    new = [StitchCutTarget('distal', 'rca', 5, 1)]  # the distal rim now lies on another branch
    assert stitch_cuts.keep_choices(old, new) == {}
    assert stitch_cuts.keep_choices(old, old.targets) == old.chosen


@pytest.fixture
def controller(qt_app, found):
    data = FusionRuntimeData()
    data.results_points_removed, data.vessel_tree, data.aligned = {}, _tree(), object()
    toolbar = AlignmentToolbar()
    page = SimpleNamespace(
        data=data,
        left_half=SimpleNamespace(alignment_toolbar=toolbar, viewer=Mock(), refresh_toolbar=Mock()),
        status_bar=Mock(),
        branches=SimpleNamespace(name=lambda vessel, branch: f'{vessel.upper()} {branch}'),
    )
    ctrl = StitchCutController(page)
    ctrl.refresh()
    return ctrl, toolbar


def _texts(toolbar):
    return [toolbar._targets.item(i).text() for i in range(toolbar._targets.count())]


def test_lists_every_target_with_its_default(controller):
    _, toolbar = controller
    assert _texts(toolbar) == [
        'Proximal rim (RCA 0) · slice 3/10 (default)',
        'Distal rim (RCA 0) · slice 8/10 (default)',
        'RCA 3 · slice 1/4 (default)',
    ]


def test_a_click_snaps_to_the_selected_targets_nearest_slice(controller):
    ctrl, toolbar = controller
    ctrl.select(2)
    ctrl.on_point_picked(1.0, 0.0, 2.2)  # nearest slice of the whole tree is on the main vessel
    assert ctrl.data.stitch_cuts.chosen == {('rca', 3): 2}
    ctrl.select(1)
    ctrl.on_point_picked(1.0, 0.0, 4.9)
    ctrl.on_step(1)
    assert ctrl.data.stitch_cuts.chosen['distal'] == 6
    assert 'slice 7/10 (chosen)' in toolbar._selection.text()
    ctrl.on_default()
    assert 'distal' not in ctrl.data.stitch_cuts.chosen


def test_a_new_removal_finds_the_targets_again_keeping_the_choices(controller):
    ctrl, _ = controller
    ctrl.on_point_picked(1.0, 0.0, 1.0)  # proximal rim
    before = ctrl.data.stitch_cuts
    ctrl.data.results_points_removed = {}  # a new snapshot
    ctrl.refresh()
    assert ctrl.data.stitch_cuts is not before
    assert ctrl.data.stitch_cuts.chosen == {'proximal': 1}
    ctrl.data.vessel_tree = _tree()  # a new tree: its slices mean something else
    ctrl.refresh()
    assert ctrl.data.stitch_cuts.chosen == {}


def test_no_targets_without_a_removal(controller):
    ctrl, toolbar = controller
    ctrl.data.results_points_removed = None
    ctrl.refresh()
    assert ctrl.data.stitch_cuts is None
    assert _texts(toolbar) == []


def test_side_branches_left_loose_are_marked_with_the_reason(controller):
    ctrl, toolbar = controller
    ctrl.on_stitched({'not_reconnected': {('rca', 3): 'contour 0 reaches into the intravascular lumen'}})
    item = toolbar._targets.item(2)
    assert item.text().startswith('⚠ RCA 3')
    assert 'lumen' in item.toolTip()
