"""The fusion case sidecar: label maps as original vertex indices, versioned file, step-by-step restore."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from domain.fusion.runtime_types import FusionRuntimeData
from fusion_pipeline import case_state as cs
from pages.fusion.controllers.case_state import CaseStateController

VERTICES = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (4.0, 0.0, 0.0)]


def _mesh(vertices):
    return SimpleNamespace(vertices=np.array(vertices, dtype=np.float64))


def _flush():
    cs._writer.submit(lambda: None).result()


def test_vertex_index_maps_points_and_follows_moved_vertices():
    index = cs.VertexIndex(VERTICES)
    assert index.indices([(2.0, 0.0, 0.0), (9.0, 9.0, 9.0), (0.0, 0.0, 0.0)]) == [2, 0]
    scaled = [(x * 2, y, z) for x, y, z in VERTICES]
    index.follow(_mesh(VERTICES), _mesh(scaled))
    assert index.indices([(4.0, 0.0, 0.0)]) == [2]
    assert index.points([2, 4]) == [(4.0, 0.0, 0.0), (8.0, 0.0, 0.0)]


def test_step_labels_keep_the_last_label_of_gone_vertices():
    index = cs.VertexIndex(VERTICES)
    previous = {'rca_points': [3, 4], 'rca_branches': {'1': [3], '2': [4]}}
    # Branch 2 (vertex 4) was deleted, vertex 1 was lassoed into rca_points.
    results = {'rca_points': [VERTICES[1], VERTICES[3]], 'rca_branches': {1: [VERTICES[1], VERTICES[3]]}}
    present = index.present(_mesh(VERTICES[:4]))
    labels = cs.step_labels(results, ['rca_points', 'rca_branches'], index, previous, present)
    assert labels == {'rca_points': [1, 3, 4], 'rca_branches': {'1': [1, 3], '2': [4]}}


def test_apply_step_labels_round_trip_skips_gone_vertices():
    index = cs.VertexIndex(VERTICES)
    results: dict = {}
    labels = {'aorta_points': [0, 4], 'lca_branches': {'0': [1], '3': [4]}}
    cs.apply_step_labels(results, labels, index, index.present(_mesh(VERTICES[:4])))
    assert results == {'aorta_points': [VERTICES[0]], 'lca_branches': {0: [VERTICES[1]]}}


def test_find_state_picks_the_newest_version_and_unchanged_writes_are_skipped(tmp_path):
    mesh_path = str(tmp_path / 'case.stl')
    for version in ('0_9_0', '0_10_0', '0_2_0'):
        (tmp_path / f'case_fusion_{version}.json').write_text('{}')
    assert cs.find_state(mesh_path).endswith('case_fusion_0_10_0.json')

    path = str(tmp_path / 'out.json')
    cs.write_state({'a': 1}, path, blocking=True)
    (tmp_path / 'out.json').write_text('touched')
    cs.write_state({'a': 1}, path, blocking=True)  # same content as the last write: skipped
    assert (tmp_path / 'out.json').read_text() == 'touched'


def _centerline(branch_ids):
    return SimpleNamespace(points=[SimpleNamespace(branch_id=b) for b in branch_ids])


def _session(tmp_path, restore=True, delete_again=True):
    """A controller on a freshly labelled mesh. With `restore`, as if Yes was answered on picking the mesh."""
    data = FusionRuntimeData()
    data.centerline_rca, data.centerline_lca = _centerline([0, 1, 2]), _centerline([0])
    data.results = {
        'mesh': _mesh(VERTICES),
        'aorta_points': [VERTICES[0]],
        'rca_points': VERTICES[1:4],
        'lca_points': [VERTICES[4]],
    }
    page = MagicMock()
    page.data = data
    page.right_half.geometry_column.mesh_path = str(tmp_path / 'case.stl')
    page.right_half.geometry_column.centerline_paths = {'rca': 'rca.vtp'}
    page.right_half.intravascular_column.input_path.return_value = ''
    ctrl = CaseStateController(page)
    page.case_state = ctrl
    saved = cs.find_state(str(tmp_path / 'case.stl'))
    if restore and saved:
        ctrl._pending = {**cs.load_state(saved), 'replay_deletions': delete_again}
    return ctrl, data


def _saved(tmp_path):
    _flush()
    return json.loads(open(cs.state_path(str(tmp_path / 'case.stl'))).read())


def _label_branches(data):
    data.results['rca_branches'] = {0: [VERTICES[1]], 1: [VERTICES[2]], 2: [VERTICES[3]]}
    data.results['lca_branches'] = {0: [VERTICES[4]]}


def test_lasso_edits_are_saved_and_restored_in_a_new_session(tmp_path):
    ctrl, data = _session(tmp_path)
    ctrl.on_label_geometry()
    _label_branches(data)
    ctrl.on_branches_labelled()
    # Lasso on Branch Labels: vertex 3 from branch 2 to branch 1, and a rename.
    data.results['rca_branches'] = {0: [VERTICES[1]], 1: [VERTICES[2], VERTICES[3]], 2: []}
    data.branch_names['rca'] = {1: 'PDA'}
    ctrl.save()
    saved = _saved(tmp_path)
    assert saved['labels']['label_branches']['rca_branches'] == {'0': [1], '1': [2, 3]}
    assert saved['branch_names']['rca'] == {'1': 'PDA'}
    assert saved['paths']['centerlines'] == {'rca': 'rca.vtp'}

    ctrl, data = _session(tmp_path)
    data.results['rca_points'] = [VERTICES[1]]  # the fresh labeling differs from the saved one
    ctrl.on_label_geometry()
    assert data.results['rca_points'] == VERTICES[1:4]
    _label_branches(data)
    ctrl.on_branches_labelled()
    assert data.results['rca_branches'] == {0: [VERTICES[1]], 1: [VERTICES[2], VERTICES[3]]}
    assert data.branch_names['rca'] == {1: 'PDA'}


def test_saved_branch_ids_count_deleted_branches_and_deletions_are_replayed(tmp_path):
    ctrl, data = _session(tmp_path)
    ctrl.on_label_geometry()
    _label_branches(data)
    ctrl.on_branches_labelled()
    ctrl.on_branch_deleted('rca', 2, True, 'PLB', [2])
    data.centerline_rca = _centerline([0, 1])  # after the deletion, Label Branches runs again
    ctrl.on_branches_labelled()
    assert _saved(tmp_path)['branch_ids'] == {'rca': [0, 1, 2], 'lca': [0]}

    ctrl, data = _session(tmp_path)
    ctrl.on_label_geometry()
    data.centerline_rca = _centerline([0, 1])  # not split like the saved one yet
    _label_branches(data)
    ctrl.on_branches_labelled()
    ctrl.page.branches.delete_branch.assert_not_called()
    on_disk = _saved(tmp_path)
    assert on_disk['labels']['label_branches']['rca_branches'] == {'0': [1], '1': [2], '2': [3]}  # still the saved
    assert [d['branch'] for d in on_disk['deleted_branches']] == [2]

    data.centerline_rca = _centerline([0, 1, 2])
    ctrl.on_branches_labelled()
    ctrl.page.branches.delete_branch.assert_not_called()  # waits for the tree, so the deletion stays undoable
    assert [d['branch'] for d in _saved(tmp_path)['deleted_branches']] == [2]
    ctrl.on_tree_ready()
    ctrl.page.branches.delete_branch.assert_called_once_with('rca', 2, True)


def test_deletions_are_not_replayed_when_unticked(tmp_path):
    ctrl, data = _session(tmp_path)
    ctrl.on_label_geometry()
    _label_branches(data)
    ctrl.on_branches_labelled()
    ctrl.on_branch_deleted('rca', 2, True, 'PLB', [2])
    _flush()

    ctrl, data = _session(tmp_path, delete_again=False)
    ctrl.on_label_geometry()
    _label_branches(data)
    ctrl.on_branches_labelled()
    ctrl.page.branches.delete_branch.assert_not_called()
    assert _saved(tmp_path)['deleted_branches'] == []


def test_without_restore_the_fresh_labels_overwrite_the_file(tmp_path):
    ctrl, data = _session(tmp_path)
    data.results['rca_points'] = [VERTICES[1]]
    ctrl.on_label_geometry()
    _flush()

    ctrl, data = _session(tmp_path, restore=False)
    ctrl.on_label_geometry()
    assert data.results['rca_points'] == VERTICES[1:4]
    assert _saved(tmp_path)['labels']['label_geometry']['rca_points'] == [1, 2, 3]


def test_scaling_factors_are_saved(tmp_path):
    ctrl, _data = _session(tmp_path)
    ctrl.on_label_geometry()
    ctrl.on_scaling_computed('rca', {'proximal_scaling': 0.4, 'distal_scaling': -0.1, 'aortic_scaling': 0.0})
    applied = {'proximal_scaling': 0.5, 'distal_scaling': -0.1, 'aortic_scaling': 0.0, 'remaining_vessels_scaling': 0}
    ctrl.on_scaling_applied('rca', applied)
    scaling = _saved(tmp_path)['scaling']
    assert scaling['vessel'] == 'rca'
    assert scaling['computed']['proximal_scaling'] == 0.4 and scaling['applied']['proximal_scaling'] == 0.5


def test_picking_a_mesh_with_a_state_reruns_the_pipeline_in_order(tmp_path):
    ctrl, data = _session(tmp_path)
    ctrl.on_label_geometry()
    ctrl.on_centerline_edit('split', 'rca', 1, point_index=12)
    _flush()

    ctrl, data = _session(tmp_path, restore=False)
    page = ctrl.page
    for name in ('aorta', 'rca', 'lca'):
        (tmp_path / f'{name}.vtp').write_text('')
    state = cs.load_state(cs.find_state(str(tmp_path / 'case.stl')))
    state['paths']['centerlines'] = {k: str(tmp_path / f'{k}.vtp') for k in ('aorta', 'rca', 'lca')}
    cs.write_state(state, cs.state_path(str(tmp_path / 'case.stl')), blocking=True)
    centerline_paths: dict = {}
    page.right_half.geometry_column.centerline_paths = centerline_paths
    page.right_half.geometry_column.set_centerline_path.side_effect = centerline_paths.__setitem__
    order: list[str] = []
    page.ccta_geometry.on_prepare_centerlines.side_effect = lambda: order.append('prepare')
    page.branches.split_branch.side_effect = lambda *a: order.append(f'split {a}') or True
    page.ccta_geometry.on_label_geometry.side_effect = lambda: order.append('label geometry')
    page.tree.on_discretize.side_effect = lambda: order.append('tree')
    data.vessel_tree = object()
    ctrl.ask_restore = MagicMock(return_value=(True, True))

    ctrl.on_mesh_selected()

    page.reset_data.assert_called_once()
    assert order == ['prepare', "split ('rca', 1, 12)", 'label geometry', 'tree']
    page.alignment.on_load_pullback.assert_not_called()  # nothing aligned in the saved state
