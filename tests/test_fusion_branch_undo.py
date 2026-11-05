"""Undo of a branch deletion on the fusion page, with the UI and multimodars faked out."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QMessageBox

from domain.fusion.runtime_types import BranchRemoval, CutSpec, FusionRuntimeData
from pages.fusion.controllers import branches as branches_module
from pages.fusion.controllers.branches import BranchController


class _Tree:
    def __init__(self, rca):
        self.rca, self.lca = rca, {}
        self.ref_pts_calls = 0

    def calculate_ref_pts(self):
        self.ref_pts_calls += 1


@pytest.fixture
def controller(monkeypatch):
    data = FusionRuntimeData()
    data.centerline_rca = SimpleNamespace(name='rca with 0, 1, 2')
    data.results = {'mesh': 'mesh before', 'rca_branches': {0: [(0, 0, 0)], 1: [(1, 1, 1)], 2: [(2, 2, 2)]}}
    data.vessel_tree = _Tree({0: 'b0', 1: 'b1', 2: 'b2'})
    data.branch_names['rca'] = {1: 'PDA', 2: 'PLB'}
    data.cuts = [CutSpec('rca', 1, 3, 'larger', 'pda outlet'), CutSpec('rca', 0, 5, 'larger', 'rca outlet')]
    data.selected_branch['rca'] = 1
    data.removal_undo.push({'mesh': 'before removal'})

    page = MagicMock()
    page.data = data
    ctrl = BranchController(page)
    page.branches = ctrl
    for name in ('refresh_centerline_scene', '_refresh_name_displays', 'refresh_labels_scene'):
        monkeypatch.setattr(ctrl, name, MagicMock())
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)

    def fake_remove(results, centerline, branch, *, stitched, final_mesh, fill_hole):
        new = {'mesh': 'mesh after', 'rca_branches': {0: results['rca_branches'][0], 2: results['rca_branches'][2]}}
        return BranchRemoval(new, SimpleNamespace(name='rca with 0, 2'), None, None, [1])

    monkeypatch.setattr(branches_module.branch_ops, 'remove_branch', fake_remove)
    return ctrl


def test_undo_delete_restores_everything_the_deletion_replaced(controller):
    data = controller.data
    centerline_before = data.centerline_rca
    controller.on_delete_requested('rca', 1, True)

    assert data.results['mesh'] == 'mesh after'
    assert 1 not in data.vessel_tree.rca and 1 not in data.branch_names['rca']
    assert len(data.removal_undo) == 0
    assert data.selected_branch['rca'] == 0
    assert len(data.branch_deletion_undo) == 1

    data.branch_names['rca'][2] = 'renamed after the deletion'
    controller.on_undo_delete()

    assert data.results['mesh'] == 'mesh before' and 1 in data.results['rca_branches']
    assert data.centerline_rca is centerline_before
    assert data.vessel_tree.rca == {0: 'b0', 1: 'b1', 2: 'b2'}
    assert data.branch_names['rca'] == {1: 'PDA', 2: 'renamed after the deletion'}
    assert data.removal_undo.items() == [{'mesh': 'before removal'}]
    assert data.selected_branch['rca'] == 1
    assert len(data.branch_deletion_undo) == 0
    controller.page.cutting.restore_branch_cuts.assert_called_once()


def test_other_edits_drop_the_deletion_undo(controller):
    controller.on_delete_requested('rca', 1, True)
    controller.clear_delete_undo()
    assert len(controller.data.branch_deletion_undo) == 0
    controller.on_undo_delete()  # nothing left: a no-op
    assert controller.data.results['mesh'] == 'mesh after'
