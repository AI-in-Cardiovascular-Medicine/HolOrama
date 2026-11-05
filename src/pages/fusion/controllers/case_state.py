"""The fusion case sidecar: rewritten on every label change, restored in one go when its mesh is picked again."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from fusion_pipeline import case_state as cs
from pages.fusion.controllers.base import StageController
from version import version_file_str


class CaseStateController(StageController):
    """Keeps the in-memory state of the sidecar and writes it.

    Picking a mesh that has a sidecar asks whether to restore it. Restoring reruns the pipeline with the
    current parameters (prepare centerlines, their split/merge edits, label geometry, label branches,
    vessel tree, pullback and alignment, overlap region), and each step takes its saved part as it runs:
    the hooks below apply pending label maps, deletions and the alignment choice."""

    def __init__(self, page) -> None:
        super().__init__(page)
        self.reset()

    def reset(self) -> None:
        """Clear All Data: forget the mesh, and drop a pending restore."""
        self._path: str | None = None
        self._index: cs.VertexIndex | None = None
        self._present: tuple[object, np.ndarray] | None = None  # (mesh, mask), see _present_mask
        self._state: dict = {}  # per labelled mesh, reset by Label Geometry
        self._labelled_ids: dict[str, list[int]] = {}  # centerline branch ids when the branches were labelled
        self._centerline_edits: list[dict] = []  # split/merge since Prepare Centerlines
        self._pullback_plan: dict | None = None  # loader settings of the loaded pullback
        self._pending: dict | None = None  # loaded state whose parts are not applied yet
        self._deferred_deletions: list[dict] = []  # saved deletions waiting for the vessel tree

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _present_mask(self) -> np.ndarray:
        """Which original vertices the current mesh has. Cached per mesh object: only removals change it."""
        assert self._index is not None and self.data.results is not None
        mesh = self.data.results['mesh']
        if self._present is None or self._present[0] is not mesh:
            self._present = (mesh, self._index.present(mesh))
        return self._present[1]

    def save(self) -> None:
        """Recompute every label map from `results` and write the file. Call after any label change."""
        if self._path is None or self._index is None or self.data.results is None:
            return
        results, present = self.data.results, self._present_mask()
        labels = self._state.setdefault('labels', {})
        for step, keys in self._state.get('keys', {}).items():
            labels[step] = cs.step_labels(results, keys, self._index, labels.get(step, {}), present)
        out = dict(self._state)  # a fresh top-level dict per write; the writer thread only reads it
        out['branch_names'] = {
            vessel: {str(b): name for b, name in names.items()} for vessel, names in self.data.branch_names.items()
        }
        out['branch_ids'] = self._saved_branch_ids()
        out['centerline_edits'] = list(self._centerline_edits)
        out['pullback_plan'] = self._pullback_plan
        out['paths'] = self._paths()
        out['version'] = version_file_str
        cs.write_state(self._with_pending(out), self._path)

    def _saved_branch_ids(self) -> dict[str, list[int]]:
        """The branch ids of the centerlines before any deletion: the ones a restore labels the branches with,
        since it deletes again only after restoring the labels."""
        removed: dict[str, set[int]] = {'rca': set(), 'lca': set()}
        for entry in self._state.get('deleted_branches', []):
            removed[entry['vessel']].update(entry.get('removed', [entry['branch']]))
        return {v: sorted(set(ids) | removed[v]) for v, ids in self._labelled_ids.items()}

    def _with_pending(self, out: dict) -> dict:
        """Saved parts not restored yet stay in the file as they were, so a restore that stops early
        does not lose them."""
        pending = self._pending or {}
        waiting = pending.get('labels', {})
        out['labels'] = {**out.get('labels', {}), **waiting}
        if cs.STEP_BRANCHES in waiting:
            out['branch_names'] = pending.get('branch_names', {})
            out['branch_ids'] = pending.get('branch_ids')
        waiting_deletions = (pending.get('deleted_branches') or []) + self._deferred_deletions
        if waiting_deletions:
            out['deleted_branches'] = waiting_deletions + out.get('deleted_branches', [])
        for key in ('alignment', 'pullback_plan', 'centerline_edits', 'scaling'):
            if not out.get(key) and pending.get(key):
                out[key] = pending[key]
        return out

    def _paths(self) -> dict:
        gc = self.right_half.geometry_column
        return {
            'mesh': gc.mesh_path,
            'centerlines': dict(gc.centerline_paths),
            'pullback': self.right_half.intravascular_column.input_path() or None,
        }

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def on_mesh_selected(self) -> None:
        """A mesh was picked. If it has a saved state, offer to restore it and rerun the pipeline from it."""
        mesh_path = self.right_half.geometry_column.mesh_path
        path = cs.find_state(mesh_path) if mesh_path else None
        state = cs.load_state(path) if path else None
        if not state:
            return
        assert path is not None
        restore, delete_again = self.ask_restore(os.path.basename(path), state)
        if not restore:
            return
        self.page.reset_data()  # a restore starts from scratch, like a fresh case
        state['replay_deletions'] = delete_again
        self._pending = state
        if self._apply_paths(state):
            self._run_pipeline(state)

    def ask_restore(self, filename: str, state: dict) -> tuple[bool, bool]:
        """(restore, delete the saved deleted branches again)."""
        steps = 'prepare centerlines, label geometry, label branches, discretize the vessel tree'
        if state.get('alignment') and state.get('pullback_plan'):
            steps += ', load the pullback and align it'
        box = QMessageBox(
            QMessageBox.Icon.Question,
            'Restore fusion state',
            f'This mesh has a saved fusion state ({filename}).\n\nRestore it? This will {steps}, '
            'with the saved labels, branch edits and alignment and the current parameter settings.\n'
            'No keeps the file until this mesh is labelled again, which overwrites it.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self.page,
        )
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        deleted = state.get('deleted_branches') or []
        check = None
        if deleted:
            names = ', '.join(d.get('name') or f"{d['vessel'].upper()} branch {d['branch']}" for d in deleted)
            check = QCheckBox(f'Delete the deleted branches again: {names}')
            check.setChecked(True)
            box.setCheckBox(check)
        restore = box.exec() == QMessageBox.StandardButton.Yes
        return restore, check is not None and check.isChecked()

    def _apply_paths(self, state: dict) -> bool:
        """Fill in the saved centerline and pullback paths. False when a centerline file is missing."""
        paths = state.get('paths') or {}
        gc = self.right_half.geometry_column
        missing = []
        for key, cl_path in (paths.get('centerlines') or {}).items():
            if cl_path and os.path.exists(cl_path):
                gc.set_centerline_path(key, cl_path)
            else:
                missing.append(cl_path or key)
        ic = self.right_half.intravascular_column
        plan = state.get('pullback_plan')
        if plan and os.path.isdir(plan.get('input_path') or ''):
            ic.set_load_plan(plan)
        elif plan:
            missing.append(plan.get('input_path') or 'pullback folder')
            state.pop('pullback_plan')
        if missing:
            self.status_bar.showMessage(f'Saved paths not found: {", ".join(missing)}.')
        return all(key in gc.centerline_paths for key in ('aorta', 'rca', 'lca'))

    def _run_pipeline(self, state: dict) -> None:
        """Rerun every step the state has data for. Stops at the first failing step (its error was shown)."""
        page = self.page

        def stopped(step: str) -> None:
            self.status_bar.showMessage(f'Restore stopped at {step}: continue from there by hand.')

        page.ccta_geometry.on_prepare_centerlines()
        if self.data.centerline_rca is None:
            return stopped('Prepare Centerlines')
        for edit in state.get('centerline_edits') or []:
            if not self._replay_centerline_edit(edit):
                return stopped(f"the {edit['op']} of {edit['vessel'].upper()} branch {edit['branch']}")
        page.ccta_geometry.on_label_geometry()  # its hooks also label branches and delete again
        if self.data.results is None:
            return stopped('Label Geometry')
        page.tree.on_discretize()
        if self.data.vessel_tree is None:
            return stopped('Discretize Vessel Tree')
        alignment = state.get('alignment')
        if alignment and state.get('pullback_plan'):
            page.alignment.on_load_pullback()
            if self.data.iv_geometry_pair is None:
                return stopped('Load Pullback')
            if alignment.get('mode') == 'manual':
                page.alignment.on_align_manual()
            else:
                page.alignment.on_align()
            if self.data.aligned is None:
                return stopped('Align')
            if cs.STEP_ANOMALOUS in state.get('labels', {}):
                page.alignment.on_label_anomalous()
        self.status_bar.showMessage('Fusion state restored.')

    def _replay_centerline_edit(self, edit: dict) -> bool:
        branches = self.page.branches
        if edit['op'] == 'split':
            return branches.split_branch(edit['vessel'], edit['branch'], edit['point_index'])
        return branches.merge_branches(edit['vessel'], edit['branch'], edit['other'])

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    def on_centerlines_prepared(self) -> None:
        """Fresh centerlines: the split/merge edits start over."""
        self._centerline_edits = []

    def on_centerline_edit(self, op: str, vessel: str, branch: int, **params) -> None:
        """A split (point_index) or merge (other) of a prepared centerline, replayed in order on restore."""
        self._centerline_edits.append({'op': op, 'vessel': vessel, 'branch': branch, **params})
        self.save()

    def on_label_geometry(self) -> None:
        """A freshly labelled mesh: new index and state, with the saved labels in place of the fresh ones
        when restoring."""
        results = self.data.results
        mesh_path = self.right_half.geometry_column.mesh_path
        if results is None or mesh_path is None:
            return
        self._index = cs.VertexIndex(results['mesh'].vertices)
        self._present = None
        self._path = cs.state_path(mesh_path)
        keys = [k for k, v in results.items() if _is_point_list(k, v)]
        self._state = {'n_vertices': len(self._index), 'keys': {cs.STEP_GEOMETRY: keys}, 'deleted_branches': []}
        self._labelled_ids = {}
        if self._pending is not None and self._pending.get('n_vertices') != len(self._index):
            self.status_bar.showMessage('Saved labels not restored: they belong to a mesh with another vertex count.')
            self._pending = None
        saved = (self._pending or {}).get('labels', {})
        if cs.STEP_GEOMETRY in saved:
            geometry = saved.pop(cs.STEP_GEOMETRY)
            cs.apply_step_labels(results, geometry, self._index, self._present_mask())
            self._state['keys'][cs.STEP_GEOMETRY] = sorted(set(keys) | set(geometry))
        self.save()

    def on_branches_labelled(self) -> None:
        """Label Branches ran (also after a split/merge). Applies the saved branch labels once the centerline
        branch ids match the saved ones, then deletes the saved deleted branches again if asked to."""
        if self._index is None or self.data.results is None:
            return
        self._state.setdefault('keys', {})[cs.STEP_BRANCHES] = list(cs.BRANCH_KEYS)
        self._labelled_ids = {v: self._branch_ids(v) for v in ('rca', 'lca')}
        pending = self._pending or {}
        saved = pending.get('labels', {}).get(cs.STEP_BRANCHES)
        replay: list[dict] = []
        if saved is not None:
            if (
                _with_deleted(pending.get('branch_ids') or {}, pending.get('deleted_branches') or [])
                == self._labelled_ids
            ):
                cs.apply_step_labels(self.data.results, saved, self._index, self._present_mask())
                for vessel, names in pending.get('branch_names', {}).items():
                    self.data.branch_names[vessel] = {int(b): n for b, n in names.items()}
                pending['labels'].pop(cs.STEP_BRANCHES)
                replay = pending.pop('deleted_branches', [])
                if not pending.get('replay_deletions'):
                    replay = []
                self.page.branches.refresh_labels_scene()
            else:
                self.status_bar.showMessage(
                    'Saved branch labels not restored yet: the centerline branches differ from the saved ones '
                    '(split/merge them the same way to restore).'
                )
        # Deleted again once the tree exists, as when they were deleted: the tree is pruned, not
        # rediscretized, so Undo Delete keeps working.
        self._deferred_deletions = replay
        self.save()
        if self.data.vessel_tree is not None:
            self._replay_deletions()

    def _replay_deletions(self) -> None:
        replay, self._deferred_deletions = self._deferred_deletions, []
        for entry in replay:
            if entry['branch'] in self._branch_ids(entry['vessel']):
                self.page.branches.delete_branch(entry['vessel'], entry['branch'], entry.get('fill_hole', True))

    def _branch_ids(self, vessel: str) -> list[int]:
        centerline = self.data.centerline(vessel)
        return sorted({p.branch_id for p in centerline.points}) if centerline is not None else []

    def on_branch_deleted(self, vessel: str, branch: int, fill_hole: bool, name: str, removed: list[int]) -> None:
        self._state.setdefault('deleted_branches', []).append(
            {'vessel': vessel, 'branch': branch, 'fill_hole': fill_hole, 'name': name, 'removed': list(removed)}
        )
        self.save()

    def on_branch_restored(self, vessel: str, removed: list[int]) -> None:
        """Undo of a deletion: forget the last deletion of those branches."""
        deleted = self._state.get('deleted_branches', [])
        for i in range(len(deleted) - 1, -1, -1):
            if deleted[i]['vessel'] == vessel and deleted[i]['branch'] in removed:
                del deleted[i]
                break
        self.save()

    def on_tree_ready(self) -> None:
        """The vessel tree exists: delete the saved deleted branches again, select the saved alignment target
        and fill in its settings."""
        self._replay_deletions()
        pending = self._pending or {}
        alignment = pending.get('alignment')
        if not alignment or pending.get('alignment_prefilled'):
            return
        pending['alignment_prefilled'] = True  # kept: the overlap step compares its target
        vessel, branch = alignment['vessel'], alignment['branch']
        ic = self.right_half.intravascular_column
        ic.set_centerline_choices(self.page.tree.centerline_choices(), vessel, branch)
        self.page.tree.select_branch(vessel, branch, alignment.get('reference_index', 0))
        ic.set_align_with_parents(alignment.get('with_parents', False))
        ic.set_manual_rotation_angle(alignment.get('rotation_deg', 0.0))
        ic.set_manual_ref_point_offset(alignment.get('ref_point_offset', 0))
        ic.set_align_settings(alignment.get('settings') or {})

    def on_pullback_loaded(self, plan: dict) -> None:
        self._pullback_plan = dict(plan)
        self.save()

    def on_aligned(self, *, vessel: str, branch: int, mode: str, rotation_deg: float, point) -> None:
        ic = self.right_half.intravascular_column
        self._state['alignment'] = {
            'vessel': vessel,
            'branch': branch,
            'reference_index': self.data.selected_reference_index[vessel],
            'with_parents': ic.align_with_parents(),
            'mode': mode,
            'rotation_deg': float(rotation_deg),
            'ref_point_offset': ic.manual_ref_point_offset() if mode == 'manual' else 0,
            'centerline_point': [float(c) for c in point],
            'settings': ic.manual_align_kwargs() if mode == 'manual' else ic.align_kwargs(),
        }
        self.save()

    def on_scaling_computed(self, vessel: str, factors: dict) -> None:
        """Find Scalings ran. Kept for analysis, like the factors actually applied (see on_scaling_applied)."""
        scaling = self._state.setdefault('scaling', {})
        scaling['vessel'] = vessel
        scaling['computed'] = {k: float(v) for k, v in factors.items()}
        self.save()

    def on_scaling_applied(self, vessel: str, factors: dict) -> None:
        """The factors the mesh was scaled with, manual edits included."""
        scaling = self._state.setdefault('scaling', {})
        scaling['vessel'] = vessel
        scaling['applied'] = {k: float(v) for k, v in factors.items()}
        self.save()

    def on_anomalous_labelled(self, keys_before) -> None:
        """Label Overlap Region added its regions. Restores the saved ones if the alignment target is the same."""
        results = self.data.results
        if self._index is None or results is None:
            return
        new_keys = [k for k, v in results.items() if k not in keys_before and _is_point_list(k, v)]
        known = self._state.setdefault('keys', {}).get(cs.STEP_ANOMALOUS, [])
        self._state['keys'][cs.STEP_ANOMALOUS] = sorted(set(known) | set(new_keys))
        pending = self._pending or {}
        saved = pending.get('labels', {}).get(cs.STEP_ANOMALOUS)
        target = pending.get('alignment') or {}
        if saved is not None and (target.get('vessel'), target.get('branch')) == (
            self.data.aligned_vessel,
            self.data.aligned_branch,
        ):
            cs.apply_step_labels(results, saved, self._index, self._present_mask())
            pending['labels'].pop(cs.STEP_ANOMALOUS)
        self.save()

    def on_scaled(self, old_mesh, new_mesh) -> None:
        """Scaling moved vertices (same order): follow them, so later label edits still map back."""
        if self._index is not None:
            self._index.follow(old_mesh, new_mesh)
            self._present = None


def _with_deleted(branch_ids: dict, deleted: list[dict]) -> dict[str, list[int]]:
    """Saved branch ids plus the deleted ones. Files written before deletions were counted in branch_ids
    may lack them (and 'removed', so only the branch itself is known there)."""
    out = {v: set(ids) for v, ids in branch_ids.items()}
    for entry in deleted:
        out.setdefault(entry['vessel'], set()).update(entry.get('removed', [entry['branch']]))
    return {v: sorted(ids) for v, ids in out.items()}


def _is_point_list(key: str, value) -> bool:
    """A vertex label list of results (not the mesh, a rim, or the per-branch dicts)."""
    return key.endswith('_points') and 'boundary' not in key and isinstance(value, list)
