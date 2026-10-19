"""Branch labels and centerline branches: label, lasso, rename, delete, split and merge."""

from __future__ import annotations

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from fusion_pipeline import branches as branch_ops
from fusion_pipeline import multimodars_api as mm_api
from pages.fusion.controllers.base import StageController
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


class BranchController(StageController):
    def __init__(self, page) -> None:
        super().__init__(page)
        self._branch_markers: list[dict] = []  # every pickable centerline point for Split
        self._selected_branch_marker: dict | None = None

    def reset(self) -> None:
        """Clear All Data: forget the split markers and the merge choices."""
        self._branch_markers = []
        self._selected_branch_marker = None
        self.left_half.branch_toolbar.set_selected_marker(None)
        self.left_half.branch_toolbar.set_branch_choices([], [])

    def on_label_branches(self) -> None:
        if not self.require(
            None not in (self.data.results, self.data.centerline_rca, self.data.centerline_lca),
            'Run Label Geometry first.',
        ):
            return
        if self.run_label_branches_pair():
            self.left_half.show_scene(FusionScene.BRANCH_LABELS)

    def run_label_branches_pair(self) -> bool:
        """Project the RCA/LCA branch structure onto label_geometry's surface points.
        Re-run after every split/merge (they reassign branch ids), which discards manual lasso edits.
        No-op until label_geometry has run. Returns whether the labels were updated."""
        if None in (self.data.results, self.data.centerline_rca, self.data.centerline_lca):
            return False
        results = self.run(
            'Labeling branches…',
            'Branches labeled — adjust them with the Lasso on the Branch Labels tab.',
            mm_api.run_label_branches_pair,
            self.data.centerline_rca,
            self.data.centerline_lca,
            self.data.results,
            **self.right_half.geometry_column.label_branches_kwargs(),
        )
        if results is None:
            return False
        self.data.results = results
        self.page.fusion.clear_removal_undo()  # snapshots carry the old branch ids
        self.refresh_labels_scene()
        return True

    def _label_layers(self) -> dict[str, tuple[str, int]]:
        """Branch Labels scene layer key -> (vessel, branch id), for every labelled branch."""
        results = self.data.results or {}
        return {
            f'{vessel}_branch_{branch}': (vessel, branch)
            for vessel in ('rca', 'lca')
            for branch in sorted(results.get(f'{vessel}_branches', {}))
        }

    def refresh_labels_scene(self) -> None:
        """One layer per labelled branch, colored and named like that branch on the Centerline Branches
        tab. Rebuilt from scratch (branch ids can disappear), keeping surviving layers' visibility."""
        viewer = self.left_half.viewer
        scene = FusionScene.BRANCH_LABELS
        saved_states = viewer.layer_states(scene)
        viewer.clear_scene(scene)
        results = self.data.results or {}
        if 'mesh' in results:
            viewer.add_mesh(
                scene, 'mesh', results['mesh'], color=colors.CCTA_MESH_COLOR, opacity=colors.BRANCH_LABELS_MESH_OPACITY
            )
        palettes = {'rca': colors.BRANCH_COLORS_RCA, 'lca': colors.BRANCH_COLORS_LCA}
        layer_labels: dict[str, str] = {}
        for key, (vessel, branch) in self._label_layers().items():
            centerline = self.data.centerline(vessel)
            cl_ids = sorted({p.branch_id for p in centerline.points}) if centerline is not None else []
            ids = cl_ids if branch in cl_ids else sorted(results[f'{vessel}_branches'])
            palette = palettes[vessel]
            color = palette[ids.index(branch) % len(palette)]
            name = self.name(vessel, branch)
            points = results[f'{vessel}_branches'][branch]
            if points:
                viewer.add_points(scene, key, np.array(points), color=color)
                layer_labels[key] = name
        for key, (visible, opacity, _color) in saved_states.items():
            viewer.set_layer_visible(scene, key, visible)
            viewer.set_layer_opacity(scene, key, opacity)
        toolbar = self.left_half.branch_labels_toolbar
        toolbar.set_layer_labels(layer_labels)
        toolbar.set_branches(
            [(self.name(vessel, branch), vessel, branch) for vessel, branch in self._label_layers().values()]
        )
        self.left_half.refresh_toolbar(scene)

    def name(self, vessel: str, branch: int) -> str:
        """Shared by the branch's centerline and vertices."""
        return branch_ops.branch_name(self.data.branch_names[vessel], vessel, branch)

    def on_lasso_mode_toggled(self, enabled: bool) -> None:
        """Same guard-then-toggle as the CCTA Geometry lasso, for the Branch Labels tab."""
        if enabled and not self._label_layers():
            ErrorMessage(self.page, 'Run Label Branches first — no branch labels to adjust.')
            self.left_half.branch_labels_toolbar.end_lasso()
            return
        self.left_half.viewer.set_lasso_mode(enabled)

    def on_lasso_closed(self) -> None:
        """Move the lasso-enclosed vertices of one branch to another branch of the same vessel."""
        viewer = self.left_half.viewer
        toolbar = self.left_half.branch_labels_toolbar
        results = self.data.results
        layers = {
            key: vb for key, vb in self._label_layers().items() if results and results[f'{vb[0]}_branches'][vb[1]]
        }
        if results is None or not layers:
            ErrorMessage(self.page, 'No branch labels available to adjust.')
            toolbar.end_lasso()
            return

        name = self.name

        sources = list(layers.values())
        source_labels = [name(*vb) for vb in sources]
        label, ok = QInputDialog.getItem(
            self.page, 'Adjust branch labels', 'Move vertices of:', source_labels, 0, False
        )
        if not ok:
            return
        vessel, source = sources[source_labels.index(label)]
        branches = results[f'{vessel}_branches']
        dest_choices = [b for b in sorted(branches) if b != source]
        if not dest_choices:
            ErrorMessage(self.page, f'{vessel.upper()} has no other branch to move them to.')
            return
        dest_labels = [name(vessel, b) for b in dest_choices]
        label, ok = QInputDialog.getItem(
            self.page,
            'Adjust branch labels',
            f'Move selected "{name(vessel, source)}" vertices to:',
            dest_labels,
            0,
            False,
        )
        if not ok:
            return
        dest = dest_choices[dest_labels.index(label)]

        points = np.array(branches[source])
        inside = viewer.points_inside_lasso(points)
        if not inside.any():
            ErrorMessage(self.page, 'No vertices fell inside the lasso.')
            return
        moved = branch_ops.move_points(branches, source, dest, inside)

        self.refresh_labels_scene()
        stale = ' Re-run Discretize Vessel Tree to use them.' if self.data.vessel_tree is not None else ''
        self.status_bar.showMessage(
            f'Moved {len(moved)} vertices: {name(vessel, source)} → {name(vessel, dest)}.{stale}'
        )
        toolbar.end_lasso()

    def _centerline_layer_labels(self) -> dict[str, str]:
        """Centerline Branches layer key -> branch name plus its id (used by the split/merge controls)."""
        labels = {}
        for vessel, centerline in (('rca', self.data.centerline_rca), ('lca', self.data.centerline_lca)):
            if centerline is None:
                continue
            for branch in sorted({p.branch_id for p in centerline.points}):
                labels[f'{vessel}_branch_{branch}'] = f'{self.name(vessel, branch)} ({branch})'
        return labels

    def on_rename_requested(self, vessel: str, branch: int) -> None:
        """One name for the branch's centerline and its vertices. Empty restores the default."""
        current = self.name(vessel, branch)
        name, ok = QInputDialog.getText(
            self.page, 'Rename branch', f'New name for {current} (empty = default):', text=current
        )
        if not ok:
            return
        name = name.strip()
        names = self.data.branch_names[vessel]
        if name:
            names[branch] = name
        else:
            names.pop(branch, None)
        self._refresh_name_displays()
        self.status_bar.showMessage(f'Renamed {current} to {self.name(vessel, branch)}.')

    def _refresh_name_displays(self) -> None:
        """Re-label every list a branch name shows up in, without recomputing anything."""
        self.refresh_labels_scene()
        self.left_half.branch_toolbar.set_layer_labels(self._centerline_layer_labels())
        self.left_half.refresh_toolbar(FusionScene.CENTERLINE_BRANCHES)
        if self.data.vessel_tree is None:
            return
        ic = self.right_half.intravascular_column
        ic.set_centerline_choices(self.page.tree.centerline_choices(), ic.reference_vessel(), ic.reference_branch())
        for vessel in ('rca', 'lca'):
            branch = self.data.selected_branch[vessel]
            self.left_half.tree_toolbar.set_branches(vessel, self.page.tree.branch_choices(vessel), branch)
            self.page.tree.select_branch(vessel, branch, self.data.selected_reference_index[vessel])

    def on_delete_requested(self, vessel: str, branch: int, fill_hole: bool) -> None:
        """Cut a side branch and its descendants off the mesh and centerline (multimodars' remove_branch).
        Other branches keep their ids and names, so the vessel tree is pruned, not rediscretized. The stitched
        and final meshes lose the branch too (final via transfer_labels), all or none."""
        centerline = self.data.centerline(vessel)
        results = self.data.results
        if not self.require(results is not None and centerline is not None, 'Run Label Branches first.'):
            return
        assert results is not None and centerline is not None
        name = self.name(vessel, branch)
        reply = QMessageBox.question(
            self.page,
            'Delete branch',
            f'Delete {name}, and every branch growing out of it, from the mesh and the '
            f'{vessel.upper()} centerline?' + (' The ostium is closed with a patch.' if fill_hole else ''),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        out = self.run(
            f'Deleting {name}…',
            f'Deleted {name}.',
            branch_ops.remove_branch,
            results,
            centerline,
            branch,
            stitched=self.data.stitched,
            final_mesh=self.data.final_mesh,
            fill_hole=fill_hole,
        )
        if out is None:
            return
        new_stitched, new_final, removed = out.stitched, out.final_mesh, out.removed
        removed_names = ', '.join(self.name(vessel, b) for b in removed)
        setattr(self.data, f'centerline_{vessel}', out.centerline)
        self.data.results = out.results
        self.page.fusion.invalidate_removal()  # stale: the mesh changed
        names = self.data.branch_names[vessel]
        self.data.branch_names[vessel] = {b: n for b, n in names.items() if b not in removed}
        tree = self.data.vessel_tree
        if tree is not None:
            branch_ops.prune_vessel_tree(tree, vessel, removed)
            self.page.cutting.drop_branches(vessel, removed)
            for b in removed:
                self.left_half.viewer.remove_layer(FusionScene.VESSEL_TREE, f'tree_{vessel}_branch_{b}')
            if self.data.selected_branch[vessel] in removed:
                self.data.selected_branch[vessel] = 0
                self.data.selected_reference_index[vessel] = 0
            self.page.tree.refresh_scene()
            self.left_half.refresh_toolbar(FusionScene.VESSEL_TREE)

        if new_stitched is not None:
            self.data.stitched = new_stitched
            self.left_half.viewer.add_mesh(
                FusionScene.CCTA_GEOMETRY, 'stitched_mesh', new_stitched['mesh'], color=colors.STITCHED_MESH_COLOR
            )
        if new_final is not None:
            self.data.final_mesh_undo.clear()  # vertex snapshots of the old topology
            self.page.final_mesh.set_final_mesh(new_final)

        self.page.ccta_geometry.refresh_scene()
        self.refresh_centerline_scene()
        self._refresh_name_displays()  # branch labels scene + tree/alignment choices
        self.page.alignment.refresh_ccta_mesh()
        also = ' from the stitched and final meshes too' if new_final is not None else ''
        also = also or (' from the stitched mesh too' if new_stitched is not None else '')
        self.status_bar.showMessage(f'Deleted {removed_names}{also}.')

    def refresh_centerline_scene(self) -> None:
        """Native VTK version of multimodars' plot_centerline_branches: RCA/LCA colored per branch, sharp
        angles numbered as split hints. Rebuilt from scratch since split/merge reassign branch ids."""
        viewer = self.left_half.viewer
        # Rebuilt from scratch, so snapshot each layer's visibility/opacity and reapply it below.
        saved_states = viewer.layer_states(FusionScene.CENTERLINE_BRANCHES)
        viewer.clear_scene(FusionScene.CENTERLINE_BRANCHES)
        self._branch_markers = []
        self._selected_branch_marker = None
        self.left_half.branch_toolbar.set_selected_marker(None)
        cos_threshold = self.left_half.branch_toolbar.cos_threshold()

        if self.data.centerline_aorta is not None:
            viewer.add_points(
                FusionScene.CENTERLINE_BRANCHES,
                'aorta',
                np.array(self.data.centerline_aorta.points_as_tuples()),
                color=colors.CENTERLINE_COLORS['centerline_aorta'],
                size=colors.BRANCHES_AORTA_POINT_SIZE,
            )

        branch_ids_by_cl: dict[str, list[int]] = {'rca': [], 'lca': []}
        sharp_angle_counts: dict[str, int] = {}
        for cl_name, cl, palette in (
            ('rca', self.data.centerline_rca, colors.BRANCH_COLORS_RCA),
            ('lca', self.data.centerline_lca, colors.BRANCH_COLORS_LCA),
        ):
            if cl is None:
                continue
            by_branch: dict[int, list[tuple[float, float, float]]] = {}
            for p in cl.points:
                by_branch.setdefault(p.branch_id, []).append((p.contour_point.x, p.contour_point.y, p.contour_point.z))
            branch_ids = sorted(by_branch)
            branch_ids_by_cl[cl_name] = branch_ids

            label_points: list[tuple[float, float, float]] = []
            label_texts: list[str] = []
            marker_number = 1
            for i, branch_id in enumerate(branch_ids):
                points = by_branch[branch_id]
                viewer.add_points(
                    FusionScene.CENTERLINE_BRANCHES,
                    f'{cl_name}_branch_{branch_id}',
                    np.array(points),
                    color=palette[i % len(palette)],
                    size=colors.POINT_SIZE,
                )
                branch_start = cl.branch_start_indices[branch_id] if branch_id < len(cl.branch_start_indices) else 0
                # Every point is pickable for Split, not just the sharp-angle markers.
                for local_index, position in enumerate(points):
                    self._branch_markers.append(
                        {
                            'centerline': cl_name,
                            'branch_id': branch_id,
                            'point_index': branch_start + local_index,
                            'position': position,
                        }
                    )
                for point_index in cl.find_sharp_angles(branch_id, cos_threshold):
                    local_index = point_index - branch_start
                    if not (0 <= local_index < len(points)):
                        continue
                    label_points.append(points[local_index])
                    label_texts.append(str(marker_number))
                    marker_number += 1
            sharp_angle_counts[cl_name] = marker_number - 1

            if label_points:
                viewer.add_points(
                    FusionScene.CENTERLINE_BRANCHES,
                    f'{cl_name}_sharp_angles',
                    np.array(label_points),
                    color=colors.SHARP_ANGLE_COLOR,
                    size=colors.SHARP_ANGLE_POINT_SIZE,
                )
                viewer.add_labels(
                    FusionScene.CENTERLINE_BRANCHES,
                    f'{cl_name}_sharp_angle_labels',
                    np.array(label_points),
                    label_texts,
                    color=colors.SHARP_ANGLE_LABEL_COLOR,
                )

        # Restore surviving layers' state before refresh_toolbar, so its checkboxes match.
        for key, (visible, opacity, _color) in saved_states.items():
            viewer.set_layer_visible(FusionScene.CENTERLINE_BRANCHES, key, visible)
            viewer.set_layer_opacity(FusionScene.CENTERLINE_BRANCHES, key, opacity)

        self.left_half.branch_toolbar.set_branch_choices(branch_ids_by_cl['rca'], branch_ids_by_cl['lca'])
        self.left_half.branch_toolbar.set_layer_labels(self._centerline_layer_labels())
        self.left_half.refresh_toolbar(FusionScene.CENTERLINE_BRANCHES)

        # The threshold only changes the markers, not branch colors, so report the count.
        total_sharp = sum(sharp_angle_counts.values())
        self.status_bar.showMessage(
            f'Sharp-angle markers (cos ≥ {cos_threshold:.2f}): '
            f"RCA {sharp_angle_counts.get('rca', 0)}, LCA {sharp_angle_counts.get('lca', 0)} "
            f'({total_sharp} total).'
        )

    def on_cos_threshold_changed(self, _value: float) -> None:
        self.refresh_centerline_scene()

    def on_marker_picked(self, x: float, y: float, z: float) -> None:
        if not self._branch_markers:
            return
        picked = np.array([x, y, z])
        best_marker, best_dist = None, float('inf')
        for marker in self._branch_markers:
            dist = float(np.linalg.norm(np.array(marker['position']) - picked))
            if dist < best_dist:
                best_dist = dist
                best_marker = marker
        self._selected_branch_marker = best_marker
        if best_marker is not None:
            description = f"{best_marker['centerline'].upper()} branch {best_marker['branch_id']} @ point {best_marker['point_index']}"
            self.left_half.branch_toolbar.set_selected_marker(description)

    def on_split_requested(self) -> None:
        marker = self._selected_branch_marker
        if not self.require(marker is not None, 'Click a sharp-angle marker in the scene first.'):
            return
        assert marker is not None
        cl_attr = 'centerline_rca' if marker['centerline'] == 'rca' else 'centerline_lca'
        cl = getattr(self.data, cl_attr)
        try:
            new_cl = cl.split_branch(marker['branch_id'], marker['point_index']).orient_by_max_z()
        except Exception as e:
            logger.exception('split_branch failed')
            ErrorMessage(self.page, str(e))
            return
        setattr(self.data, cl_attr, new_cl)
        self.data.branch_names[marker['centerline']] = {}  # split reassigns every branch id
        self.page.cutting.reset()
        self.status_bar.showMessage(
            f"Split {marker['centerline'].upper()} branch {marker['branch_id']} at point {marker['point_index']}."
        )
        self.refresh_centerline_scene()
        self.run_label_branches_pair()

    def on_merge_requested(self, cl_name: str, branch_id_a: int, branch_id_b: int) -> None:
        cl_attr = 'centerline_rca' if cl_name == 'rca' else 'centerline_lca'
        cl = getattr(self.data, cl_attr)
        if not self.require(cl is not None, 'Prepare centerlines first.'):
            return
        try:
            new_cl = cl.merge_branches(branch_id_a, branch_id_b).orient_by_max_z()
        except Exception as e:
            logger.exception('merge_branches failed')
            ErrorMessage(self.page, str(e))
            return
        setattr(self.data, cl_attr, new_cl)
        self.data.branch_names[cl_name] = {}  # merge reassigns every branch id
        self.page.cutting.reset()
        self.status_bar.showMessage(f'Merged {cl_name.upper()} branches {branch_id_a} and {branch_id_b}.')
        self.refresh_centerline_scene()
        self.run_label_branches_pair()
