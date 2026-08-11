from types import SimpleNamespace

import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from domain.fusion_types import FusionScene
from domain.runtime_types import FusionRuntimeData
from pages.fusion import colors, pipeline
from pages.fusion.left_half.left_half import LeftHalf
from pages.fusion.progress_worker import StdoutCapturingWorker
from pages.fusion.right_half.right_half import RightHalf
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


class FusionPage(QWidget):
    def __init__(self, config: SimpleNamespace, status_bar) -> None:
        super().__init__()
        self.config: SimpleNamespace = config
        self.status_bar = status_bar
        self.data = FusionRuntimeData()
        self._remesh_worker: StdoutCapturingWorker | None = None
        # Every point of the RCA/LCA centerlines currently drawn in the Centerline Branches
        # scene — pickable for Split, not just the numbered sharp-angle markers, which are
        # only a visual hint — and which one (if any) was last clicked. Rebuilt from scratch
        # by _refresh_branch_scene on every prepare/split/merge, since branch IDs get
        # reassigned after each edit.
        self._branch_markers: list[dict] = []
        self._selected_branch_marker: dict | None = None

        self.left_half = LeftHalf(self)
        self.right_half = RightHalf(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_half())
        splitter.addWidget(self.right_half())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(self._build_top_bar())
        layout.addWidget(splitter)

        self._connect_signals()

    def shutdown(self) -> None:
        if self._remesh_worker is not None:
            self._remesh_worker.wait()
        self.left_half.viewer.shutdown()

    # ------------------------------------------------------------------

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        open_case_btn = QPushButton('Open Case Folder…')
        open_case_btn.setToolTip('Sets the default browse folder for the file pickers below')
        open_case_btn.clicked.connect(self._on_open_case)
        bar.addWidget(open_case_btn)
        bar.addStretch(1)
        return bar

    def _on_open_case(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Open Case Folder')
        if not path:
            return
        self.data.case_dir = path
        self.right_half.geometry_column.set_default_dir(path)
        self.right_half.intravascular_column.set_default_dir(path)
        self.status_bar.showMessage(f'Case folder: {path}')

    def _on_clear_all_data(self) -> None:
        reply = QMessageBox.question(
            self,
            'Clear All Data',
            'Discard every loaded/computed fusion result (centerlines, vessel tree, '
            'alignment, scaling, stitched mesh) and clear the 3-D viewer?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.data = FusionRuntimeData()
        self._branch_markers = []
        self._selected_branch_marker = None
        self.left_half.branch_toolbar.set_selected_marker(None)
        self.left_half.branch_toolbar.set_branch_choices([], [])
        for scene in FusionScene:
            self.left_half.viewer.clear_scene(scene)
            self.left_half.refresh_toolbar(scene)
        self.status_bar.showMessage('Fusion data cleared.')

    def _connect_signals(self) -> None:
        gc = self.right_half.geometry_column
        gc.run_label_geometry_requested.connect(self._on_run_label_geometry)
        gc.prepare_centerlines_requested.connect(self._on_run_prepare_centerlines)
        gc.run_discretize_tree_requested.connect(self._on_run_discretize_tree)
        gc.geometry_files_changed.connect(self._on_geometry_preview)

        self.left_half.tree_toolbar.reference_selected.connect(self._select_rca_reference)
        self.left_half.tree_toolbar.lca_reference_selected.connect(self._select_lca_reference)
        self.left_half.branch_toolbar.cos_threshold_changed.connect(self._on_branch_cos_threshold_changed)
        self.left_half.branch_toolbar.split_requested.connect(self._on_split_branch_requested)
        self.left_half.branch_toolbar.merge_requested.connect(self._on_merge_branches_requested)
        self.left_half.viewer.point_picked.connect(self._on_point_picked)
        for toolbar in (
            self.left_half.geometry_toolbar,
            self.left_half.branch_toolbar,
            self.left_half.intravascular_loaded_toolbar,
            self.left_half.alignment_toolbar,
            self.left_half.tree_toolbar,
        ):
            toolbar.clear_all_data_requested.connect(self._on_clear_all_data)

        ic = self.right_half.intravascular_column
        ic.run_load_requested.connect(self._on_run_load_pullback)
        ic.run_align_requested.connect(self._on_run_align)
        ic.run_align_manual_requested.connect(self._on_run_align_manual)
        ic.reference_vessel_changed.connect(self._on_reference_vessel_changed)
        ic.reference_index_changed.connect(self._on_reference_index_changed)
        ic.run_label_anomalous_requested.connect(self._on_run_label_anomalous)

        fc = self.right_half.fusion_column
        fc.run_compute_scaling_requested.connect(self._on_run_compute_scaling)
        fc.run_apply_scaling_requested.connect(self._on_run_apply_scaling)
        fc.run_remove_points_requested.connect(self._on_run_remove_points)
        fc.run_stitch_requested.connect(self._on_run_stitch)
        fc.run_remesh_requested.connect(self._on_run_remesh)
        fc.run_smooth_requested.connect(self._on_run_smooth)
        fc.export_requested.connect(self._on_export)

    def _require(self, ok: bool, message: str) -> bool:
        if not ok:
            ErrorMessage(self, message)
        return ok

    def _run(self, busy_message: str, done_message: str, fn, *args, **kwargs):
        """Run a pipeline call with a status-bar message and a shared error path."""
        self.status_bar.showMessage(busy_message)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            logger.exception(f'Fusion pipeline step failed: {fn}')
            ErrorMessage(self, str(e))
            self.status_bar.showMessage('Failed — see log')
            return None
        self.status_bar.showMessage(done_message)
        return result

    # ------------------------------------------------------------------
    # Column 1: CCTA geometry + centerlines
    # ------------------------------------------------------------------

    def _on_geometry_preview(self) -> None:
        """Show the raw mesh + centerlines as soon as they're picked, before Prepare
        Centerlines/Run Label Geometry exist to branch/color/label anything."""
        gc = self.right_half.geometry_column
        viewer = self.left_half.viewer

        if gc.mesh_path is not None:
            try:
                mesh = pipeline.load_ccta_mesh(gc.mesh_path)
            except Exception as e:
                logger.warning(f'Could not load CCTA mesh for preview: {e}')
            else:
                viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'mesh', mesh, color=(200, 200, 200), opacity=0.4)

        for key, path in gc.centerline_paths.items():
            try:
                cl = pipeline.load_centerline(path, key.upper())
            except Exception as e:
                logger.warning(f'Could not load {key} centerline for preview: {e}')
                continue
            layer_key = f'centerline_{key}'
            viewer.add_points(
                FusionScene.CCTA_GEOMETRY,
                layer_key,
                np.array(cl.points_as_tuples()),
                color=colors.CENTERLINE_COLORS[layer_key],
                size=4.0,
            )

        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)

    def _on_run_prepare_centerlines(self) -> None:
        """Load + prepare all three centerlines: the aorta first (no reference, no branch
        detection), then RCA/LCA oriented to it (see pipeline.prepare_centerline)."""
        gc = self.right_half.geometry_column
        if not self._require(
            all(k in gc.centerline_paths for k in ('aorta', 'rca', 'lca')),
            'Load all three centerlines (aorta, RCA, LCA) first.',
        ):
            return

        def _run():
            aorta_raw = pipeline.load_centerline(gc.centerline_paths['aorta'], 'Aorta')
            aorta_cl = pipeline.prepare_centerline(aorta_raw, **gc.prepare_centerline_kwargs('aorta'))
            rca_raw = pipeline.load_centerline(gc.centerline_paths['rca'], 'RCA')
            rca_cl = pipeline.prepare_centerline(
                rca_raw, ref_centerline=aorta_cl, **gc.prepare_centerline_kwargs('rca')
            )
            lca_raw = pipeline.load_centerline(gc.centerline_paths['lca'], 'LCA')
            lca_cl = pipeline.prepare_centerline(
                lca_raw, ref_centerline=aorta_cl, **gc.prepare_centerline_kwargs('lca')
            )
            return aorta_cl, rca_cl, lca_cl

        result = self._run('Preparing centerlines…', 'Centerlines prepared.', _run)
        if result is None:
            return
        self.data.centerline_aorta, self.data.centerline_rca, self.data.centerline_lca = result
        self._refresh_geometry_scene()
        self._refresh_branch_scene()
        self.left_half.show_scene(FusionScene.CENTERLINE_BRANCHES)

    def _on_run_label_geometry(self) -> None:
        gc = self.right_half.geometry_column
        if not self._require(gc.mesh_path is not None, 'Load a CCTA mesh first.'):
            return
        if not self._require(
            None not in (self.data.centerline_aorta, self.data.centerline_rca, self.data.centerline_lca),
            'Prepare all three centerlines first.',
        ):
            return
        mesh_path = gc.mesh_path
        assert mesh_path is not None

        def _run():
            mesh = pipeline.load_ccta_mesh(mesh_path)
            return pipeline.run_label_geometry(
                mesh,
                self.data.centerline_aorta,
                self.data.centerline_rca,
                self.data.centerline_lca,
                **gc.label_geometry_kwargs(),
            )

        results = self._run('Running label_geometry…', 'label_geometry done.', _run)
        if results is None:
            return
        self.data.results = results
        self._refresh_geometry_scene()
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)
        self._run_label_branches_pair()

    def _run_label_branches_pair(self) -> None:
        """Project the current RCA/LCA branch structure onto label_geometry's labelled
        surface points. No button of its own — called right after label_geometry, and
        again after every split_branch/merge_branches edit, since both change the branch
        structure this needs to reflect. No-op until label_geometry has actually run."""
        if None in (self.data.results, self.data.centerline_rca, self.data.centerline_lca):
            return
        results = self._run(
            'Labeling branches…',
            'Branches labeled.',
            pipeline.run_label_branches_pair,
            self.data.centerline_rca,
            self.data.centerline_lca,
            self.data.results,
        )
        if results is None:
            return
        self.data.results = results

    def _on_run_discretize_tree(self) -> None:
        gc = self.right_half.geometry_column
        if not self._require(
            self.data.centerline_aorta is not None and self.data.results is not None,
            'Prepare centerlines, run Label Geometry, and Label Branches (Pair) first.',
        ):
            return
        tree = self._run(
            'Discretizing vessel tree…',
            'Vessel tree ready.',
            pipeline.run_discretize_vessel_tree,
            self.data.centerline_aorta,
            self.data.centerline_rca,
            self.data.centerline_lca,
            self.data.results,
            **gc.discretize_tree_kwargs(),
        )
        if tree is None:
            return
        self.data.vessel_tree = tree

        self.left_half.tree_toolbar.set_references(self._reference_labels('rca'))
        self.left_half.tree_toolbar.set_lca_references(self._reference_labels('lca'))

        self._refresh_tree_scene()
        self._select_rca_reference(0)
        self._select_lca_reference(0)
        self._sync_intravascular_reference_choices()
        self.left_half.show_scene(FusionScene.VESSEL_TREE)

    def _reference_labels(self, vessel: str) -> list[str]:
        """Label list for vessel_tree.rca_references/lca_references — shared by the Vessel
        Tree tab's own dropdowns and the Intravascular Alignment column's Reference
        dropdown, so both always offer the same choices for a given vessel."""
        tree = self.data.vessel_tree
        if tree is None:
            return []
        refs = tree.rca_references if vessel == 'rca' else tree.lca_references
        prefix = vessel.upper()
        return [f'{prefix} ostium'] + [f'{prefix} branch {i}' for i in range(1, len(refs))]

    def _sync_intravascular_reference_choices(self) -> None:
        """Repopulate the Intravascular Alignment column's Reference dropdown for whichever
        vessel (RCA/LCA) it currently has selected, and select whatever index that vessel
        already has active — call after the vessel_tree changes or the column's Centerline
        (RCA/LCA) selector changes."""
        ic = self.right_half.intravascular_column
        vessel = ic.reference_vessel()
        ic.set_reference_choices(self._reference_labels(vessel))
        index = self.data.selected_rca_reference_index if vessel == 'rca' else self.data.selected_lca_reference_index
        ic.set_selected_reference_index(index)

    def _select_rca_reference(self, index: int) -> None:
        """Apply reference triplet `index` (chosen via the Vessel Tree dropdown, a scene
        click, or the Intravascular Alignment column's Reference dropdown when RCA is the
        selected vessel there) and highlight it in the viewer."""
        tree = self.data.vessel_tree
        if tree is None:
            return
        try:
            triplet = tree.rca_references[index]
        except IndexError:
            logger.warning(f'Vessel tree has no rca_references[{index}].')
            return
        self.data.selected_rca_reference_index = index
        self.left_half.tree_toolbar.set_selected_index(index)
        self.left_half.viewer.add_points(
            FusionScene.VESSEL_TREE, 'selected_reference', np.array(triplet), color=(255, 255, 255), size=14.0
        )
        self._sync_intravascular_reference_ui('rca', index, triplet)

    def _select_lca_reference(self, index: int) -> None:
        """Same as _select_rca_reference but for the LCA."""
        tree = self.data.vessel_tree
        if tree is None:
            return
        try:
            triplet = tree.lca_references[index]
        except IndexError:
            logger.warning(f'Vessel tree has no lca_references[{index}].')
            return
        self.data.selected_lca_reference_index = index
        self.left_half.tree_toolbar.set_selected_lca_index(index)
        self.left_half.viewer.add_points(
            FusionScene.VESSEL_TREE, 'selected_lca_reference', np.array(triplet), color=(0, 255, 255), size=14.0
        )
        self._sync_intravascular_reference_ui('lca', index, triplet)

    def _sync_intravascular_reference_ui(self, vessel: str, index: int, triplet) -> None:
        """Keep the Intravascular Alignment column's Reference dropdown and Aortic/Superior/
        Inferior display in step with whichever RCA/LCA reference was just selected — but
        only when that column currently has this same vessel chosen, so selecting an LCA
        reference in the Vessel Tree tab doesn't clobber an in-progress RCA alignment setup
        (and vice versa)."""
        ic = self.right_half.intravascular_column
        if ic.reference_vessel() != vessel:
            return
        ic.set_selected_reference_index(index)
        ic.set_reference_points(triplet[0], triplet[1], triplet[2])

    def _on_reference_vessel_changed(self, vessel: str) -> None:
        """The Intravascular Alignment column's Centerline (RCA/LCA) selector changed —
        repopulate its Reference dropdown for the new vessel and refresh the Aortic/
        Superior/Inferior display to match whatever was already selected for it."""
        self._sync_intravascular_reference_choices()
        tree = self.data.vessel_tree
        if tree is None:
            return
        refs = tree.rca_references if vessel == 'rca' else tree.lca_references
        index = self.data.selected_rca_reference_index if vessel == 'rca' else self.data.selected_lca_reference_index
        if 0 <= index < len(refs):
            triplet = refs[index]
            self.right_half.intravascular_column.set_reference_points(triplet[0], triplet[1], triplet[2])

    def _on_reference_index_changed(self, index: int) -> None:
        """The Intravascular Alignment column's own Reference dropdown changed — route to
        the same selection path as the Vessel Tree tab so everything stays in sync."""
        if self.right_half.intravascular_column.reference_vessel() == 'rca':
            self._select_rca_reference(index)
        else:
            self._select_lca_reference(index)

    def _on_point_picked(self, x: float, y: float, z: float, scene_value: str) -> None:
        if scene_value == FusionScene.CENTERLINE_BRANCHES.value:
            self._on_branch_marker_picked(x, y, z)
            return
        if scene_value != FusionScene.VESSEL_TREE.value or self.data.vessel_tree is None:
            return
        picked = np.array([x, y, z])
        best_cl, best_index, best_dist = 'rca', 0, float('inf')
        for cl_name, refs in (
            ('rca', self.data.vessel_tree.rca_references),
            ('lca', self.data.vessel_tree.lca_references),
        ):
            for i, triplet in enumerate(refs):
                for pt in triplet:
                    dist = float(np.linalg.norm(np.array(pt) - picked))
                    if dist < best_dist:
                        best_dist = dist
                        best_cl, best_index = cl_name, i
        if best_cl == 'rca':
            self._select_rca_reference(best_index)
        else:
            self._select_lca_reference(best_index)

    def _refresh_geometry_scene(self) -> None:
        """Recreate multimodars' plot_results_key (its label_geometry/label_anomalous_region
        control_plot) as native VTK layers: translucent base mesh + one colored point cloud
        per labeled region present in results, plus the three centerlines."""
        viewer = self.left_half.viewer
        results = self.data.results
        if results is not None and 'mesh' in results:
            viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'mesh', results['mesh'], color=(200, 200, 200), opacity=0.4)
        if results is not None:
            for key, color in colors.REGION_COLORS.items():
                points = results.get(key)
                if points:
                    viewer.add_points(FusionScene.CCTA_GEOMETRY, key, np.array(points), color=color)
        for key, cl in (
            ('centerline_aorta', self.data.centerline_aorta),
            ('centerline_rca', self.data.centerline_rca),
            ('centerline_lca', self.data.centerline_lca),
        ):
            if cl is not None:
                # Points, not a polyline: points_as_tuples() concatenates every branch back
                # to back, so connecting them sequentially draws spurious lines jumping
                # between branches. Loose points also make it easy to eyeball point spacing.
                viewer.add_points(
                    FusionScene.CCTA_GEOMETRY,
                    key,
                    np.array(cl.points_as_tuples()),
                    color=colors.CENTERLINE_COLORS[key],
                    size=4.0,
                )
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def _refresh_branch_scene(self) -> None:
        """Recreate multimodars' plot_centerline_branches/plot_centerline_edges as native
        VTK layers: RCA/LCA colored per branch (see colors.BRANCH_COLORS_RCA/LCA), with
        sharp-angle positions additionally marked and numbered as a splitting hint — every
        point on every branch is picked up into self._branch_markers below, though, so
        Pick Point can split anywhere, not just at a numbered marker. Rebuilds from scratch
        every time, since split_branch/merge_branches reassign branch IDs (by descending
        length) on every edit — there's no stable id to update in place."""
        viewer = self.left_half.viewer
        viewer.clear_scene(FusionScene.CENTERLINE_BRANCHES)
        self._branch_markers = []
        self._selected_branch_marker = None
        self.left_half.branch_toolbar.set_selected_marker(None)
        cos_threshold = self.left_half.branch_toolbar.cos_threshold.value()

        if self.data.centerline_aorta is not None:
            viewer.add_points(
                FusionScene.CENTERLINE_BRANCHES,
                'aorta',
                np.array(self.data.centerline_aorta.points_as_tuples()),
                color=colors.CENTERLINE_COLORS['centerline_aorta'],
                size=3.0,
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
                    size=4.0,
                )
                branch_start = cl.branch_start_indices[branch_id] if branch_id < len(cl.branch_start_indices) else 0
                # Every point on this branch is pickable for Split, not just the numbered
                # sharp-angle ones below — lets the user split anywhere along a branch,
                # not only at spots the cos-threshold heuristic happened to flag.
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
                    size=9.0,
                )
                viewer.add_labels(
                    FusionScene.CENTERLINE_BRANCHES,
                    f'{cl_name}_sharp_angle_labels',
                    np.array(label_points),
                    label_texts,
                    color=colors.SHARP_ANGLE_LABEL_COLOR,
                )

        self.left_half.branch_toolbar.set_branch_choices(branch_ids_by_cl['rca'], branch_ids_by_cl['lca'])
        self.left_half.refresh_toolbar(FusionScene.CENTERLINE_BRANCHES)

        # find_sharp_angles only decides which points get a numbered marker in the scene —
        # it never touches branch structure/colors, so nudging the threshold can look like
        # "nothing happened" if you're watching the branch colors. Report the actual count
        # so it's obvious the threshold is doing something even when it's subtle on screen
        # (or genuinely finding nothing, e.g. after heavy smoothing removed the sharp bends).
        total_sharp = sum(sharp_angle_counts.values())
        self.status_bar.showMessage(
            f'Sharp-angle markers (cos ≥ {cos_threshold:.2f}): '
            f"RCA {sharp_angle_counts.get('rca', 0)}, LCA {sharp_angle_counts.get('lca', 0)} "
            f'({total_sharp} total).'
        )

    def _on_branch_cos_threshold_changed(self, _value: float) -> None:
        self._refresh_branch_scene()

    def _on_branch_marker_picked(self, x: float, y: float, z: float) -> None:
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

    def _on_split_branch_requested(self) -> None:
        marker = self._selected_branch_marker
        if not self._require(marker is not None, 'Click a sharp-angle marker in the scene first.'):
            return
        assert marker is not None
        cl_attr = 'centerline_rca' if marker['centerline'] == 'rca' else 'centerline_lca'
        cl = getattr(self.data, cl_attr)
        try:
            new_cl = cl.split_branch(marker['branch_id'], marker['point_index']).orient_by_max_z()
        except Exception as e:
            logger.exception('split_branch failed')
            ErrorMessage(self, str(e))
            return
        setattr(self.data, cl_attr, new_cl)
        self.status_bar.showMessage(
            f"Split {marker['centerline'].upper()} branch {marker['branch_id']} at point {marker['point_index']}."
        )
        self._refresh_branch_scene()
        self._run_label_branches_pair()

    def _on_merge_branches_requested(self, cl_name: str, branch_id_a: int, branch_id_b: int) -> None:
        cl_attr = 'centerline_rca' if cl_name == 'rca' else 'centerline_lca'
        cl = getattr(self.data, cl_attr)
        if not self._require(cl is not None, 'Prepare centerlines first.'):
            return
        try:
            new_cl = cl.merge_branches(branch_id_a, branch_id_b).orient_by_max_z()
        except Exception as e:
            logger.exception('merge_branches failed')
            ErrorMessage(self, str(e))
            return
        setattr(self.data, cl_attr, new_cl)
        self.status_bar.showMessage(f'Merged {cl_name.upper()} branches {branch_id_a} and {branch_id_b}.')
        self._refresh_branch_scene()
        self._run_label_branches_pair()

    def _refresh_tree_scene(self) -> None:
        """Recreate multimodars' plot_vessel_tree as native VTK layers."""
        tree = self.data.vessel_tree
        if tree is None:
            return
        viewer = self.left_half.viewer
        centroids: list[tuple[float, float, float]] = []

        def add_contours(key: str, contours, color: tuple[int, int, int]) -> None:
            points = [p for c in contours for p in c.points_as_tuples()]
            if points:
                viewer.add_points(FusionScene.VESSEL_TREE, key, np.array(points), color=color, size=4.0)
            centroids.extend(c.centroid for c in contours)

        add_contours('tree_aorta', tree.discretized_aorta, colors.TREE_AORTA_COLOR)
        add_contours('tree_rca_main', tree.discretized_rca_main, colors.TREE_RCA_MAIN_COLOR)
        add_contours('tree_lca_main', tree.discretized_lca_main, colors.TREE_LCA_MAIN_COLOR)
        for i, branch in enumerate(tree.rca_branches):
            color = colors.branch_ramp_color(colors.TREE_RCA_MAIN_COLOR, i, len(tree.rca_branches))
            add_contours(f'tree_rca_branch_{i + 1}', branch, color)
        for i, branch in enumerate(tree.lca_branches):
            color = colors.branch_ramp_color(colors.TREE_LCA_MAIN_COLOR, i, len(tree.lca_branches))
            add_contours(f'tree_lca_branch_{i + 1}', branch, color)
        if centroids:
            viewer.add_points(
                FusionScene.VESSEL_TREE,
                'tree_centroids',
                np.array(centroids),
                color=colors.TREE_CENTROID_COLOR,
                size=5.0,
            )

        # Reference triplets (main/ostium + 2 off-axis points used to fix rotation — see
        # colors.TREE_REF_COLORS for why we don't label them CW/CCW). One layer per triplet
        # slot, pooling RCA + LCA references, rather than one layer per triplet — otherwise
        # a tree with many side branches would flood the toolbar with tiny 1-point layers.
        ref_slots: list[list[tuple[float, float, float]]] = [[], [], []]
        for refs in (tree.rca_references, tree.lca_references):
            for triplet in refs:
                for slot in range(3):
                    ref_slots[slot].append(triplet[slot])
        for slot, pts in enumerate(ref_slots):
            if pts:
                viewer.add_points(
                    FusionScene.VESSEL_TREE,
                    f'reference_points_{slot}',
                    np.array(pts),
                    color=colors.TREE_REF_COLORS[slot],
                    size=10.0,
                )

    # ------------------------------------------------------------------
    # Column 2: intravascular alignment
    # ------------------------------------------------------------------

    def _on_run_load_pullback(self) -> None:
        ic = self.right_half.intravascular_column
        kwargs = ic.load_kwargs()
        if not self._require(bool(kwargs['input_path']), 'Select a pullback case folder first.'):
            return
        result = self._run('Loading pullback…', 'Pullback loaded.', pipeline.run_from_file_singlepair, **kwargs)
        if result is None:
            return
        geometry_pair, align_logs = result
        self.data.iv_geometry_pair = geometry_pair
        self.data.iv_align_logs = align_logs

        # Shown before centerline alignment so a bad final result can be traced back to
        # whether it was already wrong here (dia/sys self-alignment) or introduced later.
        self._add_geometry_pair_meshes(FusionScene.INTRAVASCULAR_LOADED, geometry_pair, 'raw_geom')
        self.left_half.show_scene(FusionScene.INTRAVASCULAR_LOADED)

    def _add_geometry_pair_meshes(self, scene: FusionScene, geometry_pair, key_prefix: str) -> None:
        """Loft lumen + wall meshes for both cardiac phases of a PyGeometryPair into `scene`.
        Used for both the pre-centerline-alignment (raw) and post-alignment scenes."""
        viewer = self.left_half.viewer
        for phase_key, geom, color in (
            ('a', getattr(geometry_pair, 'geom_a', None), colors.DIASTOLE_COLOR),
            ('b', getattr(geometry_pair, 'geom_b', None), colors.SYSTOLE_COLOR),
        ):
            if geom is None:
                continue
            key = f'{key_prefix}_{phase_key}'
            try:
                lumen_mesh = pipeline.frames_to_mesh(geom)
            except Exception as e:
                logger.warning(f'Could not loft a lumen mesh for {key}: {e}')
            else:
                viewer.add_mesh(scene, key, lumen_mesh, color=color, opacity=0.6)
            try:
                wall_mesh = pipeline.frames_to_mesh(geom, contour_type='Wall')
            except Exception as e:
                logger.warning(f'Could not loft a wall mesh for {key}: {e}')
            else:
                viewer.add_mesh(scene, f'{key}_wall', wall_mesh, color=(220, 220, 220), opacity=0.25)
        self.left_half.refresh_toolbar(scene)

    def _selected_align_centerline(self, vessel: str, vessel_tree):
        """(centerline, references, selected_index) for whichever vessel ('rca'/'lca') the
        Intravascular Alignment column currently has chosen in its Centerline selector."""
        if vessel == 'rca':
            return self.data.centerline_rca, vessel_tree.rca_references, self.data.selected_rca_reference_index
        return self.data.centerline_lca, vessel_tree.lca_references, self.data.selected_lca_reference_index

    def _on_run_align(self) -> None:
        ic = self.right_half.intravascular_column
        vessel = ic.reference_vessel()
        if not self._require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self._require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        vessel_tree = self.data.vessel_tree
        assert vessel_tree is not None
        centerline, references, selected_index = self._selected_align_centerline(vessel, vessel_tree)
        if not self._require(
            centerline is not None and self.data.results is not None,
            'Run label_geometry first.',
        ):
            return

        results = self.data.results
        assert centerline is not None and results is not None

        try:
            ref_points = references[selected_index]
            cl_main = centerline.get_branch(ic.branch_index())
        except (IndexError, AttributeError) as e:
            ErrorMessage(self, f'Could not resolve reference points / branch: {e}')
            return

        result = self._run(
            'Aligning intravascular geometry…',
            'Alignment done.',
            pipeline.run_align_combined,
            cl_main,
            self.data.iv_geometry_pair,
            ref_points[0],
            ref_points[1],
            ref_points[2],
            results.get(f'{vessel}_points', []),
            align_wall_anomalous=self.right_half.geometry_column.has_acute_takeoff(vessel),
            **ic.align_kwargs(),
        )
        if result is None:
            return
        self._apply_align_result(result, cl_main)

    def _on_run_align_manual(self) -> None:
        """Same preconditions as _on_run_align, but rotates by an explicit angle around a
        single reference point instead of searching angle_range_deg. Only meaningful for
        elliptic (anomalous) vessels — see pipeline.run_align_manual."""
        ic = self.right_half.intravascular_column
        vessel = ic.reference_vessel()
        if not self._require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self._require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        vessel_tree = self.data.vessel_tree
        assert vessel_tree is not None
        centerline, references, selected_index = self._selected_align_centerline(vessel, vessel_tree)
        if not self._require(centerline is not None, 'Run label_geometry first.'):
            return
        assert centerline is not None

        try:
            main_ref_pt = references[selected_index][0]
            cl_main = centerline.get_branch(ic.branch_index())
        except (IndexError, AttributeError) as e:
            ErrorMessage(self, f'Could not resolve reference point / branch: {e}')
            return
        ref_point = self._resolve_manual_ref_point(cl_main, main_ref_pt, ic.manual_ref_point_offset())

        result = self._run(
            'Aligning intravascular geometry (manual)…',
            'Manual alignment done.',
            pipeline.run_align_manual,
            cl_main,
            self.data.iv_geometry_pair,
            ic.manual_rotation_angle_deg(),
            ref_point,
            align_wall_anomalous=self.right_half.geometry_column.has_acute_takeoff(vessel),
            **ic.manual_align_kwargs(),
        )
        if result is None:
            return
        self._apply_align_result(result, cl_main)

    def _resolve_manual_ref_point(
        self, cl_main, main_ref_pt: tuple[float, float, float], offset: int
    ) -> tuple[float, float, float]:
        """offset=0 is whichever point of `cl_main` (the same single-branch RCA/LCA
        centerline passed into align_manual/align_combined) lies closest to main_ref_pt —
        not main_ref_pt itself, since that's a vessel-tree reference point and may not sit
        exactly on the centerline. +N/-N then walks N *centerline points* — not the coarser
        step_size-spaced vessel-tree contours — away from/towards point index 0.

        prepare_centerline's orient_to_reference(aorta) guarantees point index 0 of the main
        branch is always the proximal/ostium end, regardless of which reference is currently
        selected — so -N always walks towards the ostium and +N away from it. Clamped to
        [0, len(points)-1]: if the closest point is already index 0 (e.g. the 'ostium'
        reference itself is selected), negative offsets have nowhere to go and clamp back
        to it — there's nothing more proximal than the ostium to walk to."""
        points = cl_main.points_as_tuples()
        distances = [float(np.linalg.norm(np.array(p) - np.array(main_ref_pt))) for p in points]
        base_index = int(np.argmin(distances))
        raw_index = base_index + offset
        index = max(0, min(len(points) - 1, raw_index))
        if index != raw_index:
            direction = 'proximal (towards the ostium)' if raw_index < 0 else 'distal'
            self.status_bar.showMessage(
                f'Ref. point offset clamped: no centerline point further {direction} than the selected '
                f'reference — using the {"first" if raw_index < 0 else "last"} point of the branch instead.'
            )
        return points[index]

    def _apply_align_result(self, result, source_centerline) -> None:
        """`result` is align_combined/align_manual's (aligned_geometry, spacing_mm,
        total_rotation_deg) — multimodars>=0.6.0 no longer hands back a resampled
        centerline itself, just the spacing_mm it used internally, so we resample
        `source_centerline` (the same single-branch RCA/LCA centerline that was passed into
        align_combined/align_manual) ourselves to get the centerline shown alongside the
        aligned geometry. centerline_aorta gets the same treatment — any frames-correlated
        step downstream (label_anomalous_region, find_distal_and_proximal_scaling,
        find_aorta_scaling) needs its centerline argument at this same spacing_mm to line up
        with `frames`, not whatever spacing prepare_centerline originally left it at."""
        self.data.aligned, spacing_mm, total_rotation_deg = result
        self.data.resampled_centerline = source_centerline.resample(spacing_mm)
        self.data.resampled_centerline_aorta = (
            self.data.centerline_aorta.resample(spacing_mm) if self.data.centerline_aorta is not None else None
        )
        # Prefill the Manual group with whatever angle this alignment landed on (automatic
        # search or a previous manual value round-tripped back) so nudging it further starts
        # from here instead of 0.
        self.right_half.intravascular_column.set_manual_rotation_angle(total_rotation_deg)

        self.left_half.viewer.add_points(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'resampled_centerline',
            np.array(self.data.resampled_centerline.points_as_tuples()),
            color=(0, 200, 0),
            size=4.0,
        )
        # geom_a/geom_b are the two cardiac phases from from_file_singlepair's `labels`
        # kwarg (default aligned_dia/aligned_sys — see IntravascularColumn.load_kwargs) —
        # assumed diastole/systole in that order to match the app's existing color convention.
        self._add_geometry_pair_meshes(FusionScene.INTRAVASCULAR_ALIGNED, self.data.aligned, 'aligned_geom')
        self._refresh_aligned_ccta_mesh()
        self.left_half.show_scene(FusionScene.INTRAVASCULAR_ALIGNED)

    def _refresh_aligned_ccta_mesh(self) -> None:
        """Overlay the unlabeled CCTA mesh (no region colors) in the Intravascular Aligned
        scene, so the aligned IV geometry can be checked against it in place. Call again
        whenever results['mesh'] changes (scaling, point removal) to keep it in sync —
        only meaningful once alignment has happened, since that's what makes the two
        geometries share a coordinate frame in the first place."""
        if self.data.aligned is None or self.data.results is None or 'mesh' not in self.data.results:
            return
        self.left_half.viewer.add_mesh(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'ccta_mesh',
            self.data.results['mesh'],
            color=(160, 160, 160),
            opacity=0.35,
        )
        self.left_half.refresh_toolbar(FusionScene.INTRAVASCULAR_ALIGNED)

    # ------------------------------------------------------------------
    # Column 3: fusion
    # ------------------------------------------------------------------

    def _aligned_frames(self):
        if self.data.aligned is None:
            return None
        try:
            return self.data.aligned.geom_a.frames
        except AttributeError:
            return None

    def _on_run_label_anomalous(self) -> None:
        """Label Overlap Region: partitions whichever coronary the pullback was actually
        aligned onto (see the Centerline: RCA/LCA selector in column 2) into proximal/
        overlap/distal sub-regions — must match the alignment vessel, not always the RCA,
        or centerline and results_key end up describing two different vessels."""
        frames = self._aligned_frames()
        if not self._require(frames is not None, 'Align the intravascular geometry first.'):
            return
        vessel = self.right_half.intravascular_column.reference_vessel()
        centerline = self.data.resampled_centerline
        if not self._require(centerline is not None, 'Run label_geometry first.'):
            return
        results = self._run(
            'Labeling overlap region…',
            'Overlap region labeled.',
            pipeline.run_label_anomalous_region,
            centerline,
            frames,
            self.data.results,
            results_key=f'{vessel}_points',
        )
        if results is not None:
            self.data.results = results
            self._refresh_geometry_scene()  # proximal/distal/anomalous_points now exist

    def _on_run_compute_scaling(self) -> None:
        frames = self._aligned_frames()
        if not self._require(frames is not None, 'Align the intravascular geometry first.'):
            return
        vessel = self.right_half.intravascular_column.reference_vessel()
        centerline = self.data.resampled_centerline
        if not self._require(centerline is not None, 'Run label_geometry first.'):
            return
        scalings = self._run(
            'Computing scaling factors…',
            'Scaling factors computed.',
            pipeline.run_find_scalings,
            frames,
            centerline,
            self.data.resampled_centerline_aorta,
            self.data.results,
            vessel=vessel,
        )
        if scalings is None:
            return
        self.data.prox_scaling = scalings['proximal_scaling']
        self.data.distal_scaling = scalings['distal_scaling']
        self.data.aortic_scaling = scalings['aortic_scaling']
        self.right_half.fusion_column.set_scaling_results(scalings)

    def _on_run_apply_scaling(self) -> None:
        vessel = self.right_half.intravascular_column.reference_vessel()
        opposite_vessel = 'lca' if vessel == 'rca' else 'rca'
        centerline = self.data.centerline_rca if vessel == 'rca' else self.data.centerline_lca
        opposite_centerline = self.data.centerline_lca if vessel == 'rca' else self.data.centerline_rca
        if not self._require(
            None not in (self.data.prox_scaling, self.data.distal_scaling, self.data.aortic_scaling),
            'Compute scaling factors first.',
        ):
            return
        if not self._require(
            self.data.results is not None and centerline is not None and self.data.centerline_aorta is not None,
            'Run label_geometry first.',
        ):
            return

        # Read live from the spinboxes, not self.data.*_scaling — the user may have
        # edited them by hand after Compute Scaling Factors filled in the defaults.
        scaling = self.right_half.fusion_column.scaling_values()
        opposite_scaling = scaling['opposite_vessel_scaling']
        if not self._require(
            opposite_scaling == 0.0 or opposite_centerline is not None,
            f'Missing {opposite_vessel.upper()} centerline for the opposite-vessel scaling — '
            'run Label Geometry first, or set Opposite vessel (mm) back to 0.',
        ):
            return

        def _run():
            results = self.data.results
            centerline_aorta = self.data.centerline_aorta
            assert results is not None and centerline is not None and centerline_aorta is not None
            distal_scaling = scaling['distal_scaling']
            aortic_scaling = scaling['aortic_scaling']
            prox_scaling = scaling['proximal_scaling']
            mesh = results['mesh']

            scaled = pipeline.run_scale_region(mesh, results['distal_points'], centerline, distal_scaling)
            results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
            mesh = results['mesh']

            aortic_region = results['aorta_points'] + results[f'{vessel}_removed_points']
            scaled = pipeline.run_scale_region(mesh, aortic_region, centerline_aorta, aortic_scaling)
            results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
            mesh = results['mesh']

            scaled = pipeline.run_scale_region(mesh, results['proximal_points'], centerline, prox_scaling)
            results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
            mesh = results['mesh']

            # Manual-only: no intravascular data for the opposite coronary, so this is
            # skipped entirely (not scaled by 0, just left untouched) unless set by hand.
            if opposite_scaling != 0.0:
                assert opposite_centerline is not None
                scaled = pipeline.run_scale_region(
                    mesh, results[f'{opposite_vessel}_points'], opposite_centerline, opposite_scaling
                )
                results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
            return results

        results_out = self._run('Applying scaling to mesh…', 'Scaling applied.', _run)
        if results_out is None:
            return
        self.data.results = results_out
        self._refresh_geometry_scene()
        self._refresh_aligned_ccta_mesh()

    def _on_run_remove_points(self) -> None:
        fc = self.right_half.fusion_column
        if not self._require(self.data.results is not None, 'Run label_geometry first.'):
            return
        results = self._run(
            'Removing labeled points…',
            'Points removed.',
            pipeline.run_remove_labeled_points,
            self.data.results,
            fc.remove_point_keys(),
        )
        if results is not None:
            self.data.results = results
            self._refresh_geometry_scene()
            self._refresh_aligned_ccta_mesh()

    def _on_run_stitch(self) -> None:
        fc = self.right_half.fusion_column
        if not self._require(self.data.aligned is not None, 'Align the intravascular geometry first.'):
            return
        if not self._require(self.data.results is not None, 'Run label_geometry first.'):
            return
        aligned = self.data.aligned
        results = self.data.results
        assert aligned is not None and results is not None
        stitched = self._run(
            'Stitching CCTA to intravascular…',
            'Stitched.',
            pipeline.run_stitch,
            aligned.geom_a,
            results['mesh'],
            results,
            **fc.stitch_kwargs(),
        )
        if stitched is None:
            return
        self.data.stitched = stitched
        viewer = self.left_half.viewer
        viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'stitched_mesh', stitched['mesh'], color=(230, 180, 60))
        viewer.isolate_layer(FusionScene.CCTA_GEOMETRY, 'stitched_mesh')
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)

    def _on_run_remesh(self) -> None:
        fc = self.right_half.fusion_column
        if not self._require(self.data.stitched is not None, 'Stitch the geometry first.'):
            return
        if self._remesh_worker is not None and self._remesh_worker.isRunning():
            return
        stitched = self.data.stitched
        assert stitched is not None

        progress = QProgressDialog('Fixing & remeshing…', '', 0, 0, self)
        progress.setWindowTitle('Fix & Remesh')
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()
        self.status_bar.showMessage('Fixing & remeshing…')

        worker = StdoutCapturingWorker(pipeline.run_remesh, (stitched['mesh'],), fc.remesh_kwargs(), parent=self)
        worker.line_printed.connect(progress.setLabelText)
        worker.finished_ok.connect(lambda mesh: self._on_remesh_done(progress, mesh))
        worker.failed.connect(lambda message: self._on_remesh_failed(progress, message))
        self._remesh_worker = worker
        worker.start()

    def _on_remesh_done(self, progress: QProgressDialog, mesh) -> None:
        progress.close()
        self._remesh_worker = None
        self.data.final_mesh = mesh
        self.left_half.viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'final_mesh', mesh, color=(230, 230, 230))
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.status_bar.showMessage('Remeshed.')

    def _on_remesh_failed(self, progress: QProgressDialog, message: str) -> None:
        progress.close()
        self._remesh_worker = None
        logger.error(f'Fix & Remesh failed: {message}')
        ErrorMessage(self, message)
        self.status_bar.showMessage('Failed — see log')

    def _on_run_smooth(self) -> None:
        fc = self.right_half.fusion_column
        if not self._require(self.data.final_mesh is not None, 'Fix && remesh first.'):
            return
        mesh = self._run(
            'Smoothing…', 'Smoothed.', pipeline.run_taubin_smooth, self.data.final_mesh, lamb=fc.taubin_lamb()
        )
        if mesh is None:
            return
        self.data.final_mesh = mesh
        self.left_half.viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'final_mesh', mesh, color=(230, 230, 230))
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def _on_export(self, path: str) -> None:
        if not self._require(self.data.final_mesh is not None, 'Nothing to export yet — finish the pipeline first.'):
            return
        self._run('Exporting…', f'Exported: {path}', pipeline.export_mesh, self.data.final_mesh, path)
