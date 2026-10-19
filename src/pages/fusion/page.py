import copy
from types import SimpleNamespace

import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from domain.fusion.runtime_types import FusionRuntimeData
from pages.fusion import pipeline
from pages.fusion.cutting import CutController
from pages.fusion.left_half.left_half import LeftHalf
from pages.fusion.progress_worker import StdoutCapturingWorker
from pages.fusion.right_half.right_half import RightHalf
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from tools.sphere_smooth import local_smooth, local_smooth_region


class FusionPage(QWidget):
    def __init__(self, config: SimpleNamespace, status_bar) -> None:
        super().__init__()
        self.config: SimpleNamespace = config
        self.status_bar = status_bar
        self.data = FusionRuntimeData()
        self._remesh_worker: StdoutCapturingWorker | None = None
        self._branch_markers: list[dict] = []
        self._selected_branch_marker: dict | None = None

        self.left_half = LeftHalf(self)
        self.right_half = RightHalf(self)
        self.cutting = CutController(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_half())
        splitter.addWidget(self.right_half())
        # 50/50 between the 3D view and the columns, also when the window is resized.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1, 1])

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
            'alignment, scaling, stitched mesh) and clear the 3D viewer?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.data = FusionRuntimeData()
        self._reset_smoothing_state()
        self._update_removal_undo_button()
        self.cutting.reset()
        self._branch_markers = []
        self._selected_branch_marker = None
        self.left_half.branch_toolbar.set_selected_marker(None)
        self.left_half.branch_toolbar.set_branch_choices([], [])
        self.right_half.intravascular_column.set_centerline_choices([('RCA', 'rca', 0), ('LCA', 'lca', 0)], 'rca', 0)
        self.right_half.intravascular_column.set_reference_choices([])
        for vessel in ('rca', 'lca'):
            self.left_half.tree_toolbar.set_branches(vessel, [], 0)
            self.left_half.tree_toolbar.set_references(vessel, [], -1)
        for scene in FusionScene:
            self.left_half.viewer.clear_scene(scene)
            self.left_half.refresh_toolbar(scene)
        self.status_bar.showMessage('Fusion data cleared.')

    def _connect_signals(self) -> None:
        gc = self.right_half.geometry_column
        gc.run_label_geometry_requested.connect(self._on_run_label_geometry)
        gc.run_label_branches_requested.connect(self._on_run_label_branches)
        gc.prepare_centerlines_requested.connect(self._on_run_prepare_centerlines)
        gc.run_discretize_tree_requested.connect(self._on_run_discretize_tree)
        gc.geometry_files_changed.connect(self._on_geometry_preview)

        self.left_half.tree_toolbar.branch_selected.connect(self._select_branch)
        self.left_half.tree_toolbar.reference_selected.connect(self._select_reference)
        self.left_half.branch_toolbar.cos_threshold_changed.connect(self._on_branch_cos_threshold_changed)
        self.left_half.branch_toolbar.split_requested.connect(self._on_split_branch_requested)
        self.left_half.branch_toolbar.merge_requested.connect(self._on_merge_branches_requested)
        self.left_half.viewer.point_picked.connect(self._on_point_picked)
        self.left_half.geometry_toolbar.lasso_mode_toggled.connect(self._on_lasso_mode_toggled)
        self.left_half.branch_labels_toolbar.lasso_mode_toggled.connect(self._on_branch_lasso_mode_toggled)
        self.left_half.branch_labels_toolbar.rename_requested.connect(self._on_rename_branch_requested)
        self.left_half.branch_labels_toolbar.delete_requested.connect(self._on_delete_branch_requested)
        self.left_half.viewer.lasso_closed.connect(self._on_lasso_closed)
        for toolbar in (
            self.left_half.geometry_toolbar,
            self.left_half.branch_toolbar,
            self.left_half.branch_labels_toolbar,
            self.left_half.intravascular_loaded_toolbar,
            self.left_half.alignment_toolbar,
            self.left_half.tree_toolbar,
            self.left_half.cut_toolbar,
        ):
            toolbar.clear_all_data_requested.connect(self._on_clear_all_data)
        ct = self.left_half.cut_toolbar
        ct.source_changed.connect(self.cutting.on_source_changed)
        ct.step_requested.connect(self.cutting.on_step)
        ct.cut_requested.connect(self.cutting.on_cut)
        ct.rename_requested.connect(self.cutting.on_rename)
        ct.remove_requested.connect(self.cutting.on_remove)
        self.left_half.tabs.currentChanged.connect(self._on_tab_changed)

        ic = self.right_half.intravascular_column
        ic.run_load_requested.connect(self._on_run_load_pullback)
        ic.run_align_requested.connect(self._on_run_align)
        ic.run_align_manual_requested.connect(self._on_run_align_manual)
        ic.reference_centerline_changed.connect(self._on_reference_centerline_changed)
        ic.reference_index_changed.connect(self._on_reference_index_changed)
        ic.run_label_anomalous_requested.connect(self._on_run_label_anomalous)

        fc = self.right_half.fusion_column
        fc.run_compute_scaling_requested.connect(self._on_run_compute_scaling)
        fc.run_apply_scaling_requested.connect(self._on_run_apply_scaling)
        fc.run_remove_points_requested.connect(self._on_run_remove_points)
        fc.undo_remove_points_requested.connect(self._on_undo_remove_points)
        fc.run_stitch_requested.connect(self._on_run_stitch)
        fc.run_remesh_requested.connect(self._on_run_remesh)
        fc.run_smooth_requested.connect(self._on_run_smooth)
        fc.export_requested.connect(self._on_export)
        fc.export_cut_requested.connect(self.cutting.on_export)

        gt = self.left_half.geometry_toolbar
        viewer = self.left_half.viewer
        gt.sphere_mode_toggled.connect(self._on_sphere_mode_toggled)
        gt.sphere_radius_changed.connect(self._on_sphere_radius_changed)
        gt.undo_requested.connect(self._on_undo_smoothing)
        viewer.sphere_hovered.connect(self._on_sphere_hovered)
        viewer.sphere_clicked.connect(self._on_sphere_clicked)
        viewer.sphere_radius_step.connect(gt.step_sphere_radius)
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)  # fusion page only
        undo_shortcut.activated.connect(self._on_undo_shortcut)

    def _on_tab_changed(self, index: int) -> None:
        # The cuts replay lazily, here and on export, not on every upstream mesh change.
        if self.left_half.scene_at(index) == FusionScene.INLETS_OUTLETS:
            self.cutting.refresh()

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
        """Show the raw mesh and centerlines as soon as they're picked, before any processing."""
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
        """Load and prepare all three centerlines: the aorta first, then RCA/LCA oriented to it."""
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
        self.data.branch_names = {'rca': {}, 'lca': {}}  # new centerlines, new branch ids
        self.cutting.reset()
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
        self._invalidate_removal()  # stale: based on the previous labeling
        self._refresh_geometry_scene()
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)
        self._run_label_branches_pair()

    def _on_run_label_branches(self) -> None:
        if not self._require(
            None not in (self.data.results, self.data.centerline_rca, self.data.centerline_lca),
            'Run Label Geometry first.',
        ):
            return
        if self._run_label_branches_pair():
            self.left_half.show_scene(FusionScene.BRANCH_LABELS)

    def _run_label_branches_pair(self) -> bool:
        """Project the RCA/LCA branch structure onto label_geometry's surface points.
        Re-run after every split/merge (they reassign branch ids), which discards manual lasso edits.
        No-op until label_geometry has run. Returns whether the labels were updated."""
        if None in (self.data.results, self.data.centerline_rca, self.data.centerline_lca):
            return False
        results = self._run(
            'Labeling branches…',
            'Branches labeled — adjust them with the Lasso on the Branch Labels tab.',
            pipeline.run_label_branches_pair,
            self.data.centerline_rca,
            self.data.centerline_lca,
            self.data.results,
            **self.right_half.geometry_column.label_branches_kwargs(),
        )
        if results is None:
            return False
        self.data.results = results
        self._clear_removal_undo()  # snapshots carry the old branch ids
        self._refresh_branch_labels_scene()
        return True

    def _branch_label_layers(self) -> dict[str, tuple[str, int]]:
        """Branch Labels scene layer key -> (vessel, branch id), for every labelled branch."""
        results = self.data.results or {}
        return {
            f'{vessel}_branch_{branch}': (vessel, branch)
            for vessel in ('rca', 'lca')
            for branch in sorted(results.get(f'{vessel}_branches', {}))
        }

    def _refresh_branch_labels_scene(self) -> None:
        """One layer per labelled branch, colored and named like that branch on the Centerline Branches
        tab. Rebuilt from scratch (branch ids can disappear), keeping surviving layers' visibility."""
        viewer = self.left_half.viewer
        scene = FusionScene.BRANCH_LABELS
        saved_states = viewer.layer_states(scene)
        viewer.clear_scene(scene)
        results = self.data.results or {}
        if 'mesh' in results:
            viewer.add_mesh(scene, 'mesh', results['mesh'], color=(200, 200, 200), opacity=0.3)
        palettes = {'rca': colors.BRANCH_COLORS_RCA, 'lca': colors.BRANCH_COLORS_LCA}
        layer_labels: dict[str, str] = {}
        for key, (vessel, branch) in self._branch_label_layers().items():
            centerline = self.data.centerline_rca if vessel == 'rca' else self.data.centerline_lca
            cl_ids = sorted({p.branch_id for p in centerline.points}) if centerline is not None else []
            ids = cl_ids if branch in cl_ids else sorted(results[f'{vessel}_branches'])
            palette = palettes[vessel]
            color = palette[ids.index(branch) % len(palette)]
            name = self._branch_name(vessel, branch)
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
            [
                (self._branch_name(vessel, branch), vessel, branch)
                for vessel, branch in self._branch_label_layers().values()
            ]
        )
        self.left_half.refresh_toolbar(scene)

    def _on_run_discretize_tree(self) -> None:
        gc = self.right_half.geometry_column
        if not self._require(
            self.data.centerline_aorta is not None and self.data.results is not None,
            'Prepare centerlines, run Label Geometry, and Label Branches first.',
        ):
            return
        results = self.data.results
        assert results is not None
        if not self._require(
            'rca_branches' in results and 'lca_branches' in results,
            'Run Label Branches first.',
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
        self.cutting.reset()  # slice indices refer to the old tree

        self.data.selected_branch = {'rca': 0, 'lca': 0}
        self.data.selected_reference_index = {'rca': 0, 'lca': 0}

        self._refresh_tree_scene()
        ic = self.right_half.intravascular_column
        ic.set_centerline_choices(self._centerline_choices(), ic.reference_vessel(), 0)
        for vessel in ('rca', 'lca'):
            self.left_half.tree_toolbar.set_branches(vessel, self._branch_choices(vessel), 0)
            self._select_branch(vessel, 0)
        self.left_half.show_scene(FusionScene.VESSEL_TREE)

    def _tree_branches(self, vessel: str) -> dict:
        """Branch id (0 = main vessel) -> PyDiscretizedBranch. Branches that failed to discretize are missing."""
        tree = self.data.vessel_tree
        if tree is None:
            return {}
        return dict(tree.rca if vessel == 'rca' else tree.lca)

    def _branch_reference_entries(self, vessel: str, branch: int) -> list[tuple[int | None, tuple]]:
        """(child id, triplet) for every reference on one branch: its own ostium first (child None),
        then each child's takeoff, proximal to distal."""
        branches = self._tree_branches(vessel)
        node = branches.get(branch)
        if node is None:
            return []
        entries: list[tuple[int | None, tuple]] = [] if node.ostium is None else [(None, node.ostium)]
        bifurcations = node.bifurcations

        def start_contour(child: int) -> int:
            parent = branches[child].parent if child in branches else None
            return parent[1] if parent is not None else 0

        entries += [(child, bifurcations[child]) for child in sorted(bifurcations, key=start_contour)]
        return entries

    def _branch_references(self, vessel: str, branch: int) -> list:
        """Reference triplets reachable on one branch, see _branch_reference_entries."""
        return [triplet for _child, triplet in self._branch_reference_entries(vessel, branch)]

    def _branch_ids(self, vessel: str) -> list[int]:
        """Every discretized branch that has references to align onto."""
        tree = self.data.vessel_tree
        if tree is None:
            return [0]
        return [b for b in sorted(self._tree_branches(vessel)) if self._branch_reference_entries(vessel, b)]

    def _branch_name(self, vessel: str, branch: int) -> str:
        """The user-given name, else 'RCA main' / 'RCA branch 2'. Shared by the branch's centerline and vertices."""
        default = f'{vessel.upper()} main' if branch == 0 else f'{vessel.upper()} branch {branch}'
        return self.data.branch_names[vessel].get(branch) or default

    def _branch_choices(self, vessel: str) -> list[tuple[str, int]]:
        return [(self._branch_name(vessel, b), b) for b in self._branch_ids(vessel)]

    def _centerline_choices(self) -> list[tuple[str, str, int]]:
        """(label, vessel, branch) for the Intravascular Alignment column's Centerline combo."""
        return [
            (self._branch_name(vessel, b), vessel, b) for vessel in ('rca', 'lca') for b in self._branch_ids(vessel)
        ]

    def _reference_labels(self, vessel: str, branch: int) -> list[str]:
        """Labels for _branch_references(vessel, branch), for the Vessel Tree tab and the Reference dropdown."""
        name = self._branch_name(vessel, branch)
        return [
            f'{name} ostium' if child is None else f'Takeoff of {self._branch_name(vessel, child)}'
            for child, _triplet in self._branch_reference_entries(vessel, branch)
        ]

    def _select_branch(self, vessel: str, branch: int, index: int = 0) -> None:
        """Make `branch` the active branch of `vessel`, repopulate the reference dropdowns with its
        references, and select `index`."""
        if self.data.vessel_tree is None:
            return
        self.data.selected_branch[vessel] = branch
        self.left_half.tree_toolbar.set_selected_branch(vessel, branch)
        labels = self._reference_labels(vessel, branch)
        index = index if 0 <= index < len(labels) else 0
        self.left_half.tree_toolbar.set_references(vessel, labels, index)
        ic = self.right_half.intravascular_column
        if ic.reference_vessel() == vessel:
            ic.set_selected_centerline(vessel, branch)
            ic.set_reference_choices(labels)
        self._select_reference(vessel, index)

    def _select_reference(self, vessel: str, index: int) -> None:
        """Apply reference `index` of `vessel`'s active branch and highlight it in the viewer."""
        branch = self.data.selected_branch[vessel]
        refs = self._branch_references(vessel, branch)
        if not 0 <= index < len(refs):
            logger.warning(f'{vessel.upper()} branch {branch} has no reference [{index}].')
            return
        triplet = refs[index]
        self.data.selected_reference_index[vessel] = index
        self.left_half.tree_toolbar.set_selected_reference(vessel, index)
        layer, color = (
            ('selected_reference', (255, 255, 255)) if vessel == 'rca' else ('selected_lca_reference', (0, 255, 255))
        )
        self.left_half.viewer.add_points(FusionScene.VESSEL_TREE, layer, np.array(triplet), color=color, size=14.0)
        self._sync_intravascular_reference_ui(vessel, index, triplet)

    def _sync_intravascular_reference_ui(self, vessel: str, index: int, triplet) -> None:
        """Sync the Intravascular Alignment column's reference display, only if it has the same vessel
        selected (so picking an LCA reference doesn't clobber an RCA setup)."""
        ic = self.right_half.intravascular_column
        if ic.reference_vessel() != vessel:
            return
        ic.set_selected_reference_index(index)
        ic.set_reference_points(triplet[0], triplet[1], triplet[2])

    def _on_reference_centerline_changed(self, vessel: str, branch: int) -> None:
        """Switch to the chosen vessel/branch, keeping the selected reference if the branch is unchanged."""
        if self.data.vessel_tree is None:
            return
        keep = branch == self.data.selected_branch[vessel]
        self._select_branch(vessel, branch, self.data.selected_reference_index[vessel] if keep else 0)

    def _on_reference_index_changed(self, index: int) -> None:
        """Route through the same selection path as the Vessel Tree tab so everything stays in sync."""
        self._select_reference(self.right_half.intravascular_column.reference_vessel(), index)

    def _on_point_picked(self, x: float, y: float, z: float, scene_value: str) -> None:
        if scene_value == FusionScene.CENTERLINE_BRANCHES.value:
            self._on_branch_marker_picked(x, y, z)
            return
        if scene_value == FusionScene.INLETS_OUTLETS.value:
            self.cutting.on_point_picked(x, y, z)
            return
        if scene_value != FusionScene.VESSEL_TREE.value or self.data.vessel_tree is None:
            return
        picked = np.array([x, y, z])
        best, best_dist = ('rca', 0, 0), float('inf')
        for vessel in ('rca', 'lca'):
            for branch in self._branch_ids(vessel):
                for i, triplet in enumerate(self._branch_references(vessel, branch)):
                    for pt in triplet:
                        dist = float(np.linalg.norm(np.array(pt) - picked))
                        if dist < best_dist:
                            best, best_dist = (vessel, branch, i), dist
        self._select_branch(*best)

    def _refresh_geometry_scene(self) -> None:
        """Native VTK version of multimodars' plot_results_key: translucent mesh, one point cloud per
        labelled region, and the three centerlines."""
        viewer = self.left_half.viewer
        results = self.data.results
        if results is not None and 'mesh' in results:
            viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'mesh', results['mesh'], color=(200, 200, 200), opacity=0.4)
        if results is not None:
            for key, color in colors.REGION_COLORS.items():
                points = results.get(key)
                if points:
                    viewer.add_points(FusionScene.CCTA_GEOMETRY, key, np.array(points), color=color)
                else:
                    # The region was emptied (reclassify, removal): drop its stale layer.
                    viewer.remove_layer(FusionScene.CCTA_GEOMETRY, key)
        for key, cl in (
            ('centerline_aorta', self.data.centerline_aorta),
            ('centerline_rca', self.data.centerline_rca),
            ('centerline_lca', self.data.centerline_lca),
        ):
            if cl is not None:
                # Points, not a polyline: points_as_tuples() concatenates all branches, so a line
                # would jump between them.
                viewer.add_points(
                    FusionScene.CCTA_GEOMETRY,
                    key,
                    np.array(cl.points_as_tuples()),
                    color=colors.CENTERLINE_COLORS[key],
                    size=4.0,
                )
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def _on_lasso_mode_toggled(self, enabled: bool) -> None:
        # Sole slot on this signal: the guard must be able to veto before the viewer's mode flips.
        if enabled:
            results = self.data.results
            if results is None or not any(results.get(key) for key in colors.REGION_COLORS):
                ErrorMessage(self, 'Run label_geometry first — no labeled points to reclassify.')
                self.left_half.geometry_toolbar.lasso_btn.setChecked(False)
                return
        self.left_half.viewer.set_lasso_mode(enabled)

    def _on_lasso_closed(self) -> None:
        """Move the points inside the lasso from one labelled region to another."""
        if self.left_half.branch_labels_toolbar.lasso_btn.isChecked():
            self._on_branch_lasso_closed()
            return
        viewer = self.left_half.viewer
        results = self.data.results
        if results is None:
            return

        sources = [key for key in colors.REGION_COLORS if results.get(key)]
        if not sources:
            ErrorMessage(self, 'No labeled points available to reclassify.')
            self.left_half.geometry_toolbar.lasso_btn.setChecked(False)
            return

        if len(sources) == 1:
            source = sources[0]
        else:
            source_labels = [colors.region_label(key) for key in sources]
            label, ok = QInputDialog.getItem(
                self, 'Reclassify points', 'Select points to reclassify:', source_labels, 0, False
            )
            if not ok:
                return
            source = sources[source_labels.index(label)]

        dest_choices = [key for key in colors.REGION_COLORS if key != source]
        dest_labels = [colors.region_label(key) for key in dest_choices]
        label, ok = QInputDialog.getItem(
            self,
            'Reclassify points',
            f'Relabel selected "{colors.region_label(source)}" points as:',
            dest_labels,
            0,
            False,
        )
        if not ok:
            return
        dest = dest_choices[dest_labels.index(label)]

        points = np.array(results[source])
        inside = viewer.points_inside_lasso(points)
        if not inside.any():
            ErrorMessage(self, 'No points fell inside the lasso.')
            return

        moved = [tuple(p) for p in points[inside]]
        results[source] = [tuple(p) for p in points[~inside]]
        results[dest] = list(results.get(dest) or []) + moved
        self._sync_branch_labels(moved, source, dest)

        self._refresh_geometry_scene()
        self.status_bar.showMessage(
            f'Reclassified {len(moved)} point(s): {colors.region_label(source)} → {colors.region_label(dest)}.'
        )
        self.left_half.geometry_toolbar.lasso_btn.setChecked(False)  # ends lasso mode, clears overlay

    def _sync_branch_labels(self, moved: list, source: str, dest: str) -> None:
        """Keep the branch labels partitioning rca/lca_points after a CCTA Geometry lasso move: drop moved
        points from the source vessel's branches and give joined points their nearest centerline point's branch.
        Done in place so manual Branch Labels edits survive."""
        results = self.data.results
        if results is None:
            return
        vessel_keys = {'rca_points': 'rca', 'lca_points': 'lca'}
        moved_set = set(moved)
        if source in vessel_keys and f'{vessel_keys[source]}_branches' in results:
            key = f'{vessel_keys[source]}_branches'
            results[key] = {b: [p for p in pts if p not in moved_set] for b, pts in results[key].items()}
        if dest in vessel_keys and f'{vessel_keys[dest]}_branches' in results:
            vessel = vessel_keys[dest]
            centerline = self.data.centerline_rca if vessel == 'rca' else self.data.centerline_lca
            if centerline is None:
                return
            cl_points = np.array(centerline.points_as_tuples())
            cl_branch = np.array([p.branch_id for p in centerline.points])
            branches = {b: list(pts) for b, pts in results[f'{vessel}_branches'].items()}
            for point in moved:
                nearest = int(np.argmin(np.linalg.norm(cl_points - np.array(point), axis=1)))
                branches.setdefault(int(cl_branch[nearest]), []).append(point)
            results[f'{vessel}_branches'] = branches
        self._refresh_branch_labels_scene()

    def _on_branch_lasso_mode_toggled(self, enabled: bool) -> None:
        """Same guard-then-toggle as _on_lasso_mode_toggled, for the Branch Labels tab."""
        if enabled and not self._branch_label_layers():
            ErrorMessage(self, 'Run Label Branches first — no branch labels to adjust.')
            self.left_half.branch_labels_toolbar.lasso_btn.setChecked(False)
            return
        self.left_half.viewer.set_lasso_mode(enabled)

    def _on_branch_lasso_closed(self) -> None:
        """Move the lasso-enclosed vertices of one branch to another branch of the same vessel."""
        viewer = self.left_half.viewer
        toolbar = self.left_half.branch_labels_toolbar
        results = self.data.results
        layers = {
            key: vb
            for key, vb in self._branch_label_layers().items()
            if results and results[f'{vb[0]}_branches'][vb[1]]
        }
        if results is None or not layers:
            ErrorMessage(self, 'No branch labels available to adjust.')
            toolbar.lasso_btn.setChecked(False)
            return

        name = self._branch_name

        sources = list(layers.values())
        source_labels = [name(*vb) for vb in sources]
        label, ok = QInputDialog.getItem(self, 'Adjust branch labels', 'Move vertices of:', source_labels, 0, False)
        if not ok:
            return
        vessel, source = sources[source_labels.index(label)]
        branches = results[f'{vessel}_branches']
        dest_choices = [b for b in sorted(branches) if b != source]
        if not dest_choices:
            ErrorMessage(self, f'{vessel.upper()} has no other branch to move them to.')
            return
        dest_labels = [name(vessel, b) for b in dest_choices]
        label, ok = QInputDialog.getItem(
            self, 'Adjust branch labels', f'Move selected "{name(vessel, source)}" vertices to:', dest_labels, 0, False
        )
        if not ok:
            return
        dest = dest_choices[dest_labels.index(label)]

        points = np.array(branches[source])
        inside = viewer.points_inside_lasso(points)
        if not inside.any():
            ErrorMessage(self, 'No vertices fell inside the lasso.')
            return
        moved = [tuple(p) for p in points[inside]]
        branches[source] = [tuple(p) for p in points[~inside]]
        branches[dest] = list(branches.get(dest) or []) + moved

        self._refresh_branch_labels_scene()
        stale = ' Re-run Discretize Vessel Tree to use them.' if self.data.vessel_tree is not None else ''
        self.status_bar.showMessage(
            f'Moved {len(moved)} vertices: {name(vessel, source)} → {name(vessel, dest)}.{stale}'
        )
        toolbar.lasso_btn.setChecked(False)  # ends lasso mode, clears overlay

    def _centerline_branch_layer_labels(self) -> dict[str, str]:
        """Centerline Branches layer key -> branch name plus its id (used by the split/merge controls)."""
        labels = {}
        for vessel, centerline in (('rca', self.data.centerline_rca), ('lca', self.data.centerline_lca)):
            if centerline is None:
                continue
            for branch in sorted({p.branch_id for p in centerline.points}):
                labels[f'{vessel}_branch_{branch}'] = f'{self._branch_name(vessel, branch)} ({branch})'
        return labels

    def _on_rename_branch_requested(self, vessel: str, branch: int) -> None:
        """One name for the branch's centerline and its vertices. Empty restores the default."""
        current = self._branch_name(vessel, branch)
        name, ok = QInputDialog.getText(
            self, 'Rename branch', f'New name for {current} (empty = default):', text=current
        )
        if not ok:
            return
        name = name.strip()
        names = self.data.branch_names[vessel]
        if name:
            names[branch] = name
        else:
            names.pop(branch, None)
        self._refresh_branch_name_displays()
        self.status_bar.showMessage(f'Renamed {current} to {self._branch_name(vessel, branch)}.')

    def _refresh_branch_name_displays(self) -> None:
        """Re-label every list a branch name shows up in, without recomputing anything."""
        self._refresh_branch_labels_scene()
        self.left_half.branch_toolbar.set_layer_labels(self._centerline_branch_layer_labels())
        self.left_half.refresh_toolbar(FusionScene.CENTERLINE_BRANCHES)
        if self.data.vessel_tree is None:
            return
        ic = self.right_half.intravascular_column
        ic.set_centerline_choices(self._centerline_choices(), ic.reference_vessel(), ic.reference_branch())
        for vessel in ('rca', 'lca'):
            branch = self.data.selected_branch[vessel]
            self.left_half.tree_toolbar.set_branches(vessel, self._branch_choices(vessel), branch)
            self._select_branch(vessel, branch, self.data.selected_reference_index[vessel])

    def _on_delete_branch_requested(self, vessel: str, branch: int, fill_hole: bool) -> None:
        """Cut a side branch and its descendants off the mesh and centerline (multimodars' remove_branch).
        Other branches keep their ids and names, so the vessel tree is pruned, not rediscretized. The stitched
        and final meshes lose the branch too (final via transfer_labels), all or none."""
        centerline = self.data.centerline_rca if vessel == 'rca' else self.data.centerline_lca
        results = self.data.results
        if not self._require(results is not None and centerline is not None, 'Run Label Branches first.'):
            return
        assert results is not None and centerline is not None
        name = self._branch_name(vessel, branch)
        reply = QMessageBox.question(
            self,
            'Delete branch',
            f'Delete {name}, and every branch growing out of it, from the mesh and the '
            f'{vessel.upper()} centerline?' + (' The ostium is closed with a patch.' if fill_hole else ''),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        stitched, final_mesh = self.data.stitched, self.data.final_mesh

        def _run():
            new_results, new_centerline = pipeline.run_remove_branch(results, centerline, branch, fill_hole=fill_hole)
            new_stitched = new_final = None
            if final_mesh is not None and stitched is not None:
                labelled = pipeline.transfer_labels(stitched, final_mesh)
                new_final = pipeline.run_remove_branch(labelled, centerline, branch, fill_hole=fill_hole)[0]['mesh']
            if stitched is not None:
                new_stitched = pipeline.run_remove_branch(stitched, centerline, branch, fill_hole=fill_hole)[0]
            return new_results, new_centerline, new_stitched, new_final

        out = self._run(f'Deleting {name}…', f'Deleted {name}.', _run)
        if out is None:
            return
        new_results, new_centerline, new_stitched, new_final = out
        removed = sorted(set(centerline.branch_ids()) - set(new_centerline.branch_ids()))
        removed_names = ', '.join(self._branch_name(vessel, b) for b in removed)
        setattr(self.data, f'centerline_{vessel}', new_centerline)
        self.data.results = new_results
        self._invalidate_removal()  # stale: the mesh changed
        names = self.data.branch_names[vessel]
        self.data.branch_names[vessel] = {b: n for b, n in names.items() if b not in removed}
        tree = self.data.vessel_tree
        if tree is not None:
            pipeline.prune_vessel_tree(tree, vessel, removed)
            self.cutting.drop_branches(vessel, removed)
            for b in removed:
                self.left_half.viewer.remove_layer(FusionScene.VESSEL_TREE, f'tree_{vessel}_branch_{b}')
            if self.data.selected_branch[vessel] in removed:
                self.data.selected_branch[vessel] = 0
                self.data.selected_reference_index[vessel] = 0
            self._refresh_tree_scene()
            self.left_half.refresh_toolbar(FusionScene.VESSEL_TREE)

        if new_stitched is not None:
            self.data.stitched = new_stitched
            self.left_half.viewer.add_mesh(
                FusionScene.CCTA_GEOMETRY, 'stitched_mesh', new_stitched['mesh'], color=(230, 180, 60)
            )
        if new_final is not None:
            self.data.final_mesh_undo.clear()  # vertex snapshots of the old topology
            self._set_final_mesh(new_final)

        self._refresh_geometry_scene()
        self._refresh_branch_scene()
        self._refresh_branch_name_displays()  # branch labels scene + tree/alignment choices
        self._refresh_aligned_ccta_mesh()
        also = ' from the stitched and final meshes too' if new_final is not None else ''
        also = also or (' from the stitched mesh too' if new_stitched is not None else '')
        self.status_bar.showMessage(f'Deleted {removed_names}{also}.')

    def _refresh_branch_scene(self) -> None:
        """Native VTK version of multimodars' plot_centerline_branches: RCA/LCA colored per branch, sharp
        angles numbered as split hints. Rebuilt from scratch since split/merge reassign branch ids."""
        viewer = self.left_half.viewer
        # Rebuilt from scratch, so snapshot each layer's visibility/opacity and reapply it below.
        saved_states = viewer.layer_states(FusionScene.CENTERLINE_BRANCHES)
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
                    size=9.0,
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
        self.left_half.branch_toolbar.set_layer_labels(self._centerline_branch_layer_labels())
        self.left_half.refresh_toolbar(FusionScene.CENTERLINE_BRANCHES)

        # The threshold only changes the markers, not branch colors, so report the count.
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
        self.data.branch_names[marker['centerline']] = {}  # split reassigns every branch id
        self.cutting.reset()
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
        self.data.branch_names[cl_name] = {}  # merge reassigns every branch id
        self.cutting.reset()
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
        for vessel, main_color in (('rca', colors.TREE_RCA_MAIN_COLOR), ('lca', colors.TREE_LCA_MAIN_COLOR)):
            branches = self._tree_branches(vessel)
            if 0 in branches:
                add_contours(f'tree_{vessel}_main', branches[0].contours, main_color)
            side_ids = sorted(b for b in branches if b != 0)
            for i, branch_id in enumerate(side_ids):
                color = colors.branch_ramp_color(main_color, i, len(side_ids))
                add_contours(f'tree_{vessel}_branch_{branch_id}', branches[branch_id].contours, color)
        if centroids:
            viewer.add_points(
                FusionScene.VESSEL_TREE,
                'tree_centroids',
                np.array(centroids),
                color=colors.TREE_CENTROID_COLOR,
                size=5.0,
            )

        # Reference triplets: one layer per triplet slot, pooled over all branches, to keep the toolbar short.
        ref_slots: list[list[tuple[float, float, float]]] = [[], [], []]
        all_refs = [
            triplet
            for vessel in ('rca', 'lca')
            for branch in self._tree_branches(vessel)
            for triplet in self._branch_references(vessel, branch)
        ]
        for triplet in all_refs:
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
        plan = ic.load_plan()
        if not self._require(bool(plan['input_path']), 'Select a pullback case folder first.'):
            return
        common = {
            'input_path': plan['input_path'],
            'step_rotation_deg': plan['step_rotation_deg'],
            'sample_size': plan['sample_size'],
            'n_points': plan['n_points'],
        }

        if plan['mode'] == 'oct':
            # OCT: single geometry loaded from the exported tagged-frame arrays.
            result = self._run(
                'Loading OCT pullback…', 'Pullback loaded.', pipeline.run_from_array, labels=plan['label'], **common
            )
        else:
            dia, sys = plan['diastole'], plan['systole']
            if not self._require(dia or sys, 'Select Diastole, Systole, or both.'):
                return
            if dia and sys:
                # Both phases -> diastole/systole pair.
                result = self._run(
                    'Loading pullback (dia + sys)…',
                    'Pullback loaded.',
                    pipeline.run_from_file_singlepair,
                    labels=[plan['dia_label'], plan['sys_label']],
                    **common,
                )
            else:
                # Exactly one phase → single geometry, diastole flag picks which.
                label = plan['dia_label'] if dia else plan['sys_label']
                result = self._run(
                    'Loading pullback…',
                    'Pullback loaded.',
                    pipeline.run_from_file_single,
                    label=label,
                    diastole=dia,
                    **common,
                )
        if result is None:
            return
        geometry, align_logs = result
        # A PyGeometryPair (dia/sys) or a single PyGeometry (OCT / one IVUS phase), see _phase_geoms.
        self.data.iv_geometry_pair = geometry
        self.data.iv_align_logs = align_logs

        # Shown before centerline alignment, so errors can be traced to the dia/sys self-alignment.
        self._add_geometry_pair_meshes(FusionScene.INTRAVASCULAR_LOADED, geometry, 'raw_geom')
        self.left_half.show_scene(FusionScene.INTRAVASCULAR_LOADED)

    @staticmethod
    def _phase_geoms(geometry):
        """Normalize a pullback result to (phase_key, geom, color) rows. A PyGeometryPair gives
        diastole (a) and systole (b), a single PyGeometry (OCT or one IVUS phase) is one phase-a row."""
        geom_a = getattr(geometry, 'geom_a', None)
        geom_b = getattr(geometry, 'geom_b', None)
        if geom_a is None and geom_b is None:
            return [('a', geometry, colors.DIASTOLE_COLOR)]
        rows = []
        if geom_a is not None:
            rows.append(('a', geom_a, colors.DIASTOLE_COLOR))
        if geom_b is not None:
            rows.append(('b', geom_b, colors.SYSTOLE_COLOR))
        return rows

    @staticmethod
    def _primary_geom(geometry):
        """The single PyGeometry to read frames from / stitch: geom_a for a pair, else the geometry itself."""
        if geometry is None:
            return None
        geom_a = getattr(geometry, 'geom_a', None)
        return geometry if geom_a is None else geom_a

    def _add_geometry_pair_meshes(self, scene: FusionScene, geometry_pair, key_prefix: str) -> None:
        """Loft lumen and wall meshes for each phase of `geometry_pair` into `scene`."""
        viewer = self.left_half.viewer
        for phase_key, geom, color in self._phase_geoms(geometry_pair):
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

    def _selected_align_centerline(self, vessel: str, branch: int):
        """(centerline, references, selected_index) for the chosen vessel/branch. The centerline is the
        whole vessel, the references only the branch's own."""
        centerline = self.data.centerline_rca if vessel == 'rca' else self.data.centerline_lca
        return centerline, self._branch_references(vessel, branch), self.data.selected_reference_index[vessel]

    def _align_takeoff_anomalous(self, vessel: str, branch: int) -> bool:
        """Acute takeoff only applies to the main vessel (a side branch leaves its parent, not the aorta)."""
        return branch == 0 and self.right_half.geometry_column.has_acute_takeoff(vessel)

    def _on_run_align(self) -> None:
        ic = self.right_half.intravascular_column
        vessel, branch = ic.reference_vessel(), ic.reference_branch()
        if not self._require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self._require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        centerline, references, selected_index = self._selected_align_centerline(vessel, branch)
        if not self._require(
            centerline is not None and self.data.results is not None,
            'Run label_geometry first.',
        ):
            return

        results = self.data.results
        assert centerline is not None and results is not None

        try:
            ref_points = references[selected_index]
            cl_main = self._alignment_branch(centerline, branch)
        except (IndexError, AttributeError, ValueError, TypeError) as e:
            ErrorMessage(self, f'Could not resolve reference points / branch: {e}')
            return
        # A side branch is scored against its own surface points, not the whole vessel's.
        points = results.get(f'{vessel}_branches', {}).get(branch) if branch else None

        result = self._run(
            'Aligning intravascular geometry…',
            'Alignment done.',
            pipeline.run_align_combined,
            cl_main,
            self.data.iv_geometry_pair,
            ref_points[0],
            ref_points[1],
            ref_points[2],
            points or results.get(f'{vessel}_points', []),
            align_wall_anomalous=self._align_takeoff_anomalous(vessel, branch),
            **ic.align_kwargs(),
        )
        if result is None:
            return
        self._apply_align_result(
            result, vessel, branch, centerline, cl_main, self._resolve_manual_ref_point(cl_main, ref_points[0], 0)
        )

    def _on_run_align_manual(self) -> None:
        """Like _on_run_align, but rotates by an explicit angle around one reference point instead of
        searching. Only meaningful for elliptic (anomalous) vessels."""
        ic = self.right_half.intravascular_column
        vessel, branch = ic.reference_vessel(), ic.reference_branch()
        if not self._require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self._require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        centerline, references, selected_index = self._selected_align_centerline(vessel, branch)
        if not self._require(centerline is not None, 'Run label_geometry first.'):
            return
        assert centerline is not None

        try:
            main_ref_pt = references[selected_index][0]
            cl_main = self._alignment_branch(centerline, branch)
        except (IndexError, AttributeError, ValueError, TypeError) as e:
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
            align_wall_anomalous=self._align_takeoff_anomalous(vessel, branch),
            **ic.manual_align_kwargs(),
        )
        if result is None:
            return
        # Mark the unshifted reference (offset 0), not ref_point: the IV geometry is placed
        # on ref_point, so marking it would move the dot along with the vessel.
        self._apply_align_result(
            result, vessel, branch, centerline, cl_main, self._resolve_manual_ref_point(cl_main, main_ref_pt, 0)
        )

    def _alignment_branch(self, centerline, branch: int):
        """The single-branch centerline to align onto. With "Include parent vessel", the path from the
        vessel ostium through the parents to the side branch's tip, so proximal frames land on the parent."""
        if self.right_half.intravascular_column.align_with_parents():
            return centerline.get_branch(branch, with_parents=True)
        return centerline.get_branch(branch)

    def _resolve_manual_ref_point(
        self, cl_main, main_ref_pt: tuple[float, float, float], offset: int
    ) -> tuple[float, float, float]:
        """The centerline point closest to main_ref_pt, shifted by `offset` centerline points
        (negative = towards the ostium). Clamped to the centerline's ends."""
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

    def _apply_align_result(
        self,
        result,
        vessel: str,
        branch: int,
        vessel_centerline,
        source_centerline,
        ostium_point: tuple[float, float, float],
    ) -> None:
        """`result` is (aligned_geometry, spacing_mm, total_rotation_deg). `source_centerline` is the
        single-branch centerline passed to the aligner, kept as-is: resampling it to spacing_mm would offset it
        from the one multimodars used. `vessel_centerline` is the whole vessel (for label_anomalous_region).
        `ostium_point` is a fixed marker, so an offset shows up as the IV geometry shifting relative to it."""
        self.data.aligned, _spacing_mm, total_rotation_deg = result
        self.data.aligned_centerline = source_centerline
        self.data.aligned_vessel = vessel
        self.data.aligned_branch = branch
        self.data.aligned_vessel_centerline = vessel_centerline
        # Prefill the Manual angle with this result so further nudging starts from here.
        self.right_half.intravascular_column.set_manual_rotation_angle(total_rotation_deg)

        viewer = self.left_half.viewer
        viewer.add_points(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'aligned_centerline',
            np.array(source_centerline.points_as_tuples()),
            color=(0, 200, 0),
            size=4.0,
        )
        viewer.add_points(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'ostium_point',
            np.array([ostium_point]),
            color=colors.TREE_REF_COLORS[0],
            size=14.0,
        )
        # geom_a/geom_b are assumed diastole/systole (load_plan label order) to match the colors.
        self._add_geometry_pair_meshes(FusionScene.INTRAVASCULAR_ALIGNED, self.data.aligned, 'aligned_geom')
        self._refresh_aligned_ccta_mesh()
        self.left_half.show_scene(FusionScene.INTRAVASCULAR_ALIGNED)

    def _refresh_aligned_ccta_mesh(self) -> None:
        """Overlay the unlabelled CCTA mesh in the Intravascular Aligned scene. Call whenever results['mesh']
        changes. No-op before alignment."""
        if self.data.aligned is None or self.data.results is None or 'mesh' not in self.data.results:
            return
        self.left_half.viewer.add_mesh(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'ccta_mesh',
            self.data.results['mesh'],
            color=(160, 160, 160),
            opacity=0.35,
        )
        self._refresh_aligned_boundary_rings()
        self.left_half.refresh_toolbar(FusionScene.INTRAVASCULAR_ALIGNED)

    def _refresh_aligned_boundary_rings(self) -> None:
        """Show Remove Labeled Points' rims (boundary_points_N) in the Intravascular Aligned scene to judge
        the seam point counts. Old ring layers are always dropped first, since the ring count can change."""
        viewer = self.left_half.viewer
        scene = FusionScene.INTRAVASCULAR_ALIGNED
        # Remembered so a re-added layer keeps the user's checkbox state.
        was_visible = {key: state[0] for key, state in viewer.layer_states(scene).items()}
        for key in was_visible:
            if key.startswith('boundary_ring_'):
                viewer.remove_layer(scene, key)
        results = self.data.results or {}
        ring_number = 1
        while f'boundary_points_{ring_number}' in results:
            points = np.asarray(results[f'boundary_points_{ring_number}'], dtype=np.float64)
            if len(points):
                color = colors.BOUNDARY_RING_COLORS[(ring_number - 1) % len(colors.BOUNDARY_RING_COLORS)]
                key = f'boundary_ring_{ring_number}'
                viewer.add_points(scene, key, points, color=color, size=8.0, visible=was_visible.get(key, True))
            ring_number += 1

    # ------------------------------------------------------------------
    # Column 3: fusion
    # ------------------------------------------------------------------

    def _aligned_frames(self):
        geom = self._primary_geom(self.data.aligned)
        if geom is None:
            return None
        try:
            return geom.frames
        except AttributeError:
            return None

    def _on_run_label_anomalous(self) -> None:
        """Split the aligned branch's pullback path into proximal/overlap/distal regions. Uses the
        vessel/branch stored at alignment time, not the current selector. Needs the whole vessel centerline.
        Off-path vertices go to off_path_points. Re-align after a split/merge (they reassign branch ids)."""
        frames = self._aligned_frames()
        if not self._require(frames is not None, 'Align the intravascular geometry first.'):
            return
        centerline = self.data.aligned_vessel_centerline
        if not self._require(centerline is not None, 'Run label_geometry first.'):
            return
        results = self._run(
            'Labeling overlap region…',
            'Overlap region labeled.',
            pipeline.run_label_anomalous_region,
            centerline,
            frames,
            self.data.results,
            branch_id=self.data.aligned_branch,
        )
        if results is not None:
            self.data.results = results
            self._invalidate_removal()  # stale: re-run Remove Labeled Points
            # proximal/distal/anomalous (shown as overlap)/off_path_points now exist
            self._refresh_geometry_scene()
            # Hide the aligned vessel's own region, which would cover them.
            vessel_layer = f'{self.data.aligned_vessel}_points'
            if vessel_layer in self.left_half.viewer.layer_states(FusionScene.CCTA_GEOMETRY):
                self.left_half.viewer.set_layer_visible(FusionScene.CCTA_GEOMETRY, vessel_layer, False)
                self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def _on_run_compute_scaling(self) -> None:
        frames = self._aligned_frames()
        if not self._require(frames is not None, 'Align the intravascular geometry first.'):
            return
        vessel = self.right_half.intravascular_column.reference_vessel()
        # The aligned branch, not the whole vessel: near a junction, vertices could snap to another branch's
        # centerline and be pushed sideways.
        centerline = self.data.aligned_centerline
        if not self._require(centerline is not None, 'Run label_geometry first.'):
            return
        scalings = self._run(
            'Computing scaling factors…',
            'Scaling factors computed.',
            pipeline.run_find_scalings,
            frames,
            centerline,
            self.data.centerline_aorta,
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
        # The whole vessel, not aligned_centerline: proximal/distal_points include the side
        # branches (and, for a side-branch pullback, the parent vessel), which must each be
        # morphed around their own branch's centerline.
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

        # Read from the spinboxes: the user may have edited the computed values.
        scaling = self.right_half.fusion_column.scaling_values()
        remaining_scaling = scaling['remaining_vessels_scaling']
        if not self._require(
            remaining_scaling == 0.0 or opposite_centerline is not None,
            f'Missing {opposite_vessel.upper()} centerline for the remaining-vessels scaling — '
            'run Label Geometry first, or set Remaining vessels (mm) back to 0.',
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

            # Manual-only (no IV data for these vessels): skipped unless set. Each part is morphed around its own
            # centerline: the opposite coronary, and this coronary's off_path_points.
            if remaining_scaling != 0.0:
                assert opposite_centerline is not None
                for region, region_centerline in (
                    (results[f'{opposite_vessel}_points'], opposite_centerline),
                    (results.get('off_path_points', []), centerline),
                ):
                    if not region:
                        continue
                    scaled = pipeline.run_scale_region(mesh, region, region_centerline, remaining_scaling)
                    results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
                    mesh = results['mesh']
            return results

        results_out = self._run('Applying scaling to mesh…', 'Scaling applied.', _run)
        if results_out is None:
            return
        self.data.results = results_out
        self._invalidate_removal()  # stale: re-run Remove Labeled Points on the scaled mesh
        self._refresh_geometry_scene()
        self._refresh_branch_labels_scene()  # scaled mesh + its synced branch labels
        self._refresh_aligned_ccta_mesh()

    def _on_run_remove_points(self) -> None:
        fc = self.right_half.fusion_column
        if not self._require(self.data.results is not None, 'Run label_geometry first.'):
            return
        # Taken before the call, in case remove_labeled_points_from_mesh touches its input.
        before = copy.deepcopy(self.data.results)
        assert before is not None
        results = self._run(
            'Removing labeled points…',
            'Points removed.',
            pipeline.run_remove_labeled_points,
            self.data.results,
            fc.remove_point_keys(),
        )
        if results is not None:
            undo = self.data.removal_undo
            undo.append(before)
            del undo[: max(len(undo) - self._REMOVAL_UNDO_LIMIT, 0)]
            self._set_removed_results(results)
            if self.data.aligned is not None:  # the aligned scene is empty until alignment
                self.left_half.show_scene(FusionScene.INTRAVASCULAR_ALIGNED)

    # Full results snapshots (mesh + every label list), far heavier than a vertex array.
    _REMOVAL_UNDO_LIMIT = 5

    def _on_undo_remove_points(self) -> None:
        if not self.data.removal_undo:
            return
        results = self.data.removal_undo.pop()
        # With removals left, the restored state is a post-removal one Stitch can start from.
        # Otherwise it is the labelled mesh from before any removal.
        if self.data.removal_undo:
            self._set_removed_results(results)
        else:
            self.data.results = results
            self.data.results_points_removed = None
            self._clear_stitch_outputs()
            self._refresh_geometry_scene()
            self._refresh_branch_labels_scene()
            self._refresh_aligned_ccta_mesh()
            self._update_removal_undo_button()
        self.status_bar.showMessage(f'Undid point removal ({len(self.data.removal_undo)} left).')

    def _set_removed_results(self, results: dict) -> None:
        """Make `results` (the output of a removal) current, as the snapshot Stitch starts from."""
        self.data.results = results
        self.data.results_points_removed = copy.deepcopy(results)
        self.right_half.fusion_column.set_seam_points_limit(pipeline.max_seam_points(results))
        self._clear_stitch_outputs()
        self._refresh_geometry_scene()
        self._refresh_branch_labels_scene()
        self._refresh_aligned_ccta_mesh()
        self._update_removal_undo_button()

    def _invalidate_removal(self) -> None:
        """`results` was replaced by something other than a removal: the post-removal
        snapshot and the removal undo stack no longer match it."""
        self.data.results_points_removed = None
        self._clear_removal_undo()

    def _clear_removal_undo(self) -> None:
        self.data.removal_undo.clear()
        self._update_removal_undo_button()

    def _update_removal_undo_button(self) -> None:
        self.right_half.fusion_column.undo_remove_btn.setEnabled(bool(self.data.removal_undo))

    def _on_undo_shortcut(self) -> None:
        """Ctrl+Z: undo smoothing first, then point removals, but only while nothing is stitched (that
        would drop the stitched mesh). The Undo button next to Remove still does it."""
        if self.data.final_mesh_undo:
            self._on_undo_smoothing()
        elif self.data.stitched is None and self.data.final_mesh is None:
            self._on_undo_remove_points()

    def _clear_stitch_outputs(self) -> None:
        """Drop the stitched mesh and everything derived from it, so no stale final_mesh survives a re-stitch."""
        self.data.stitched = None
        self.data.final_mesh = None
        self._reset_smoothing_state()
        viewer = self.left_half.viewer
        viewer.remove_layer(FusionScene.CCTA_GEOMETRY, 'stitched_mesh')
        viewer.remove_layer(FusionScene.CCTA_GEOMETRY, 'final_mesh')
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def _on_run_stitch(self) -> None:
        fc = self.right_half.fusion_column
        if not self._require(self.data.aligned is not None, 'Align the intravascular geometry first.'):
            return
        if not self._require(self.data.results_points_removed is not None, 'Remove labeled points first.'):
            return
        if self._remesh_worker is not None and self._remesh_worker.isRunning():
            return
        aligned_geom = self._primary_geom(self.data.aligned)
        snapshot = self.data.results_points_removed
        assert aligned_geom is not None and snapshot is not None

        def _run():
            # Stitch mutates results in place, so pass a copy to keep the snapshot reusable.
            results = copy.deepcopy(snapshot)
            # The tree (discretized before scaling/removal, as multimodars expects) lets the
            # stitch reconnect side branches leaving the removed segment. It is only read.
            return pipeline.run_stitch(
                aligned_geom, results['mesh'], results, tree=self.data.vessel_tree, **fc.stitch_kwargs()
            )

        stitched = self._run('Stitching CCTA to intravascular…', 'Stitched.', _run)
        if stitched is None:
            return
        self._clear_stitch_outputs()
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
        self.data.final_mesh_undo.clear()
        self._set_final_mesh(mesh)
        self.status_bar.showMessage('Remeshed.')

    def _on_remesh_failed(self, progress: QProgressDialog, message: str) -> None:
        progress.close()
        self._remesh_worker = None
        logger.error(f'Fix & Remesh failed: {message}')
        ErrorMessage(self, message)
        self.status_bar.showMessage('Failed — see log')

    def _on_run_smooth(self) -> None:
        fc = self.right_half.fusion_column
        final_mesh = self.data.final_mesh
        if not self._require(final_mesh is not None, 'Fix and remesh first.'):
            return
        assert final_mesh is not None
        # run_taubin_smooth works in place, so snapshot the vertices before, not after.
        before = final_mesh.vertices.copy()
        mesh = self._run('Smoothing…', 'Smoothed.', pipeline.run_taubin_smooth, final_mesh, lamb=fc.taubin_lamb())
        if mesh is None:
            return
        self._push_smoothing_undo(before)
        self._set_final_mesh(mesh)

    # ------------------------------------------------------------------
    # Local sphere smoothing of the final mesh (+ undo)
    # ------------------------------------------------------------------

    _SMOOTHING_UNDO_LIMIT = 20  # vertex snapshots, each a full (N, 3) float array

    def _set_final_mesh(self, mesh) -> None:
        self.data.final_mesh = mesh
        self.left_half.viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'final_mesh', mesh, color=(230, 230, 230))
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.left_half.geometry_toolbar.set_undo_available(bool(self.data.final_mesh_undo))

    def _push_smoothing_undo(self, vertices: np.ndarray) -> None:
        undo = self.data.final_mesh_undo
        undo.append(vertices)
        del undo[: max(len(undo) - self._SMOOTHING_UNDO_LIMIT, 0)]

    def _reset_smoothing_state(self) -> None:
        """No final mesh any more (re-stitch, Clear All Data): drop undo and end brushing."""
        self.data.final_mesh_undo.clear()
        gt = self.left_half.geometry_toolbar
        gt.set_undo_available(False)
        gt.sphere_btn.setChecked(False)  # its toggled slot turns the viewer's sphere mode off

    def _on_sphere_mode_toggled(self, enabled: bool) -> None:
        viewer = self.left_half.viewer
        if enabled:
            if self.data.final_mesh is None:
                ErrorMessage(self, 'Fix and remesh first — the sphere brush smooths the final mesh.')
                self.left_half.geometry_toolbar.sphere_btn.setChecked(False)
                return
            viewer.set_sphere_radius(self.left_half.geometry_toolbar.sphere_radius.value())
            viewer.set_sphere_mode(True, FusionScene.CCTA_GEOMETRY, 'final_mesh')
            self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)
            self.status_bar.showMessage('Sphere smooth: hover to preview, click to smooth, Ctrl+wheel for radius.')
        else:
            viewer.set_sphere_mode(False)

    def _on_sphere_radius_changed(self, radius_mm: float) -> None:
        viewer = self.left_half.viewer
        viewer.set_sphere_radius(radius_mm)
        viewer.refresh_sphere_hover()  # highlight must follow the new radius

    def _on_sphere_hovered(self, x: float, y: float, z: float) -> None:
        mesh = self.data.final_mesh
        if mesh is None:
            return
        radius = self.left_half.geometry_toolbar.sphere_radius.value()
        region, _ = local_smooth_region(mesh, (x, y, z), radius)
        self.left_half.viewer.set_sphere_highlight(np.asarray(mesh.vertices)[region])

    def _on_sphere_clicked(self, x: float, y: float, z: float) -> None:
        mesh = self.data.final_mesh
        if mesh is None:
            return
        gt = self.left_half.geometry_toolbar
        smoothed = self._run(
            'Smoothing locally…',
            'Locally smoothed.',
            local_smooth,
            mesh,
            (x, y, z),
            gt.sphere_radius.value(),
            iterations=gt.sphere_iterations.value(),
            lamb=self.right_half.fusion_column.taubin_lamb(),
        )
        if smoothed is None:
            return
        self._push_smoothing_undo(mesh.vertices.copy())
        self._set_final_mesh(smoothed)
        self.left_half.viewer.refresh_sphere_hover()

    def _on_undo_smoothing(self) -> None:
        mesh = self.data.final_mesh
        if mesh is None or not self.data.final_mesh_undo:
            return
        restored = mesh.copy()
        restored.vertices = self.data.final_mesh_undo.pop()
        self._set_final_mesh(restored)
        self.left_half.viewer.refresh_sphere_hover()
        self.status_bar.showMessage(f'Undid smoothing step ({len(self.data.final_mesh_undo)} left).')

    def _on_export(self, path: str) -> None:
        if not self._require(self.data.final_mesh is not None, 'Nothing to export yet — finish the pipeline first.'):
            return
        self._run('Exporting…', f'Exported: {path}', pipeline.export_mesh, self.data.final_mesh, path)
