from types import SimpleNamespace

import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
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
from pages.fusion.right_half.right_half import RightHalf
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


class FusionPage(QWidget):
    def __init__(self, config: SimpleNamespace, status_bar) -> None:
        super().__init__()
        self.config: SimpleNamespace = config
        self.status_bar = status_bar
        self.data = FusionRuntimeData()

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

    def _connect_signals(self) -> None:
        gc = self.right_half.geometry_column
        gc.run_label_geometry_requested.connect(self._on_run_label_geometry)
        gc.run_prepare_centerlines_requested.connect(self._on_run_prepare_centerlines)
        gc.run_discretize_tree_requested.connect(self._on_run_discretize_tree)

        self.left_half.tree_toolbar.reference_selected.connect(self._select_rca_reference)
        self.left_half.viewer.point_picked.connect(self._on_point_picked)

        ic = self.right_half.intravascular_column
        ic.run_load_requested.connect(self._on_run_load_pullback)
        ic.run_align_requested.connect(self._on_run_align)

        fc = self.right_half.fusion_column
        fc.run_label_anomalous_requested.connect(self._on_run_label_anomalous)
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

    def _run_with_progress(self, title: str, busy_message: str, done_message: str, fn, *args, **kwargs):
        """Like _run(), plus an indeterminate busy dialog — for the one pipeline step
        (fix_and_remesh_stitched_mesh) that isn't Rust-backed and can take several
        seconds with no intermediate progress to report, unlike everything else here."""
        progress = QProgressDialog(busy_message, '', 0, 0, self)
        progress.setWindowTitle(title)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()
        QApplication.processEvents()  # second flush processes the paint event queued by show
        try:
            return self._run(busy_message, done_message, fn, *args, **kwargs)
        finally:
            progress.close()

    # ------------------------------------------------------------------
    # Column 1: CCTA geometry + centerlines
    # ------------------------------------------------------------------

    def _on_run_label_geometry(self) -> None:
        gc = self.right_half.geometry_column
        if not self._require(gc.mesh_path is not None, 'Load a CCTA mesh first.'):
            return
        if not self._require(
            all(k in gc.centerline_paths for k in ('aorta', 'rca', 'lca')),
            'Load all three centerlines (aorta, RCA, LCA) first.',
        ):
            return
        mesh_path = gc.mesh_path
        assert mesh_path is not None

        def _run():
            cl_aorta = pipeline.read_centerline_vtp(gc.centerline_paths['aorta'])
            cl_rca = pipeline.read_centerline_vtp(gc.centerline_paths['rca'])
            cl_lca = pipeline.read_centerline_vtp(gc.centerline_paths['lca'])
            return pipeline.run_label_geometry(mesh_path, cl_aorta, cl_rca, cl_lca, **gc.label_geometry_kwargs())

        result = self._run('Running label_geometry…', 'label_geometry done.', _run)
        if result is None:
            return
        results, (cl_rca, cl_lca, cl_aorta) = result
        self.data.results = results
        self.data.centerline_rca = cl_rca
        self.data.centerline_lca = cl_lca
        self.data.centerline_aorta = cl_aorta
        self._refresh_geometry_scene()
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)

    def _on_run_prepare_centerlines(self) -> None:
        if not self._require(
            self.data.results is not None and self.data.centerline_rca is not None,
            'Run label_geometry first.',
        ):
            return
        result = self._run(
            'Preparing centerlines…',
            'Centerlines prepared.',
            pipeline.run_prepare_centerlines,
            self.data.centerline_rca,
            self.data.centerline_lca,
            self.data.results,
        )
        if result is None:
            return
        self.data.centerline_rca, self.data.centerline_lca, self.data.results = result
        self._refresh_geometry_scene()

    def _on_run_discretize_tree(self) -> None:
        gc = self.right_half.geometry_column
        if not self._require(
            self.data.centerline_aorta is not None and self.data.results is not None,
            'Run label_geometry (and prepare_centerlines) first.',
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

        reference_labels = ['RCA ostium'] + [f'RCA branch {i}' for i in range(1, len(tree.rca_references))]
        self.left_half.tree_toolbar.set_references(reference_labels)

        self._refresh_tree_scene()
        self._select_rca_reference(0)
        self.left_half.show_scene(FusionScene.VESSEL_TREE)

    def _select_rca_reference(self, index: int) -> None:
        """Apply reference triplet `index` (chosen via the dropdown or a scene click) as
        the alignment reference points, and highlight it in the viewer."""
        tree = self.data.vessel_tree
        if tree is None:
            return
        try:
            triplet = tree.rca_references[index]
        except IndexError:
            logger.warning(f'Vessel tree has no rca_references[{index}].')
            return
        self.data.selected_rca_reference_index = index
        self.right_half.intravascular_column.set_reference_points(triplet[0], triplet[1], triplet[2])
        self.left_half.tree_toolbar.set_selected_index(index)
        self.left_half.viewer.add_points(
            FusionScene.VESSEL_TREE, 'selected_reference', np.array(triplet), color=(255, 255, 255), size=14.0
        )

    def _on_point_picked(self, x: float, y: float, z: float, scene_value: str) -> None:
        if scene_value != FusionScene.VESSEL_TREE.value or self.data.vessel_tree is None:
            return
        picked = np.array([x, y, z])
        best_index, best_dist = 0, float('inf')
        for i, triplet in enumerate(self.data.vessel_tree.rca_references):
            for pt in triplet:
                dist = float(np.linalg.norm(np.array(pt) - picked))
                if dist < best_dist:
                    best_dist = dist
                    best_index = i
        self._select_rca_reference(best_index)

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

    def _on_run_align(self) -> None:
        ic = self.right_half.intravascular_column
        if not self._require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self._require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        if not self._require(
            self.data.centerline_rca is not None and self.data.results is not None,
            'Run label_geometry first.',
        ):
            return

        vessel_tree = self.data.vessel_tree
        centerline_rca = self.data.centerline_rca
        results = self.data.results
        assert vessel_tree is not None and centerline_rca is not None and results is not None

        try:
            ref_points = vessel_tree.rca_references[self.data.selected_rca_reference_index]
            rca_cl_main = centerline_rca.get_branch(ic.branch_index())
        except (IndexError, AttributeError) as e:
            ErrorMessage(self, f'Could not resolve reference points / branch: {e}')
            return

        result = self._run(
            'Aligning intravascular geometry…',
            'Alignment done.',
            pipeline.run_align_combined,
            rca_cl_main,
            self.data.iv_geometry_pair,
            ref_points[0],
            ref_points[1],
            ref_points[2],
            results.get('rca_points', []),
            align_wall_anomalous=self.right_half.geometry_column.is_anomalous(),
            **ic.align_kwargs(),
        )
        if result is None:
            return
        self.data.aligned, self.data.resampled_centerline = result

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
        frames = self._aligned_frames()
        if not self._require(frames is not None, 'Align the intravascular geometry first.'):
            return
        results = self._run(
            'Labeling anomalous region…',
            'Anomalous region labeled.',
            pipeline.run_label_anomalous_region,
            self.data.centerline_rca,
            frames,
            self.data.results,
        )
        if results is not None:
            self.data.results = results
            self._refresh_geometry_scene()  # proximal/distal/anomalous_points now exist

    def _on_run_compute_scaling(self) -> None:
        frames = self._aligned_frames()
        if not self._require(frames is not None, 'Align the intravascular geometry first.'):
            return
        scalings = self._run(
            'Computing scaling factors…',
            'Scaling factors computed.',
            pipeline.run_find_scalings,
            frames,
            self.data.centerline_rca,
            self.data.centerline_aorta,
            self.data.results,
        )
        if scalings is None:
            return
        self.data.prox_scaling = scalings['proximal_scaling']
        self.data.distal_scaling = scalings['distal_scaling']
        self.data.aortic_scaling = scalings['aortic_scaling']
        self.data.aortic_wall_scaling = scalings['aortic_wall_scaling']
        self.right_half.fusion_column.set_scaling_results(scalings)

    def _on_run_apply_scaling(self) -> None:
        if not self._require(
            None not in (self.data.prox_scaling, self.data.distal_scaling, self.data.aortic_scaling),
            'Compute scaling factors first.',
        ):
            return
        if not self._require(
            self.data.results is not None
            and self.data.centerline_rca is not None
            and self.data.centerline_aorta is not None,
            'Run label_geometry first.',
        ):
            return

        # Read live from the spinboxes, not self.data.*_scaling — the user may have
        # edited them by hand after Compute Scaling Factors filled in the defaults.
        scaling = self.right_half.fusion_column.scaling_values()

        def _run():
            results = self.data.results
            centerline_rca = self.data.centerline_rca
            centerline_aorta = self.data.centerline_aorta
            assert results is not None and centerline_rca is not None and centerline_aorta is not None
            distal_scaling = scaling['distal_scaling']
            aortic_scaling = scaling['aortic_scaling']
            prox_scaling = scaling['proximal_scaling']
            mesh = results['mesh']

            scaled = pipeline.run_scale_region(mesh, results['distal_points'], centerline_rca, distal_scaling)
            results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
            mesh = results['mesh']

            aortic_region = results['aorta_points'] + results['rca_removed_points']
            scaled = pipeline.run_scale_region(mesh, aortic_region, centerline_aorta, aortic_scaling)
            results = pipeline.run_sync_results_to_mesh(results, mesh, scaled)
            mesh = results['mesh']

            scaled = pipeline.run_scale_region(mesh, results['proximal_points'], centerline_rca, prox_scaling)
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
        stitched = self.data.stitched
        assert stitched is not None
        mesh = self._run_with_progress(
            'Fix & Remesh',
            'Fixing & remeshing… (pure-Python step, can take a few seconds)',
            'Remeshed.',
            pipeline.run_remesh,
            stitched['mesh'],
            **fc.remesh_kwargs(),
        )
        if mesh is None:
            return
        self.data.final_mesh = mesh
        self.left_half.viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'final_mesh', mesh, color=(230, 230, 230))
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

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
