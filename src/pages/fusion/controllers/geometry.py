"""Column 1 mesh and centerlines, the CCTA Geometry scene and its region lasso."""

from __future__ import annotations

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QInputDialog

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from fusion_pipeline import branches as branch_ops
from fusion_pipeline import geometry as geo
from fusion_pipeline import multimodars_api as mm_api
from pages.fusion.controllers.base import StageController
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


class GeometryController(StageController):
    def on_preview(self) -> None:
        """Show the raw mesh and centerlines as soon as they're picked, before any processing."""
        gc = self.right_half.geometry_column
        viewer = self.left_half.viewer

        if gc.mesh_path is not None:
            try:
                mesh = geo.load_ccta_mesh(gc.mesh_path)
            except Exception as e:
                logger.warning(f'Could not load CCTA mesh for preview: {e}')
            else:
                viewer.add_mesh(
                    FusionScene.CCTA_GEOMETRY,
                    'mesh',
                    mesh,
                    color=colors.CCTA_MESH_COLOR,
                    opacity=colors.CCTA_MESH_OPACITY,
                )

        for key, path in gc.centerline_paths.items():
            try:
                cl = mm_api.load_centerline(path, key.upper())
            except Exception as e:
                logger.warning(f'Could not load {key} centerline for preview: {e}')
                continue
            layer_key = f'centerline_{key}'
            viewer.add_points(
                FusionScene.CCTA_GEOMETRY,
                layer_key,
                np.array(cl.points_as_tuples()),
                color=colors.CENTERLINE_COLORS[layer_key],
                size=colors.POINT_SIZE,
            )

        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)

    def on_prepare_centerlines(self) -> None:
        """Load and prepare all three centerlines: the aorta first, then RCA/LCA oriented to it."""
        gc = self.right_half.geometry_column
        if not self.require(
            all(k in gc.centerline_paths for k in ('aorta', 'rca', 'lca')),
            'Load all three centerlines (aorta, RCA, LCA) first.',
        ):
            return

        def _run():
            aorta_raw = mm_api.load_centerline(gc.centerline_paths['aorta'], 'Aorta')
            aorta_cl = mm_api.prepare_centerline(aorta_raw, **gc.prepare_centerline_kwargs('aorta'))
            rca_raw = mm_api.load_centerline(gc.centerline_paths['rca'], 'RCA')
            rca_cl = mm_api.prepare_centerline(rca_raw, ref_centerline=aorta_cl, **gc.prepare_centerline_kwargs('rca'))
            lca_raw = mm_api.load_centerline(gc.centerline_paths['lca'], 'LCA')
            lca_cl = mm_api.prepare_centerline(lca_raw, ref_centerline=aorta_cl, **gc.prepare_centerline_kwargs('lca'))
            return aorta_cl, rca_cl, lca_cl

        result = self.run('Preparing centerlines…', 'Centerlines prepared.', _run)
        if result is None:
            return
        self.data.centerline_aorta, self.data.centerline_rca, self.data.centerline_lca = result
        self.data.branch_names = {'rca': {}, 'lca': {}}  # new centerlines, new branch ids
        self.page.cutting.reset()
        self.refresh_scene()
        self.page.branches.refresh_centerline_scene()
        self.left_half.show_scene(FusionScene.CENTERLINE_BRANCHES)

    def on_label_geometry(self) -> None:
        gc = self.right_half.geometry_column
        if not self.require(gc.mesh_path is not None, 'Load a CCTA mesh first.'):
            return
        if not self.require(
            None not in (self.data.centerline_aorta, self.data.centerline_rca, self.data.centerline_lca),
            'Prepare all three centerlines first.',
        ):
            return
        mesh_path = gc.mesh_path
        assert mesh_path is not None

        def _run():
            mesh = geo.load_ccta_mesh(mesh_path)
            return mm_api.run_label_geometry(
                mesh,
                self.data.centerline_aorta,
                self.data.centerline_rca,
                self.data.centerline_lca,
                **gc.label_geometry_kwargs(),
            )

        results = self.run('Running label_geometry…', 'label_geometry done.', _run)
        if results is None:
            return
        self.data.results = results
        self.page.fusion.invalidate_removal()  # stale: based on the previous labeling
        self.refresh_scene()
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)
        self.page.branches.run_label_branches_pair()

    def refresh_scene(self) -> None:
        """Native VTK version of multimodars' plot_results_key: translucent mesh, one point cloud per
        labelled region, and the three centerlines."""
        viewer = self.left_half.viewer
        results = self.data.results
        if results is not None and 'mesh' in results:
            viewer.add_mesh(
                FusionScene.CCTA_GEOMETRY,
                'mesh',
                results['mesh'],
                color=colors.CCTA_MESH_COLOR,
                opacity=colors.CCTA_MESH_OPACITY,
            )
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
                    size=colors.POINT_SIZE,
                )
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def on_lasso_mode_toggled(self, enabled: bool) -> None:
        # Sole slot on this signal: the guard must be able to veto before the viewer's mode flips.
        if enabled:
            results = self.data.results
            if results is None or not any(results.get(key) for key in colors.REGION_COLORS):
                ErrorMessage(self.page, 'Run label_geometry first — no labeled points to reclassify.')
                self.left_half.geometry_toolbar.end_lasso()
                return
        self.left_half.viewer.set_lasso_mode(enabled)

    def on_lasso_closed(self) -> None:
        """Move the points inside the lasso from one labelled region to another."""
        viewer = self.left_half.viewer
        results = self.data.results
        if results is None:
            return

        sources = [key for key in colors.REGION_COLORS if results.get(key)]
        if not sources:
            ErrorMessage(self.page, 'No labeled points available to reclassify.')
            self.left_half.geometry_toolbar.end_lasso()
            return

        if len(sources) == 1:
            source = sources[0]
        else:
            source_labels = [colors.region_label(key) for key in sources]
            label, ok = QInputDialog.getItem(
                self.page, 'Reclassify points', 'Select points to reclassify:', source_labels, 0, False
            )
            if not ok:
                return
            source = sources[source_labels.index(label)]

        dest_choices = [key for key in colors.REGION_COLORS if key != source]
        dest_labels = [colors.region_label(key) for key in dest_choices]
        label, ok = QInputDialog.getItem(
            self.page,
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
            ErrorMessage(self.page, 'No points fell inside the lasso.')
            return

        moved = branch_ops.move_points(results, source, dest, inside)
        branch_ops.sync_branch_labels(results, moved, source, dest, self.data.centerline)

        self.refresh_scene()
        self.page.branches.refresh_labels_scene()
        self.status_bar.showMessage(
            f'Reclassified {len(moved)} point(s): {colors.region_label(source)} → {colors.region_label(dest)}.'
        )
        self.left_half.geometry_toolbar.end_lasso()
