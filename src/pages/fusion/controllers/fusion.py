"""Column 3 up to stitching: scaling, labeled point removal (+ undo), stitch."""

from __future__ import annotations

import copy

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from fusion_pipeline import geometry as geo
from fusion_pipeline import multimodars_api as mm_api
from fusion_pipeline import scaling as scaling_ops
from pages.fusion.controllers.base import StageController


class FusionController(StageController):
    def on_compute_scaling(self) -> None:
        frames = self.page.alignment.aligned_frames()
        if not self.require(frames is not None, 'Align the intravascular geometry first.'):
            return
        vessel = self.right_half.intravascular_column.reference_vessel()
        # The aligned branch, not the whole vessel: near a junction, vertices could snap to another branch's
        # centerline and be pushed sideways.
        centerline = self.data.aligned_centerline
        if not self.require(centerline is not None, 'Run label_geometry first.'):
            return
        scalings = self.run(
            'Computing scaling factors…',
            'Scaling factors computed.',
            mm_api.run_find_scalings,
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

    def on_apply_scaling(self) -> None:
        vessel = self.right_half.intravascular_column.reference_vessel()
        opposite_vessel = scaling_ops.OPPOSITE_VESSEL[vessel]
        centerline = self.data.centerline(vessel)
        opposite_centerline = self.data.centerline(opposite_vessel)
        if not self.require(
            None not in (self.data.prox_scaling, self.data.distal_scaling, self.data.aortic_scaling),
            'Compute scaling factors first.',
        ):
            return
        if not self.require(
            self.data.results is not None and centerline is not None and self.data.centerline_aorta is not None,
            'Run label_geometry first.',
        ):
            return

        # Read from the spinboxes: the user may have edited the computed values.
        scaling = self.right_half.fusion_column.scaling_values()
        remaining_scaling = scaling['remaining_vessels_scaling']
        if not self.require(
            remaining_scaling == 0.0 or opposite_centerline is not None,
            f'Missing {opposite_vessel.upper()} centerline for the remaining-vessels scaling — '
            'run Label Geometry first, or set Remaining vessels (mm) back to 0.',
        ):
            return

        results_out = self.run(
            'Applying scaling to mesh…',
            'Scaling applied.',
            scaling_ops.apply_scaling,
            self.data.results,
            vessel,
            scaling,
            centerline=centerline,
            centerline_aorta=self.data.centerline_aorta,
            opposite_centerline=opposite_centerline,
        )
        if results_out is None:
            return
        self.data.results = results_out
        self.invalidate_removal()  # stale: re-run Remove Labeled Points on the scaled mesh
        self.page.ccta_geometry.refresh_scene()
        self.page.branches.refresh_labels_scene()  # scaled mesh + its synced branch labels
        self.page.alignment.refresh_ccta_mesh()

    def on_remove_points(self) -> None:
        fc = self.right_half.fusion_column
        if not self.require(self.data.results is not None, 'Run label_geometry first.'):
            return
        # Taken before the call, in case remove_labeled_points_from_mesh touches its input.
        before = copy.deepcopy(self.data.results)
        assert before is not None
        results = self.run(
            'Removing labeled points…',
            'Points removed.',
            mm_api.run_remove_labeled_points,
            self.data.results,
            fc.remove_point_keys(),
        )
        if results is not None:
            self.data.removal_undo.push(before)
            self._set_removed_results(results)
            if self.data.aligned is not None:  # the aligned scene is empty until alignment
                self.left_half.show_scene(FusionScene.INTRAVASCULAR_ALIGNED)

    def on_undo_remove_points(self) -> None:
        results = self.data.removal_undo.pop()
        if results is None:
            return
        # With removals left, the restored state is a post-removal one Stitch can start from.
        # Otherwise it is the labelled mesh from before any removal.
        if self.data.removal_undo:
            self._set_removed_results(results)
        else:
            self.data.results = results
            self.data.results_points_removed = None
            self.clear_stitch_outputs()
            self.page.ccta_geometry.refresh_scene()
            self.page.branches.refresh_labels_scene()
            self.page.alignment.refresh_ccta_mesh()
            self.update_removal_undo_button()
        self.status_bar.showMessage(f'Undid point removal ({len(self.data.removal_undo)} left).')

    def _set_removed_results(self, results: dict) -> None:
        """Make `results` (the output of a removal) current, as the snapshot Stitch starts from."""
        self.data.results = results
        self.data.results_points_removed = copy.deepcopy(results)
        self.right_half.fusion_column.set_seam_points_limit(geo.max_seam_points(results))
        self.clear_stitch_outputs()
        self.page.ccta_geometry.refresh_scene()
        self.page.branches.refresh_labels_scene()
        self.page.alignment.refresh_ccta_mesh()
        self.update_removal_undo_button()

    def invalidate_removal(self) -> None:
        """`results` was replaced by something other than a removal: the post-removal
        snapshot and the removal undo stack no longer match it."""
        self.data.results_points_removed = None
        self.clear_removal_undo()

    def clear_removal_undo(self) -> None:
        self.data.removal_undo.clear()
        self.update_removal_undo_button()

    def update_removal_undo_button(self) -> None:
        self.right_half.fusion_column.set_undo_remove_available(bool(self.data.removal_undo))

    def clear_stitch_outputs(self) -> None:
        """Drop the stitched mesh and everything derived from it, so no stale final_mesh survives a re-stitch."""
        self.data.stitched = None
        self.data.final_mesh = None
        self.page.final_mesh.reset_smoothing_state()
        viewer = self.left_half.viewer
        viewer.remove_layer(FusionScene.CCTA_GEOMETRY, 'stitched_mesh')
        viewer.remove_layer(FusionScene.CCTA_GEOMETRY, 'final_mesh')
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)

    def on_stitch(self) -> None:
        fc = self.right_half.fusion_column
        if not self.require(self.data.aligned is not None, 'Align the intravascular geometry first.'):
            return
        if not self.require(self.data.results_points_removed is not None, 'Remove labeled points first.'):
            return
        if self.page.final_mesh.remesh_running():
            return
        aligned_geom = geo.primary_geometry(self.data.aligned)
        snapshot = self.data.results_points_removed
        assert aligned_geom is not None and snapshot is not None

        def _run():
            # Stitch mutates results in place, so pass a copy to keep the snapshot reusable.
            results = copy.deepcopy(snapshot)
            # The tree (discretized before scaling/removal, as multimodars expects) lets the
            # stitch reconnect side branches leaving the removed segment. It is only read.
            return mm_api.run_stitch(
                aligned_geom, results['mesh'], results, tree=self.data.vessel_tree, **fc.stitch_kwargs()
            )

        stitched = self.run('Stitching CCTA to intravascular…', 'Stitched.', _run)
        if stitched is None:
            return
        self.clear_stitch_outputs()
        self.data.stitched = stitched
        viewer = self.left_half.viewer
        viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'stitched_mesh', stitched['mesh'], color=colors.STITCHED_MESH_COLOR)
        viewer.isolate_layer(FusionScene.CCTA_GEOMETRY, 'stitched_mesh')
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)
