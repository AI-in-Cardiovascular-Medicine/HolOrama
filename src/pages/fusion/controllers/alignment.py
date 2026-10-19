"""Column 2: load the pullback, align it, label the anomalous region."""

from __future__ import annotations

import numpy as np
from loguru import logger

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from fusion_pipeline import branches as branch_ops
from fusion_pipeline import geometry as geo
from fusion_pipeline import multimodars_api as mm_api
from pages.fusion.controllers.base import StageController
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


class AlignmentController(StageController):
    def on_load_pullback(self) -> None:
        ic = self.right_half.intravascular_column
        plan = ic.load_plan()
        if not self.require(bool(plan['input_path']), 'Select a pullback case folder first.'):
            return
        common = {
            'input_path': plan['input_path'],
            'step_rotation_deg': plan['step_rotation_deg'],
            'sample_size': plan['sample_size'],
            'n_points': plan['n_points'],
        }

        if plan['mode'] == 'oct':
            # OCT: single geometry loaded from the exported tagged-frame arrays.
            result = self.run(
                'Loading OCT pullback…', 'Pullback loaded.', mm_api.run_from_array, labels=plan['label'], **common
            )
        else:
            dia, sys = plan['diastole'], plan['systole']
            if not self.require(dia or sys, 'Select Diastole, Systole, or both.'):
                return
            if dia and sys:
                # Both phases -> diastole/systole pair.
                result = self.run(
                    'Loading pullback (dia + sys)…',
                    'Pullback loaded.',
                    mm_api.run_from_file_singlepair,
                    labels=[plan['dia_label'], plan['sys_label']],
                    **common,
                )
            else:
                # Exactly one phase -> single geometry, diastole flag picks which.
                label = plan['dia_label'] if dia else plan['sys_label']
                result = self.run(
                    'Loading pullback…',
                    'Pullback loaded.',
                    mm_api.run_from_file_single,
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
        """(phase_key, geom, color) rows of a pullback, see geo.phases."""
        phase_colors = {'a': colors.DIASTOLE_COLOR, 'b': colors.SYSTOLE_COLOR}
        return [(key, geom, phase_colors[key]) for key, geom in geo.phases(geometry)]

    def _add_geometry_pair_meshes(self, scene: FusionScene, geometry_pair, key_prefix: str) -> None:
        """Loft lumen and wall meshes for each phase of `geometry_pair` into `scene`."""
        viewer = self.left_half.viewer
        for phase_key, geom, color in self._phase_geoms(geometry_pair):
            if geom is None:
                continue
            key = f'{key_prefix}_{phase_key}'
            try:
                lumen_mesh = geo.frames_to_mesh(geom)
            except Exception as e:
                logger.warning(f'Could not loft a lumen mesh for {key}: {e}')
            else:
                viewer.add_mesh(scene, key, lumen_mesh, color=color, opacity=colors.IV_LUMEN_OPACITY)
            try:
                wall_mesh = geo.frames_to_mesh(geom, contour_type='Wall')
            except Exception as e:
                logger.warning(f'Could not loft a wall mesh for {key}: {e}')
            else:
                viewer.add_mesh(
                    scene, f'{key}_wall', wall_mesh, color=colors.IV_WALL_COLOR, opacity=colors.IV_WALL_OPACITY
                )
        self.left_half.refresh_toolbar(scene)

    def _selected_align_centerline(self, vessel: str, branch: int):
        """(centerline, references, selected_index) for the chosen vessel/branch. The centerline is the
        whole vessel, the references only the branch's own."""
        centerline = self.data.centerline(vessel)
        return (
            centerline,
            branch_ops.references(self.data.vessel_tree, vessel, branch),
            self.data.selected_reference_index[vessel],
        )

    def _align_takeoff_anomalous(self, vessel: str, branch: int) -> bool:
        """Acute takeoff only applies to the main vessel (a side branch leaves its parent, not the aorta)."""
        return branch == 0 and self.right_half.geometry_column.has_acute_takeoff(vessel)

    def on_align(self) -> None:
        ic = self.right_half.intravascular_column
        vessel, branch = ic.reference_vessel(), ic.reference_branch()
        if not self.require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self.require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        centerline, references, selected_index = self._selected_align_centerline(vessel, branch)
        if not self.require(
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
            ErrorMessage(self.page, f'Could not resolve reference points / branch: {e}')
            return
        # A side branch is scored against its own surface points, not the whole vessel's.
        points = results.get(f'{vessel}_branches', {}).get(branch) if branch else None

        result = self.run(
            'Aligning intravascular geometry…',
            'Alignment done.',
            mm_api.run_align_combined,
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

    def on_align_manual(self) -> None:
        """Like _on_run_align, but rotates by an explicit angle around one reference point instead of
        searching. Only meaningful for elliptic (anomalous) vessels."""
        ic = self.right_half.intravascular_column
        vessel, branch = ic.reference_vessel(), ic.reference_branch()
        if not self.require(self.data.iv_geometry_pair is not None, 'Load a pullback first.'):
            return
        if not self.require(self.data.vessel_tree is not None, 'Discretize the vessel tree first.'):
            return
        centerline, references, selected_index = self._selected_align_centerline(vessel, branch)
        if not self.require(centerline is not None, 'Run label_geometry first.'):
            return
        assert centerline is not None

        try:
            main_ref_pt = references[selected_index][0]
            cl_main = self._alignment_branch(centerline, branch)
        except (IndexError, AttributeError, ValueError, TypeError) as e:
            ErrorMessage(self.page, f'Could not resolve reference point / branch: {e}')
            return
        ref_point = self._resolve_manual_ref_point(cl_main, main_ref_pt, ic.manual_ref_point_offset())

        result = self.run(
            'Aligning intravascular geometry (manual)…',
            'Manual alignment done.',
            mm_api.run_align_manual,
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
        index, raw_index = geo.shifted_nearest_index(points, main_ref_pt, offset)
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
            color=colors.ALIGNED_CENTERLINE_COLOR,
            size=colors.POINT_SIZE,
        )
        viewer.add_points(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'ostium_point',
            np.array([ostium_point]),
            color=colors.TREE_REF_COLORS[0],
            size=colors.SELECTED_REF_POINT_SIZE,
        )
        # geom_a/geom_b are assumed diastole/systole (load_plan label order) to match the colors.
        self._add_geometry_pair_meshes(FusionScene.INTRAVASCULAR_ALIGNED, self.data.aligned, 'aligned_geom')
        self.refresh_ccta_mesh()
        self.left_half.show_scene(FusionScene.INTRAVASCULAR_ALIGNED)

    def refresh_ccta_mesh(self) -> None:
        """Overlay the unlabelled CCTA mesh in the Intravascular Aligned scene. Call whenever results['mesh']
        changes. No-op before alignment."""
        if self.data.aligned is None or self.data.results is None or 'mesh' not in self.data.results:
            return
        self.left_half.viewer.add_mesh(
            FusionScene.INTRAVASCULAR_ALIGNED,
            'ccta_mesh',
            self.data.results['mesh'],
            color=colors.ALIGNED_CCTA_MESH_COLOR,
            opacity=colors.ALIGNED_CCTA_MESH_OPACITY,
        )
        self._refresh_boundary_rings()
        self.left_half.refresh_toolbar(FusionScene.INTRAVASCULAR_ALIGNED)

    def _refresh_boundary_rings(self) -> None:
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
                viewer.add_points(
                    scene,
                    key,
                    points,
                    color=color,
                    size=colors.BOUNDARY_RING_POINT_SIZE,
                    visible=was_visible.get(key, True),
                )
            ring_number += 1

    def aligned_frames(self):
        geom = geo.primary_geometry(self.data.aligned)
        if geom is None:
            return None
        try:
            return geom.frames
        except AttributeError:
            return None

    def on_label_anomalous(self) -> None:
        """Split the aligned branch's pullback path into proximal/overlap/distal regions. Uses the
        vessel/branch stored at alignment time, not the current selector. Needs the whole vessel centerline.
        Off-path vertices go to off_path_points. Re-align after a split/merge (they reassign branch ids)."""
        frames = self.aligned_frames()
        if not self.require(frames is not None, 'Align the intravascular geometry first.'):
            return
        centerline = self.data.aligned_vessel_centerline
        if not self.require(centerline is not None, 'Run label_geometry first.'):
            return
        results = self.run(
            'Labeling overlap region…',
            'Overlap region labeled.',
            mm_api.run_label_anomalous_region,
            centerline,
            frames,
            self.data.results,
            branch_id=self.data.aligned_branch,
        )
        if results is not None:
            self.data.results = results
            self.page.fusion.invalidate_removal()  # stale: re-run Remove Labeled Points
            # proximal/distal/anomalous (shown as overlap)/off_path_points now exist
            self.page.ccta_geometry.refresh_scene()
            # Hide the aligned vessel's own region, which would cover them.
            vessel_layer = f'{self.data.aligned_vessel}_points'
            if vessel_layer in self.left_half.viewer.layer_states(FusionScene.CCTA_GEOMETRY):
                self.left_half.viewer.set_layer_visible(FusionScene.CCTA_GEOMETRY, vessel_layer, False)
                self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
