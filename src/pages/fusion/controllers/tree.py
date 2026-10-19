"""Vessel tree discretization, its scene and the reference (branch + triplet) selection."""

from __future__ import annotations

import numpy as np
from loguru import logger

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from fusion_pipeline import branches as branch_ops
from fusion_pipeline import multimodars_api as mm_api
from pages.fusion.controllers.base import StageController


class TreeController(StageController):
    def on_discretize(self) -> None:
        gc = self.right_half.geometry_column
        if not self.require(
            self.data.centerline_aorta is not None and self.data.results is not None,
            'Prepare centerlines, run Label Geometry, and Label Branches first.',
        ):
            return
        results = self.data.results
        assert results is not None
        if not self.require(
            'rca_branches' in results and 'lca_branches' in results,
            'Run Label Branches first.',
        ):
            return
        tree = self.run(
            'Discretizing vessel tree…',
            'Vessel tree ready.',
            mm_api.run_discretize_vessel_tree,
            self.data.centerline_aorta,
            self.data.centerline_rca,
            self.data.centerline_lca,
            self.data.results,
            **gc.discretize_tree_kwargs(),
        )
        if tree is None:
            return
        self.data.vessel_tree = tree
        self.page.cutting.reset()  # slice indices refer to the old tree

        self.data.selected_branch = {'rca': 0, 'lca': 0}
        self.data.selected_reference_index = {'rca': 0, 'lca': 0}

        self.refresh_scene()
        ic = self.right_half.intravascular_column
        ic.set_centerline_choices(self.centerline_choices(), ic.reference_vessel(), 0)
        for vessel in ('rca', 'lca'):
            self.left_half.tree_toolbar.set_branches(vessel, self.branch_choices(vessel), 0)
            self.select_branch(vessel, 0)
        self.left_half.show_scene(FusionScene.VESSEL_TREE)

    def branch_choices(self, vessel: str) -> list[tuple[str, int]]:
        return [
            (self.page.branches.name(vessel, b), b)
            for b in branch_ops.ids_with_references(self.data.vessel_tree, vessel)
        ]

    def centerline_choices(self) -> list[tuple[str, str, int]]:
        """(label, vessel, branch) for the Intravascular Alignment column's Centerline combo."""
        return [
            (self.page.branches.name(vessel, b), vessel, b)
            for vessel in ('rca', 'lca')
            for b in branch_ops.ids_with_references(self.data.vessel_tree, vessel)
        ]

    def _reference_labels(self, vessel: str, branch: int) -> list[str]:
        """Labels for the references of one branch, for the Vessel Tree tab and the Reference dropdown."""
        name = self.page.branches.name(vessel, branch)
        return [
            f'{name} ostium' if child is None else f'Takeoff of {self.page.branches.name(vessel, child)}'
            for child, _triplet in branch_ops.reference_entries(self.data.vessel_tree, vessel, branch)
        ]

    def select_branch(self, vessel: str, branch: int, index: int = 0) -> None:
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
        self.select_reference(vessel, index)

    def select_reference(self, vessel: str, index: int) -> None:
        """Apply reference `index` of `vessel`'s active branch and highlight it in the viewer."""
        branch = self.data.selected_branch[vessel]
        refs = branch_ops.references(self.data.vessel_tree, vessel, branch)
        if not 0 <= index < len(refs):
            logger.warning(f'{vessel.upper()} branch {branch} has no reference [{index}].')
            return
        triplet = refs[index]
        self.data.selected_reference_index[vessel] = index
        self.left_half.tree_toolbar.set_selected_reference(vessel, index)
        layer = 'selected_reference' if vessel == 'rca' else 'selected_lca_reference'
        self.left_half.viewer.add_points(
            FusionScene.VESSEL_TREE,
            layer,
            np.array(triplet),
            color=colors.SELECTED_REF_COLORS[vessel],
            size=colors.SELECTED_REF_POINT_SIZE,
        )
        self._sync_intravascular_reference_ui(vessel, index, triplet)

    def _sync_intravascular_reference_ui(self, vessel: str, index: int, triplet) -> None:
        """Sync the Intravascular Alignment column's reference display, only if it has the same vessel
        selected (so picking an LCA reference doesn't clobber an RCA setup)."""
        ic = self.right_half.intravascular_column
        if ic.reference_vessel() != vessel:
            return
        ic.set_selected_reference_index(index)
        ic.set_reference_points(triplet[0], triplet[1], triplet[2])

    def on_reference_centerline_changed(self, vessel: str, branch: int) -> None:
        """Switch to the chosen vessel/branch, keeping the selected reference if the branch is unchanged."""
        if self.data.vessel_tree is None:
            return
        keep = branch == self.data.selected_branch[vessel]
        self.select_branch(vessel, branch, self.data.selected_reference_index[vessel] if keep else 0)

    def on_reference_index_changed(self, index: int) -> None:
        """Route through the same selection path as the Vessel Tree tab so everything stays in sync."""
        self.select_reference(self.right_half.intravascular_column.reference_vessel(), index)

    def on_point_picked(self, x: float, y: float, z: float) -> None:
        """Select the reference whose triplet has a point closest to the click."""
        if self.data.vessel_tree is None:
            return
        nearest = branch_ops.nearest_reference(self.data.vessel_tree, (x, y, z))
        self.select_branch(*(nearest or ('rca', 0, 0)))

    def refresh_scene(self) -> None:
        """Recreate multimodars' plot_vessel_tree as native VTK layers."""
        tree = self.data.vessel_tree
        if tree is None:
            return
        viewer = self.left_half.viewer
        centroids: list[tuple[float, float, float]] = []

        def add_contours(key: str, contours, color: tuple[int, int, int]) -> None:
            points = [p for c in contours for p in c.points_as_tuples()]
            if points:
                viewer.add_points(FusionScene.VESSEL_TREE, key, np.array(points), color=color, size=colors.POINT_SIZE)
            centroids.extend(c.centroid for c in contours)

        add_contours('tree_aorta', tree.discretized_aorta, colors.TREE_AORTA_COLOR)
        for vessel, main_color in (('rca', colors.TREE_RCA_MAIN_COLOR), ('lca', colors.TREE_LCA_MAIN_COLOR)):
            branches = branch_ops.tree_branches(self.data.vessel_tree, vessel)
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
                size=colors.TREE_CENTROID_POINT_SIZE,
            )

        # Reference triplets: one layer per triplet slot, pooled over all branches, to keep the toolbar short.
        ref_slots: list[list[tuple[float, float, float]]] = [[], [], []]
        all_refs = [
            triplet
            for vessel in ('rca', 'lca')
            for branch in branch_ops.tree_branches(self.data.vessel_tree, vessel)
            for triplet in branch_ops.references(self.data.vessel_tree, vessel, branch)
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
                    size=colors.TREE_REF_POINT_SIZE,
                )
