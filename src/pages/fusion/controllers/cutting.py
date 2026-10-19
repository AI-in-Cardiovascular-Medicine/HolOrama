"""Inlets & Outlets tab: picks slices, keeps the cut list and shows the cut geometry.
The cutting itself is in fusion_pipeline.cuts."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any

import numpy as np
import trimesh
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

from domain.fusion import display_types as colors
from domain.fusion.runtime_types import CutResult, CutSpec
from domain.fusion.types import FusionScene
from fusion_pipeline.cuts import SliceKey, apply_cuts, export_cut_geometry, mesh_fingerprint, slice_points, tree_slices
from pages.fusion.controllers.base import StageController

VESSELS = ('aorta', 'rca', 'lca')
# Source meshes, latest first: the default when the user hasn't chosen one.
SOURCES = {'final': 'Final mesh', 'stitched': 'Stitched mesh', 'ccta': 'CCTA mesh'}


@contextmanager
def _busy():
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


class CutController(StageController):
    """Drives the Inlets & Outlets tab and Export Cut Geometry. State lives in page.data,
    so Clear All Data resets it. Only the per-tree slice cache is kept here."""

    def __init__(self, page) -> None:
        super().__init__(page)
        # (tree, slices, slice keys, every contour point (N, 3), its slice key index (N,))
        self._tree_cache: tuple | None = None

    @property
    def toolbar(self):
        return self.left_half.cut_toolbar

    @property
    def viewer(self):
        return self.left_half.viewer

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """The tree or the centerlines changed: slice indices and branch ids of the cuts no
        longer mean the same, so drop them."""
        if self.data.cuts:
            logger.info(f'Dropping {len(self.data.cuts)} cut(s): the vessel tree or centerlines changed.')
        self.data.cuts = []
        self.data.cut_selection = None
        self.data.cut_result = None
        self._tree_cache = None
        self.viewer.clear_scene(FusionScene.INLETS_OUTLETS)
        self.left_half.refresh_toolbar(FusionScene.INLETS_OUTLETS)
        self.toolbar.set_selection(None)
        self.toolbar.set_cuts([])

    def drop_branches(self, vessel: str, branch_ids) -> None:
        """Forget the cuts on deleted branches, keep every other cut."""
        gone = set(branch_ids)
        on_gone = [s.name for s in self.data.cuts if s.vessel == vessel and s.branch in gone]
        if on_gone:
            logger.info(f'Dropping cut(s) on deleted branches: {", ".join(on_gone)}')
        self.data.cuts = [s for s in self.data.cuts if not (s.vessel == vessel and s.branch in gone)]
        selection = self.data.cut_selection
        if selection is not None and selection[0] == vessel and selection[1] in gone:
            self.data.cut_selection = None
        self.data.cut_result = None
        self._tree_cache = None  # same tree object, fewer branches

    def _tree(self) -> tuple | None:
        tree = self.data.vessel_tree
        if tree is None:
            return None
        if self._tree_cache is None or self._tree_cache[0] is not tree:
            slices = tree_slices(tree)
            self._tree_cache = (tree, slices, *slice_points(slices))
        return self._tree_cache

    def _slices(self) -> dict:
        cache = self._tree()
        return cache[1] if cache is not None else {}

    def _source_meshes(self) -> dict[str, Any]:
        d = self.data
        meshes = {
            'final': d.final_mesh,
            'stitched': (d.stitched or {}).get('mesh'),
            'ccta': (d.results or {}).get('mesh'),
        }
        return {key: mesh for key, mesh in meshes.items() if mesh is not None}

    def _source(self) -> tuple[str, Any] | None:
        meshes = self._source_meshes()
        if not meshes:
            return None
        key = self.data.cut_source if self.data.cut_source in meshes else next(iter(meshes))
        return key, meshes[key]

    def _centerlines(self) -> dict[str, Any]:
        d = self.data
        centerlines = {'aorta': d.centerline_aorta, 'rca': d.centerline_rca, 'lca': d.centerline_lca}
        return {vessel: cl for vessel, cl in centerlines.items() if cl is not None}

    def _result(self) -> CutResult | None:
        """The cuts replayed on the source mesh, recomputed only when the mesh or the cuts
        changed since the cached one."""
        source, tree = self._source(), self.data.vessel_tree
        if source is None or tree is None:
            self.data.cut_result = None
            return None
        source_key, mesh = source
        key = (source_key, mesh_fingerprint(mesh), tuple(spec.geometry for spec in self.data.cuts))
        cached = self.data.cut_result
        if cached is not None and cached.key == key:
            return cached
        start = CutResult(mesh=mesh, centerlines=self._centerlines())
        if self.data.cuts:
            self.status_bar.showMessage(f'Applying {len(self.data.cuts)} cut(s) to the {SOURCES[source_key]}…')
        with _busy():
            result = apply_cuts(start, self._slices(), tree, self.data.cuts)
        result.key = key
        self.data.cut_result = result
        if result.failed:
            self.status_bar.showMessage(f'{len(result.failed)} cut(s) could not be applied — see the cut list.')
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Bring the scene and the toolbar up to date (tab shown, or the cuts changed)."""
        source = self._source()
        self.toolbar.set_sources([(SOURCES[key], key) for key in self._source_meshes()], source[0] if source else None)
        result = self._result()
        scene = FusionScene.INLETS_OUTLETS
        viewer = self.viewer

        mesh = result.mesh if result is not None else (source[1] if source is not None else None)
        if mesh is not None:
            viewer.add_mesh(scene, 'mesh', mesh, color=colors.CUT_MESH_COLOR, opacity=colors.CUT_MESH_OPACITY)
        else:
            viewer.remove_layer(scene, 'mesh')
        caps = list(result.caps.values()) if result is not None else []
        if caps:
            viewer.add_mesh(scene, 'caps', trimesh.util.concatenate(caps), color=colors.CUT_CAP_COLOR)
        else:
            viewer.remove_layer(scene, 'caps')

        centerlines = result.centerlines if result is not None else self._centerlines()
        for vessel in VESSELS:
            key = f'centerline_{vessel}'
            centerline = centerlines.get(vessel)
            if centerline is not None:
                points = np.array(centerline.points_as_tuples())
                viewer.add_points(scene, key, points, color=colors.CENTERLINE_COLORS[key], size=colors.POINT_SIZE)
            else:
                viewer.remove_layer(scene, key)

        cache = self._tree()
        alive = self._alive_points(cache)
        for vessel in VESSELS:
            key = f'slices_{vessel}'
            if cache is None:
                viewer.remove_layer(scene, key)
                continue
            keys, points, owners = cache[2], cache[3], cache[4]
            of_vessel = np.array([k[0] == vessel for k in keys], dtype=bool)
            mask = alive & of_vessel[owners] if len(owners) else alive
            if mask.any():
                viewer.add_points(
                    scene, key, points[mask], color=colors.CUT_SLICE_COLORS[vessel], size=colors.POINT_SIZE
                )
            else:
                viewer.remove_layer(scene, key)

        self._draw_selection()
        self._sync_cut_list()
        self.left_half.refresh_toolbar(scene)

    def _alive_points(self, cache) -> np.ndarray:
        """Per contour point of the tree: whether its slice is still there to cut."""
        if cache is None:
            return np.zeros(0, dtype=bool)
        result = self.data.cut_result
        dropped = result.dropped if result is not None else set()
        alive_keys = np.array([k not in dropped for k in cache[2]], dtype=bool)
        return alive_keys[cache[4]] if len(cache[4]) else np.zeros(0, dtype=bool)

    def _draw_selection(self) -> None:
        scene = FusionScene.INLETS_OUTLETS
        selection = self.data.cut_selection
        contour = self._contour(selection) if selection is not None else None
        if contour is None:
            self.data.cut_selection = None
            self.viewer.remove_layer(scene, 'selected_slice')
            self.toolbar.set_selection(None)
            return
        ring = np.array(contour.points_as_tuples())
        self.viewer.add_polyline(
            scene,
            'selected_slice',
            np.vstack([ring, ring[:1]]),
            color=colors.CUT_SELECTED_SLICE_COLOR,
            line_width=colors.CUT_SELECTED_SLICE_LINE_WIDTH,
            on_top=True,  # coronary rings are small and sit on their own slice points
        )

    def _sync_cut_list(self) -> None:
        result = self.data.cut_result
        failed = result.failed if result is not None else {}
        self.toolbar.set_cuts(
            [
                (
                    f'{spec.name} — {self._slice_label((spec.vessel, spec.branch, spec.slice_idx))}, keep {spec.keep}',
                    failed.get(i),
                )
                for i, spec in enumerate(self.data.cuts)
            ]
        )

    def _contour(self, key: SliceKey):
        vessel, branch, j = key
        contours = self._slices().get((vessel, branch), [])
        return contours[j] if 0 <= j < len(contours) else None

    def _branch_label(self, vessel: str, branch: int) -> str:
        return 'Aorta' if vessel == 'aorta' else self.page.branches.name(vessel, branch)

    def _slice_label(self, key: SliceKey) -> str:
        vessel, branch, j = key
        n = len(self._slices().get((vessel, branch), []))
        return f'{self._branch_label(vessel, branch)} · slice {j + 1}/{n}'

    def _default_name(self, key: SliceKey) -> str:
        vessel, branch, j = key
        if vessel == 'aorta':
            n = len(self._slices().get(('aorta', 0), []))
            return 'Aorta proximal' if j < n / 2 else 'Aorta distal'
        return f'{self._branch_label(vessel, branch)} outlet'

    def _unique_name(self, name: str, skip: int | None = None) -> str:
        taken = {spec.name for i, spec in enumerate(self.data.cuts) if i != skip}
        unique, n = name, 2
        while unique in taken:
            unique, n = f'{name} {n}', n + 1
        return unique

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select(self, key: SliceKey) -> None:
        self.data.cut_selection = key
        self.toolbar.set_selection(self._slice_label(key), self._unique_name(self._default_name(key)))
        self._draw_selection()

    def on_point_picked(self, x: float, y: float, z: float) -> None:
        """Snap a click to the slice with the nearest ring point (not the nearest centre, which
        near a bifurcation can belong to the neighbouring branch), ignoring cut-off slices."""
        cache = self._tree()
        if cache is None:
            return
        alive = self._alive_points(cache)
        if not alive.any():
            return
        dist = np.linalg.norm(cache[3] - np.array([x, y, z]), axis=1)
        dist[~alive] = np.inf
        self.select(cache[2][cache[4][int(np.argmin(dist))]])

    def on_step(self, step: int) -> None:
        """Move the selection along its branch, skipping slices already cut off."""
        selection = self.data.cut_selection
        if selection is None:
            return
        vessel, branch, j = selection
        n = len(self._slices().get((vessel, branch), []))
        result = self.data.cut_result
        dropped = result.dropped if result is not None else set()
        j += step
        while 0 <= j < n:
            if (vessel, branch, j) not in dropped:
                self.select((vessel, branch, j))
                return
            j += step

    def on_source_changed(self, key: str) -> None:
        self.data.cut_source = key
        self.refresh()

    def on_cut(self, keep: str, name: str) -> None:
        selection = self.data.cut_selection
        if not self.require(selection is not None, 'Pick a slice to cut at first (Pick Point).'):
            return
        assert selection is not None
        result = self._result()
        if not self.require(result is not None, 'Discretize the vessel tree and build a mesh first.'):
            return
        assert result is not None
        vessel, branch, slice_idx = selection
        name = self._unique_name(name.strip() or self._default_name(selection))
        spec = CutSpec(vessel, branch, slice_idx, keep=keep, name=name)
        specs = [*self.data.cuts, spec]
        self.status_bar.showMessage(f'Cutting at {self._slice_label(selection)}…')
        with _busy():
            cut = apply_cuts(result, self._slices(), self.data.vessel_tree, specs, first=len(specs) - 1)
        failure = cut.failed.get(len(specs) - 1)
        if failure is not None:
            self.require(False, f'Could not cut at {self._slice_label(selection)}:\n{failure}')
            self.status_bar.showMessage('Cut failed.')
            return
        cut.key = (*result.key[:2], tuple(s.geometry for s in specs))
        self.data.cuts = specs
        self.data.cut_result = cut
        self.data.cut_selection = None
        self.refresh()
        self.status_bar.showMessage(f'Cut {spec.name!r} at {self._slice_label(selection)}.')

    def on_remove(self, index: int) -> None:
        spec = self.data.cuts.pop(index)
        self.refresh()  # replays the remaining cuts
        self.status_bar.showMessage(f'Removed cut {spec.name!r}.')

    def on_rename(self, index: int) -> None:
        spec = self.data.cuts[index]
        name, ok = QInputDialog.getText(self.page, 'Rename cut', 'Cap name:', text=spec.name)
        if not ok or not name.strip():
            return
        self.data.cuts[index] = dataclasses.replace(spec, name=self._unique_name(name.strip(), skip=index))
        self._sync_cut_list()  # the name doesn't change the geometry, so no replay

    def on_export(self, path: str) -> None:
        if not self.require(bool(self.data.cuts), 'Make a cut on the Inlets & Outlets tab first.'):
            return
        result = self._result()
        if not self.require(result is not None, 'Discretize the vessel tree and build a mesh first.'):
            return
        assert result is not None
        if result.failed:
            names = ', '.join(self.data.cuts[i].name for i in sorted(result.failed))
            reply = QMessageBox.question(
                self.page,
                'Export cut geometry',
                f'These cuts could not be applied on this mesh and will be missing: {names}. Export anyway?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._sync_cut_list()
        written = self.run(
            'Exporting cut geometry…',
            f'Exported cut geometry: {path}',
            export_cut_geometry,
            result,
            self.data.cuts,
            path,
        )
        if written is not None:
            logger.info('Exported cut geometry:\n' + '\n'.join(written))
            self.status_bar.showMessage(f'Exported {len(written)} files next to {path}')
