"""The final mesh: fix and remesh, smoothing, sphere brush (+ undo), export."""

from __future__ import annotations

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QProgressDialog

from domain.fusion import display_types as colors
from domain.fusion.types import FusionScene
from fusion_pipeline import geometry as geo
from fusion_pipeline import multimodars_api as mm_api
from gui.progress_worker import StdoutCapturingWorker
from pages.fusion.controllers.base import StageController
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from tools.sphere_smooth import local_smooth, local_smooth_region


class FinalMeshController(StageController):
    def __init__(self, page) -> None:
        super().__init__(page)
        self._remesh_worker: StdoutCapturingWorker | None = None

    def remesh_running(self) -> bool:
        return self._remesh_worker is not None and self._remesh_worker.isRunning()

    def shutdown(self) -> None:
        if self._remesh_worker is not None:
            self._remesh_worker.wait()

    def on_remesh(self) -> None:
        fc = self.right_half.fusion_column
        if not self.require(self.data.stitched is not None, 'Stitch the geometry first.'):
            return
        if self.remesh_running():
            return
        stitched = self.data.stitched
        assert stitched is not None

        progress = QProgressDialog('Fixing & remeshing…', '', 0, 0, self.page)
        progress.setWindowTitle('Fix & Remesh')
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()
        self.status_bar.showMessage('Fixing & remeshing…')

        worker = StdoutCapturingWorker(mm_api.run_remesh, (stitched['mesh'],), fc.remesh_kwargs(), parent=self.page)
        worker.line_printed.connect(progress.setLabelText)
        worker.finished_ok.connect(lambda mesh: self._on_remesh_done(progress, mesh))
        worker.failed.connect(lambda message: self._on_remesh_failed(progress, message))
        self._remesh_worker = worker
        worker.start()

    def _on_remesh_done(self, progress: QProgressDialog, mesh) -> None:
        progress.close()
        self._remesh_worker = None
        self.data.final_mesh_undo.clear()
        self.set_final_mesh(mesh)
        self.status_bar.showMessage('Remeshed.')

    def _on_remesh_failed(self, progress: QProgressDialog, message: str) -> None:
        progress.close()
        self._remesh_worker = None
        logger.error(f'Fix & Remesh failed: {message}')
        ErrorMessage(self.page, message)
        self.status_bar.showMessage('Failed — see log')

    def on_smooth(self) -> None:
        fc = self.right_half.fusion_column
        final_mesh = self.data.final_mesh
        if not self.require(final_mesh is not None, 'Fix and remesh first.'):
            return
        assert final_mesh is not None
        # taubin_smooth works in place, so snapshot the vertices before, not after.
        before = final_mesh.vertices.copy()
        mesh = self.run('Smoothing…', 'Smoothed.', geo.taubin_smooth, final_mesh, lamb=fc.taubin_lamb())
        if mesh is None:
            return
        self.data.final_mesh_undo.push(before)
        self.set_final_mesh(mesh)

    def set_final_mesh(self, mesh) -> None:
        self.data.final_mesh = mesh
        self.left_half.viewer.add_mesh(FusionScene.CCTA_GEOMETRY, 'final_mesh', mesh, color=colors.FINAL_MESH_COLOR)
        self.left_half.refresh_toolbar(FusionScene.CCTA_GEOMETRY)
        self.left_half.geometry_toolbar.set_undo_available(bool(self.data.final_mesh_undo))

    def reset_smoothing_state(self) -> None:
        """No final mesh any more (re-stitch, Clear All Data): drop undo and end brushing."""
        self.data.final_mesh_undo.clear()
        gt = self.left_half.geometry_toolbar
        gt.set_undo_available(False)
        gt.end_sphere_brush()

    def on_sphere_mode_toggled(self, enabled: bool) -> None:
        viewer = self.left_half.viewer
        if enabled:
            if self.data.final_mesh is None:
                ErrorMessage(self.page, 'Fix and remesh first — the sphere brush smooths the final mesh.')
                self.left_half.geometry_toolbar.end_sphere_brush()
                return
            viewer.set_sphere_radius(self.left_half.geometry_toolbar.sphere_radius_mm())
            viewer.set_sphere_mode(True, FusionScene.CCTA_GEOMETRY, 'final_mesh')
            self.left_half.show_scene(FusionScene.CCTA_GEOMETRY)
            self.status_bar.showMessage('Sphere smooth: hover to preview, click to smooth, Ctrl+wheel for radius.')
        else:
            viewer.set_sphere_mode(False)

    def on_sphere_radius_changed(self, radius_mm: float) -> None:
        viewer = self.left_half.viewer
        viewer.set_sphere_radius(radius_mm)
        viewer.refresh_sphere_hover()  # highlight must follow the new radius

    def on_sphere_hovered(self, x: float, y: float, z: float) -> None:
        mesh = self.data.final_mesh
        if mesh is None:
            return
        radius = self.left_half.geometry_toolbar.sphere_radius_mm()
        region, _ = local_smooth_region(mesh, (x, y, z), radius)
        self.left_half.viewer.set_sphere_highlight(np.asarray(mesh.vertices)[region])

    def on_sphere_clicked(self, x: float, y: float, z: float) -> None:
        mesh = self.data.final_mesh
        if mesh is None:
            return
        gt = self.left_half.geometry_toolbar
        smoothed = self.run(
            'Smoothing locally…',
            'Locally smoothed.',
            local_smooth,
            mesh,
            (x, y, z),
            gt.sphere_radius_mm(),
            iterations=gt.sphere_iterations(),
            lamb=self.right_half.fusion_column.taubin_lamb(),
        )
        if smoothed is None:
            return
        self.data.final_mesh_undo.push(mesh.vertices.copy())
        self.set_final_mesh(smoothed)
        self.left_half.viewer.refresh_sphere_hover()

    def on_undo_smoothing(self) -> None:
        mesh = self.data.final_mesh
        vertices = self.data.final_mesh_undo.pop() if mesh is not None else None
        if mesh is None or vertices is None:
            return
        restored = mesh.copy()
        restored.vertices = vertices
        self.set_final_mesh(restored)
        self.left_half.viewer.refresh_sphere_hover()
        self.status_bar.showMessage(f'Undid smoothing step ({len(self.data.final_mesh_undo)} left).')

    def on_export(self, path: str) -> None:
        if not self.require(self.data.final_mesh is not None, 'Nothing to export yet — finish the pipeline first.'):
            return
        self.run('Exporting…', f'Exported: {path}', geo.export_mesh, self.data.final_mesh, path)
