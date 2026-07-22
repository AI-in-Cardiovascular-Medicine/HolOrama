import glob
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import SimpleITK as sitk
import trimesh
from loguru import logger
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from domain.runtime_types import CctaRuntimeData
from gui.active_page import ActivePage
from input_output.input.ccta_io import (
    read_ct_volume,
    read_mask_volume,
    read_nifti_volume,
)
from input_output.output.stl_export import export_nifti, export_stl
from pages.ccta import cut_geometry, cut_state_io, vmtk_runner
from pages.ccta.progress_worker import StdoutCapturingWorker
from pages.ccta.left_half.cut_geometry_viewer import CutGeometryViewer3D
from pages.ccta.left_half.display import CctaDisplay
from pages.ccta.left_half.display_3d import CctaViewer3D
from pages.ccta.right_half.brush_panel import BrushPanel
from pages.ccta.right_half.mask_panel import MaskPanel
from pages.ccta.right_half.stl_extraction_panel import StlExtractionPanel
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from version import version_file_str

if TYPE_CHECKING:
    from gui.app import Master

# (lvot_anchor, lvot_normal, aorta_anchor, aorta_normal), all in voxel-index (z, y, x)
# space — the plane geometry _compute_cut_planes derives from the drawn cut lines.
_CutPlanes = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class _CenterlinePrereqs:
    """Everything _start_centerline_worker needs, gathered and validated up front by
    _validate_centerline_prereqs so the two concerns — can we run this? and actually
    running it — don't have to be read together as one function."""

    cut_mesh: trimesh.Trimesh
    cut_mesh_inlet: np.ndarray
    cut_mesh_outlet: np.ndarray
    rca_points: list[np.ndarray]
    lca_points: list[np.ndarray]
    venv_path: str
    build_path: str
    distro: str
    out_dir: str
    stem: str


class CctaPage(QWidget):
    def __init__(self, config: SimpleNamespace, status_bar) -> None:
        super().__init__()
        self.config: SimpleNamespace = config
        self.status_bar = status_bar
        self.data = CctaRuntimeData()
        self._last_image_dir: str | None = None
        self._source_path: str | None = None
        self._mask_dirty: bool = False
        self._cut_state_dirty: bool = False
        self._cut_labels: tuple[int, int, int] | None = None  # (cor, aorta, lv) from the last Build Cut Geometry
        self._centerlines_worker: StdoutCapturingWorker | None = None

        self._axial = CctaDisplay('axial')
        self._coronal = CctaDisplay('coronal')
        self._sagittal = CctaDisplay('sagittal')

        self._axial_label = QLabel('Axial')
        self._coronal_label = QLabel('Coronal')
        self._sagittal_label = QLabel('Sagittal')
        for lbl in (self._axial_label, self._coronal_label, self._sagittal_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        views = QWidget()
        grid = QGridLayout(views)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.addWidget(self._panel(self._axial, self._axial_label), 0, 0)
        grid.addWidget(self._panel(self._sagittal, self._sagittal_label), 0, 1)
        grid.addWidget(self._panel(self._coronal, self._coronal_label), 1, 0)
        self._3d_viewer = CctaViewer3D()
        grid.addWidget(self._3d_viewer, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Cut geometry gets its own tab (own VTK render window) rather than sharing
        # the segmentation 3-D view — it has its own mask/picking, unrelated to the
        # per-label segmentation actors shown in "Segmentation".
        self._cut_viewer = CutGeometryViewer3D()

        self._tabs = QTabWidget()
        self._tabs.addTab(views, 'Segmentation')
        self._tabs.addTab(self._cut_viewer, 'Cut Geometry')

        # Right panel: Mask labels (top, stretches) + Brush controls (bottom, fixed)
        self._mask_tab = MaskPanel()
        self._mask_tab.alpha_changed.connect(self._on_mask_alpha_changed)
        self._mask_tab.label_visibility_changed.connect(self._on_label_visibility_changed)
        self._mask_tab.label_colors_changed.connect(self._on_label_colors_changed)
        self._mask_tab.label_name_changed.connect(self._on_label_name_changed)

        self._brush_panel = BrushPanel()
        self._brush_panel.brush_enabled_changed.connect(self._on_brush_enabled_changed)
        self._brush_panel.geometry_changed.connect(self._on_brush_geometry_changed)
        self._mask_tab.label_name_changed.connect(self._brush_panel.update_label_name)
        self._mask_tab.set_brush_panel(self._brush_panel)

        self._stl_panel = StlExtractionPanel()
        self._stl_panel.line_draw_requested.connect(self._on_line_draw_requested)
        self._stl_panel.extract_requested.connect(self._on_extract_requested)
        self._stl_panel.build_cut_geometry_requested.connect(self._on_build_cut_geometry)
        self._stl_panel.outlet_point_mode_requested.connect(self._on_outlet_point_mode_requested)
        self._stl_panel.clear_outlet_points_requested.connect(self._cut_viewer.clear_points)
        self._mask_tab.label_name_changed.connect(self._stl_panel.update_label_name)
        self._cut_line_0: tuple | None = None  # (p1_zyx, p2_zyx) from axial view   (LVOT)
        self._cut_line_1: tuple | None = None  # (p1_zyx, p2_zyx) from coronal view (LVOT)
        self._aorta_cut_line: tuple | None = None  # (p1_zyx, p2_zyx) from coronal view (aorta top, required)

        right_col = QWidget()
        right_vbox = QVBoxLayout(right_col)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)
        right_vbox.addWidget(self._mask_tab, 1)
        right_vbox.addWidget(self._stl_panel, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._tabs)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([10000, 220])

        QVBoxLayout(self).addWidget(splitter)

        for display in (self._axial, self._coronal, self._sagittal):
            display.cursor_moved.connect(self._on_cursor_moved)
            display.windowing_changed.connect(self._on_windowing_changed)
            display.mask_painted.connect(self._on_mask_painted)
            display.mask_about_to_change.connect(self._on_mask_about_to_change)
        self._3d_viewer.cursor_moved.connect(self._on_cursor_moved)
        self._3d_viewer.mask_erased.connect(self._on_3d_mask_erased)
        self._3d_viewer.mask_about_to_change.connect(self._on_mask_about_to_change)
        self._cut_viewer.outlet_points_changed.connect(self._stl_panel.set_outlet_point_count)
        self._cut_viewer.outlet_points_changed.connect(self._on_outlet_points_changed)
        self._cut_viewer.smooth_requested.connect(self._on_smooth_requested)
        self._cut_viewer.reduce_mesh_requested.connect(self._on_reduce_mesh_requested)
        self._cut_viewer.calculate_centerlines_requested.connect(self._on_calculate_centerlines)

        self._pending_coronal_cut: int = 1  # 1 = LVOT, 2 = aorta top
        self._axial.line_drawn.connect(lambda p1, p2: self._on_line_drawn(0, p1, p2))
        self._coronal.line_drawn.connect(lambda p1, p2: self._on_line_drawn(self._pending_coronal_cut, p1, p2))

        timer: QTimer = QTimer(self)
        timer.timeout.connect(self._auto_save)
        timer.start(self.config.save.autosave_interval)

    def shutdown(self) -> None:
        self._3d_viewer.shutdown()
        self._cut_viewer.shutdown()

    def open_folder(self) -> None:
        master = cast('Master', self.window())
        master._switch_page(ActivePage.CCTA.value)

        source = self._choose_data_source()
        if source is None:
            return

        # Path confirmed — reinstantiate for a guaranteed clean state.
        # `self` must not be used after this point.
        master.reload_ccta()
        master.ccta_page._load_case(*source)

    def _choose_data_source(self) -> tuple[str, str] | None:
        """Ask DICOM-folder vs NIfTI-file, then prompt for the matching path.
        Returns (path, mode), or None if the user cancels at either step."""
        msg = QMessageBox(self)
        msg.setWindowTitle('Open CT Data')
        msg.setText('Select data format:')
        dicom_btn = msg.addButton('DICOM Folder', QMessageBox.ButtonRole.ActionRole)
        nifti_btn = msg.addButton('NIfTI File', QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == dicom_btn:
            path = QFileDialog.getExistingDirectory(
                self,
                'Open DICOM Folder',
                '',
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            return (path, 'dicom') if path else None
        if clicked == nifti_btn:
            path, _ = QFileDialog.getOpenFileName(
                self,
                'Open NIfTI File',
                '',
                'NIfTI files (*.nii *.nii.gz);;All Files (*)',
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            return (path, 'nifti') if path else None
        return None

    def _load_case(self, path: str, mode: str) -> None:
        """Read the volume chosen by _choose_data_source and populate this (freshly
        reinstantiated, per open_folder) page with it."""
        result = self._read_volume_with_progress(path, mode)
        if result is None:
            return
        volume, metadata = result
        self._apply_loaded_volume(volume, metadata, mode)
        self._finish_loading_for_mode(path, mode)

    def _read_volume_with_progress(self, path: str, mode: str) -> tuple[np.ndarray, dict] | None:
        """Read the CT volume from disk behind a modal progress dialog. Shows an
        ErrorMessage and returns None if reading fails."""
        progress = QProgressDialog('', '', 0, 0, self)
        progress.setWindowTitle('Loading CCTA')
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()

        try:
            if mode == 'dicom':
                progress.setLabelText('Scanning folder...')

                def _cb(current: int, total: int) -> None:
                    if progress.maximum() == 0:
                        progress.setMaximum(total)
                    progress.setValue(current)
                    progress.setLabelText(f'Reading slice {current} / {total}...')
                    QApplication.processEvents()

                return read_ct_volume(path, progress_cb=_cb)
            progress.setLabelText('Reading NIfTI...')
            QApplication.processEvents()
            return read_nifti_volume(path)
        except ValueError as e:
            ErrorMessage(self, str(e))
            self.status_bar.showMessage('Ready')
            return None
        finally:
            progress.close()

    def _apply_loaded_volume(self, volume: np.ndarray, metadata: dict, mode: str) -> None:
        """Push a freshly-read volume into this page's displays/data and report it on
        the status bar. Source-path/mask bookkeeping is mode-specific — handled by
        _finish_loading_for_mode instead."""
        master = cast('Master', self.window())
        dz = metadata['slice_thickness']
        dy, dx = metadata['pixel_spacing']
        self.data.volume = volume
        self.data.voxel_spacing = (dz, dy, dx)

        ccta_meta = metadata.get('ccta_metadata')
        if ccta_meta is not None:
            master.ccta_metadata = ccta_meta

        for display in (self._axial, self._coronal, self._sagittal):
            display.set_volume(volume, self.data.voxel_spacing)

        Z, Y, X = volume.shape
        self._update_labels(Z // 2, Y // 2, X // 2, Z, Y, X)
        self._initialize_empty_mask()
        fmt = 'NIfTI' if mode == 'nifti' else 'CCTA'
        self.status_bar.showMessage(f'{fmt}: {Z} slices  |  pixel spacing {dy:.3f} mm  |  slice thickness {dz:.3f} mm')

    def _finish_loading_for_mode(self, path: str, mode: str) -> None:
        """Mode-specific source-path bookkeeping plus mask auto-load/prompt, run
        after the volume itself is already applied."""
        if mode == 'nifti':
            self._last_image_dir = os.path.dirname(os.path.abspath(path))
            nifti_base = path
            for ext in ('.nii.gz', '.nii'):
                if nifti_base.endswith(ext):
                    nifti_base = nifti_base[: -len(ext)]
                    break
            self._source_path = nifti_base
            if not self._try_auto_load_mask():
                reply = QMessageBox.question(
                    self,
                    'Load Mask?',
                    'Would you like to load a segmentation mask for this volume?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.open_mask()
        else:
            self._source_path = os.path.join(path, os.path.basename(path))
            self._try_auto_load_mask()

    def _try_auto_load_mask(self) -> bool:
        """Load the most recent versioned mask for the current source path, if one exists."""
        if self._source_path is None:
            return False
        matches = glob.glob(f'{self._source_path}_ccta_seg_*.nii.gz')
        if not matches:
            return False
        mask_path = max(matches, key=os.path.getmtime)
        try:
            mask, _ = read_mask_volume(mask_path)
        except ValueError:
            return False
        self._apply_mask(mask, clear_undo=True)
        self.status_bar.showMessage(f'Mask auto-loaded: {os.path.basename(mask_path)}')
        self._try_load_cut_state()
        return True

    def _try_load_cut_state(self) -> None:
        """Restore previously drawn cut lines, label choices, and RCA/LCA outlet
        points for this case (if any were saved), and automatically rebuild the cut
        geometry from them — so re-opening a case picks up right where Build Cut
        Geometry / outlet-point work left off, without redrawing anything by hand."""
        if self._source_path is None:
            return
        state = cut_state_io.load_cut_state(self._source_path)
        if state is None:
            return

        labels = self._restore_cut_state_onto_ui(state)
        if labels is None:
            return
        cor, aorta, lv = labels
        self._stl_panel.set_selected_labels(cor, aorta, lv)
        # switch_tab=False: this runs on passive file-open, not an explicit "Build Cut
        # Geometry" click, so it shouldn't yank the user into the Cut Geometry tab.
        self._on_build_cut_geometry(cor, aorta, lv, switch_tab=False)
        if state['rca_points']:
            self._cut_viewer.set_points('rca', state['rca_points'])
        if state['lca_points']:
            self._cut_viewer.set_points('lca', state['lca_points'])
        self.status_bar.showMessage('Restored cut geometry from previous session.')

    def _restore_cut_state_onto_ui(self, state: dict) -> tuple[int, int, int] | None:
        """Write a loaded cut-state dict's label names and cut lines onto the mask
        panel / STL panel / overlays. Returns the (cor, aorta, lv) labels if they're
        all still present in the current mask and the cut lines are complete enough
        to rebuild the cut geometry from, else None."""
        if state['label_names']:
            self._mask_tab.set_label_names(state['label_names'])

        self._cut_line_0 = state['cut_line_0']
        self._cut_line_1 = state['cut_line_1']
        self._aorta_cut_line = state['aorta_cut_line']
        for i, line in enumerate((self._cut_line_0, self._cut_line_1, self._aorta_cut_line)):
            if line is not None:
                self._stl_panel.set_line_drawn(i)
        self._refresh_cut_line_overlays()

        cor, aorta, lv = state['cor_label'], state['aorta_label'], state['lv_label']
        labels_present = cor is not None and aorta is not None and lv is not None
        labels_valid = labels_present and all(lbl in self.data.labels for lbl in (cor, aorta, lv))
        if not (labels_valid and self._cut_lines_ready()):
            return None
        return cor, aorta, lv

    def _apply_mask(self, mask: np.ndarray, clear_undo: bool = False) -> None:
        """Apply a loaded mask array to all displays and panels."""
        if clear_undo:
            self.data.mask_undo.clear()
        self.data.mask = mask
        self.data.labels = sorted(int(v) for v in np.unique(mask) if v != 0)
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_mask(mask, self.data.labels)
        self._mask_tab.set_labels(self.data.labels)
        self._brush_panel.set_labels(self.data.labels)
        self._stl_panel.set_labels(self.data.labels, self._mask_tab.label_names())
        if self.data.voxel_spacing is not None:
            self._3d_viewer.set_mask(mask, self.data.labels, self.data.voxel_spacing)

    def open_mask(self) -> None:
        if self.data.volume is None:
            ErrorMessage(self, 'Load a CT volume before opening a mask.')
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open CCTA Mask',
            self._last_image_dir or '..',
            'NIfTI files (*.nii *.nii.gz);;All Files (*)',
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return

        try:
            mask, _ = read_mask_volume(path)
        except ValueError as e:
            ErrorMessage(self, str(e))
            return

        self._apply_mask(mask, clear_undo=True)
        self.status_bar.showMessage(f'Mask loaded: {len(self.data.labels)} label(s) — {self.data.labels}')

    def save_mask(self) -> None:
        if self.data.mask is None or self._source_path is None:
            ErrorMessage(self, 'No mask loaded.')
            return
        self._mask_dirty = False
        spacing = self.data.voxel_spacing if self.data.voxel_spacing else (1.0, 1.0, 1.0)
        out_path = f'{self._source_path}_ccta_seg_{version_file_str}.nii.gz'
        self._save_mask_snapshot(self.data.mask.copy(), spacing, out_path)
        self.status_bar.showMessage(f'Mask saved: {os.path.basename(out_path)}')

    def _auto_save(self) -> None:
        if self._mask_dirty and self.data.mask is not None and self._source_path is not None:
            self._mask_dirty = False
            snapshot = self.data.mask.copy()
            spacing = self.data.voxel_spacing if self.data.voxel_spacing else (1.0, 1.0, 1.0)
            out_path = f'{self._source_path}_ccta_seg_{version_file_str}.nii.gz'
            threading.Thread(target=self._save_mask_snapshot, args=(snapshot, spacing, out_path), daemon=True).start()

        if self._cut_state_dirty and self._source_path is not None:
            self._cut_state_dirty = False
            self._save_cut_state()

    def _save_cut_state(self) -> None:
        assert self._source_path is not None  # callers only invoke this after checking it
        cor, aorta, lv = self._cut_labels if self._cut_labels is not None else (None, None, None)
        cut_state_io.save_cut_state(
            self._source_path,
            cor,
            aorta,
            lv,
            self._cut_line_0,
            self._cut_line_1,
            self._aorta_cut_line,
            self._cut_viewer.rca_points_voxel(),
            self._cut_viewer.lca_points_voxel(),
            self._mask_tab.label_names(),
        )

    @staticmethod
    def _save_mask_snapshot(snapshot: np.ndarray, spacing: tuple, out_path: str) -> None:
        dz, dy, dx = spacing
        out_dir = os.path.dirname(out_path) or '.'
        tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.nii.gz')
        os.close(tmp_fd)
        try:
            img = sitk.GetImageFromArray(snapshot)  # snapshot is (Z, Y, X); sitk reverses to (X, Y, Z)
            img.SetSpacing((dx, dy, dz))  # sitk spacing order: (x, y, z)
            sitk.WriteImage(img, tmp_path)
            shutil.move(tmp_path, out_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.exception('Failed to auto-save CCTA mask')
        else:
            logger.info(f'Auto-saved CCTA mask to {out_path}')

    def _on_brush_enabled_changed(self, enabled: bool) -> None:
        if enabled:
            geo = self._brush_panel.current_geometry()
            if geo is not None:
                for d in (self._axial, self._coronal, self._sagittal):
                    d.enable_brush(geo)
        else:
            for d in (self._axial, self._coronal, self._sagittal):
                d.disable_brush()

    def _initialize_empty_mask(self, n_labels: int = 4) -> None:
        """Create a blank mask for the loaded volume so the brush works without a mask file."""
        assert self.data.volume is not None
        Z, Y, X = self.data.volume.shape
        mask = np.zeros((Z, Y, X), dtype=np.uint8)
        self.data.mask_undo.clear()
        self.data.mask = mask
        self.data.labels = list(range(1, n_labels + 1))
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_mask(mask, self.data.labels)
        self._mask_tab.set_labels(self.data.labels)
        self._brush_panel.set_labels(self.data.labels)
        self._stl_panel.set_labels(self.data.labels, self._mask_tab.label_names())

    def reset_to_neutral(self) -> None:
        """Return to neutral state: deactivate brush and cancel any active line draw."""
        for d in (self._axial, self._coronal, self._sagittal):
            d.stop_line_draw()
        self._brush_panel.set_enabled(False)

    def _on_brush_geometry_changed(self, geometry) -> None:
        for d in (self._axial, self._coronal, self._sagittal):
            d.update_brush(geometry)

    def _on_mask_painted(self) -> None:
        self._mask_dirty = True
        sender_display = self.sender()
        for d in (self._axial, self._coronal, self._sagittal):
            if d is not sender_display:
                d._render()

    def _on_3d_mask_erased(self) -> None:
        self._mask_dirty = True
        for d in (self._axial, self._coronal, self._sagittal):
            d._render()

    def _on_mask_about_to_change(self) -> None:
        """Record the mask right before a brush stroke or 3-D erase mutates it, for Ctrl+Z."""
        if self.data.mask is not None:
            self.data.mask_undo.push(self.data.mask.copy())

    def undo_last_mask_edit(self) -> None:
        snapshot = self.data.mask_undo.pop()
        if snapshot is None:
            return
        self._apply_mask_edit(snapshot)
        self._mask_dirty = True
        self.status_bar.showMessage('Mask edit undone')

    def _apply_mask_edit(self, mask: np.ndarray) -> None:
        """Swap in mask voxel data for an undo, same as a brush stroke or lasso erase
        would: update the shared array without touching label visibility, custom colors,
        or names (unlike _apply_mask, which is for loading a genuinely new mask)."""
        self.data.mask = mask
        for display in (self._axial, self._coronal, self._sagittal):
            display.update_mask_data(mask)
        if self.data.voxel_spacing is not None:
            self._3d_viewer.update_mask_data(mask, self.data.voxel_spacing)

    def _on_mask_alpha_changed(self, alpha: float) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_mask_alpha(alpha)

    def _on_label_visibility_changed(self, label: int, visible: bool) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_label_visible(label, visible)
        self._3d_viewer.set_label_visible(label, visible)

    def _on_label_colors_changed(self, colors: list) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_label_colors(colors)
        self._3d_viewer.set_label_colors(colors)
        self._brush_panel.set_label_colors(colors)

    def _on_windowing_changed(self, level: int, width: int) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_windowing(level, width)

    def _on_cursor_moved(self, z: int, y: int, x: int) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_cursor(z, y, x)
        self._3d_viewer.set_cursor(z, y, x)
        if self.data.volume is not None:
            Z, Y, X = self.data.volume.shape
            self._update_labels(z, y, x, Z, Y, X)

    def reset_zoom(self) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.reset_zoom()

    def reset_windowing(self) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.reset_windowing()

    def _update_labels(self, z: int, y: int, x: int, Z: int, Y: int, X: int) -> None:
        self._axial_label.setText(f'Axial  Z: {z + 1} / {Z}')
        self._sagittal_label.setText(f'Sagittal  X: {x + 1} / {X}')
        self._coronal_label.setText(f'Coronal  Y: {y + 1} / {Y}')

    def _on_line_draw_requested(self, index: int) -> None:
        # index 0: axial (LVOT), index 1: coronal (LVOT), index 2: coronal (aorta top)
        if index == 0:
            self._axial.start_line_draw()
        else:
            self._pending_coronal_cut = index
            self._coronal.start_line_draw()

    def _on_line_drawn(self, index: int, p1: tuple, p2: tuple) -> None:
        if index == 0:
            self._cut_line_0 = (p1, p2)
        elif index == 1:
            self._cut_line_1 = (p1, p2)
        else:
            self._aorta_cut_line = (p1, p2)
        self._cut_state_dirty = True
        self._stl_panel.set_line_drawn(index)
        self._refresh_cut_line_overlays()

    def _refresh_cut_line_overlays(self) -> None:
        axial_lines = [self._cut_line_0] if self._cut_line_0 else []
        coronal_lines = [line for line in (self._cut_line_1, self._aorta_cut_line) if line is not None]
        self._axial.set_cut_lines(axial_lines)
        self._coronal.set_cut_lines(coronal_lines)

    def _cut_lines_ready(self) -> bool:
        return self._cut_line_0 is not None and self._cut_line_1 is not None and self._aorta_cut_line is not None

    def _compute_cut_planes(self) -> _CutPlanes | None:
        """Returns (lvot_anchor, lvot_normal, aorta_anchor, aorta_normal), all in voxel-
        index (z, y, x) space, or None (with an ErrorMessage already shown) if the cut
        lines aren't ready or don't define a valid plane. Independent of any label
        choice, so it's shared by _compute_combined_mask (which also needs per-label
        centroids to pick the cut's keep-side) and find_inlet_outlet_centroids (which
        only needs the plane geometry itself, not the mask)."""
        if not self._cut_lines_ready():
            ErrorMessage(self, 'All three cut lines must be drawn first.')
            return None
        assert self._cut_line_0 is not None and self._cut_line_1 is not None and self._aorta_cut_line is not None

        # LVOT cut plane: line 0 (axial, moves in y/x) × line 1 (coronal, moves in z/x).
        p00 = np.array(self._cut_line_0[0], dtype=float)
        p01 = np.array(self._cut_line_0[1], dtype=float)
        p10 = np.array(self._cut_line_1[0], dtype=float)
        p11 = np.array(self._cut_line_1[1], dtype=float)
        d0 = p01 - p00
        d1 = p11 - p10
        lvot_normal = np.cross(d0, d1)
        if np.linalg.norm(lvot_normal) < 1e-6:
            ErrorMessage(self, 'LVOT cut lines are parallel — cannot define a plane. Please redraw.')
            return None
        lvot_anchor = (p00 + p01) / 2

        # Aorta top cut: single coronal line, plane extends through all Y.
        q0 = np.array(self._aorta_cut_line[0], dtype=float)
        q1 = np.array(self._aorta_cut_line[1], dtype=float)
        d_aorta = q1 - q0
        aorta_normal = np.cross(d_aorta, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(aorta_normal) < 1e-6:
            ErrorMessage(self, 'Aorta top cut line is degenerate — cannot define a plane. Please redraw.')
            return None
        aorta_anchor = (q0 + q1) / 2

        return lvot_anchor, lvot_normal, aorta_anchor, aorta_normal

    def _compute_combined_mask(
        self, cor_label: int, aorta_label: int, lv_label: int
    ) -> tuple[np.ndarray, _CutPlanes] | None:
        """Build the combined coronaries|aorta|LVOT mask from the two LVOT cut lines
        and the aorta-top cut line (all three required), plus the plane geometry used
        to cut it (callers that need both — Build Cut Geometry — would otherwise have
        to call _compute_cut_planes() a second time to get what this already computed
        internally). Shows an ErrorMessage and returns None on any failure — shared by
        Extract && Export and Build Cut Geometry so both always cut identically."""
        if self.data.mask is None or self.data.voxel_spacing is None:
            ErrorMessage(self, 'No mask or volume loaded.')
            return None
        planes = self._compute_cut_planes()
        if planes is None:
            return None
        lvot_anchor, lvot_normal, aorta_anchor, aorta_normal = planes

        mask = self.data.mask
        coronaries = mask == cor_label
        aorta = mask == aorta_label
        lv = mask == lv_label

        aorta_voxels = np.argwhere(aorta)
        if len(aorta_voxels) == 0:
            ErrorMessage(self, 'Aorta mask is empty.')
            return None
        aorta_centroid = aorta_voxels.mean(axis=0)

        Z, Y, X = mask.shape
        iz, iy, ix = np.mgrid[0:Z, 0:Y, 0:X]
        coords = np.stack([iz, iy, ix], axis=-1).astype(float)
        lvot_dist = cut_geometry.signed_distance_to_plane(coords, lvot_anchor, lvot_normal)
        aorta_side = np.dot(lvot_normal, aorta_centroid - lvot_anchor)
        lvot = lv & ((lvot_dist > 0) == (aorta_side > 0))

        coronaries_voxels = np.argwhere(coronaries)
        ref_centroid = coronaries_voxels.mean(axis=0) if len(coronaries_voxels) > 0 else aorta_centroid
        ref_side = np.dot(aorta_normal, ref_centroid - aorta_anchor)
        aorta_dist = cut_geometry.signed_distance_to_plane(coords, aorta_anchor, aorta_normal)
        aorta = aorta & ((aorta_dist > 0) == (ref_side > 0))

        return (coronaries | aorta | lvot).astype(np.uint8), planes

    def _on_extract_requested(self, cor_label: int, aorta_label: int, lv_label: int, fmt: str) -> None:
        if self.data.mask is None or self.data.voxel_spacing is None:
            ErrorMessage(self, 'No mask or volume loaded.')
            return
        if not self._cut_lines_ready():
            ErrorMessage(self, 'All three cut lines must be drawn first.')
            return

        # Ask for destination first so the user isn't waiting on computation before seeing the dialog.
        if fmt == 'nifti':
            path, _ = QFileDialog.getSaveFileName(
                self,
                'Save NIfTI Mask',
                '',
                'NIfTI (*.nii.gz)',
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if not path:
                return
            if not path.endswith('.nii.gz'):
                path += '.nii.gz'
        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                'Save STL',
                '',
                'STL (*.stl)',
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if not path:
                return
            if not path.endswith('.stl'):
                path += '.stl'

        label = 'NIfTI' if fmt == 'nifti' else 'STL'
        progress = QProgressDialog('Computing cut geometry…', None, 0, 0, self)
        progress.setWindowTitle(f'Export {label}')
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()

        # STL: if the cut geometry has already been built (and possibly smoothed) in
        # 3-D, export exactly that mesh — what's shown in the viewer is what's on
        # disk, and smoothing is otherwise lost since it never touches the voxel mask.
        # NIfTI is a voxel format, so it always re-derives the combined mask fresh.
        if fmt == 'stl' and self.data.cut_mesh is not None:
            progress.setLabelText(f'Writing {label}…')
            QApplication.processEvents()
            self.data.cut_mesh.export(path)
        else:
            result = self._compute_combined_mask(cor_label, aorta_label, lv_label)
            if result is None:
                progress.close()
                return
            combined, _planes = result

            progress.setLabelText(f'Writing {label}…')
            QApplication.processEvents()

            if fmt == 'nifti':
                export_nifti(combined, self.data.voxel_spacing, path)
            else:
                export_stl(combined, self.data.voxel_spacing, path)

        progress.close()
        self.status_bar.showMessage(f'Exported: {os.path.basename(path)}')

    def _on_build_cut_geometry(self, cor_label: int, aorta_label: int, lv_label: int, switch_tab: bool = True) -> None:
        result = self._compute_combined_mask(cor_label, aorta_label, lv_label)
        if result is None:
            return
        combined, (lvot_anchor, lvot_normal, aorta_anchor, aorta_normal) = result
        assert self.data.voxel_spacing is not None  # _compute_combined_mask already required this

        self._stl_panel.set_building(True)
        QApplication.processEvents()
        try:
            mesh = cut_geometry.build_cut_mesh(combined, self.data.voxel_spacing)
            inlet, outlet = cut_geometry.find_inlet_outlet_centroids(
                mesh, self.data.voxel_spacing, combined.shape, lvot_anchor, lvot_normal, aorta_anchor, aorta_normal
            )
        except Exception as e:
            logger.exception('Build Cut Geometry failed')
            ErrorMessage(self, f'Build Cut Geometry failed: {e}')
            return
        finally:
            self._stl_panel.set_building(False)

        self.data.cut_mesh = mesh
        self.data.cut_mesh_inlet = inlet
        self.data.cut_mesh_outlet = outlet
        self._cut_labels = (cor_label, aorta_label, lv_label)
        self._cut_state_dirty = True
        self._cut_viewer.set_cut_mesh(mesh, inlet, outlet, combined, self.data.voxel_spacing)
        if switch_tab:
            self._tabs.setCurrentWidget(self._cut_viewer)  # jump straight to the result
        self.status_bar.showMessage('Cut geometry built.')

    def _on_outlet_points_changed(self, _category: str, _count: int) -> None:
        self._cut_state_dirty = True

    def _on_label_name_changed(self, _label: int, _name: str) -> None:
        self._cut_state_dirty = True

    def _on_outlet_point_mode_requested(self, category: str) -> None:
        if category and self.data.cut_mesh is None:
            ErrorMessage(self, 'Build the cut geometry first.')
            self._stl_panel.reset_outlet_mode()
            return
        self._cut_viewer.set_point_mode(category)

    def _apply_mesh_op(
        self,
        op: Callable[[trimesh.Trimesh], trimesh.Trimesh],
        op_name: str,
        status_message: Callable[[trimesh.Trimesh], str],
    ) -> None:
        """Shared skeleton for Smooth/Reduce Mesh: both replace self.data.cut_mesh
        with the result of a single trimesh -> trimesh operation, relocate the
        inlet/outlet centroids on the result, and push it to the viewer. `op_name`
        labels the error if `op` raises; `status_message` receives the *new* mesh so
        callers can report on it (e.g. its new face count)."""
        if self.data.cut_mesh is None:
            ErrorMessage(self, 'Build the cut geometry first.')
            return
        assert self.data.mask is not None and self.data.voxel_spacing is not None  # implied by cut_mesh existing
        planes = self._compute_cut_planes()
        if planes is None:
            return
        lvot_anchor, lvot_normal, aorta_anchor, aorta_normal = planes

        try:
            mesh = op(self.data.cut_mesh)
            inlet, outlet = cut_geometry.find_inlet_outlet_centroids(
                mesh,
                self.data.voxel_spacing,
                self.data.mask.shape,
                lvot_anchor,
                lvot_normal,
                aorta_anchor,
                aorta_normal,
            )
        except Exception as e:
            logger.exception(f'{op_name} failed')
            ErrorMessage(self, f'{op_name} failed: {e}')
            return

        self.data.cut_mesh = mesh
        self.data.cut_mesh_inlet = inlet
        self.data.cut_mesh_outlet = outlet
        self._cut_viewer.update_cut_mesh(mesh, inlet, outlet)
        self.status_bar.showMessage(status_message(mesh))

    def _on_smooth_requested(self, lamb: float) -> None:
        self._apply_mesh_op(
            lambda mesh: cut_geometry.smooth_mesh(mesh, lamb=lamb),
            'Smoothing',
            lambda _mesh: 'Cut geometry smoothed.',
        )

    def _on_reduce_mesh_requested(self, target_reduction: float) -> None:
        if self.data.cut_mesh is None:
            ErrorMessage(self, 'Build the cut geometry first.')
            return
        before = len(self.data.cut_mesh.faces)
        self._apply_mesh_op(
            lambda mesh: cut_geometry.reduce_mesh(mesh, target_reduction),
            'Mesh reduction',
            lambda mesh: f'Mesh reduced: {before} -> {len(mesh.faces)} faces.',
        )

    def _on_calculate_centerlines(self) -> None:
        if self._centerlines_worker is not None and self._centerlines_worker.isRunning():
            return
        prereqs = self._validate_centerline_prereqs()
        if prereqs is not None:
            self._start_centerline_worker(prereqs)

    def _validate_centerline_prereqs(self) -> _CenterlinePrereqs | None:
        """Every guard Calculate Centerlines needs before it can run: a built cut
        mesh, a case file to derive output paths from, at least one RCA and one LCA
        outlet point, and a working vmtk install. Shows the relevant ErrorMessage and
        returns None on the first one that fails."""
        if self.data.cut_mesh is None:
            ErrorMessage(self, 'Build the cut geometry first.')
            return None
        assert self.data.cut_mesh_inlet is not None and self.data.cut_mesh_outlet is not None
        if self._source_path is None:
            ErrorMessage(self, 'No case file loaded — cannot determine where to write centerlines.')
            return None

        rca_points = self._cut_viewer.rca_points_world()
        lca_points = self._cut_viewer.lca_points_world()
        if not rca_points:
            ErrorMessage(self, 'Add at least one RCA outlet point first.')
            return None
        if not lca_points:
            ErrorMessage(self, 'Add at least one LCA outlet point first.')
            return None

        venv_path = self.config.vmtk.venv_path
        build_path = self.config.vmtk.build_path
        distro = self.config.vmtk.wsl_distro
        ok, reason = vmtk_runner.check_vmtk_available(venv_path, build_path, distro)
        if not ok:
            ErrorMessage(self, f'vmtk not found. {reason}')
            return None

        return _CenterlinePrereqs(
            cut_mesh=self.data.cut_mesh,
            cut_mesh_inlet=self.data.cut_mesh_inlet,
            cut_mesh_outlet=self.data.cut_mesh_outlet,
            rca_points=rca_points,
            lca_points=lca_points,
            venv_path=venv_path,
            build_path=build_path,
            distro=distro,
            out_dir=os.path.dirname(self._source_path),
            stem=os.path.basename(self._source_path),
        )

    def _start_centerline_worker(self, prereqs: _CenterlinePrereqs) -> None:
        """Runs vmtk in a background QThread. This can take minutes — vmtkcenterlines'
        Voronoi-diagram step is slow and often silent — so running it on the main
        thread would freeze the whole app with no way to tell "slow" from "stuck".
        StdoutCapturingWorker forwards every print() line (vmtk_runner streams vmtk's
        own output plus its own start/finish/heartbeat markers) live to both the
        console and the progress dialog."""
        stl_path = os.path.join(prereqs.out_dir, f'{prereqs.stem}_root_smooth.stl')
        ao_csv = os.path.join(prereqs.out_dir, 'ao.csv')
        rca_csv = os.path.join(prereqs.out_dir, 'rca.csv')
        lca_csv = os.path.join(prereqs.out_dir, 'lca.csv')

        progress = QProgressDialog('Computing centerlines…', None, 0, 0, self)
        progress.setWindowTitle('Calculate Centerlines')
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.show()

        def _do_work() -> tuple[str, dict[str, str]]:
            prereqs.cut_mesh.export(stl_path)
            vmtk_runner.write_point_csv(ao_csv, [prereqs.cut_mesh_inlet, prereqs.cut_mesh_outlet])
            vmtk_runner.write_point_csv(rca_csv, prereqs.rca_points)
            vmtk_runner.write_point_csv(lca_csv, prereqs.lca_points)

            paths = vmtk_runner.run_centerlines(
                stl_path,
                prereqs.out_dir,
                ao_source=prereqs.cut_mesh_inlet,
                ao_target=prereqs.cut_mesh_outlet,
                rca_targets=prereqs.rca_points,
                lca_targets=prereqs.lca_points,
                venv_path=prereqs.venv_path,
                build_path=prereqs.build_path,
                distro=prereqs.distro,
                log_cb=print,
            )
            return prereqs.out_dir, paths

        # Capture the real stdout *before* the worker starts and redirects it —
        # StdoutCapturingWorker swaps sys.stdout for its own capture object for the
        # run's duration, so connecting straight to print() here would have each
        # printed line re-enter the capture and re-emit forever.
        real_stdout = sys.stdout

        def _print_to_console(line: str) -> None:
            real_stdout.write(line + '\n')
            real_stdout.flush()

        worker = StdoutCapturingWorker(_do_work, (), {}, parent=self)
        worker.line_printed.connect(progress.setLabelText)
        worker.line_printed.connect(_print_to_console)
        worker.finished_ok.connect(lambda result: self._on_calculate_centerlines_done(progress, result))
        worker.failed.connect(lambda message: self._on_calculate_centerlines_failed(progress, message))
        self._centerlines_worker = worker
        worker.start()

    def _on_calculate_centerlines_done(self, progress: QProgressDialog, result: tuple[str, dict[str, str]]) -> None:
        progress.close()
        self._centerlines_worker = None
        out_dir, paths = result
        self._cut_viewer.set_centerlines(paths)  # so the user can check the result before trusting it
        self.status_bar.showMessage(f'Centerlines saved to {out_dir}')

    def _on_calculate_centerlines_failed(self, progress: QProgressDialog, message: str) -> None:
        progress.close()
        self._centerlines_worker = None
        logger.error(f'Centerline calculation failed: {message}')
        ErrorMessage(self, f'Centerline calculation failed: {message}')

    @staticmethod
    def _panel(display: CctaDisplay, label: QLabel) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(display, 1)
        layout.addWidget(label)
        return w
