import glob
import os
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING, cast
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
from loguru import logger
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QApplication,
    QGridLayout,
    QSplitter,
)
from PyQt6.QtCore import Qt, QTimer

from pages.ccta.left_half.display import CctaDisplay
from pages.ccta.left_half.display_3d import CctaViewer3D
from pages.ccta.right_half.mask_panel import MaskPanel
from pages.ccta.right_half.brush_panel import BrushPanel
from pages.ccta.right_half.stl_extraction_panel import StlExtractionPanel
from input_output.input.ccta_io import read_ct_volume, read_nifti_volume, read_mask_volume
from input_output.output.stl_export import export_nifti, export_stl
from version import version_file_str
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from gui.active_page import ActivePage
from domain.runtime_types import CctaRuntimeData

if TYPE_CHECKING:
    from gui.app import Master


class CctaPage(QWidget):
    def __init__(self, config: SimpleNamespace, status_bar) -> None:
        super().__init__()
        self.config: SimpleNamespace = config
        self.status_bar = status_bar
        self.data = CctaRuntimeData()
        self._last_image_dir: str | None = None
        self._source_path: str | None = None
        self._mask_dirty: bool = False

        self._axial = CctaDisplay('axial')
        self._coronal = CctaDisplay('coronal')
        self._sagittal = CctaDisplay('sagittal')

        self._axial_label = QLabel('Axial')
        self._coronal_label = QLabel('Coronal')
        self._sagittal_label = QLabel('Sagittal')
        for lbl in (self._axial_label, self._coronal_label, self._sagittal_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 4-view grid
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

        # Right panel: Mask labels (top, stretches) + Brush controls (bottom, fixed)
        self._mask_tab = MaskPanel()
        self._mask_tab.alpha_changed.connect(self._on_mask_alpha_changed)
        self._mask_tab.label_visibility_changed.connect(self._on_label_visibility_changed)

        self._brush_panel = BrushPanel()
        self._brush_panel.brush_enabled_changed.connect(self._on_brush_enabled_changed)
        self._brush_panel.geometry_changed.connect(self._on_brush_geometry_changed)
        self._mask_tab.label_name_changed.connect(self._brush_panel.update_label_name)
        self._mask_tab.set_brush_panel(self._brush_panel)

        self._stl_panel = StlExtractionPanel()
        self._stl_panel.line_draw_requested.connect(self._on_line_draw_requested)
        self._stl_panel.extract_requested.connect(self._on_extract_requested)
        self._mask_tab.label_name_changed.connect(self._stl_panel.update_label_name)
        self._cut_line_0: tuple | None = None  # (p1_zyx, p2_zyx) from axial view   (LVOT)
        self._cut_line_1: tuple | None = None  # (p1_zyx, p2_zyx) from coronal view (LVOT)
        self._aorta_cut_line: tuple | None = None  # (p1_zyx, p2_zyx) from coronal view (aorta top, optional)

        right_col = QWidget()
        right_vbox = QVBoxLayout(right_col)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)
        right_vbox.addWidget(self._mask_tab, 1)
        right_vbox.addWidget(self._stl_panel, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(views)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([10000, 220])

        QVBoxLayout(self).addWidget(splitter)

        for display in (self._axial, self._coronal, self._sagittal):
            display.cursor_moved.connect(self._on_cursor_moved)
            display.windowing_changed.connect(self._on_windowing_changed)
            display.mask_painted.connect(self._on_mask_painted)

        self._pending_coronal_cut: int = 1  # 1 = LVOT, 2 = aorta top
        self._axial.line_drawn.connect(lambda p1, p2: self._on_line_drawn(0, p1, p2))
        self._coronal.line_drawn.connect(lambda p1, p2: self._on_line_drawn(self._pending_coronal_cut, p1, p2))

        timer: QTimer = QTimer(self)
        timer.timeout.connect(self._auto_save)
        timer.start(self.config.save.autosave_interval)

    def shutdown(self) -> None:
        self._3d_viewer.shutdown()

    def open_folder(self) -> None:
        master = cast('Master', self.window())
        master._switch_page(ActivePage.CCTA.value)

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
            if not path:
                return
            mode = 'dicom'
        elif clicked == nifti_btn:
            path, _ = QFileDialog.getOpenFileName(
                self,
                'Open NIfTI File',
                '',
                'NIfTI files (*.nii *.nii.gz);;All Files (*)',
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if not path:
                return
            mode = 'nifti'
        else:
            return

        # Path confirmed — reinstantiate for a guaranteed clean state.
        # `self` must not be used after this point.
        master.reload_ccta()
        page = master.ccta_page

        progress = QProgressDialog('', '', 0, 0, page)
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

                volume, metadata = read_ct_volume(path, progress_cb=_cb)
            else:
                progress.setLabelText('Reading NIfTI...')
                QApplication.processEvents()
                volume, metadata = read_nifti_volume(path)
        except ValueError as e:
            progress.close()
            ErrorMessage(page, str(e))
            page.status_bar.showMessage('Ready')
            return
        finally:
            progress.close()

        dz = metadata['slice_thickness']
        dy, dx = metadata['pixel_spacing']
        page.data.volume = volume
        page.data.voxel_spacing = (dz, dy, dx)

        ccta_meta = metadata.get('ccta_metadata')
        if ccta_meta is not None:
            master.ccta_metadata = ccta_meta

        for display in (page._axial, page._coronal, page._sagittal):
            display.set_volume(volume, page.data.voxel_spacing)

        Z, Y, X = volume.shape
        page._update_labels(Z // 2, Y // 2, X // 2, Z, Y, X)
        page._initialize_empty_mask()
        fmt = 'NIfTI' if mode == 'nifti' else 'CCTA'
        page.status_bar.showMessage(f'{fmt}: {Z} slices  |  pixel spacing {dy:.3f} mm  |  slice thickness {dz:.3f} mm')

        if mode == 'nifti':
            page._last_image_dir = os.path.dirname(os.path.abspath(path))
            nifti_base = path
            for ext in ('.nii.gz', '.nii'):
                if nifti_base.endswith(ext):
                    nifti_base = nifti_base[: -len(ext)]
                    break
            page._source_path = nifti_base
            if not page._try_auto_load_mask():
                reply = QMessageBox.question(
                    page,
                    'Load Mask?',
                    'Would you like to load a segmentation mask for this volume?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    page.open_mask()
        else:
            page._source_path = os.path.join(path, os.path.basename(path))
            page._try_auto_load_mask()

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
        self._apply_mask(mask)
        self.status_bar.showMessage(f'Mask auto-loaded: {os.path.basename(mask_path)}')
        return True

    def _apply_mask(self, mask: np.ndarray) -> None:
        """Apply a loaded mask array to all displays and panels."""
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

        self._apply_mask(mask)
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
        if not self._mask_dirty or self.data.mask is None or self._source_path is None:
            return
        self._mask_dirty = False
        snapshot = self.data.mask.copy()
        spacing = self.data.voxel_spacing if self.data.voxel_spacing else (1.0, 1.0, 1.0)
        out_path = f'{self._source_path}_ccta_seg_{version_file_str}.nii.gz'
        threading.Thread(target=self._save_mask_snapshot, args=(snapshot, spacing, out_path), daemon=True).start()

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

    def _on_mask_alpha_changed(self, alpha: float) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_mask_alpha(alpha)

    def _on_label_visibility_changed(self, label: int, visible: bool) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_label_visible(label, visible)
        self._3d_viewer.set_label_visible(label, visible)

    def _on_windowing_changed(self, level: int, width: int) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_windowing(level, width)

    def _on_cursor_moved(self, z: int, y: int, x: int) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_cursor(z, y, x)
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
        self._stl_panel.set_line_drawn(index)
        self._refresh_cut_line_overlays()

    def _refresh_cut_line_overlays(self) -> None:
        axial_lines = [self._cut_line_0] if self._cut_line_0 else []
        coronal_lines = [line for line in (self._cut_line_1, self._aorta_cut_line) if line is not None]
        self._axial.set_cut_lines(axial_lines)
        self._coronal.set_cut_lines(coronal_lines)

    def _on_extract_requested(self, cor_label: int, aorta_label: int, lv_label: int, fmt: str) -> None:
        if self.data.mask is None or self.data.voxel_spacing is None:
            ErrorMessage(self, 'No mask or volume loaded.')
            return
        if self._cut_line_0 is None or self._cut_line_1 is None:
            ErrorMessage(self, 'Both cut lines must be drawn first.')
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
        progress = QProgressDialog(f'Preparing {label} export…', 'Cancel', 0, 4, self)
        progress.setWindowTitle(f'Export {label}')
        progress.setMinimumDuration(0)
        progress.setModal(True)
        progress.setValue(0)
        QApplication.processEvents()

        mask = self.data.mask
        coronaries = mask == cor_label
        aorta = mask == aorta_label
        lv = mask == lv_label

        aorta_voxels = np.argwhere(aorta)
        if len(aorta_voxels) == 0:
            progress.close()
            ErrorMessage(self, 'Aorta mask is empty.')
            return
        aorta_centroid = aorta_voxels.mean(axis=0)

        # LVOT cut plane: line 0 (axial, moves in y/x) × line 1 (coronal, moves in z/x).
        p00 = np.array(self._cut_line_0[0], dtype=float)
        p01 = np.array(self._cut_line_0[1], dtype=float)
        p10 = np.array(self._cut_line_1[0], dtype=float)
        p11 = np.array(self._cut_line_1[1], dtype=float)
        d0 = p01 - p00
        d1 = p11 - p10
        lvot_normal = np.cross(d0, d1)
        if np.linalg.norm(lvot_normal) < 1e-6:
            progress.close()
            ErrorMessage(self, 'LVOT cut lines are parallel — cannot define a plane. Please redraw.')
            return
        lvot_anchor = (p00 + p01) / 2

        progress.setLabelText('Computing LVOT cut…')
        progress.setValue(1)
        QApplication.processEvents()
        if progress.wasCanceled():
            return

        Z, Y, X = mask.shape
        iz, iy, ix = np.mgrid[0:Z, 0:Y, 0:X]
        coords = np.stack([iz, iy, ix], axis=-1).astype(float)
        lvot_dist = ((coords - lvot_anchor) * lvot_normal).sum(axis=-1)
        aorta_side = np.dot(lvot_normal, aorta_centroid - lvot_anchor)
        lvot = lv & ((lvot_dist > 0) == (aorta_side > 0))

        progress.setLabelText('Applying aorta cut…')
        progress.setValue(2)
        QApplication.processEvents()
        if progress.wasCanceled():
            return

        # Optional aorta top cut: single coronal line, plane extends through all Y.
        if self._aorta_cut_line is not None:
            q0 = np.array(self._aorta_cut_line[0], dtype=float)
            q1 = np.array(self._aorta_cut_line[1], dtype=float)
            d_aorta = q1 - q0
            aorta_cut_normal = np.cross(d_aorta, np.array([0.0, 1.0, 0.0]))
            if np.linalg.norm(aorta_cut_normal) > 1e-6:
                aorta_anchor = (q0 + q1) / 2
                coronaries_voxels = np.argwhere(coronaries)
                ref_centroid = coronaries_voxels.mean(axis=0) if len(coronaries_voxels) > 0 else aorta_centroid
                ref_side = np.dot(aorta_cut_normal, ref_centroid - aorta_anchor)
                aorta_dist = ((coords - aorta_anchor) * aorta_cut_normal).sum(axis=-1)
                aorta = aorta & ((aorta_dist > 0) == (ref_side > 0))

        combined = (coronaries | aorta | lvot).astype(np.uint8)

        progress.setLabelText(f'Writing {label}…')
        progress.setValue(3)
        QApplication.processEvents()
        if progress.wasCanceled():
            return

        if fmt == 'nifti':
            export_nifti(combined, self.data.voxel_spacing, path)
        else:
            export_stl(combined, self.data.voxel_spacing, path)

        progress.setValue(4)
        progress.close()
        self.status_bar.showMessage(f'Exported: {os.path.basename(path)}')

    @staticmethod
    def _panel(display: CctaDisplay, label: QLabel) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(display, 1)
        layout.addWidget(label)
        return w
