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
from input_output.input.ccta_io import read_ct_volume, read_nifti_volume, read_mask_volume
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(views)
        splitter.addWidget(self._mask_tab)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([10000, 220])

        QVBoxLayout(self).addWidget(splitter)

        for display in (self._axial, self._coronal, self._sagittal):
            display.cursor_moved.connect(self._on_cursor_moved)
            display.windowing_changed.connect(self._on_windowing_changed)
            display.mask_painted.connect(self._on_mask_painted)

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

        self.data.mask = mask
        self.data.labels = sorted(int(v) for v in np.unique(mask) if v != 0)

        for display in (self._axial, self._coronal, self._sagittal):
            display.set_mask(mask, self.data.labels)
        self._mask_tab.set_labels(self.data.labels)
        self._brush_panel.set_labels(self.data.labels)
        if self.data.voxel_spacing is not None:
            self._3d_viewer.set_mask(mask, self.data.labels, self.data.voxel_spacing)

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

    @staticmethod
    def _panel(display: CctaDisplay, label: QLabel) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(display, 1)
        layout.addWidget(label)
        return w
