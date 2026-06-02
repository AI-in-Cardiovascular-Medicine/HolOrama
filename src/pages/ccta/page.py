from typing import TYPE_CHECKING, cast

import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QDialogButtonBox,
    QPushButton,
    QProgressDialog,
    QApplication,
    QGridLayout,
    QSplitter,
)
from PyQt6.QtCore import Qt

from pages.ccta.display import CctaDisplay
from pages.ccta.display_3d import _3DViewerCCTA
from pages.ccta.tool_tab.tab_gui import MaskControlTab
from input_output.input.dicom_dir import read_ct_volume, read_nifti_volume, read_mask_volume
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from gui.active_page import ActivePage
from domain.runtime_types import CctaRuntimeData

if TYPE_CHECKING:
    from gui.app import Master


class _CTOpenDialog(QFileDialog):
    """File dialog that lets the user pick either a NIfTI file or a DICOM folder.

    - Navigate to a .nii / .nii.gz file and click "Open" → NIfTI mode.
    - Browse into any folder and click "Open DICOM Folder" → DICOM mode
      (uses the directory currently shown in the dialog).
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, 'Open CT Data', '..')
        self.setOption(QFileDialog.Option.DontUseNativeDialog)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilters(['NIfTI files (*.nii *.nii.gz)', 'All Files (*)'])

        self._dicom_path: str | None = None

        dicom_btn = QPushButton('Open DICOM Folder')
        dicom_btn.clicked.connect(self._select_dicom)
        btn_box = self.findChild(QDialogButtonBox)
        if btn_box is not None:
            btn_box.addButton(dicom_btn, QDialogButtonBox.ButtonRole.ActionRole)

    def _select_dicom(self) -> None:
        self._dicom_path = self.directory().absolutePath()
        self.accept()

    def result_path_and_mode(self) -> tuple[str, str] | None:
        """Return (path, 'dicom'|'nifti'), or None if nothing was selected."""
        if self._dicom_path is not None:
            return self._dicom_path, 'dicom'
        files = self.selectedFiles()
        if files:
            return files[0], 'nifti'
        return None


class CctaPage(QWidget):
    def __init__(self, status_bar) -> None:
        super().__init__()
        self.status_bar = status_bar
        self.data = CctaRuntimeData()

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
        self._3d_viewer = _3DViewerCCTA()
        grid.addWidget(self._3d_viewer, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Mask control panel
        self._mask_tab = MaskControlTab()
        self._mask_tab.alpha_changed.connect(self._on_mask_alpha_changed)
        self._mask_tab.label_visibility_changed.connect(self._on_label_visibility_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(views)
        splitter.addWidget(self._mask_tab)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([10000, 210])

        QVBoxLayout(self).addWidget(splitter)

        for display in (self._axial, self._coronal, self._sagittal):
            display.cursor_moved.connect(self._on_cursor_moved)
            display.windowing_changed.connect(self._on_windowing_changed)

    def open_folder(self) -> None:
        cast('Master', self.window())._switch_page(ActivePage.CCTA.value)

        dialog = _CTOpenDialog(self)
        if not dialog.exec():
            return
        result = dialog.result_path_and_mode()
        if result is None:
            return
        path, mode = result

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

                volume, metadata = read_ct_volume(path, progress_cb=_cb)
            else:
                progress.setLabelText('Reading NIfTI...')
                QApplication.processEvents()
                volume, metadata = read_nifti_volume(path)
        except ValueError as e:
            progress.close()
            ErrorMessage(self, str(e))
            self.status_bar.showMessage('Ready')
            return
        finally:
            progress.close()

        dz = metadata['slice_thickness']
        dy, dx = metadata['pixel_spacing']
        self.data.volume = volume
        self.data.voxel_spacing = (dz, dy, dx)
        self.data.mask = None
        self.data.labels = []

        for display in (self._axial, self._coronal, self._sagittal):
            display.set_volume(volume, self.data.voxel_spacing)
            display.clear_mask()
        self._mask_tab.clear_labels()

        Z, Y, X = volume.shape
        self._update_labels(Z // 2, Y // 2, X // 2, Z, Y, X)
        fmt = 'NIfTI' if mode == 'nifti' else 'CCTA'
        self.status_bar.showMessage(f'{fmt}: {Z} slices  |  pixel spacing {dy:.3f} mm  |  slice thickness {dz:.3f} mm')

    def open_mask(self) -> None:
        if self.data.volume is None:
            ErrorMessage(self, 'Load a CT volume before opening a mask.')
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open CCTA Mask',
            '..',
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

        self.status_bar.showMessage(f'Mask loaded: {len(self.data.labels)} label(s) — {self.data.labels}')

    def _on_mask_alpha_changed(self, alpha: float) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_mask_alpha(alpha)

    def _on_label_visibility_changed(self, label: int, visible: bool) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_label_visible(label, visible)

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

    @staticmethod
    def _cpr_placeholder() -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel('3D / CPR')
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        return w
