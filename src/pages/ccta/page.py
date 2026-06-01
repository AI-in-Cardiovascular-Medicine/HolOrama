import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QProgressDialog,
    QApplication,
    QGridLayout,
)
from PyQt6.QtCore import Qt

from pages.ccta.display import CctaDisplay
from input_output.input.dicom_dir import read_ct_volume
from pages.intravascular.popup_windows.message_boxes import ErrorMessage


class CctaPage(QWidget):
    def __init__(self, status_bar) -> None:
        super().__init__()
        self.status_bar = status_bar
        self.volume: np.ndarray | None = None

        self._axial = CctaDisplay('axial')
        self._coronal = CctaDisplay('coronal')
        self._sagittal = CctaDisplay('sagittal')

        self._axial_label = QLabel('Axial')
        self._coronal_label = QLabel('Coronal')
        self._sagittal_label = QLabel('Sagittal')
        for lbl in (self._axial_label, self._coronal_label, self._sagittal_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.addWidget(self._panel(self._axial, self._axial_label), 0, 0)
        grid.addWidget(self._panel(self._sagittal, self._sagittal_label), 0, 1)
        grid.addWidget(self._panel(self._coronal, self._coronal_label), 1, 0)
        grid.addWidget(self._cpr_placeholder(), 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for display in (self._axial, self._coronal, self._sagittal):
            display.cursor_moved.connect(self._on_cursor_moved)

    # ----------------------------------------------------------------- public

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, 'Open CCTA Folder', '..', options=QFileDialog.Option.DontUseNativeDialog
        )
        if not folder:
            return

        progress = QProgressDialog('Scanning folder...', '', 0, 0, self)
        progress.setWindowTitle('Loading CCTA')
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setModal(True)
        progress.show()
        QApplication.processEvents()

        def _cb(current: int, total: int) -> None:
            if progress.maximum() == 0:
                progress.setMaximum(total)
            progress.setValue(current)
            progress.setLabelText(f'Reading slice {current} / {total}...')
            QApplication.processEvents()

        try:
            volume, metadata = read_ct_volume(folder, progress_cb=_cb)
        except ValueError as e:
            progress.close()
            ErrorMessage(self, str(e))
            self.status_bar.showMessage('Ready')
            return
        finally:
            progress.close()

        self.volume = volume
        dz = metadata['slice_thickness']
        dy, dx = metadata['pixel_spacing']
        voxel_spacing = (dz, dy, dx)

        for display in (self._axial, self._coronal, self._sagittal):
            display.set_volume(volume, voxel_spacing)

        Z, Y, X = volume.shape
        self._update_labels(Z // 2, Y // 2, X // 2, Z, Y, X)
        self.status_bar.showMessage(f'CCTA: {Z} slices  |  pixel spacing {dy:.3f} mm  |  slice thickness {dz:.3f} mm')

    # ---------------------------------------------------------- slot handlers

    def _on_cursor_moved(self, z: int, y: int, x: int) -> None:
        for display in (self._axial, self._coronal, self._sagittal):
            display.set_cursor(z, y, x)
        if self.volume is not None:
            Z, Y, X = self.volume.shape
            self._update_labels(z, y, x, Z, Y, X)

    def _update_labels(self, z: int, y: int, x: int, Z: int, Y: int, X: int) -> None:
        self._axial_label.setText(f'Axial  Z: {z + 1} / {Z}')
        self._sagittal_label.setText(f'Sagittal  X: {x + 1} / {X}')
        self._coronal_label.setText(f'Coronal  Y: {y + 1} / {Y}')

    # ----------------------------------------------------------------- layout helpers

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
