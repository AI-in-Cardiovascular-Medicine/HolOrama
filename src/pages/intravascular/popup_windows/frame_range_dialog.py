from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QFormLayout,
    QRadioButton,
    QHBoxLayout,
    QWidget,
)


class FrameRangeDialog(QDialog):
    def __init__(self, main_window, step: bool = False):
        super().__init__(main_window)
        self.main_window = main_window
        self.lower_limit = QLineEdit(self)
        self.lower_limit.setText('1')
        self.upper_limit = QLineEdit(self)
        self.upper_limit.setText(str(main_window.runtime_data.images.shape[0]))

        if step:
            self.radio_mm = QRadioButton('mm', self)
            self.radio_frames = QRadioButton('Frames', self)
            self.radio_mm.setChecked(True)

            self.step_mm = QLineEdit(self)
            self.step_mm.setText('1')
            self.step_frames = QLineEdit(self)
            self.step_frames.setText('10')
            self.step_frames.setEnabled(False)

            self.radio_mm.toggled.connect(self._on_mode_toggled)

        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow('Lower limit', self.lower_limit)
        layout.addRow('Upper limit', self.upper_limit)
        if step:
            radio_row = QWidget(self)
            radio_layout = QHBoxLayout(radio_row)
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_layout.addWidget(self.radio_mm)
            radio_layout.addWidget(self.radio_frames)
            layout.addRow('Step mode', radio_row)
            layout.addRow('Step (mm)', self.step_mm)
            layout.addRow('Step (frames)', self.step_frames)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def _on_mode_toggled(self, mm_checked: bool):
        self.step_mm.setEnabled(mm_checked)
        self.step_frames.setEnabled(not mm_checked)

    def getInputs(self):
        lower_limit = int(self.lower_limit.text()) - 1
        lower_limit = max(0, lower_limit)
        upper_limit = int(self.upper_limit.text())
        upper_limit = min(self.main_window.runtime_data.images.shape[0], upper_limit)

        if lower_limit >= upper_limit:
            lower_limit, upper_limit = upper_limit, lower_limit
        return lower_limit, upper_limit

    def getStepMm(self) -> float:
        return float(self.step_mm.text())

    def getStepFrames(self) -> int:
        return int(self.step_frames.text())

    def isStepByMm(self) -> bool:
        return self.radio_mm.isChecked()


class StartFramesDialog(QDialog):
    def __init__(self, main_window, label1='First diastolic frame', label2='First systolic frame'):
        super().__init__(main_window)
        self.main_window = main_window

        self.diastolic_start = QLineEdit(self)
        self.systolic_start = QLineEdit(self)

        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow(label1, self.diastolic_start)
        layout.addRow(label2, self.systolic_start)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        # Set non-modal mode
        self.setWindowModality(Qt.WindowModality.NonModal)

    def getInputs(self):
        # Retrieve and return input values
        diastolic = int(self.diastolic_start.text()) - 1
        systolic = int(self.systolic_start.text()) - 1
        return diastolic, systolic
