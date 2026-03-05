from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QLineEdit, QDialogButtonBox, QFormLayout

class FrameRangeDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.lower_limit = QLineEdit(self)
        self.lower_limit.setText('1')
        self.upper_limit = QLineEdit(self)
        self.upper_limit.setText(str(main_window.images.shape[0]))

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, 
            self
        )

        layout = QFormLayout(self)
        layout.addRow('Lower limit', self.lower_limit)
        layout.addRow('Upper limit', self.upper_limit)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        lower_limit = int(self.lower_limit.text()) - 1
        lower_limit = max(0, lower_limit)
        upper_limit = int(self.upper_limit.text())
        upper_limit = min(self.main_window.images.shape[0], upper_limit)

        if lower_limit >= upper_limit:
            lower_limit, upper_limit = upper_limit, lower_limit
        return lower_limit, upper_limit


class StartFramesDialog(QDialog):
    def __init__(self, main_window, label1='First diastolic frame', label2='First systolic frame'):
        super().__init__(main_window)
        self.main_window = main_window

        self.diastolic_start = QLineEdit(self)
        self.systolic_start = QLineEdit(self)

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, 
            self
        )
        
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
