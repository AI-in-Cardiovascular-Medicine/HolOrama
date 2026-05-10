import itertools
import numpy as np
from loguru import logger

from gating.signal_processing import identify_extrema
from gui.popup_windows.frame_range_dialog import StartFramesDialog

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QRadioButton, QDialogButtonBox


class GatingMethodDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Select Gating Methods")
        layout = QVBoxLayout(self)

        # Image-based signal options
        self.image_group = QGroupBox("Image-based Signal")
        self.image_maxima = QRadioButton("Maxima")
        self.image_extrema = QRadioButton("Extrema")
        self.image_maxima.setChecked(True)
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_maxima)
        image_layout.addWidget(self.image_extrema)
        self.image_group.setLayout(image_layout)

        # Contour-based signal options
        self.contour_group = QGroupBox("Contour-based Signal")
        self.contour_maxima = QRadioButton("Maxima")
        self.contour_extrema = QRadioButton("Extrema")
        self.contour_extrema.setChecked(True)
        contour_layout = QVBoxLayout()
        contour_layout.addWidget(self.contour_maxima)
        contour_layout.addWidget(self.contour_extrema)
        self.contour_group.setLayout(contour_layout)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.image_group)
        layout.addWidget(self.contour_group)
        layout.addWidget(self.buttons)

    def get_methods(self):
        """Returns selected methods as tuple (image_method, contour_method)."""
        image_method = "maxima" if self.image_maxima.isChecked() else "extrema"
        contour_method = "maxima" if self.contour_maxima.isChecked() else "extrema"
        return image_method, contour_method


class AutomaticGating:
    def __init__(self, main_window, report_data) -> None:
        self.main_window = main_window
        self.report_data = report_data
        self.x = self.report_data['frame'].values  # want 1-based indexing for GUI

    def automatic_gating(self, image_based_signal, contour_based_signal):
        """
        Automatically gates the frames based on the image and contour signals.
        User still needs to choose if signals need maxima or extrema. Default is set
        to maxima for image based-gating and extrema for contour-based since in most
        cases correct.
        The gating is based on the following assumptions:
        - Diastole frames can depict more distal parts of the coronary artery, and AAOCA undergoes
            more compression during systole, hence sum of elliptic ratio is higher for systolic frames.

        Parameters:
        - image_based_signal (numpy.ndarray): The signal containing image-correlation/blurriness (filtered).
        - contour_based_signal (numpy.ndarray): The signal containing contour measurements (filtered).
        """
        dialog = GatingMethodDialog(self.main_window)
        if dialog.exec():
            image_method, contour_method = dialog.get_methods()
            temp_indices = identify_extrema(self.main_window, image_based_signal)
            image_indices = temp_indices[1] if image_method == "maxima" else temp_indices[0]
            if contour_method == "maxima":
                temp_indices = identify_extrema(self.main_window, contour_based_signal)
                contour_indices = temp_indices[1]
            else:
                contour_indices = temp_indices[0]

            # Create a list with indices most likely presenting systole/diastole
            # Take common intersection
            final_indices = np.intersect1d(image_indices, contour_indices)
            # print(f"Image based indices: {image_indices}")
            # print(f"Contour based indices: {contour_indices}")
            # print(f"Combined indices: {final_indices}")
            # _write_csv_signals(image_based_signal,
            #                   contour_based_signal,
            #                   image_indices,
            #                   contour_indices,
            #                   final_indices)
            # start by initializing every second
            first_half = final_indices[::2].tolist()
            second_half = final_indices[1::2].tolist()

            # systolic contours always have higher elliptic ratio intramural because of compression
            sum_first_half = sum(
                [
                    self.report_data.loc[self.report_data['frame'] == frame, 'elliptic_ratio'].values[0]
                    for frame in first_half
                ]
            )
            sum_second_half = sum(
                [
                    self.report_data.loc[self.report_data['frame'] == frame, 'elliptic_ratio'].values[0]
                    for frame in second_half
                ]
            )

            # reset all phases
            for fd in self.main_window.data.values():
                fd.phase = '-'
            self.main_window.gated_frames_dia = []
            self.main_window.gated_frames_sys = []
            self.main_window.diastolic_frame_box.setChecked(False)
            self.main_window.systolic_frame_box.setChecked(False)

            if sum_first_half > sum_second_half:
                self.main_window.gated_frames_dia = second_half
                self.main_window.gated_frames_sys = first_half
                self.main_window.gated_frames_dia.sort()
                self.main_window.gated_frames_sys.sort()
            else:
                self.main_window.gated_frames_dia = first_half
                self.main_window.gated_frames_sys = second_half
                self.main_window.gated_frames_dia.sort()
                self.main_window.gated_frames_sys.sort()

            for frame in self.main_window.gated_frames_dia:
                self.main_window.data[frame].phase = 'D'
            for frame in self.main_window.gated_frames_sys:
                self.main_window.data[frame].phase = 'S'


def _write_csv_signals(image_signal, contour_signal, image_indices, contour_indices, combined_indices):
    import pandas as pd

    df = pd.DataFrame(
        {
            'frame': np.arange(len(image_signal)),
            'image_signal': image_signal,
            'contour_signal': contour_signal,
            'image_indices': np.zeros(len(image_signal)),
            'contour_indices': np.zeros(len(image_signal)),
            'combined_indices': np.zeros(len(image_signal)),
        }
    )
    # replace the 0 with ones if frame is in image_indices, contour_indices, combined_indices
    df.loc[df['frame'].isin(image_indices), 'image_indices'] = 1
    df.loc[df['frame'].isin(contour_indices), 'contour_indices'] = 1
    df.loc[df['frame'].isin(combined_indices), 'combined_indices'] = 1
    df.to_csv('/home/yungselm/Documents/AAOCASeg/stats_data/signals.csv')
