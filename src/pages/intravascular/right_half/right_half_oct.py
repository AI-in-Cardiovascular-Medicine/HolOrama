"""Right half shown for OCT pullbacks.

OCT pullbacks have no cardiac gating, so where the IVUS half (right_half_ivus.py)
marks diastolic/systolic frames and plots the gating signal, this one tags frames of
interest, rates their quality, and shows the idealized pullback schematic instead.
The shell that swaps between the two is right_half.py.
"""

import bisect
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from domain.all_types import OCT_QUALITY_LABELS
from pages.intravascular.popup_windows.frame_range_dialog import FrameRangeDialog
from pages.intravascular.right_half.common import LongitudinalSlot, build_lower_buttons
from pages.intravascular.utils.oct_plot import OCTPlot


class RightHalfOct(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        mw = main_window

        self.tagged_frame_button = QCheckBox('Tagged Frame')
        self.tagged_frame_button.setChecked(False)
        self.tagged_frame_button.stateChanged.connect(partial(toggle_tagged_frame, mw))
        self.use_tagged_button = QPushButton('Tagged Frames')
        self.use_tagged_button.setStyleSheet('background-color: yellow')
        self.use_tagged_button.clicked.connect(partial(use_tagged, mw))

        self.oct_quality_buttons: dict[str, QPushButton] = {}
        self.oct_quality_button_group = QButtonGroup(self)
        self.oct_quality_button_group.setExclusive(True)
        for label in OCT_QUALITY_LABELS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(partial(set_oct_quality, mw, label))
            self.oct_quality_buttons[label] = btn
            self.oct_quality_button_group.addButton(btn)
        self.oct_quality_buttons[OCT_QUALITY_LABELS[-1]].setChecked(True)

        self.oct_plot = OCTPlot(mw)  # OCT counterpart of the gating plot

        vbox = QVBoxLayout(self)

        checkboxes = QHBoxLayout()
        checkboxes.addWidget(self.tagged_frame_button)
        checkboxes.addWidget(self.use_tagged_button)
        # compare_btn = QPushButton('Compare Frames')
        # compare_btn.setToolTip('Open a small display to compare two frames')
        # compare_btn.clicked.connect(partial(open_small_display, mw))
        # checkboxes.addWidget(compare_btn)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        checkboxes.addWidget(separator)
        for label in OCT_QUALITY_LABELS:
            checkboxes.addWidget(self.oct_quality_buttons[label])
        vbox.addLayout(checkboxes)

        self.lview_slot = LongitudinalSlot()

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.oct_plot)  # the IVUS gating plot's slot: OCT has no gating to show
        self.splitter.addWidget(self.lview_slot)
        self.splitter.setStretchFactor(0, mw.config.intravascular.gating_display_stretch)
        self.splitter.setStretchFactor(1, mw.config.intravascular.lview_display_stretch)
        vbox.addWidget(self.splitter)

        tag_button = QPushButton('Tag Frames by Distance')
        tag_button.setToolTip('Tag frames at regular distance intervals within a frame range')
        tag_button.clicked.connect(partial(tag_frames_by_distance, mw))
        vbox.addLayout(build_lower_buttons(mw, tag_button))

        # Connected once and left connected: the handler is a no-op unless an OCT
        # pullback is loaded, which is cheaper than reconnecting on every switch.
        mw.display_slider.valueChanged.connect(self._on_frame_changed)

    # ------------------------------------------------------------------
    # Shell API
    # ------------------------------------------------------------------

    def activate(self):
        mw = self.main_window
        self.lview_slot.attach(mw.longitudinal_view)
        mw.runtime_data.gated_frames = mw.runtime_data.tagged_frames
        self.oct_plot.set_frame(mw.display_slider.value())
        self.oct_plot.refresh()

    def _on_frame_changed(self, frame):
        """Update Tagged Frame checkbox, quality buttons and schematic marker on slider moves (OCT only)."""
        mw = self.main_window
        if not (mw.image_displayed and mw.runtime_data.metadata.get('modality') == 'OCT'):
            return
        self.tagged_frame_button.blockSignals(True)
        self.tagged_frame_button.setChecked(frame in mw.runtime_data.tagged_frames)
        self.tagged_frame_button.blockSignals(False)
        quality = mw.runtime_data.frame_data_dct[frame].quality
        self.oct_quality_buttons[quality].setChecked(True)
        self.oct_plot.set_frame(frame)

    def deactivate(self):
        self.lview_slot.detach(self.main_window.longitudinal_view)
        self.oct_plot.reset()  # drops the old pullback and stops its background measuring


# ---------------------------------------------------------------------------
# OCT callbacks
# ---------------------------------------------------------------------------


def tag_frames_by_distance(main_window):
    if not main_window.image_displayed:
        return
    dialog = FrameRangeDialog(main_window, step=True)
    dialog.setWindowTitle('Tag Frames by Distance')
    if not dialog.exec():
        return

    lower_limit, upper_limit = dialog.getInputs()

    if dialog.isStepByMm():
        step_mm = dialog.getStepMm()
        if step_mm <= 0.0:
            return
        speed = main_window.runtime_data.metadata['pullback_speed']  # mm/s
        frame_rate = main_window.runtime_data.metadata['frame_rate']  # frames/s
        step_frames = step_mm / (speed / frame_rate)
    else:
        step_frames = dialog.getStepFrames()
        if step_frames <= 0:
            return

    for idx in main_window.runtime_data.tagged_frames:
        main_window.runtime_data.frame_data_dct[idx].phase = '-'
    main_window.runtime_data.tagged_frames.clear()

    i = 0
    while True:
        idx = lower_limit + round(i * step_frames)
        if idx >= upper_limit:
            break
        if idx not in main_window.runtime_data.tagged_frames:
            bisect.insort_left(main_window.runtime_data.tagged_frames, idx)
        main_window.runtime_data.frame_data_dct[idx].phase = 'T'
        i += 1

    main_window.display.update_display()


def toggle_tagged_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            if frame not in main_window.runtime_data.tagged_frames:
                bisect.insort_left(main_window.runtime_data.tagged_frames, frame)
            main_window.runtime_data.frame_data_dct[frame].phase = 'T'
        else:
            try:
                main_window.runtime_data.tagged_frames.remove(frame)
            except ValueError:
                pass
            if main_window.runtime_data.frame_data_dct[frame].phase == 'T':
                main_window.runtime_data.frame_data_dct[frame].phase = '-'
        main_window.display.update_display()


def use_tagged(main_window):
    if main_window.image_displayed:
        main_window.runtime_data.gated_frames = main_window.runtime_data.tagged_frames


def set_oct_quality(main_window, label):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        main_window.runtime_data.frame_data_dct[frame].quality = label
        main_window.save_contours_soon()
