"""Right half shown for OCT pullbacks.

OCT pullbacks have no cardiac gating, so where the IVUS half (right_half_ivus.py)
marks diastolic/systolic frames and plots the gating signal, this one tags frames of
interest, rates their quality, and shows the idealized pullback schematic instead.
The shell that swaps between the two is right_half.py.
"""

import bisect
from functools import partial
from math import lcm

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from domain.all_types import OCT_QUALITY_LABELS
from pages.intravascular.popup_windows.frame_range_dialog import FrameRangeDialog
from pages.intravascular.right_half.common import LongitudinalSlot, build_lower_buttons
from pages.intravascular.utils.oct_plot import OCTPlot

# Per-frame flags shown next to the tagging controls: FrameData attribute -> checkbox text.
# All of them are mutually exclusive with the quality rating (see RightHalfOct._on_flag_toggled).
OCT_FRAME_FLAGS = (
    ('guiding_catheter', 'Guiding Catheter'),
    ('unanalyzable', 'Unanalyzable'),
    ('unlabeled', 'Unlabeled'),
)


def _separator() -> QFrame:
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.VLine)
    separator.setFrameShadow(QFrame.Shadow.Sunken)
    return separator


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

        self.frame_flag_buttons: dict[str, QCheckBox] = {}
        for attr, text in OCT_FRAME_FLAGS:
            box = QCheckBox(text)
            box.toggled.connect(partial(self._on_flag_toggled, attr))
            self.frame_flag_buttons[attr] = box

        # Exclusive group, but nothing is checked until the user rates a frame; the way back
        # to 'no rating' is checking a flag, not clicking the checked button again.
        self.oct_quality_buttons: dict[str, QPushButton] = {}
        self.oct_quality_button_group = QButtonGroup(self)
        self.oct_quality_button_group.setExclusive(True)
        for label in OCT_QUALITY_LABELS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(partial(self._on_quality_clicked, label))
            self.oct_quality_buttons[label] = btn
            self.oct_quality_button_group.addButton(btn)

        self.oct_plot = OCTPlot(mw)  # OCT counterpart of the gating plot

        vbox = QVBoxLayout(self)

        # Both rows live in one grid so they line up: the separators share column 2, and the
        # area right of them is split into lcm(3, 5) = 15 equal columns, so the three flags
        # and the five quality buttons spread across exactly the same width.
        controls = QGridLayout()
        columns = lcm(len(OCT_FRAME_FLAGS), len(OCT_QUALITY_LABELS))
        flag_span = columns // len(OCT_FRAME_FLAGS)
        quality_span = columns // len(OCT_QUALITY_LABELS)

        controls.addWidget(self.tagged_frame_button, 0, 0)
        controls.addWidget(self.use_tagged_button, 0, 1)
        # compare_btn = QPushButton('Compare Frames')
        # compare_btn.setToolTip('Open a small display to compare two frames')
        # compare_btn.clicked.connect(partial(open_small_display, mw))
        controls.addWidget(_separator(), 0, 2)
        for i, (attr, _) in enumerate(OCT_FRAME_FLAGS):
            controls.addWidget(self.frame_flag_buttons[attr], 0, 3 + i * flag_span, 1, flag_span)

        controls.addWidget(QLabel('Frame Quality'), 1, 0, 1, 2)
        controls.addWidget(_separator(), 1, 2)
        for i, label in enumerate(OCT_QUALITY_LABELS):
            controls.addWidget(self.oct_quality_buttons[label], 1, 3 + i * quality_span, 1, quality_span)

        for column in range(3, 3 + columns):  # the leading columns stay at their content width
            controls.setColumnStretch(column, 1)
        vbox.addLayout(controls)

        self.lview_slot = LongitudinalSlot()

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.oct_plot)
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
        self._on_frame_changed(mw.display_slider.value())
        self.oct_plot.refresh()

    def _on_frame_changed(self, frame):
        """Update Tagged Frame checkbox, flags, quality buttons and schematic marker on slider moves (OCT only)."""
        mw = self.main_window
        if not (mw.image_displayed and mw.runtime_data.metadata.get('modality') == 'OCT'):
            return
        self.tagged_frame_button.blockSignals(True)
        self.tagged_frame_button.setChecked(frame in mw.runtime_data.tagged_frames)
        self.tagged_frame_button.blockSignals(False)
        frame_data = (mw.runtime_data.frame_data_dct or {}).get(frame)
        if frame_data is not None:
            for attr, box in self.frame_flag_buttons.items():
                box.blockSignals(True)
                box.setChecked(getattr(frame_data, attr))
                box.blockSignals(False)
            self._show_quality(frame_data.quality)
        self.oct_plot.set_frame(frame)

    def deactivate(self):
        self.lview_slot.detach(self.main_window.longitudinal_view)
        self.oct_plot.reset()  # drops the old pullback and stops its background measuring

    # ------------------------------------------------------------------
    # Quality / flag interlock
    # ------------------------------------------------------------------

    def _show_quality(self, quality):
        """Reflect `quality` ('' = unrated) in the button row without firing any handler."""
        group = self.oct_quality_button_group
        group.setExclusive(False)  # an exclusive group refuses to leave every button unchecked
        for label, btn in self.oct_quality_buttons.items():
            btn.setChecked(label == quality)
        group.setExclusive(True)

    def _on_quality_clicked(self, label):
        """A rating and the flags are mutually exclusive: rating a frame clears every flag."""
        set_oct_quality(self.main_window, label)
        for box in self.frame_flag_buttons.values():
            box.setChecked(False)  # each writes through _on_flag_toggled

    def _on_flag_toggled(self, attr, checked):
        """...and the other way round: any flag drops the rating, the only way back to unrated."""
        set_oct_flag(self.main_window, attr, checked)
        if checked:
            self._show_quality('')
            set_oct_quality(self.main_window, '')
        elif not self._checked_quality() and not any(b.isChecked() for b in self.frame_flag_buttons.values()):
            # Clearing the last thing on a frame leaves no label at all, so it is unlabeled again.
            self.frame_flag_buttons['unlabeled'].setChecked(True)

    def _checked_quality(self) -> str:
        """The rating currently shown, or '' when the frame is unrated."""
        button = self.oct_quality_button_group.checkedButton()
        return button.text() if button is not None else ''


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


def set_oct_flag(main_window, attr, checked):
    """Write one of the OCT_FRAME_FLAGS booleans onto the current frame."""
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        setattr(main_window.runtime_data.frame_data_dct[frame], attr, bool(checked))
        main_window.save_contours_soon()
