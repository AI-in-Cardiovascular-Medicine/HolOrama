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

# Per-frame flags shown next to the tagging controls: FrameData attribute -> button text.
# They and the quality ratings are one exclusive choice split over two rows — a frame carries
# exactly one label, so picking any button here clears whatever else the frame had.
OCT_FRAME_FLAGS = (
    ('guiding_catheter', 'Guiding Catheter'),
    ('unanalyzable', 'Unanalyzable'),
    ('unlabeled', 'Unlabeled'),
)

# A guiding-catheter frame shows the catheter rather than the vessel, so there is nothing
# on it worth carrying into the report: it never also holds a tag. Both directions are
# enforced — labelling a frame (or a whole range) as guiding catheter drops its tag, and
# tagging skips frames that carry the label.
_NO_TAG_TOOLTIP = 'The guiding catheter fills this frame, so there is nothing to tag'


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
        self.tagged_frame_button.stateChanged.connect(self._on_tagged_frame_toggled)
        self.use_tagged_button = QPushButton('Tagged Frames')
        self.use_tagged_button.setStyleSheet('background-color: yellow')
        self.use_tagged_button.clicked.connect(partial(use_tagged, mw))

        self.catheter_range_button = QPushButton('Catheter Range')  # an action, not a label
        self.catheter_range_button.setStyleSheet('background-color: orange')
        self.catheter_range_button.setToolTip('Flag this frame and every frame after it as guiding catheter')
        self.catheter_range_button.clicked.connect(self._on_catheter_range)

        self.clear_catheter_range_button = QPushButton('Clear Catheter Range')
        self.clear_catheter_range_button.setToolTip('Set every frame labelled as guiding catheter back to unlabeled')
        self.clear_catheter_range_button.clicked.connect(self._on_clear_catheter_range)
        self.clear_catheter_range_button.setEnabled(False)  # nothing to revert until a pullback is loaded

        # One group across both rows: the flags and the quality ratings are the same choice,
        # so Qt keeps exactly one of the eight checked and unchecks the rest for us.
        self.frame_label_group = QButtonGroup(self)
        self.frame_label_group.setExclusive(True)

        self.frame_flag_buttons: dict[str, QPushButton] = {}
        for attr, text in OCT_FRAME_FLAGS:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(partial(self._on_flag_clicked, attr))
            self.frame_flag_buttons[attr] = btn
            self.frame_label_group.addButton(btn)

        self.oct_quality_buttons: dict[str, QPushButton] = {}
        for label in OCT_QUALITY_LABELS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(partial(self._on_quality_clicked, label))
            self.oct_quality_buttons[label] = btn
            self.frame_label_group.addButton(btn)

        self.oct_plot = OCTPlot(mw)  # OCT counterpart of the gating plot

        vbox = QVBoxLayout(self)

        # Both rows live in one grid so they line up: the separators share column 2, and the
        # label area right of them is split into lcm(3, 5) = 15 equal columns, so the three
        # frame labels and the five quality ratings spread across exactly the same width.
        # Catheter Range and the button that reverts it share an action column past a
        # full-height divider at the end of the label area: they act on a whole range of
        # frames rather than labelling the one on screen.
        controls = QGridLayout()
        columns = lcm(len(OCT_FRAME_FLAGS), len(OCT_QUALITY_LABELS))
        flag_span = columns // len(OCT_FRAME_FLAGS)
        quality_span = columns // len(OCT_QUALITY_LABELS)
        divider_column = 3 + columns

        controls.addWidget(self.tagged_frame_button, 0, 0)
        controls.addWidget(self.use_tagged_button, 0, 1)
        controls.addWidget(_separator(), 0, 2)
        for i, (attr, _) in enumerate(OCT_FRAME_FLAGS):
            controls.addWidget(self.frame_flag_buttons[attr], 0, 3 + i * flag_span, 1, flag_span)

        controls.addWidget(QLabel('Frame Quality'), 1, 0, 1, 2)
        controls.addWidget(_separator(), 1, 2)
        for i, label in enumerate(OCT_QUALITY_LABELS):
            controls.addWidget(self.oct_quality_buttons[label], 1, 3 + i * quality_span, 1, quality_span)

        controls.addWidget(_separator(), 0, divider_column, 2, 1)
        controls.addWidget(self.catheter_range_button, 0, divider_column + 1)
        controls.addWidget(self.clear_catheter_range_button, 1, divider_column + 1)

        # Only the label area takes the slack; the leading and action columns keep their
        # content width, so both rows stay the same width and stay aligned with each other.
        for column in range(3, divider_column):
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
        tag_button.setToolTip(
            'Tag frames a fixed distance apart, counted out from the frame on screen, within a frame range'
        )
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
        self._show_tagged(frame)
        frame_data = (mw.runtime_data.frame_data_dct or {}).get(frame)
        if frame_data is not None:
            self._show_label(frame_data)
        self.clear_catheter_range_button.setEnabled(bool(self.oct_plot.catheter_range_flags().any()))
        self.oct_plot.set_frame(frame)
        self.oct_plot.update()  # the veil is read live, so a relabel shows without a rebuild

    def _show_tagged(self, frame):
        """Mirror the frame's tag in the checkbox, and whether it may carry one at all."""
        mw = self.main_window
        catheter = _is_catheter_frame(mw.runtime_data, frame)
        self.tagged_frame_button.blockSignals(True)
        self.tagged_frame_button.setChecked(frame in mw.runtime_data.tagged_frames)
        self.tagged_frame_button.blockSignals(False)
        self.tagged_frame_button.setEnabled(not catheter)
        self.tagged_frame_button.setToolTip(_NO_TAG_TOOLTIP if catheter else '')

    def _on_tagged_frame_toggled(self, state):
        """Tag or untag the frame on screen, then show what the data actually took — a
        guiding-catheter frame refuses the tag."""
        toggle_tagged_frame(self.main_window, state)
        self._show_tagged(self.main_window.display_slider.value())

    def deactivate(self):
        self.lview_slot.detach(self.main_window.longitudinal_view)
        self.oct_plot.reset()  # drops the old pullback and stops its background measuring

    # ------------------------------------------------------------------
    # Frame label — one exclusive choice across both rows
    # ------------------------------------------------------------------

    def _label_button(self, frame_data):
        """The one button standing for this frame's label, or None when it has none."""
        if frame_data.quality:
            return self.oct_quality_buttons.get(frame_data.quality)
        for attr, _ in OCT_FRAME_FLAGS:
            if getattr(frame_data, attr):
                return self.frame_flag_buttons[attr]
        return None

    def _show_label(self, frame_data):
        """Check the button this frame's label calls for, and nothing else."""
        target = self._label_button(frame_data)
        group = self.frame_label_group
        group.setExclusive(False)  # an exclusive group refuses to leave every button unchecked
        for button in group.buttons():
            button.setChecked(button is target)
        group.setExclusive(True)

    def _on_quality_clicked(self, label):
        set_oct_label(self.main_window, quality=label)

    def _on_flag_clicked(self, attr):
        set_oct_label(self.main_window, flag=attr)

    def _on_catheter_range(self):
        """Flag this frame and everything after it, then show the result on the current frame."""
        set_catheter_range(self.main_window)
        self.main_window.display.update_display()  # the range may have dropped tags
        self._on_frame_changed(self.main_window.display_slider.value())

    def _on_clear_catheter_range(self):
        """Undo the whole range: every guiding catheter frame goes back to unlabeled."""
        clear_catheter_range(self.main_window)
        self._on_frame_changed(self.main_window.display_slider.value())


# ---------------------------------------------------------------------------
# OCT callbacks
# ---------------------------------------------------------------------------


def _is_catheter_frame(runtime_data, frame: int) -> bool:
    """Whether `frame` is labelled as guiding catheter, which rules out tagging it."""
    frame_data = (runtime_data.frame_data_dct or {}).get(frame)
    return bool(frame_data is not None and frame_data.guiding_catheter)


def _untag_frame(runtime_data, frame: int) -> None:
    """Take the tag off `frame`, keeping tagged_frames and the frame's phase in step."""
    try:
        runtime_data.tagged_frames.remove(frame)
    except ValueError:
        pass
    frame_data = (runtime_data.frame_data_dct or {}).get(frame)
    if frame_data is not None and frame_data.phase == 'T':
        frame_data.phase = '-'


def _frames_at_distance(anchor: int, step_frames: float, lower_limit: int, upper_limit: int) -> list:
    """The frames `step_frames` apart that fall in [lower_limit, upper_limit), counted out
    from `anchor` in both directions.

    The frame on screen is the reference rather than the start of the range: sitting on
    frame 375 of 380 and tagging every 4th frame tags 375, 379 and 371, 367, ... down —
    not the range's first frame and every 4th one after it, which lands on a different set
    of frames entirely and usually misses the one being looked at. An anchor outside the
    range still sets which frames inside it are hit, so narrowing the range afterwards
    keeps the same frames rather than shifting all of them.

    Each position is `anchor + round(n * step_frames)`, so a fractional step — which is
    what a step in mm comes out as — spaces the frames evenly instead of drifting the way
    repeated rounding would.
    """
    frames = set()
    step = 0
    while (frame := anchor + round(step * step_frames)) < upper_limit:  # the anchor, then upwards
        if frame >= lower_limit:
            frames.add(frame)
        step += 1
    step = -1
    while (frame := anchor + round(step * step_frames)) >= lower_limit:  # downwards
        if frame < upper_limit:
            frames.add(frame)
        step -= 1
    return sorted(frames)


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
    # Cleared in place, not reassigned: activate() aliases gated_frames to this same list.
    main_window.runtime_data.tagged_frames.clear()

    frames = [
        frame
        for frame in _frames_at_distance(main_window.display_slider.value(), step_frames, lower_limit, upper_limit)
        # The guiding-catheter stretch is skipped rather than shifting the tags along it,
        # so the spacing of the rest is the one that was asked for.
        if not _is_catheter_frame(main_window.runtime_data, frame)
    ]
    main_window.runtime_data.tagged_frames.extend(frames)  # already sorted and unique
    for idx in frames:
        main_window.runtime_data.frame_data_dct[idx].phase = 'T'

    main_window.display.update_display()


def toggle_tagged_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            if _is_catheter_frame(main_window.runtime_data, frame):
                return  # the catheter fills it; the checkbox is disabled, so this is a guard
            if frame not in main_window.runtime_data.tagged_frames:
                bisect.insort_left(main_window.runtime_data.tagged_frames, frame)
            main_window.runtime_data.frame_data_dct[frame].phase = 'T'
        else:
            _untag_frame(main_window.runtime_data, frame)
        main_window.display.update_display()


def use_tagged(main_window):
    if main_window.image_displayed:
        main_window.runtime_data.gated_frames = main_window.runtime_data.tagged_frames


def apply_oct_label(frame_data, quality: str = '', flag: str = '') -> None:
    """Make `quality` or `flag` the frame's one label, clearing whatever else it carried."""
    frame_data.quality = quality
    for attr, _ in OCT_FRAME_FLAGS:
        setattr(frame_data, attr, attr == flag)


def set_oct_label(main_window, quality: str = '', flag: str = '') -> None:
    """Label the current frame with one quality rating or one flag, never both."""
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        apply_oct_label(main_window.runtime_data.frame_data_dct[frame], quality, flag)
        if flag == 'guiding_catheter':
            _untag_frame(main_window.runtime_data, frame)
        main_window.save_contours_soon()


def set_catheter_range(main_window):
    """Flag the current frame and every frame after it as guiding catheter.

    The catheter occupies the whole tail of a pullback once it is reached, so those frames
    are labelled in one go rather than one at a time. Each one is relabelled exactly as a
    click would: guiding catheter replaces whatever label the frame had, and drops its tag
    with it — the range is the stretch nothing can be read from, tags included. Clearing
    the range later does not bring those tags back.
    """
    if not main_window.image_displayed:
        return
    frame_data_dct = main_window.runtime_data.frame_data_dct
    for frame in range(main_window.display_slider.value(), main_window.display_slider.maximum() + 1):
        frame_data = frame_data_dct.get(frame)
        if frame_data is not None:
            apply_oct_label(frame_data, flag='guiding_catheter')
            _untag_frame(main_window.runtime_data, frame)
    main_window.save_contours_soon()


def clear_catheter_range(main_window):
    """Undo set_catheter_range: every guiding catheter frame goes back to unlabeled.

    Not restricted to the frames the last click covered — the label is the range, so
    whatever carries it anywhere in the pullback is what gets cleared.
    """
    if not main_window.image_displayed:
        return
    for frame_data in main_window.runtime_data.frame_data_dct.values():
        if frame_data.guiding_catheter:
            apply_oct_label(frame_data, flag='unlabeled')
    main_window.save_contours_soon()
