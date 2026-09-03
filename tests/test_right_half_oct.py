"""Tests for the OCT right half (pages.intravascular.right_half.right_half_oct).

Covers the frame label — one exclusive choice spread over the flag row and the quality
row — plus the two range actions that write it in bulk, and the grey veil the schematic
draws over the stretch labelled as guiding catheter.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QSlider, QVBoxLayout, QWidget

import pages.intravascular.right_half.right_half_oct as right_half_oct
from domain.all_types import OCT_QUALITY_LABELS
from domain.io_types import FrameData
from pages.intravascular.right_half.right_half_oct import (
    OCT_FRAME_FLAGS,
    RightHalfOct,
    apply_oct_label,
    clear_catheter_range,
    set_catheter_range,
    set_oct_label,
)

N_FRAMES = 10


def _label_of(frame_data) -> tuple:
    return (frame_data.quality, frame_data.guiding_catheter, frame_data.unanalyzable, frame_data.unlabeled)


class _StubPlot(QWidget):
    """Stands in for OCTPlot: the widget only calls these four."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def set_frame(self, frame):
        pass

    def refresh(self):
        pass

    def reset(self):
        pass

    def catheter_range_flags(self):
        dct = self.main_window.runtime_data.frame_data_dct
        return np.array([bool(dct[i].guiding_catheter) for i in sorted(dct)], dtype=bool)


@pytest.fixture
def oct_half(qt_app, monkeypatch):
    """RightHalfOct on a stub main_window, with the schematic and the lower buttons stubbed."""
    monkeypatch.setattr(right_half_oct, 'OCTPlot', _StubPlot)
    monkeypatch.setattr(right_half_oct, 'build_lower_buttons', lambda main_window, button: QVBoxLayout())

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMaximum(N_FRAMES - 1)
    main_window = SimpleNamespace(
        image_displayed=True,
        runtime_data=SimpleNamespace(
            frame_data_dct={i: FrameData() for i in range(N_FRAMES)},
            metadata={'modality': 'OCT', 'num_frames': N_FRAMES},
            tagged_frames=[],
            gated_frames=[],
        ),
        display=SimpleNamespace(update_display=lambda: None),
        longitudinal_view=QWidget(),
        display_slider=slider,
        config=SimpleNamespace(intravascular=SimpleNamespace(gating_display_stretch=3, lview_display_stretch=1)),
        save_contours_soon=lambda: None,
    )
    widget = RightHalfOct(main_window)
    widget.activate()
    return widget


class TestApplyOctLabel:
    def test_a_quality_clears_every_flag(self):
        frame_data = FrameData(guiding_catheter=True, unanalyzable=True, unlabeled=True)
        apply_oct_label(frame_data, quality='Bad')
        assert _label_of(frame_data) == ('Bad', False, False, False)

    def test_a_flag_clears_the_rating_and_the_other_flags(self):
        frame_data = FrameData(quality='Good', unanalyzable=True)
        apply_oct_label(frame_data, flag='guiding_catheter')
        assert _label_of(frame_data) == ('', True, False, False)

    def test_no_argument_leaves_the_frame_unlabeled(self):
        frame_data = FrameData(quality='Good', unlabeled=False)
        apply_oct_label(frame_data, flag='unlabeled')
        assert _label_of(frame_data) == ('', False, False, True)

    def test_writes_through_the_slider_frame_only(self):
        dct = {i: FrameData() for i in range(3)}
        main_window = SimpleNamespace(
            image_displayed=True,
            runtime_data=SimpleNamespace(frame_data_dct=dct),
            display_slider=SimpleNamespace(value=lambda: 1),
            save_contours_soon=lambda: None,
        )
        set_oct_label(main_window, quality='Ok')
        assert dct[1].quality == 'Ok'
        assert dct[0].unlabeled is True and dct[2].unlabeled is True


class TestLabelButtons:
    def test_both_rows_are_one_exclusive_group_of_checkable_buttons(self, oct_half):
        buttons = list(oct_half.frame_flag_buttons.values()) + list(oct_half.oct_quality_buttons.values())
        assert all(isinstance(b, QPushButton) and b.isCheckable() for b in buttons)
        assert oct_half.frame_label_group.exclusive()
        assert set(oct_half.frame_label_group.buttons()) == set(buttons)
        assert len(buttons) == len(OCT_FRAME_FLAGS) + len(OCT_QUALITY_LABELS)

    def test_the_range_actions_are_not_part_of_the_choice(self, oct_half):
        for button in (oct_half.catheter_range_button, oct_half.clear_catheter_range_button):
            assert not button.isCheckable()
            assert button not in oct_half.frame_label_group.buttons()

    def test_a_fresh_frame_starts_unlabeled_with_no_rating(self, oct_half):
        assert oct_half.frame_label_group.checkedButton().text() == 'Unlabeled'
        assert _label_of(oct_half.main_window.runtime_data.frame_data_dct[0]) == ('', False, False, True)

    @pytest.mark.parametrize(
        'attr_or_label, is_flag, expected',
        [
            ('guiding_catheter', True, ('', True, False, False)),
            ('unanalyzable', True, ('', False, True, False)),
            ('unlabeled', True, ('', False, False, True)),
            ('Bad', False, ('Bad', False, False, False)),
            ('Very Good', False, ('Very Good', False, False, False)),
        ],
    )
    def test_clicking_any_button_leaves_exactly_one_label(self, oct_half, attr_or_label, is_flag, expected):
        frame_data = oct_half.main_window.runtime_data.frame_data_dct[0]
        apply_oct_label(frame_data, flag='unanalyzable')  # something else to displace
        oct_half._on_frame_changed(0)

        buttons = oct_half.frame_flag_buttons if is_flag else oct_half.oct_quality_buttons
        buttons[attr_or_label].click()

        assert _label_of(frame_data) == expected
        assert sum(1 for b in oct_half.frame_label_group.buttons() if b.isChecked()) == 1

    def test_switching_frames_shows_that_frame_s_label(self, oct_half):
        dct = oct_half.main_window.runtime_data.frame_data_dct
        apply_oct_label(dct[4], quality='Good')
        apply_oct_label(dct[5], flag='guiding_catheter')

        oct_half.main_window.display_slider.setValue(4)
        assert oct_half.frame_label_group.checkedButton().text() == 'Good'
        oct_half.main_window.display_slider.setValue(5)
        assert oct_half.frame_label_group.checkedButton().text() == 'Guiding Catheter'

    def test_syncing_a_frame_does_not_write_to_it(self, oct_half):
        dct = oct_half.main_window.runtime_data.frame_data_dct
        apply_oct_label(dct[7], quality='Ok')
        oct_half.main_window.display_slider.setValue(7)
        assert _label_of(dct[7]) == ('Ok', False, False, False)


class TestCatheterRange:
    def test_labels_the_current_frame_and_everything_after_it(self, oct_half):
        dct = oct_half.main_window.runtime_data.frame_data_dct
        for frame_data in dct.values():
            apply_oct_label(frame_data, quality='Good')

        oct_half.main_window.display_slider.setValue(6)
        oct_half.catheter_range_button.click()

        for i in range(6):
            assert _label_of(dct[i]) == ('Good', False, False, False), i
        for i in range(6, N_FRAMES):
            assert _label_of(dct[i]) == ('', True, False, False), i

    def test_shows_on_the_current_frame_without_a_slider_move(self, oct_half):
        oct_half.main_window.display_slider.setValue(6)
        oct_half.catheter_range_button.click()
        assert oct_half.frame_label_group.checkedButton().text() == 'Guiding Catheter'

    def test_on_the_last_frame_it_labels_only_that_frame(self, oct_half):
        dct = oct_half.main_window.runtime_data.frame_data_dct
        oct_half.main_window.display_slider.setValue(N_FRAMES - 1)
        oct_half.catheter_range_button.click()
        assert dct[N_FRAMES - 1].guiding_catheter is True
        assert not any(dct[i].guiding_catheter for i in range(N_FRAMES - 1))

    def test_clear_takes_the_label_off_every_frame_that_has_it(self, oct_half):
        dct = oct_half.main_window.runtime_data.frame_data_dct
        apply_oct_label(dct[1], flag='guiding_catheter')  # an isolated one, outside the run
        oct_half.main_window.display_slider.setValue(6)
        oct_half.catheter_range_button.click()

        oct_half.clear_catheter_range_button.click()

        assert not any(frame_data.guiding_catheter for frame_data in dct.values())
        assert all(dct[i].unlabeled for i in [1] + list(range(6, N_FRAMES)))

    def test_clear_leaves_other_labels_alone(self, oct_half):
        dct = oct_half.main_window.runtime_data.frame_data_dct
        apply_oct_label(dct[0], quality='Bad')
        apply_oct_label(dct[1], flag='unanalyzable')
        oct_half.main_window.display_slider.setValue(6)
        oct_half.catheter_range_button.click()

        clear_catheter_range(oct_half.main_window)

        assert _label_of(dct[0]) == ('Bad', False, False, False)
        assert _label_of(dct[1]) == ('', False, True, False)

    def test_clear_starts_disabled_before_a_pullback_is_loaded(self, oct_half, monkeypatch):
        monkeypatch.setattr(right_half_oct, 'OCTPlot', _StubPlot)
        monkeypatch.setattr(right_half_oct, 'build_lower_buttons', lambda main_window, button: QVBoxLayout())
        fresh = RightHalfOct(oct_half.main_window)  # constructed, never activated
        assert not fresh.clear_catheter_range_button.isEnabled()

    def test_clear_is_disabled_while_no_frame_carries_the_label(self, oct_half):
        assert not oct_half.clear_catheter_range_button.isEnabled()
        oct_half.main_window.display_slider.setValue(6)
        oct_half.catheter_range_button.click()
        assert oct_half.clear_catheter_range_button.isEnabled()
        oct_half.clear_catheter_range_button.click()
        assert not oct_half.clear_catheter_range_button.isEnabled()

    def test_both_are_no_ops_with_nothing_loaded(self):
        dct = {0: FrameData()}
        main_window = SimpleNamespace(
            image_displayed=False,
            runtime_data=SimpleNamespace(frame_data_dct=dct),
            display_slider=SimpleNamespace(value=lambda: 0, maximum=lambda: 0),
            save_contours_soon=lambda: None,
        )
        set_catheter_range(main_window)
        clear_catheter_range(main_window)
        assert _label_of(dct[0]) == ('', False, False, True)


class TestLayout:
    def test_the_two_label_rows_line_up_and_span_the_same_width(self, oct_half):
        oct_half.resize(1000, 600)
        oct_half.show()
        try:
            oct_half.window().windowHandle()  # force a layout pass
            oct_half.layout().activate()
            flags = [oct_half.frame_flag_buttons[attr] for attr, _ in OCT_FRAME_FLAGS]
            qualities = [oct_half.oct_quality_buttons[label] for label in OCT_QUALITY_LABELS]
            assert flags[0].x() == qualities[0].x()
            assert flags[-1].x() + flags[-1].width() == qualities[-1].x() + qualities[-1].width()
        finally:
            oct_half.hide()

    def test_clear_sits_directly_below_the_button_it_reverts(self, oct_half):
        oct_half.resize(1000, 600)
        oct_half.show()
        try:
            oct_half.layout().activate()
            catheter, clear = oct_half.catheter_range_button, oct_half.clear_catheter_range_button
            assert catheter.x() == clear.x() and catheter.width() == clear.width()
            assert catheter.y() + catheter.height() <= clear.y()
            # ...and both sit right of the divider, past every label button
            rightmost_label = max(
                (b.x() + b.width() for b in oct_half.frame_label_group.buttons()),
            )
            assert catheter.x() > rightmost_label
        finally:
            oct_half.hide()

    def test_the_schematic_has_the_splitter_pane_to_itself(self, oct_half):
        assert oct_half.splitter.widget(0) is oct_half.oct_plot


class TestFramesAtDistance:
    """Tag Frames by Distance counts out from the frame on screen, not from the range."""

    def test_the_users_case(self):
        # 380 frames, sitting on frame 375 (index 374), every 4th frame.
        frames = right_half_oct._frames_at_distance(374, 4, 0, 380)
        assert frames[-4:] == [366, 370, 374, 378]  # i.e. frames 367, 371, 375, 379
        assert 374 in frames

    def test_it_reaches_both_ends_of_the_range(self):
        frames = right_half_oct._frames_at_distance(10, 5, 0, 26)
        assert frames == [0, 5, 10, 15, 20, 25]

    def test_the_anchor_is_always_tagged(self):
        for anchor in (0, 1, 7, 29):
            assert anchor in right_half_oct._frames_at_distance(anchor, 4, 0, 30)

    def test_nothing_falls_outside_the_range(self):
        frames = right_half_oct._frames_at_distance(20, 3, 10, 25)
        assert frames == [11, 14, 17, 20, 23]

    def test_a_step_of_one_tags_the_whole_range(self):
        assert right_half_oct._frames_at_distance(3, 1, 0, 6) == [0, 1, 2, 3, 4, 5]

    def test_an_anchor_outside_the_range_still_sets_the_spacing(self):
        # Frames stay where they would be with the anchor in range: 30, 34, 38, ...
        assert right_half_oct._frames_at_distance(50, 4, 30, 45) == [30, 34, 38, 42]

    def test_a_fractional_step_does_not_drift(self):
        # 2.5 frames apart, which is what a step in mm comes out as: the gaps alternate
        # 2 and 3 instead of rounding to 2 every time and losing half a frame per step.
        frames = right_half_oct._frames_at_distance(10, 2.5, 0, 21)
        gaps = [b - a for a, b in zip(frames, frames[1:])]

        assert set(gaps) == {2, 3}
        assert (frames[-1] - frames[0]) / (len(frames) - 1) == pytest.approx(2.5)

    def test_an_empty_range_tags_nothing(self):
        assert right_half_oct._frames_at_distance(5, 4, 5, 5) == []


class TestTagFramesByDistance:
    """The whole action, with the dialog answered for it."""

    def _run(self, oct_half, monkeypatch, anchor, step_frames, lower=1, upper=N_FRAMES):
        main_window = oct_half.main_window
        main_window.display_slider.setValue(anchor)

        class _Dialog:
            def __init__(self, *args, **kwargs):
                pass

            def setWindowTitle(self, title):
                pass

            def exec(self):
                return True

            def getInputs(self):
                return lower - 1, upper

            def isStepByMm(self):
                return False

            def getStepFrames(self):
                return step_frames

        monkeypatch.setattr(right_half_oct, 'FrameRangeDialog', _Dialog)
        right_half_oct.tag_frames_by_distance(main_window)
        return main_window

    def test_it_tags_outwards_from_the_frame_on_screen(self, oct_half, monkeypatch):
        main_window = self._run(oct_half, monkeypatch, anchor=N_FRAMES - 2, step_frames=3)

        # Ten frames, sitting on index 8, every 3rd: upwards runs out of pullback at once,
        # so the tags are the ones counted back down from it.
        assert main_window.runtime_data.tagged_frames == [2, 5, 8]

    def test_the_tagged_frames_stay_sorted_and_carry_the_phase(self, oct_half, monkeypatch):
        main_window = self._run(oct_half, monkeypatch, anchor=4, step_frames=2)
        tagged = main_window.runtime_data.tagged_frames

        assert tagged == sorted(tagged)
        assert all(main_window.runtime_data.frame_data_dct[frame].phase == 'T' for frame in tagged)

    def test_it_replaces_a_previous_tagging(self, oct_half, monkeypatch):
        first = self._run(oct_half, monkeypatch, anchor=0, step_frames=2).runtime_data.tagged_frames[:]
        main_window = self._run(oct_half, monkeypatch, anchor=1, step_frames=2)

        assert main_window.runtime_data.tagged_frames != first
        untagged = set(first) - set(main_window.runtime_data.tagged_frames)
        assert all(main_window.runtime_data.frame_data_dct[frame].phase == '-' for frame in untagged)

    def test_the_gated_frames_alias_still_points_at_the_same_list(self, oct_half, monkeypatch):
        main_window = self._run(oct_half, monkeypatch, anchor=3, step_frames=4)

        assert main_window.runtime_data.gated_frames is main_window.runtime_data.tagged_frames


class TestCatheterRangeAndTags:
    """A guiding-catheter frame shows the catheter, not the vessel, so it never also
    carries a tag — enforced whichever of the two is set second."""

    def _tag(self, main_window, *frames):
        for frame in frames:
            main_window.runtime_data.tagged_frames.append(frame)
            main_window.runtime_data.frame_data_dct[frame].phase = 'T'
        main_window.runtime_data.tagged_frames.sort()

    def test_the_range_drops_the_tags_inside_it(self, oct_half):
        main_window = oct_half.main_window
        self._tag(main_window, 1, 5, 7)
        main_window.display_slider.setValue(5)

        oct_half._on_catheter_range()

        assert main_window.runtime_data.tagged_frames == [1]  # only the frame before it survives

    def test_the_dropped_tags_lose_their_phase_too(self, oct_half):
        main_window = oct_half.main_window
        self._tag(main_window, 8)
        main_window.display_slider.setValue(8)

        oct_half._on_catheter_range()

        assert main_window.runtime_data.frame_data_dct[8].phase == '-'

    def test_labelling_one_frame_as_catheter_drops_its_tag(self, oct_half):
        main_window = oct_half.main_window
        self._tag(main_window, 4)
        main_window.display_slider.setValue(4)

        oct_half.frame_flag_buttons['guiding_catheter'].click()

        assert main_window.runtime_data.tagged_frames == []
        assert main_window.runtime_data.frame_data_dct[4].phase == '-'

    def test_a_quality_rating_leaves_the_tag_alone(self, oct_half):
        main_window = oct_half.main_window
        self._tag(main_window, 4)
        main_window.display_slider.setValue(4)

        oct_half.oct_quality_buttons['Good'].click()

        assert main_window.runtime_data.tagged_frames == [4]

    def test_the_tag_checkbox_is_closed_off_on_a_catheter_frame(self, oct_half):
        main_window = oct_half.main_window
        apply_oct_label(main_window.runtime_data.frame_data_dct[6], flag='guiding_catheter')

        oct_half._on_frame_changed(6)
        assert not oct_half.tagged_frame_button.isEnabled()
        assert oct_half.tagged_frame_button.toolTip()

        oct_half._on_frame_changed(5)
        assert oct_half.tagged_frame_button.isEnabled()
        assert not oct_half.tagged_frame_button.toolTip()

    def test_tagging_a_catheter_frame_does_nothing(self, oct_half):
        main_window = oct_half.main_window
        apply_oct_label(main_window.runtime_data.frame_data_dct[6], flag='guiding_catheter')
        main_window.display_slider.setValue(6)

        oct_half._on_tagged_frame_toggled(True)  # what the checkbox would send

        assert main_window.runtime_data.tagged_frames == []
        assert main_window.runtime_data.frame_data_dct[6].phase != 'T'
        assert not oct_half.tagged_frame_button.isChecked()  # and the box shows it

    def test_untagging_still_works_on_a_frame_that_became_catheter(self, oct_half):
        main_window = oct_half.main_window
        self._tag(main_window, 3)
        main_window.runtime_data.frame_data_dct[3].guiding_catheter = True
        main_window.display_slider.setValue(3)

        oct_half._on_tagged_frame_toggled(False)

        assert main_window.runtime_data.tagged_frames == []

    def test_tag_by_distance_skips_the_catheter_stretch(self, oct_half, monkeypatch):
        main_window = oct_half.main_window
        for frame in range(6, N_FRAMES):
            apply_oct_label(main_window.runtime_data.frame_data_dct[frame], flag='guiding_catheter')
        main_window.display_slider.setValue(0)

        class _Dialog:
            def __init__(self, *args, **kwargs):
                pass

            def setWindowTitle(self, title):
                pass

            def exec(self):
                return True

            def getInputs(self):
                return 0, N_FRAMES

            def isStepByMm(self):
                return False

            def getStepFrames(self):
                return 2

        monkeypatch.setattr(right_half_oct, 'FrameRangeDialog', _Dialog)
        right_half_oct.tag_frames_by_distance(main_window)

        assert main_window.runtime_data.tagged_frames == [0, 2, 4]
        assert all(main_window.runtime_data.frame_data_dct[frame].phase != 'T' for frame in range(6, N_FRAMES))

    def test_clearing_the_range_lets_the_frame_be_tagged_again(self, oct_half):
        main_window = oct_half.main_window
        main_window.display_slider.setValue(7)
        oct_half._on_catheter_range()

        oct_half._on_clear_catheter_range()

        assert oct_half.tagged_frame_button.isEnabled()
        oct_half._on_tagged_frame_toggled(True)
        assert main_window.runtime_data.tagged_frames == [7]
