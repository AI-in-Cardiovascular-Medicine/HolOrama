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

    def test_clear_sits_under_the_schematic_in_the_splitter_s_top_pane(self, oct_half):
        pane = oct_half.splitter.widget(0)
        assert oct_half.oct_plot.parent() is pane
        assert oct_half.clear_catheter_range_button.parent() is pane
