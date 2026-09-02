"""Tests for Delete All On Frame (pages.intravascular.utils.contours_gui.delete_all_on_frame)
and the whole-frame undo entry it relies on.

The point of the single snapshot is that the undo stack is short: one entry per contour
type would evict the rest of the history and take a press per type to walk back.
"""

from types import SimpleNamespace

import pytest

import pages.intravascular.utils.contours_gui as contours_gui
from domain.all_types import ContourType
from domain.io_types import (
    FRAME_ANNOTATION_FIELDS,
    Contour,
    FrameData,
    Measure,
    Measurements,
    clear_frame_annotations,
)
from domain.runtime_types import RuntimeData
from domain.undo import FrameAnnotationSnapshot, push_contour_snapshot
from gui.shortcuts import undo_last_contour_edit
from pages.intravascular.utils.contours_gui import delete_all_on_frame

CONTOURS = ('lumen', 'eem', 'calcium', 'branch', 'lipid', 'macrophage')


def _drawn_frame() -> FrameData:
    """A frame carrying something of every kind the user can draw."""
    frame_data = FrameData(phase='T', quality='Good', unlabeled=False)
    for key in CONTOURS:
        setattr(
            frame_data,
            key,
            Contour(contours=[([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])], closed=[True], start_coords=[[]], end_coords=[[]]),
        )
    frame_data.wire = Contour(contours=[([0.3], [0.9])], closed=[False], start_coords=[[]], end_coords=[[]])
    frame_data.lumen.measurements = Measurements(area=7.5, circumference=9.0, minor_axis=2.0)
    frame_data.measurement_1 = Measure(points=((1.0, 2.0), (3.0, 4.0)), length=2.8)
    frame_data.measurement_2 = Measure(points=((5.0, 6.0), (7.0, 8.0)), length=2.8)
    frame_data.reference = (11.0, 12.0)
    frame_data.centroid = (2.0, 5.0)
    frame_data.closest_points = ((1.0, 1.0), (2.0, 2.0))
    frame_data.farthest_points = ((0.0, 0.0), (9.0, 9.0))
    return frame_data


@pytest.fixture
def main_window():
    """Stub main_window with four drawn-on frames, sitting on frame 2."""
    runtime_data = RuntimeData()
    runtime_data.frame_data_dct = {i: _drawn_frame() for i in range(4)}
    window = SimpleNamespace(
        image_displayed=True,
        runtime_data=runtime_data,
        display=SimpleNamespace(
            frame=2,
            working_spline='stale',
            active_contour_index=3,
            updates=0,
            contour_key=lambda: ContourType.LUMEN.value,
        ),
        display_slider=SimpleNamespace(set_value=lambda value: None),
        longitudinal_view=SimpleNamespace(plot_areas=lambda: None),
        saves=0,
    )
    window.display.update_display = lambda: setattr(window.display, 'updates', window.display.updates + 1)
    window.save_contours_soon = lambda: setattr(window, 'saves', window.saves + 1)
    return window


class TestAnnotationFields:
    def test_every_drawable_contour_type_is_covered(self):
        assert {ct.value for ct in ContourType} <= set(FRAME_ANNOTATION_FIELDS)

    def test_the_lumen_derived_values_go_too(self):
        assert {'centroid', 'closest_points', 'farthest_points'} <= set(FRAME_ANNOTATION_FIELDS)

    def test_every_field_exists_on_frame_data(self):
        blank = FrameData()
        for name in FRAME_ANNOTATION_FIELDS:
            assert hasattr(blank, name), name

    def test_the_frame_s_own_labels_are_not_annotations(self):
        for name in ('phase', 'quality', 'guiding_catheter', 'unanalyzable', 'unlabeled'):
            assert name not in FRAME_ANNOTATION_FIELDS


class TestClearFrameAnnotations:
    def test_resets_every_field_to_the_untouched_default(self):
        frame_data = _drawn_frame()
        clear_frame_annotations(frame_data)
        blank = FrameData()
        for name in FRAME_ANNOTATION_FIELDS:
            got, want = getattr(frame_data, name), getattr(blank, name)
            assert type(got) is type(want), name
            if isinstance(want, Contour):
                assert got.contours == [] and got.closed == [] and got.measurements == Measurements(), name
            else:
                assert got == want, name

    def test_keeps_the_phase_and_the_oct_label(self):
        frame_data = _drawn_frame()
        clear_frame_annotations(frame_data)
        assert frame_data.phase == 'T'
        assert frame_data.quality == 'Good' and frame_data.unlabeled is False

    def test_each_frame_gets_its_own_fresh_contours(self):
        first, second = _drawn_frame(), _drawn_frame()
        clear_frame_annotations(first)
        clear_frame_annotations(second)
        first.lumen.contours.append(([1.0], [2.0]))
        assert second.lumen.contours == []


class TestDeleteAllOnFrame:
    def test_clears_the_frame_on_screen(self, main_window):
        delete_all_on_frame(main_window)
        frame_data = main_window.runtime_data.frame_data_dct[2]
        for name in FRAME_ANNOTATION_FIELDS:
            value = getattr(frame_data, name)
            assert value.contours == [] if isinstance(value, Contour) else value is None, name
        assert frame_data.lumen.measurements == Measurements()

    def test_leaves_every_other_frame_alone(self, main_window):
        delete_all_on_frame(main_window)
        for i in (0, 1, 3):
            assert main_window.runtime_data.frame_data_dct[i].lumen.contours
            assert main_window.runtime_data.frame_data_dct[i].measurement_1 is not None

    def test_drops_the_in_progress_drawing_state(self, main_window):
        delete_all_on_frame(main_window)
        assert main_window.display.working_spline is None
        assert main_window.display.active_contour_index == 0

    def test_queues_a_save_and_redraws(self, main_window):
        delete_all_on_frame(main_window)
        assert main_window.saves == 1
        assert main_window.display.updates == 1
        assert main_window.runtime_data.unsaved_changes is True

    def test_records_a_single_undo_entry(self, main_window):
        delete_all_on_frame(main_window)
        stack = main_window.runtime_data.contour_undo
        assert stack.can_undo
        snapshot = stack.pop()
        assert isinstance(snapshot, FrameAnnotationSnapshot) and snapshot.frame == 2
        assert not stack.can_undo, 'one entry, not one per contour type'

    def test_one_undo_restores_the_whole_frame(self, main_window):
        before = {name: getattr(main_window.runtime_data.frame_data_dct[2], name) for name in FRAME_ANNOTATION_FIELDS}
        delete_all_on_frame(main_window)

        undo_last_contour_edit(main_window)

        restored = main_window.runtime_data.frame_data_dct[2]
        for name in FRAME_ANNOTATION_FIELDS:
            got, want = getattr(restored, name), before[name]
            if isinstance(want, Contour):
                assert got.contours == want.contours and got.measurements == want.measurements, name
            else:
                assert got == want, name

    def test_the_snapshot_is_a_deep_copy(self, main_window):
        delete_all_on_frame(main_window)
        main_window.runtime_data.frame_data_dct[2].lumen.contours.append(([9.0], [9.0]))

        undo_last_contour_edit(main_window)

        assert main_window.runtime_data.frame_data_dct[2].lumen.contours == [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]

    def test_per_contour_undo_still_works(self, main_window):
        push_contour_snapshot(main_window.runtime_data, 1, 'lumen', 0)
        main_window.runtime_data.frame_data_dct[1].lumen.contours = []

        undo_last_contour_edit(main_window)

        assert main_window.runtime_data.frame_data_dct[1].lumen.contours == [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]

    def test_does_nothing_without_a_file_loaded(self, main_window, monkeypatch):
        shown = []
        monkeypatch.setattr(contours_gui, 'ErrorMessage', lambda window, text: shown.append(text))
        main_window.image_displayed = False

        delete_all_on_frame(main_window)

        assert main_window.saves == 0
        assert not main_window.runtime_data.contour_undo.can_undo
        assert shown, 'the user is told why nothing happened'

    def test_does_nothing_on_a_frame_that_has_no_data(self, main_window):
        main_window.display.frame = 99

        delete_all_on_frame(main_window)

        assert main_window.saves == 0
        assert not main_window.runtime_data.contour_undo.can_undo
