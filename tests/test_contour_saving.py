"""Tests for persisting frame data (input_output.output.contours.write_contours).

Two things are pinned down here: the file a save lands in is the one the loader looks
for, and a save actually happens for every kind of edit — the write path itself, plus
the flag every contour-changing operation sets and the flush that runs before the page
or the app goes away.
"""

import json
import os
from types import SimpleNamespace

import pytest

from domain.io_types import FrameData
from domain.runtime_types import RuntimeData
from domain.undo import push_contour_snapshot
from input_output.input.contours import read_contours
from input_output.output.contours import write_contours
from version import CONTOURS_VERSION_TAG


def _stem_for(file_name: str) -> str:
    """The stem read_image stores in main_window.file_name for a chosen file."""
    root, ext = os.path.splitext(file_name)
    if ext == '.gz':
        root = os.path.splitext(root)[0]
    return root


@pytest.fixture
def window(tmp_path):
    """Minimal stand-in for IntravascularPage: what write_contours/read_contours touch."""

    def _make(file_name='pullback.dcm', frames=None):
        runtime = RuntimeData()
        runtime.frame_data_dct = frames if frames is not None else {0: FrameData(), 1: FrameData()}
        runtime.metadata = {'num_frames': len(runtime.frame_data_dct)}
        return SimpleNamespace(
            image_displayed=True,
            file_name=_stem_for(str(tmp_path / file_name)),
            runtime_data=runtime,
            display=SimpleNamespace(image_size=800),
            hide_contours_box=SimpleNamespace(setChecked=lambda value: None),
            contours_drawn=False,
        )

    return _make


class TestSaveFileName:
    @pytest.mark.parametrize(
        'file_name',
        [
            'pullback.dcm',
            '1.2.840.113619.2.55.3.604688.dcm',  # dotted DICOM UID, as OCT exports are named
            'OCT_pullback.2024-05-03.dcm',
            'case.nii.gz',
            'IM_0001',  # no extension at all
        ],
    )
    def test_saved_file_is_the_one_the_loader_reads_back(self, window, file_name):
        main_window = window(file_name)
        main_window.runtime_data.frame_data_dct[0].lumen.contours = [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]

        write_contours(main_window)

        expected = f'{main_window.file_name}_contours_{CONTOURS_VERSION_TAG}.json'
        assert os.path.exists(expected), f'nothing written to the stem the loader globs: {expected}'
        assert read_contours(main_window, main_window.file_name) is True
        assert main_window.runtime_data.frame_data_dct[0].lumen.contours[0][0] == [1.0, 2.0, 3.0]

    def test_two_pullbacks_sharing_a_prefix_keep_separate_files(self, window):
        """'A.one.dcm' and 'A.two.dcm' must not both save as 'A_contours_*.json'."""
        first = window('A.one.dcm')
        first.runtime_data.frame_data_dct[0].lumen.contours = [([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])]
        write_contours(first)

        second = window('A.two.dcm')
        second.runtime_data.frame_data_dct[0].lumen.contours = [([9.0, 8.0, 7.0], [9.0, 8.0, 7.0])]
        write_contours(second)

        assert read_contours(first, first.file_name) is True
        assert first.runtime_data.frame_data_dct[0].lumen.contours[0][0] == [1.0, 2.0, 3.0]


class TestWriteBehaviour:
    def test_autosave_skips_an_unchanged_pullback(self, window, tmp_path):
        main_window = window()
        write_contours(main_window)
        out_path = f'{main_window.file_name}_contours_{CONTOURS_VERSION_TAG}.json'
        first_mtime = os.path.getmtime(out_path)

        os.remove(out_path)
        write_contours(main_window, force=False)  # nothing changed since the last write

        assert not os.path.exists(out_path)
        assert first_mtime  # the earlier write did happen

    def test_autosave_writes_once_the_content_moves(self, window):
        main_window = window()
        write_contours(main_window)
        main_window.runtime_data.frame_data_dct[1].lumen.contours = [([7.0], [8.0])]

        write_contours(main_window, force=False, blocking=True)

        out_path = f'{main_window.file_name}_contours_{CONTOURS_VERSION_TAG}.json'
        with open(out_path) as f:
            saved = json.load(f)
        assert saved['1']['lumen']['contours'] == [[[7.0], [8.0]]]

    def test_blocking_write_lands_before_returning(self, window):
        """What the close/reload flush relies on: no daemon thread to outlive."""
        main_window = window()
        main_window.runtime_data.frame_data_dct[0].phase = 'T'
        write_contours(main_window, force=False, blocking=True)
        assert os.path.exists(f'{main_window.file_name}_contours_{CONTOURS_VERSION_TAG}.json')

    def test_nothing_is_written_without_an_image(self, window):
        main_window = window()
        main_window.image_displayed = False
        write_contours(main_window, force=False)
        assert not os.path.exists(f'{main_window.file_name}_contours_{CONTOURS_VERSION_TAG}.json')

    def test_quality_and_phase_reach_the_file(self, window):
        """OCT-only edits: a quality label and a tagged frame are frame data like any other."""
        main_window = window()
        main_window.runtime_data.frame_data_dct[0].quality = 'Bad'
        main_window.runtime_data.frame_data_dct[0].phase = 'T'
        write_contours(main_window)

        assert read_contours(main_window, main_window.file_name) is True
        assert main_window.runtime_data.frame_data_dct[0].quality == 'Bad'
        assert main_window.runtime_data.frame_data_dct[0].phase == 'T'


class TestUnsavedFlag:
    def test_taking_an_undo_snapshot_flags_unsaved_changes(self):
        runtime = RuntimeData()
        runtime.frame_data_dct = {0: FrameData()}
        assert not runtime.unsaved_changes

        push_contour_snapshot(runtime, 0, 'lumen', 0)

        assert runtime.unsaved_changes  # every snapshot means an edit is about to land

    def test_snapshot_flags_even_when_the_frame_is_missing(self):
        runtime = RuntimeData()
        runtime.frame_data_dct = {}
        push_contour_snapshot(runtime, 7, 'lumen', 0)
        assert runtime.unsaved_changes

    def test_mark_unsaved_is_idempotent(self):
        runtime = RuntimeData()
        runtime.mark_unsaved()
        runtime.mark_unsaved()
        assert runtime.unsaved_changes


class TestPageSaveTriggers:
    """The page-level plumbing: an edit flags itself, the debounce writes, teardown flushes."""

    @pytest.fixture
    def page(self, qt_app, tmp_path):
        import numpy as np
        import yaml

        from pages.intravascular.page import IntravascularPage

        def _to_namespace(obj):
            if isinstance(obj, dict):
                return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
            return obj

        config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.yaml')
        with open(config_path, encoding='utf-8') as f:
            config = _to_namespace(yaml.safe_load(f))

        widget = IntravascularPage(
            config,
            SimpleNamespace(addMenu=lambda *args: None),
            SimpleNamespace(showMessage=lambda *args: None),
        )
        widget._pending_save_timer.setInterval(10)  # no need to wait out the real debounce

        n_frames, dim = 4, 64
        widget.file_name = str(tmp_path / '1.2.840.113619.2.55.3.604688')  # dotted, like OCT exports
        widget.image_displayed = True
        widget.runtime_data.frame_data_dct = {i: FrameData() for i in range(n_frames)}
        widget.runtime_data.metadata = {
            'num_frames': n_frames,
            'resolution': 0.02,
            'dimension': dim,
            'modality': 'OCT',
        }
        widget.runtime_data.images = np.full((n_frames, dim, dim), 100, dtype=np.uint8)
        yield widget
        widget._pending_save_timer.stop()
        widget._autosave_timer.stop()

    @staticmethod
    def _out_path(page):
        return f'{page.file_name}_contours_{CONTOURS_VERSION_TAG}.json'

    @staticmethod
    def _settle(ms=200):
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def test_edit_reaches_disk_without_an_explicit_save(self, page):
        page.runtime_data.frame_data_dct[1].lumen.contours = [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]
        page.save_contours_soon()

        assert page.runtime_data.unsaved_changes
        assert not os.path.exists(self._out_path(page))  # debounced, not immediate

        self._settle()

        assert os.path.exists(self._out_path(page))
        assert not page.runtime_data.unsaved_changes

    def test_burst_of_edits_collapses_into_one_write(self, page):
        for i in range(5):
            page.runtime_data.frame_data_dct[1].phase = 'T' if i % 2 else '-'
            page.save_contours_soon()
        self._settle()

        assert os.path.exists(self._out_path(page))
        assert not page.runtime_data.unsaved_changes

    def test_flush_writes_an_edit_that_never_waited_for_the_debounce(self, page):
        page.runtime_data.frame_data_dct[2].lumen.contours = [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]
        page.save_contours_soon()

        page.flush_contours()  # what closeEvent / reload_intravascular do

        with open(self._out_path(page)) as f:
            saved = json.load(f)
        assert saved['2']['lumen']['contours'] == [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        assert not page.runtime_data.unsaved_changes

    def test_flush_with_nothing_pending_does_not_write(self, page):
        page.flush_contours()  # establishes the baseline hash
        os.remove(self._out_path(page))

        page.flush_contours()

        assert not os.path.exists(self._out_path(page))

    def test_oct_label_edit_triggers_a_save(self, page):
        """A non-contour, OCT-only edit — and one that refreshes nothing on screen."""
        from pages.intravascular.right_half.right_half_oct import set_oct_label

        page.display_slider.setMaximum(3)
        page.display_slider.blockSignals(True)
        page.display_slider.setValue(2)
        page.display_slider.blockSignals(False)

        set_oct_label(page, quality='Bad')
        self._settle()

        with open(self._out_path(page)) as f:
            saved = json.load(f)['2']
        assert saved['quality'] == 'Bad'
        assert saved['unlabeled'] is False  # a rating and the flags are one exclusive choice

    def test_oct_flag_edit_triggers_a_save(self, page):
        """The flags share the quality's exclusive group, so setting one clears the rating."""
        from pages.intravascular.right_half.right_half_oct import set_oct_label

        page.display_slider.setMaximum(3)
        page.display_slider.blockSignals(True)
        page.display_slider.setValue(2)
        page.display_slider.blockSignals(False)

        set_oct_label(page, quality='Bad')
        set_oct_label(page, flag='guiding_catheter')
        self._settle()

        with open(self._out_path(page)) as f:
            saved = json.load(f)['2']
        assert saved['guiding_catheter'] is True
        assert saved['quality'] == '' and saved['unlabeled'] is False and saved['unanalyzable'] is False

    def test_reset_state_cancels_a_pending_save(self, page):
        page.runtime_data.frame_data_dct[1].phase = 'T'
        page.save_contours_soon()
        page.reset_state()  # a new file is being loaded into this page
        self._settle()

        # nothing may be written after the reset: file_name and the data are gone
        assert not os.path.exists(self._out_path(page))
