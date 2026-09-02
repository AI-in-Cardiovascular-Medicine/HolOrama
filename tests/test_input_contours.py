import pytest

from domain.io_types import Contour, iter_wires, set_wire_points, wire_points
from input_output.input.contours import (
    _build_contour,
    _build_frame_data,
    _build_measure,
    _build_wire,
    _contour_file_sort_key,
    _normalize_coord_entry,
)


class TestContourFileSortKey:
    def test_holorama_file_outranks_higher_legacy_version(self):
        # Post-rename 0.1.0 must win over a pre-rename AIVUS-CAA 1.8.0 file.
        files = ['scan_contours_1_8_0.json', 'scan_contours_ho_0_1_0.json']
        assert max(files, key=_contour_file_sort_key) == 'scan_contours_ho_0_1_0.json'

    def test_numeric_version_beats_string_order_within_same_group(self):
        # '0_10_0' would lose to '0_9_0' under plain string comparison.
        files = ['scan_contours_ho_0_9_0.json', 'scan_contours_ho_0_10_0.json']
        assert max(files, key=_contour_file_sort_key) == 'scan_contours_ho_0_10_0.json'

    def test_legacy_files_compared_numerically(self):
        files = ['scan_contours_0_7_4.json', 'scan_contours_1_3_2.json', 'scan_contours_1_1_1.json']
        assert max(files, key=_contour_file_sort_key) == 'scan_contours_1_3_2.json'

    def test_unrecognized_filename_sorts_lowest(self):
        files = ['scan_contours_weird.json', 'scan_contours_ho_0_1_0.json']
        assert max(files, key=_contour_file_sort_key) == 'scan_contours_ho_0_1_0.json'


class TestNormalizeCoordEntry:
    def test_none_returns_empty_list(self):
        assert _normalize_coord_entry(None) == []

    def test_empty_list_returns_empty_list(self):
        assert _normalize_coord_entry([]) == []

    def test_list_of_pairs(self):
        result = _normalize_coord_entry([[1.0, 2.0], [3.0, 4.0]])
        assert result == [(1.0, 2.0), (3.0, 4.0)]

    def test_filters_none_entries_in_list(self):
        result = _normalize_coord_entry([[1.0, 2.0], None, [3.0, 4.0]])
        assert result == [(1.0, 2.0), (3.0, 4.0)]


class TestBuildContour:
    def test_none_returns_empty_contour(self):
        c = _build_contour(None)
        assert c.contours == []

    def test_empty_dict_returns_empty_contour(self):
        c = _build_contour({})
        assert c.contours == []

    def test_basic_contour(self):
        raw = {'contours': [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])], 'closed': [True]}
        c = _build_contour(raw)
        assert len(c.contours) == 1
        assert c.closed == [True]

    def test_strips_duplicate_closing_point(self):
        raw = {'contours': [([1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 4.0])]}
        c = _build_contour(raw)
        x, y = c.contours[0]
        assert x == [1.0, 2.0, 3.0]
        assert y == [4.0, 5.0, 6.0]

    def test_preserves_start_end_coords(self):
        raw = {
            'contours': [([1.0, 2.0], [3.0, 4.0])],
            'start_coords': [[[1.0, 3.0]]],
            'end_coords': [[[2.0, 4.0]]],
        }
        c = _build_contour(raw)
        assert c.start_coords == [[(1.0, 3.0)]]
        assert c.end_coords == [[(2.0, 4.0)]]


class TestBuildWire:
    def test_none_returns_empty_wire(self):
        assert iter_wires(_build_wire(None)) == []

    def test_legacy_single_wire_becomes_first_entry(self):
        # Files written before multi-wire support stored one wire as [[x1, y1], [x2, y2]].
        w = _build_wire([[316.0, 318.0], [372.0, 298.0]])
        assert iter_wires(w) == [[(316.0, 318.0), (372.0, 298.0)]]

    def test_current_format_keeps_every_wire(self):
        raw = {
            'contours': [([1.0, 2.0], [3.0, 4.0]), ([5.0, 6.0], [7.0, 8.0])],
            'closed': [False, False],
            'start_coords': [],
            'end_coords': [],
            'measurements': {},
        }
        assert iter_wires(_build_wire(raw)) == [[(1.0, 3.0), (2.0, 4.0)], [(5.0, 7.0), (6.0, 8.0)]]

    def test_does_not_strip_two_identical_points(self):
        # _build_contour's duplicate-closing-point rule must not apply to wires.
        raw = {'contours': [([1.0, 1.0], [2.0, 2.0])]}
        assert iter_wires(_build_wire(raw)) == [[(1.0, 2.0), (1.0, 2.0)]]


class TestWireHelpers:
    def test_set_wire_points_grows_aligned_lists(self):
        w = Contour()
        set_wire_points(w, 1, [(1.0, 2.0), (3.0, 4.0)])
        assert len(w.contours) == 2
        assert w.closed == [False, False]
        assert w.start_coords == [[], []]
        assert w.end_coords == [[], []]
        assert wire_points(w, 1) == [(1.0, 2.0), (3.0, 4.0)]

    def test_wire_points_out_of_range(self):
        assert wire_points(Contour(), 0) == []

    def test_iter_wires_skips_empty_entries(self):
        w = Contour()
        set_wire_points(w, 1, [(1.0, 2.0), (3.0, 4.0)])
        assert iter_wires(w) == [[(1.0, 2.0), (3.0, 4.0)]]

    def test_iter_wires_accepts_pre_migration_tuple(self):
        assert iter_wires(((1.0, 2.0), (3.0, 4.0))) == [[(1.0, 2.0), (3.0, 4.0)]]


class TestBuildMeasure:
    def test_none_returns_none(self):
        assert _build_measure(None) is None

    def test_dict_format(self):
        raw = {'points': ((1.0, 2.0), (3.0, 4.0)), 'length': 5.0}
        m = _build_measure(raw)
        assert m.points == ((1.0, 2.0), (3.0, 4.0))
        assert m.length == 5.0

    def test_dict_format_without_length(self):
        m = _build_measure({'points': ((0.0, 0.0), (1.0, 1.0))})
        assert m.length is None


class TestBuildFrameData:
    def test_parses_integer_keys(self):
        raw = {
            '0': {'phase': 'D'},
            '1': {'phase': 'S'},
            'gating_signal': {},
        }
        result = _build_frame_data(raw)
        assert set(result.keys()) == {0, 1}
        assert result[0].phase == 'D'
        assert result[1].phase == 'S'

    def test_skips_non_integer_keys(self):
        raw = {'0': {'phase': '-'}, 'gating_signal': {}, 'metadata': {}}
        result = _build_frame_data(raw)
        assert list(result.keys()) == [0]

    def test_full_frame_contour_fields(self):
        raw = {
            '0': {
                'phase': 'D',
                'lumen': {
                    'contours': [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])],
                    'closed': [True],
                    'start_coords': [],
                    'end_coords': [],
                    'measurements': {},
                },
            }
        }
        result = _build_frame_data(raw)
        assert len(result[0].lumen.contours) == 1
        assert result[0].lumen.closed == [True]


EEM = {
    'contours': [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])],
    'closed': [True],
    'start_coords': [],
    'end_coords': [],
    'measurements': {},
}
NO_EEM = {'contours': [], 'closed': [], 'start_coords': [], 'end_coords': [], 'measurements': {}}


class TestFrameLabelMigration:
    """Files older than 0.11.0 gave every frame a 'Very Good' quality whether it had been
    reviewed or not, so the EEM stands in for 'was this frame ever analyzed'."""

    def test_pre_flags_frame_without_eem_loads_unlabeled(self):
        raw = {'0': {'quality': 'Very Good', 'eem': NO_EEM}}
        frame = _build_frame_data(raw, pre_flags=True)[0]
        assert frame.quality == '' and frame.unlabeled is True

    def test_pre_flags_frame_with_eem_keeps_the_label_from_the_file(self):
        raw = {'0': {'quality': 'Bad', 'eem': EEM}}
        frame = _build_frame_data(raw, pre_flags=True)[0]
        assert frame.quality == 'Bad' and frame.unlabeled is False

    def test_eem_holding_only_empty_entries_does_not_count_as_analyzed(self):
        raw = {'0': {'quality': 'Ok', 'eem': {'contours': [([], [])], 'measurements': {}}}}
        frame = _build_frame_data(raw, pre_flags=True)[0]
        assert frame.quality == '' and frame.unlabeled is True

    def test_analyzed_frame_without_a_usable_quality_still_loads_unlabeled(self):
        # The reported case: an EEM but no rating to keep, which must not leave the frame
        # with no rating and no flag either — nothing would be ticked in the UI at all.
        for raw in ({'0': {'eem': EEM}}, {'0': {'quality': '', 'eem': EEM}}):
            frame = _build_frame_data(raw, pre_flags=True)[0]
            assert frame.quality == '' and frame.unlabeled is True

    def test_current_files_skip_the_migration(self):
        raw = {'0': {'quality': 'Good', 'unlabeled': False, 'eem': NO_EEM}}
        frame = _build_frame_data(raw, pre_flags=False)[0]
        assert frame.quality == 'Good' and frame.unlabeled is False


class TestFrameLabelExclusivity:
    """A rating and the three flags are one choice, so exactly one of the four is ever set."""

    @pytest.mark.parametrize(
        'stored, expected',
        [
            (
                {'quality': 'Good', 'guiding_catheter': True, 'unanalyzable': True, 'unlabeled': True},
                ('Good', False, False, False),
            ),
            (
                {'quality': '', 'guiding_catheter': True, 'unanalyzable': True, 'unlabeled': True},
                ('', True, False, False),
            ),
            (
                {'quality': '', 'guiding_catheter': False, 'unanalyzable': True, 'unlabeled': True},
                ('', False, True, False),
            ),
            (
                {'quality': '', 'guiding_catheter': False, 'unanalyzable': False, 'unlabeled': False},
                ('', False, False, True),
            ),
        ],
    )
    def test_contradictory_combinations_are_normalised(self, stored, expected):
        frame = _build_frame_data({'0': {**stored, 'eem': EEM}}, pre_flags=False)[0]
        assert (frame.quality, frame.guiding_catheter, frame.unanalyzable, frame.unlabeled) == expected

    def test_a_frame_is_never_loaded_with_no_label_at_all(self):
        frame = _build_frame_data({'0': {'eem': EEM}}, pre_flags=False)[0]
        assert sum([bool(frame.quality), frame.guiding_catheter, frame.unanalyzable, frame.unlabeled]) == 1
