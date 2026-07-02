from input_output.input.contours import (
    _normalize_coord_entry,
    _build_contour,
    _build_measure,
    _build_frame_data,
    _contour_file_sort_key,
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
