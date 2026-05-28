from input_output.input.contours import (
    _normalize_coord_entry,
    _is_legacy,
    _build_contour,
    _build_contour_legacy,
    _build_measure,
    _build_frame_data,
    _build_frame_data_legacy,
)


class TestNormalizeCoordEntry:
    def test_none_returns_empty_list(self):
        assert _normalize_coord_entry(None) == []

    def test_empty_list_returns_empty_list(self):
        assert _normalize_coord_entry([]) == []

    def test_single_pair_as_list(self):
        assert _normalize_coord_entry([1.0, 2.0]) == [(1.0, 2.0)]

    def test_single_pair_as_tuple(self):
        assert _normalize_coord_entry((3.0, 4.0)) == [(3.0, 4.0)]

    def test_list_of_pairs(self):
        result = _normalize_coord_entry([[1.0, 2.0], [3.0, 4.0]])
        assert result == [(1.0, 2.0), (3.0, 4.0)]

    def test_filters_none_entries_in_list(self):
        result = _normalize_coord_entry([[1.0, 2.0], None, [3.0, 4.0]])
        assert result == [(1.0, 2.0), (3.0, 4.0)]


class TestIsLegacy:
    def test_lumen_list_is_legacy(self):
        assert _is_legacy({'lumen': [[1, 2], [3, 4]]}) is True

    def test_lumen_tuple_is_legacy(self):
        assert _is_legacy({'lumen': ([1, 2], [3, 4])}) is True

    def test_lumen_dict_is_not_legacy(self):
        assert _is_legacy({'lumen': {'contours': []}}) is False

    def test_lumen_missing_is_not_legacy(self):
        assert _is_legacy({'other': 'data'}) is False


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


class TestBuildContourLegacy:
    def test_missing_key_returns_empty_contour(self):
        assert _build_contour_legacy({}, 'lumen', 0).contours == []

    def test_builds_from_flat_xy_lists_and_strips_closing_point(self):
        # Legacy format: (x_per_frame, y_per_frame); last point is a closing duplicate
        raw = {
            'lumen': (
                [[1.0, 2.0, 3.0, 1.0], []],
                [[4.0, 5.0, 6.0, 4.0], []],
            )
        }
        c = _build_contour_legacy(raw, 'lumen', 0)
        x, y = c.contours[0]
        assert x == [1.0, 2.0, 3.0]
        assert y == [4.0, 5.0, 6.0]

    def test_empty_frame_returns_empty_contour(self):
        raw = {'lumen': ([[1.0, 2.0], []], [[3.0, 4.0], []])}
        c = _build_contour_legacy(raw, 'lumen', 1)
        assert c.contours == []

    def test_frame_out_of_range_returns_empty_contour(self):
        raw = {'lumen': ([[1.0]], [[2.0]])}
        c = _build_contour_legacy(raw, 'lumen', 99)
        assert c.contours == []


class TestBuildMeasure:
    def test_none_returns_none(self):
        assert _build_measure(None) is None

    def test_legacy_list_four_values_unscaled_by_factor(self):
        # [x1, y1, x2, y2] in display coordinates; scaling_factor=2 → divide by 2
        m = _build_measure([10.0, 20.0, 30.0, 40.0], scaling_factor=2.0)
        assert m is not None
        assert m.points == ((5.0, 10.0), (15.0, 20.0))

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


class TestBuildFrameDataLegacy:
    def _raw(self, num_frames=3):
        return {
            'phases': ['D', '-', 'S'],
            'lumen': (
                [[1.0, 2.0, 1.0], [], []],
                [[3.0, 4.0, 3.0], [], []],
            ),
            'reference': [None] * num_frames,
            'measures': [[None, None]] * num_frames,
            'measure_lengths': [[None, None]] * num_frames,
        }

    def test_creates_frame_data_for_each_frame(self):
        result = _build_frame_data_legacy(self._raw(), num_frames=3)
        assert len(result) == 3

    def test_phases_assigned_correctly(self):
        result = _build_frame_data_legacy(self._raw(), num_frames=3)
        assert result[0].phase == 'D'
        assert result[1].phase == '-'
        assert result[2].phase == 'S'

    def test_lumen_contour_closing_point_stripped(self):
        result = _build_frame_data_legacy(self._raw(), num_frames=3)
        assert len(result[0].lumen.contours) == 1
        x, y = result[0].lumen.contours[0]
        assert x == [1.0, 2.0]
        assert y == [3.0, 4.0]

    def test_empty_frame_has_no_lumen_contour(self):
        result = _build_frame_data_legacy(self._raw(), num_frames=3)
        assert result[1].lumen.contours == []
