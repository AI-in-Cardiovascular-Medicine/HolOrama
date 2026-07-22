import numpy as np
import pytest

from tools.geometry import OpenSplineGeometry, SplineGeometry

# 4-point diamond; enough unique points for cubic closed spline (k=3 needs n>3)
_DX = [0.0, 1.0, 0.0, -1.0]
_DY = [1.0, 0.0, -1.0, 0.0]


def closed_spline(n_pts=100) -> SplineGeometry:
    return SplineGeometry(list(_DX), list(_DY), n_pts, None, None, is_closed=True)


class TestSplineGeometryInit:
    def test_mismatched_xy_raises(self):
        with pytest.raises(ValueError):
            SplineGeometry([1.0, 2.0], [1.0], 100, None, None)

    def test_closed_auto_adds_closing_point(self):
        sg = closed_spline()
        assert sg.knot_points_x[0] == sg.knot_points_x[-1]
        assert sg.knot_points_y[0] == sg.knot_points_y[-1]

    def test_closed_already_closed_no_extra_duplicate(self):
        x = list(_DX) + [_DX[0]]
        y = list(_DY) + [_DY[0]]
        sg = SplineGeometry(x, y, 100, None, None, is_closed=True)
        # Already closed: _ensure_closed should not append another duplicate
        assert len(sg.knot_points_x) == len(x)

    def test_empty_closed_spline_keeps_empty_full_contour(self):
        sg = SplineGeometry([], [], 100, None, None, is_closed=True)
        x, y = sg.full_contour
        assert len(x) == 0


class TestSplineGeometryInterpolation:
    def test_interpolation_yields_requested_point_count(self):
        sg = closed_spline(n_pts=200)
        x, y = sg.full_contour
        assert len(x) == 200
        assert len(y) == 200

    def test_interpolation_returns_numpy_arrays(self):
        x, y = closed_spline().interpolate()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_manual_interpolation_updates_full_contour(self):
        sg = SplineGeometry(list(_DX), list(_DY), 50, None, None, is_closed=False)
        sg.interpolate()
        x, _ = sg.full_contour
        assert len(x) == 50


class TestSplineGeometryClassMethods:
    def test_from_points(self):
        points = list(zip(_DX, _DY))
        sg = SplineGeometry.from_points(points, 100, is_closed=True)
        assert sg.knot_points_x[0] == _DX[0]
        assert sg.n_interpolated_points == 100
        assert len(sg.full_contour[0]) == 100

    def test_from_points_empty(self):
        sg = SplineGeometry.from_points([], 100, is_closed=True)
        assert sg.knot_points_x == []

    def test_from_arrays(self):
        sg = SplineGeometry.from_arrays(_DX, _DY, 100, is_closed=True)
        assert sg.knot_points_y[0] == _DY[0]
        assert len(sg.full_contour[0]) == 100


class TestSplineGeometryMutation:
    def test_insert_point_increases_knot_count_by_one(self):
        sg = closed_spline()
        initial = len(sg.knot_points_x)
        sg.insert_point(0.5, 0.5)
        assert len(sg.knot_points_x) == initial + 1

    def test_scale_returns_new_instance_with_scaled_knots(self):
        sg = closed_spline()
        scaled = sg.scale(2.0)
        assert scaled is not sg
        assert scaled.knot_points_x[0] == pytest.approx(sg.knot_points_x[0] * 2)
        assert scaled.knot_points_y[0] == pytest.approx(sg.knot_points_y[0] * 2)

    def test_get_closest_contour_index_at_exact_point(self):
        sg = closed_spline()
        x, y = sg.full_contour
        idx = sg.get_closest_contour_index(float(x[0]), float(y[0]), threshold=1.0)
        assert idx == 0

    def test_get_closest_contour_index_outside_threshold_returns_none(self):
        sg = closed_spline()
        assert sg.get_closest_contour_index(100.0, 100.0, threshold=1.0) is None

    def test_to_unscaled_divides_full_contour_by_factor(self):
        sg = closed_spline()
        ux, uy = sg.to_unscaled(2.0)
        x, y = sg.full_contour
        np.testing.assert_allclose(ux, [v / 2.0 for v in x])
        np.testing.assert_allclose(uy, [v / 2.0 for v in y])


class TestOpenSplineGeometry:
    def test_is_not_closed(self):
        sg = OpenSplineGeometry(list(_DX), list(_DY), 100)
        assert sg.is_closed is False

    def test_no_closing_point_added(self):
        sg = OpenSplineGeometry([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 100)
        # First and last points should differ
        assert sg.knot_points_x[0] != sg.knot_points_x[-1] or sg.knot_points_y[0] != sg.knot_points_y[-1]

    def test_full_contour_empty_before_manual_interpolation(self):
        sg = OpenSplineGeometry([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 100)
        x, _ = sg.full_contour
        assert len(x) == 0

    def test_interpolation_produces_correct_n_points(self):
        sg = OpenSplineGeometry(list(_DX), list(_DY), 100)
        x, y = sg.interpolate()
        assert len(x) == 100
        assert len(y) == 100
