"""Tests for changing a contour's knot count (domain.intravascular.knot_resampling) and its
Points control / Shift+Wheel (pages.intravascular.utils.contours_gui).

The shape survives, a count once reached returns exactly, and only the selected contour
changes (one Ctrl+Z).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

from domain.intravascular.types import ContourType
from domain.intravascular.io_types import Contour, FrameData
from domain.intravascular.knot_resampling import KnotHistory
from domain.intravascular.runtime_types import RuntimeData
from gui.shortcuts import undo_last_contour_edit
from pages.intravascular.utils.contours_gui import (
    selected_contour_knot_count,
    set_selected_contour_knots,
    step_selected_contour_knots,
)

MIN_KNOTS, MAX_KNOTS = 3, 40  # config: n_interactive_points_range


def _lobed(n=20, radius=50.0, lobes=4, depth=0.25, cx=200.0, cy=200.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = radius * (1 + depth * np.sin(lobes * angles))
    return [float(cx + r * math.cos(a)) for r, a in zip(radii, angles)], [
        float(cy + r * math.sin(a)) for r, a in zip(radii, angles)
    ]


def _arc(n=12):
    angles = np.linspace(0, np.pi, n)
    return [float(100 + 60 * math.cos(a)) for a in angles], [float(100 + 30 * math.sin(a)) for a in angles]


def _curve(xs, ys, closed, samples=2000):
    """The drawn spline through the knots, densely."""
    xs, ys = list(xs), list(ys)
    if closed:
        xs, ys = xs + [xs[0]], ys + [ys[0]]
    tck, _ = splprep(np.array([xs, ys]), s=0.0, k=min(3, len(xs) - 1), per=int(closed))
    return np.column_stack(splev(np.linspace(0, 1, samples), tck))


def _strays(knots, reference_knots, closed=True) -> float:
    """Max gap (px) between the splines through `knots` and `reference_knots`."""
    distances, _ = cKDTree(_curve(*knots, closed)).query(_curve(*reference_knots, closed))
    return float(distances.max())


class TestKnotHistory:
    def test_the_original_count_gives_the_original_knots(self):
        xs, ys = _lobed()
        assert KnotHistory(xs, ys, True, (MIN_KNOTS, MAX_KNOTS)).knots(20) == (xs, ys)

    @pytest.mark.parametrize('count', [3, 7, 12, 19, 21, 33, 40])
    def test_every_count_has_that_many_knots(self, count):
        assert len(KnotHistory(*_lobed(), True, (MIN_KNOTS, MAX_KNOTS)).knots(count)[0]) == count

    def test_counts_stay_between_the_limits(self):
        history = KnotHistory(*_lobed(), True, (MIN_KNOTS, MAX_KNOTS))
        assert len(history.knots(1)[0]) == MIN_KNOTS
        assert len(history.knots(99)[0]) == MAX_KNOTS

    def test_down_and_up_again_loses_nothing(self):
        original = _lobed()
        history = KnotHistory(*original, True, (MIN_KNOTS, MAX_KNOTS))
        on_the_way_down = {count: history.knots(count) for count in range(19, 2, -1)}
        on_the_way_up = {count: history.knots(count) for count in range(3, 41)}
        assert history.knots(20) == original
        for count, knots in on_the_way_down.items():
            assert on_the_way_up[count] == knots, count

    def test_thinning_keeps_the_shape(self):
        original = _lobed()
        history = KnotHistory(*original, True, (MIN_KNOTS, MAX_KNOTS))
        assert _strays(history.knots(16), original) < 1.0
        assert _strays(history.knots(12), original) < 4.0

    def test_thinning_beats_spreading_the_knots_evenly(self):
        # A dent the knots crowd into: the even spread misses it, the thinning keeps it
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        radii = 50 - 20 * np.exp(-(((angles - np.pi) / 0.35) ** 2))
        original = list(200 + radii * np.cos(angles)), list(200 + radii * np.sin(angles))
        thinned = KnotHistory(*original, True, (MIN_KNOTS, MAX_KNOTS)).knots(10)
        dense = _curve(*original, True, samples=1000)
        evenly = list(dense[::100, 0]), list(dense[::100, 1])
        assert _strays(thinned, original) < _strays(evenly, original)

    def test_added_knots_lie_on_the_shape(self):
        original = _lobed()
        thickened = KnotHistory(*original, True, (MIN_KNOTS, MAX_KNOTS)).knots(30)
        assert _strays(thickened, original) < 0.5
        distances, _ = cKDTree(_curve(*original, True)).query(np.column_stack(thickened))
        assert distances.max() < 0.25  # every knot, the new ones too, on the original spline

    def test_an_open_contour_keeps_its_ends(self):
        xs, ys = _arc()
        history = KnotHistory(xs, ys, False, (MIN_KNOTS, MAX_KNOTS))
        for count in (3, 6, 25):
            new_xs, new_ys = history.knots(count)
            assert (new_xs[0], new_ys[0]) == pytest.approx((xs[0], ys[0]))
            assert (new_xs[-1], new_ys[-1]) == pytest.approx((xs[-1], ys[-1]))

    def test_labelled_knots_stay(self):
        xs, ys = _lobed()
        history = KnotHistory(xs, ys, True, (MIN_KNOTS, MAX_KNOTS), pinned=[(xs[5], ys[5]), (xs[11], ys[11])])
        new_xs, new_ys = history.knots(3)
        for i in (5, 11):
            assert any(math.hypot(x - xs[i], y - ys[i]) < 1e-6 for x, y in zip(new_xs, new_ys)), i

    def test_a_closing_repeat_is_not_a_knot(self):
        xs, ys = _lobed()
        history = KnotHistory(xs + [xs[0]], ys + [ys[0]], True, (MIN_KNOTS, MAX_KNOTS))
        assert history.original_count == 20
        assert history.is_original(xs + [xs[0]], ys + [ys[0]])

    def test_it_knows_the_contours_it_handed_out(self):
        history = KnotHistory(*_lobed(), True, (MIN_KNOTS, MAX_KNOTS))
        thinned = history.knots(10)
        assert history.holds(*thinned)
        moved = ([thinned[0][0] + 3.0] + thinned[0][1:], thinned[1])
        assert not history.holds(*moved)


# -- the Points control and Shift+Wheel --------------------------------------------------


@pytest.fixture
def main_window():
    """Stub window: a lumen and two calcium contours, second selected."""
    runtime_data = RuntimeData()
    frame_data = FrameData()
    frame_data.lumen = Contour(contours=[_lobed()], closed=[True], start_coords=[[]], end_coords=[[]])
    frame_data.contours['calcium'] = Contour(
        contours=[_lobed(n=10, radius=10, cx=150), _lobed(n=20, radius=20, cx=260)],
        closed=[True, True],
        start_coords=[[], []],
        end_coords=[[], []],
    )
    runtime_data.frame_data_dct = {0: frame_data}
    display = SimpleNamespace(
        frame=0,
        active_contour_type=ContourType('calcium'),
        active_contour_index=1,
        drawing_mode=False,
        active_point_index=None,
        working_spline=None,
        knot_histories={},
        knot_count_range=(MIN_KNOTS, MAX_KNOTS),
        redraws=0,
    )
    display.contour_key = lambda contour_type=None: (contour_type or display.active_contour_type).value
    display._frame_contour = lambda: frame_data.contour(display.contour_key())
    display.display_image = lambda **_: setattr(display, 'redraws', display.redraws + 1)
    display.update_display = lambda: None
    window = SimpleNamespace(
        image_displayed=True,
        runtime_data=runtime_data,
        display=display,
        display_slider=SimpleNamespace(set_value=lambda value: None),
        longitudinal_view=SimpleNamespace(plot_areas=lambda: None),
        status_bar=SimpleNamespace(showMessage=lambda text: None),
        save_contours_soon=lambda: None,
    )
    return window


def _calcium(main_window, index):
    return main_window.runtime_data.frame_data_dct[0].contours['calcium'].contours[index]


class TestPointsControl:
    def test_counts_the_selected_contour(self, main_window):
        assert selected_contour_knot_count(main_window) == 20

    def test_changes_only_the_selected_contour(self, main_window):
        other, lumen = _calcium(main_window, 0), main_window.runtime_data.frame_data_dct[0].lumen.contours[0]
        set_selected_contour_knots(main_window, 8)
        assert len(_calcium(main_window, 1)[0]) == 8
        assert _calcium(main_window, 0) == other
        assert main_window.runtime_data.frame_data_dct[0].lumen.contours[0] == lumen

    def test_the_wheel_down_and_up_again_gives_the_original_back(self, main_window):
        original = _calcium(main_window, 1)
        for _ in range(12):
            step_selected_contour_knots(main_window, -1)
        assert len(_calcium(main_window, 1)[0]) == 8
        for _ in range(12):
            step_selected_contour_knots(main_window, +1)
        assert _calcium(main_window, 1) == original

    def test_one_ctrl_z_undoes_a_whole_run(self, main_window):
        original = _calcium(main_window, 1)
        for _ in range(6):
            step_selected_contour_knots(main_window, -1)
        set_selected_contour_knots(main_window, 30)

        undo_last_contour_edit(main_window)

        assert _calcium(main_window, 1) == original
        assert not main_window.runtime_data.contour_undo.can_undo, 'one entry, not one per step'

    def test_after_a_ctrl_z_it_carries_on_from_what_is_back(self, main_window):
        set_selected_contour_knots(main_window, 10)
        undo_last_contour_edit(main_window)
        set_selected_contour_knots(main_window, 10)
        assert main_window.runtime_data.contour_undo.can_undo
        undo_last_contour_edit(main_window)
        assert len(_calcium(main_window, 1)[0]) == 20

    def test_an_edit_in_between_starts_a_new_history(self, main_window):
        set_selected_contour_knots(main_window, 10)
        xs, ys = _calcium(main_window, 1)
        dragged = ([xs[0] + 5.0] + list(xs[1:]), list(ys))
        main_window.runtime_data.frame_data_dct[0].contours['calcium'].contours[1] = dragged

        set_selected_contour_knots(main_window, 20)

        new_xs, new_ys = _calcium(main_window, 1)
        assert len(new_xs) == 20
        assert any(math.hypot(x - dragged[0][0], y - dragged[1][0]) < 1e-6 for x, y in zip(new_xs, new_ys))

    def test_the_limits_are_no_change(self, main_window):
        set_selected_contour_knots(main_window, MAX_KNOTS)
        redraws = main_window.display.redraws
        step_selected_contour_knots(main_window, +1)
        assert main_window.display.redraws == redraws

    def test_a_sector_is_left_alone(self, main_window):
        main_window.display.active_contour_type = ContourType('wire')
        main_window.display.active_contour_index = 0
        main_window.runtime_data.frame_data_dct[0].contours['wire'] = Contour(
            contours=[([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])], closed=[False], start_coords=[[]], end_coords=[[]]
        )
        assert selected_contour_knot_count(main_window) is None
        set_selected_contour_knots(main_window, 10)
        assert main_window.display.redraws == 0

    def test_nothing_while_a_contour_is_being_drawn(self, main_window):
        main_window.display.drawing_mode = True
        set_selected_contour_knots(main_window, 10)
        assert len(_calcium(main_window, 1)[0]) == 20
