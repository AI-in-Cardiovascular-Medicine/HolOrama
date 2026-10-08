"""Tests for the two ends of an open contour in pages.intravascular.left_half.display.

An open contour (a calcium arc) runs from its first knot, the start (yellow, with a line to
the image edge), to its last, the end (red, likewise). Dragging, adding, deleting or scaling
knots must keep it so, or the arc's extent is misread.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QGraphicsLineItem

from domain.intravascular.types import ContourType
from domain.intravascular.io_types import Contour, FrameData
from domain.intravascular.runtime_types import RuntimeData

DIM = 200
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'src' / 'config.yaml'
CALCIUM = ContourType('calcium')


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _to_namespace(value) for key, value in obj.items()})
    return obj


def _arc(n=8, radius=50.0, from_deg=200.0, to_deg=320.0):
    centre = DIM / 2
    angles = [math.radians(from_deg + (to_deg - from_deg) * i / (n - 1)) for i in range(n)]
    return [centre + radius * math.cos(a) for a in angles], [centre + radius * math.sin(a) for a in angles]


@pytest.fixture
def display(qt_app):
    """Real Display on a stub window, an open calcium arc selected."""
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = _to_namespace(yaml.safe_load(f))

    xs, ys = _arc()
    arc = Contour(contours=[(xs, ys)], closed=[False], start_coords=[[(xs[0], ys[0])]], end_coords=[[(xs[-1], ys[-1])]])
    frames = {0: FrameData()}
    frames[0].contours['calcium'] = arc
    runtime = RuntimeData()
    runtime.frame_data_dct = frames
    runtime.metadata = {'num_frames': 1, 'resolution': 0.02, 'modality': 'OCT', 'dimension': DIM}
    runtime.images = np.full((1, DIM, DIM), 100, dtype=np.uint8)
    main_window = SimpleNamespace(
        config=config,
        runtime_data=runtime,
        hide_contours=False,
        hide_special_points=True,
        colormap_enabled=False,
        mask_mode_box=None,
        longitudinal_view=SimpleNamespace(
            set_data=lambda images: None,
            plot_areas=lambda: None,
            hide_lview_contours=lambda: None,
            show_lview_contours=lambda: None,
            update_marker=lambda frame: None,
        ),
        image_displayed=True,
        file_name='test',
        status_bar=SimpleNamespace(showMessage=lambda *args: None),
        left_half=SimpleNamespace(set_active_contour_type_ui=lambda ct: None, sync_knot_count=lambda: None),
    )
    main_window.save_contours_soon = runtime.mark_unsaved

    from pages.intravascular.left_half.display import Display

    widget = Display(main_window)
    main_window.display = widget
    widget.set_data(runtime.images)
    widget.set_active_contour_type(CALCIUM)
    widget.display_image(update_contours=True)
    return SimpleNamespace(widget=widget, arc=arc)


def _knots(arc):
    return arc.contours[0][0], arc.contours[0][1]


def _assert_ends_shown(display):
    """First knot alone yellow, last alone red, each with its edge line, and those are the
    stored start and end."""
    widget, arc = display.widget, display.arc
    xs, ys = _knots(arc)
    colours = [point.color for point in widget.points_to_draw]
    assert len(colours) == len(xs)
    assert colours[0] == widget.start_color
    assert colours[-1] == widget.end_color
    assert widget.start_color not in colours[1:] and widget.end_color not in colours[:-1]

    assert arc.start_coords[0] == [pytest.approx((xs[0], ys[0]))]
    assert arc.end_coords[0] == [pytest.approx((xs[-1], ys[-1]))]

    sf = widget.scaling_factor
    line_starts = [
        (item.line().p1().x() / sf, item.line().p1().y() / sf)
        for item in widget.graphics_scene.items()
        if type(item) is QGraphicsLineItem
    ]
    for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
        assert any(math.hypot(x - lx, y - ly) < 1e-6 for lx, ly in line_starts), (x, y)


def test_a_fresh_arc_shows_both_ends(display):
    _assert_ends_shown(display)


def test_a_knot_next_to_the_start_is_not_a_second_start(display):
    xs, ys = _knots(display.arc)
    # A knot inside the start's snap radius
    xs.insert(1, xs[0] + 0.5)
    ys.insert(1, ys[0] + 0.5)
    display.widget.display_image(update_contours=True)
    _assert_ends_shown(display)


def test_scaling_the_arc_takes_its_ends_along(display):
    for _ in range(10):
        display.widget._scale_active_contour(+120)
    _assert_ends_shown(display)


def test_deleting_the_knot_before_the_end_keeps_the_end(display):
    xs, ys = _knots(display.arc)
    xs.insert(len(xs) - 1, xs[-1] - 0.5)  # close enough to the end to be taken for it
    ys.insert(len(ys) - 1, ys[-1])
    display.widget.display_image(update_contours=True)

    display.widget._delete_point(display.widget.points_to_draw[-2])

    _assert_ends_shown(display)
    assert len(_knots(display.arc)[0]) == len(_arc()[0])


def test_deleting_the_last_knot_makes_the_one_before_the_end(display):
    display.widget._delete_point(display.widget.points_to_draw[-1])
    _assert_ends_shown(display)


def test_a_click_on_the_very_start_of_the_path_adds_no_knot_before_it(display):
    spline = display.widget.working_spline
    first = (spline.geometry.knot_points_x[0], spline.geometry.knot_points_y[0])
    index = spline.update_point(QPointF(first[0] - 3.0, first[1]), -1, path_index=0)
    assert index >= 1
    assert (spline.geometry.knot_points_x[0], spline.geometry.knot_points_y[0]) == first


def test_a_click_on_the_very_end_of_the_path_adds_no_knot_after_it(display):
    spline = display.widget.working_spline
    last = (spline.geometry.knot_points_x[-1], spline.geometry.knot_points_y[-1])
    path_end = len(spline.geometry.full_contour[0]) - 1
    index = spline.update_point(QPointF(last[0] + 3.0, last[1]), -1, path_index=path_end)
    assert index <= len(spline.geometry.knot_points_x) - 2
    assert (spline.geometry.knot_points_x[-1], spline.geometry.knot_points_y[-1]) == last
