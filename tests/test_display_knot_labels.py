"""Tests for labelling a closed contour's knot points start / end (the double-click popup
in pages.intravascular.left_half.display).

The pair delimits the uncertain arc of a contour, so a knot carries one label at most and
switching one for the other has to drop the old one — otherwise the same knot ends up in
both lists and the arc it delimits is read from itself to itself.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from domain.all_types import ContourType
from domain.io_types import Contour, FrameData
from domain.runtime_types import RuntimeData

DIM = 200
N_FRAMES = 2
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'src' / 'config.yaml'


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _to_namespace(value) for key, value in obj.items()})
    return obj


def _ring(radius=40.0, n=12):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centre = DIM / 2
    return (
        [float(centre + radius * math.cos(a)) for a in angles],
        [float(centre + radius * math.sin(a)) for a in angles],
    )


@pytest.fixture
def display(qt_app):
    """A real Display on a stub main_window, showing one closed lumen contour."""
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = _to_namespace(yaml.safe_load(f))

    frames = {i: FrameData() for i in range(N_FRAMES)}
    frames[0].lumen = Contour(contours=[_ring()], closed=[True], start_coords=[[]], end_coords=[[]])
    runtime = RuntimeData()
    runtime.frame_data_dct = frames
    runtime.metadata = {'num_frames': N_FRAMES, 'resolution': 0.02, 'modality': 'OCT', 'dimension': DIM}
    runtime.images = np.full((N_FRAMES, DIM, DIM), 100, dtype=np.uint8)
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
        left_half=SimpleNamespace(set_active_contour_type_ui=lambda ct: None),
    )
    main_window.save_contours_soon = runtime.mark_unsaved

    from pages.intravascular.left_half.display import Display

    widget = Display(main_window)
    main_window.display = widget
    widget.set_data(runtime.images)
    return SimpleNamespace(widget=widget, contour=frames[0].lumen, knot=_ring()[0][0], knot_y=_ring()[1][0])


def _menu_state(widget, contour, kx, ky):
    """What the popup would offer for this knot: (Mark as Start, Mark as End, Remove)."""
    is_start, is_end = widget._knot_labels(contour, 0, kx, ky)
    return (not is_start, not is_end, is_start or is_end)


class TestWhatThePopupOffers:
    def test_an_unlabelled_knot_can_become_either(self, display):
        assert _menu_state(display.widget, display.contour, display.knot, display.knot_y) == (True, True, False)

    def test_a_start_can_become_an_end_or_lose_its_label(self, display):
        widget = display.widget
        widget._label_knot(display.contour.start_coords, 0, display.knot, display.knot_y)

        assert _menu_state(widget, display.contour, display.knot, display.knot_y) == (False, True, True)

    def test_an_end_can_become_a_start_or_lose_its_label(self, display):
        widget = display.widget
        widget._label_knot(display.contour.end_coords, 0, display.knot, display.knot_y)

        assert _menu_state(widget, display.contour, display.knot, display.knot_y) == (True, False, True)

    def test_another_knot_of_the_same_contour_is_unaffected(self, display):
        widget = display.widget
        widget._label_knot(display.contour.start_coords, 0, display.knot, display.knot_y)
        other_x, other_y = _ring()[0][6], _ring()[1][6]

        assert _menu_state(widget, display.contour, other_x, other_y) == (True, True, False)


class TestSwitchingALabel:
    def test_switching_start_to_end_leaves_it_in_one_list_only(self, display):
        widget = display.widget
        contour = display.contour
        widget._label_knot(contour.start_coords, 0, display.knot, display.knot_y)

        # What the popup does for "Mark as End": drop whatever it carried, then label it.
        widget._unlabel_knot(contour, 0, display.knot, display.knot_y)
        widget._label_knot(contour.end_coords, 0, display.knot, display.knot_y)

        assert contour.start_coords[0] == []
        assert contour.end_coords[0] == [(display.knot, display.knot_y)]

    def test_switching_back_again_works_too(self, display):
        widget = display.widget
        contour = display.contour
        widget._label_knot(contour.end_coords, 0, display.knot, display.knot_y)

        widget._unlabel_knot(contour, 0, display.knot, display.knot_y)
        widget._label_knot(contour.start_coords, 0, display.knot, display.knot_y)

        assert contour.start_coords[0] == [(display.knot, display.knot_y)]
        assert contour.end_coords[0] == []

    def test_removing_a_label_clears_both_lists(self, display):
        widget = display.widget
        contour = display.contour
        widget._label_knot(contour.start_coords, 0, display.knot, display.knot_y)

        widget._unlabel_knot(contour, 0, display.knot, display.knot_y)

        assert contour.start_coords[0] == []
        assert contour.end_coords[0] == []

    def test_a_second_knot_keeps_its_own_label_through_a_switch(self, display):
        widget = display.widget
        contour = display.contour
        other_x, other_y = _ring()[0][6], _ring()[1][6]
        widget._label_knot(contour.start_coords, 0, display.knot, display.knot_y)
        widget._label_knot(contour.end_coords, 0, other_x, other_y)

        widget._unlabel_knot(contour, 0, display.knot, display.knot_y)
        widget._label_knot(contour.end_coords, 0, display.knot, display.knot_y)

        assert contour.start_coords[0] == []
        assert sorted(contour.end_coords[0]) == sorted([(other_x, other_y), (display.knot, display.knot_y)])


class TestLabelListGrowth:
    def test_labelling_a_later_contour_grows_the_list(self, display):
        contour = Contour(contours=[_ring(), _ring(30.0)], closed=[True, True])
        display.widget._label_knot(contour.start_coords, 1, 1.0, 2.0)

        assert contour.start_coords == [[], [(1.0, 2.0)]]

    def test_unlabelling_a_contour_with_no_lists_yet_is_harmless(self, display):
        contour = Contour(contours=[_ring()], closed=[True])
        display.widget._unlabel_knot(contour, 0, 1.0, 2.0)

        assert contour.start_coords == []
        assert contour.end_coords == []


class TestDeletingALabelledKnot:
    def test_the_label_goes_with_the_knot(self, display):
        widget = display.widget
        contour = display.contour
        widget.set_active_contour_type(ContourType.LUMEN)
        widget.display_image(update_contours=True)
        knot = widget.points_to_draw[0]
        widget._label_knot(contour.start_coords, 0, knot.x / widget.scaling_factor, knot.y / widget.scaling_factor)
        knots_before = len(contour.contours[0][0])

        widget._delete_point(knot)

        assert len(contour.contours[0][0]) == knots_before - 1
        assert contour.start_coords[0] == []
