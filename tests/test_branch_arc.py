"""A side branch over the lumen is kept as the open arc outside it, and still fills up to the lumen."""

from pathlib import Path

import numpy as np
import yaml

from domain.intravascular.io_types import Contour, FrameData
from input_output.output.imgs_masks import contours_to_mask
from tools.branch_arc import branch_diameter, trim_outside_lumen

with open(Path(__file__).resolve().parents[1] / 'src' / 'config.yaml', encoding='utf-8') as _f:
    _CONFIG = yaml.safe_load(_f)['intravascular']
SAMPLES = _CONFIG['n_points_contour']
KNOT_RANGE = tuple(_CONFIG['n_interactive_points_range'])
BRANCH_LABEL = 7  # the default preset's
LUMEN_LABEL = 1


def _circle(r, cx, cy, n=20):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (cx + r * np.cos(a)).tolist(), (cy + r * np.sin(a)).tolist()


LUMEN = _circle(80, 250, 250)
BRANCH = _circle(40, 340, 250)


def _mask(branch, closed):
    fd = FrameData(
        contours={
            'lumen': Contour(contours=[LUMEN], closed=[True]),
            'branch': Contour(contours=[branch], closed=[closed]),
        },
        centroid=(250, 250),
    )
    return contours_to_mask(np.zeros((1, 500, 500)), [0], {0: fd})[0]


def test_arc_ends_on_the_lumen_border():
    xs, ys = trim_outside_lumen(*BRANCH, *LUMEN, SAMPLES, KNOT_RANGE)
    for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
        assert abs(np.hypot(x - 250, y - 250) - 80) < 0.5
    assert all(np.hypot(x - 250, y - 250) >= 79.5 for x, y in zip(xs, ys))


def test_no_trim_without_a_crossing():
    assert trim_outside_lumen(*_circle(20, 400, 250), *LUMEN, SAMPLES, KNOT_RANGE) is None
    assert trim_outside_lumen(*_circle(20, 250, 250), *LUMEN, SAMPLES, KNOT_RANGE) is None


def test_open_arc_fills_like_the_closed_branch():
    closed = _mask(BRANCH, True)
    opened = _mask(trim_outside_lumen(*BRANCH, *LUMEN, SAMPLES, KNOT_RANGE), False)
    assert ((closed == BRANCH_LABEL) ^ (opened == BRANCH_LABEL)).sum() < 20
    assert ((closed == LUMEN_LABEL) == (opened == LUMEN_LABEL)).all()


def test_closed_branch_diameter_is_its_longest_distance():
    diameter, start, end = branch_diameter(*BRANCH, True, *LUMEN, (250, 250), SAMPLES)
    assert abs(diameter - 80) < 0.5
    assert abs(np.hypot(*(start - end)) - diameter) < 1e-9


def test_open_branch_diameter_is_the_widest_chord_across_the_axis():
    arc = trim_outside_lumen(*BRANCH, *LUMEN, SAMPLES, KNOT_RANGE)
    diameter, start, end = branch_diameter(*arc, False, *LUMEN, (250, 250), SAMPLES)
    assert abs(diameter - 80) < 1.0
    assert abs(start[0] - end[0]) < 1e-6  # orthogonal to the horizontal axis lumen centroid -> arc middle


def test_open_branch_without_a_lumen_has_no_diameter():
    arc = trim_outside_lumen(*BRANCH, *LUMEN, SAMPLES, KNOT_RANGE)
    assert branch_diameter(*arc, False, None, None, None, SAMPLES) is None
