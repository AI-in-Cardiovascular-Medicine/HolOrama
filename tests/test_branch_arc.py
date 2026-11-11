"""A side branch over the lumen is kept as the open arc outside it, and still fills up to the lumen."""

import numpy as np

from domain.intravascular.io_types import Contour, FrameData
from input_output.output.imgs_masks import contours_to_mask
from tools.branch_arc import trim_outside_lumen

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
    xs, ys = trim_outside_lumen(*BRANCH, *LUMEN, 20)
    for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
        assert abs(np.hypot(x - 250, y - 250) - 80) < 0.5
    assert all(np.hypot(x - 250, y - 250) >= 79.5 for x, y in zip(xs, ys))


def test_no_trim_without_a_crossing():
    assert trim_outside_lumen(*_circle(20, 400, 250), *LUMEN, 20) is None
    assert trim_outside_lumen(*_circle(20, 250, 250), *LUMEN, 20) is None


def test_open_arc_fills_like_the_closed_branch():
    closed = _mask(BRANCH, True)
    opened = _mask(trim_outside_lumen(*BRANCH, *LUMEN, 20), False)
    assert ((closed == BRANCH_LABEL) ^ (opened == BRANCH_LABEL)).sum() < 20
    assert ((closed == LUMEN_LABEL) == (opened == LUMEN_LABEL)).all()
