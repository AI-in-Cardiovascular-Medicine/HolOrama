"""A side branch leaving the lumen sets one reference point per run of frames it shows on."""

from types import SimpleNamespace

import numpy as np
import pytest

from domain.intravascular.io_types import Contour, FrameData
from pages.intravascular.utils.branch_references import update_branch_references
from tools.branch_arc import trim_outside_lumen

RESOLUTION = 0.02  # mm per px, so a branch of radius 40 px is 1.6 mm wide
CENTRE = (250.0, 250.0)


def _circle(r, cx, cy, n=20):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (cx + r * np.cos(a)).tolist(), (cy + r * np.sin(a)).tolist()


LUMEN = _circle(80, *CENTRE)


def _frame(branch_radius=None, phase='-', closed=False) -> FrameData:
    contours = {'lumen': Contour(contours=[LUMEN], closed=[True])}
    if branch_radius is not None:
        branch = _circle(branch_radius, 340, 250)
        if not closed:
            branch = trim_outside_lumen(*branch, *LUMEN, 200, (3, 40))
        contours['branch'] = Contour(contours=[branch], closed=[closed])
    return FrameData(phase=phase, contours=contours, centroid=CENTRE)


def _window(frames: dict[int, FrameData]):
    return SimpleNamespace(
        runtime_data=SimpleNamespace(frame_data_dct=frames, metadata={'resolution': RESOLUTION}),
        config=SimpleNamespace(intravascular=SimpleNamespace(branch_reference_min_diameter_mm=1.0)),
        display=SimpleNamespace(n_points_contour=200),
    )


def _referenced(frames):
    return sorted(f for f, fd in frames.items() if fd.reference is not None)


def test_a_run_of_neighbouring_frames_gets_one_reference_on_its_widest_branch():
    frames = {i: _frame() for i in range(12)}
    frames.update({3: _frame(30), 4: _frame(40), 5: _frame(35), 9: _frame(40)})
    assert update_branch_references(_window(frames))
    assert _referenced(frames) == [4, 9]
    assert all(frames[f].reference_auto for f in (4, 9))


def test_the_reference_sits_in_the_branch_on_its_axis():
    frames = {0: _frame(40)}
    update_branch_references(_window(frames))
    x, y = frames[0].reference
    assert y == pytest.approx(250, abs=0.5)  # on the axis lumen centroid -> arc middle
    assert 330 < x < 380  # between the lumen border and the arc


def test_narrow_and_closed_branches_set_none():
    frames = {0: _frame(20), 1: _frame(), 2: _frame(40, closed=True)}
    assert not update_branch_references(_window(frames))
    assert _referenced(frames) == []


def test_a_hand_placed_reference_keeps_its_run_free_of_automatic_ones():
    frames = {0: _frame(30), 1: _frame(40)}
    frames[0].reference = (1.0, 2.0)
    update_branch_references(_window(frames))
    assert frames[0].reference == (1.0, 2.0) and not frames[0].reference_auto
    assert frames[1].reference is None


def test_an_automatic_reference_goes_when_its_branch_does():
    frames = {0: _frame(40)}
    window = _window(frames)
    update_branch_references(window)
    frames[0].contours.pop('branch')
    assert update_branch_references(window)
    assert frames[0].reference is None and not frames[0].reference_auto


def test_tagged_frames_are_runs_within_their_phase():
    # Diastolic frames 0, 10, 20 neighbour each other in their group; the systolic one is its own run.
    frames = {i: _frame() for i in range(25)}
    frames.update({0: _frame(30, 'D'), 10: _frame(40, 'D'), 20: _frame(35, 'D'), 5: _frame(40, 'S')})
    frames[15] = _frame(40)  # untagged, so ignored once frames are tagged
    update_branch_references(_window(frames))
    assert _referenced(frames) == [5, 10]
