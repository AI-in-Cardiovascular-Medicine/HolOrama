"""Side branches and in-wall structures exported per type for multimodars: frame, x, y, z, contour, closed."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from domain.intravascular.contour_presets import active_preset
from domain.intravascular.io_types import Contour, FrameData
from input_output.output.reports import save_structure_csv
from input_output.output.structure_contours import frame_contours
from tools.branch_arc import trim_outside_lumen

RESOLUTION = 0.02
DIMENSION = 500
SHAPE = (DIMENSION, DIMENSION)
SAMPLES = 200
CENTRE = (250.0, 250.0)


def _circle(r, cx=CENTRE[0], cy=CENTRE[1], n=20, start=0.0, stop=2 * np.pi, endpoint=False):
    a = np.linspace(start, stop, n, endpoint=endpoint)
    return (cx + r * np.cos(a)).tolist(), (cy + r * np.sin(a)).tolist()


LUMEN = _circle(80)
EEM = _circle(150)


def _frame(**structures) -> FrameData:
    contours = {'lumen': Contour(contours=[LUMEN], closed=[True]), 'eem': Contour(contours=[EEM], closed=[True])}
    for key, entries in structures.items():
        contours[key] = Contour(contours=[e for e, _ in entries], closed=[c for _, c in entries])
    return FrameData(phase='D', contours=contours, centroid=CENTRE)


def _contours(frame, key):
    return frame_contours(frame, active_preset()[key], active_preset(), SHAPE, SAMPLES)


def _radii(points):
    return np.hypot(points[:, 0] - CENTRE[0], points[:, 1] - CENTRE[1])


def test_side_branches_are_written_as_drawn_open_arcs_ending_on_the_lumen():
    arc = trim_outside_lumen(*_circle(40, cx=340), *LUMEN, SAMPLES, (3, 40))
    blob = _circle(20, cx=250, cy=420)
    (open_arc, open_closed), (closed_branch, closed_closed) = _contours(
        _frame(branch=[(arc, False), (blob, True)]), 'branch'
    )
    assert (open_closed, closed_closed) == (False, True)
    assert _radii(open_arc[[0, -1]]) == pytest.approx([80, 80], abs=0.5)  # both ends on the lumen border
    assert len(closed_branch) == SAMPLES - 1  # the first point is not repeated at the end
    assert not np.allclose(closed_branch[0], closed_branch[-1])


def test_an_open_plaque_arc_is_closed_against_its_container():
    arc = _circle(100, n=8, stop=np.pi / 2, endpoint=True)  # a quarter arc in the wall
    ((outline, closed),) = _contours(_frame(calcium=[(arc, False)]), 'calcium')
    assert closed
    radii = _radii(outline)
    assert radii.min() == pytest.approx(100, abs=2) and radii.max() == pytest.approx(150, abs=2)  # arc to EEM


def test_a_closed_plaque_is_written_as_drawn():
    nodule = _circle(15, cx=360)
    ((points, closed),) = _contours(_frame(calcium=[(nodule, True)]), 'calcium')
    assert closed and len(points) == SAMPLES - 1
    assert np.hypot(points[:, 0] - 360, points[:, 1] - 250) == pytest.approx(np.full(len(points), 15), abs=0.5)


def test_a_ring_round_the_lumen_is_split_into_two_closed_halves():
    ring = _circle(110)  # a circumferential calcification: the ring from here out to the EEM
    halves = _contours(_frame(calcium=[(ring, True)]), 'calcium')
    assert len(halves) == 2 and all(closed for _, closed in halves)
    for points, _ in halves:
        radii = _radii(points)
        assert radii.min() > 100 and radii.max() < 152  # between the drawn ring and the EEM, not the lumen
    tops = [points[:, 1].mean() < CENTRE[1] for points, _ in halves]
    assert sorted(tops) == [False, True]  # one above the lumen centroid, one below


def test_the_csv_numbers_each_frames_contours_and_skips_types_not_drawn(tmp_path):
    arc = trim_outside_lumen(*_circle(40, cx=340), *LUMEN, SAMPLES, (3, 40))
    frames = {
        0: _frame(branch=[(arc, False)], calcium=[(_circle(15, cx=360), True), (_circle(15, cx=140), True)]),
        1: _frame(),
    }
    window = SimpleNamespace(
        runtime_data=SimpleNamespace(
            frame_data_dct=frames,
            metadata={'resolution': RESOLUTION, 'dimension': DIMENSION, 'pullback_length': np.array([0.5, 1.0])},
            images=np.zeros((2, DIMENSION, DIMENSION)),
        ),
        display=SimpleNamespace(n_points_contour=SAMPLES),
        file_name=str(tmp_path / 'case'),
    )
    save_structure_csv(window, 'diastolic', [0, 1])

    folder = tmp_path / 'case_csv_files'
    assert sorted(p.name for p in folder.iterdir()) == [
        'branch_diastolic_contours.csv',
        'calcium_diastolic_contours.csv',
    ]
    with open(folder / 'calcium_diastolic_contours.csv', newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    assert {len(r) for r in rows} == {6}
    assert {(int(r[0]), int(r[4]), int(r[5])) for r in rows} == {(1, 0, 1), (1, 1, 1)}  # 1-based frame, two contours
    x, y = float(rows[0][1]), float(rows[0][2])
    assert np.hypot(x / RESOLUTION - 360, (DIMENSION * RESOLUTION - y) / RESOLUTION - 250) == pytest.approx(15, abs=0.5)
    with open(folder / 'branch_diastolic_contours.csv', newline='') as f:
        assert {r[5] for r in csv.reader(f, delimiter='\t')} == {'0'}  # the open arc
