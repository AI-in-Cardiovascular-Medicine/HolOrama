"""Side branches, plaques and reference points exported as multimodars annotations."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from domain.intravascular.io_types import Contour, FrameData
from input_output.output.annotations import pullback_annotations
from input_output.output.reports import save_csv_files
from tools.branch_arc import trim_outside_lumen

RESOLUTION = 0.02  # mm per px, a branch of radius 40 px is 1.6 mm wide
DIMENSION = 500
CENTRE = (250.0, 250.0)


def _circle(r, cx, cy, n=20):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (cx + r * np.cos(a)).tolist(), (cy + r * np.sin(a)).tolist()


LUMEN = _circle(80, *CENTRE)


def _frame(branch_radius=None, phase='D', calcium=False, reference=None, auto=False) -> FrameData:
    contours = {'lumen': Contour(contours=[LUMEN], closed=[True])}
    if branch_radius is not None:
        arc = trim_outside_lumen(*_circle(branch_radius, 340, 250), *LUMEN, 200, (3, 40))
        contours['branch'] = Contour(contours=[arc], closed=[False])
    if calcium:
        contours['calcium'] = Contour(contours=[_circle(15, 180, 180)], closed=[True])
    return FrameData(phase=phase, contours=contours, centroid=CENTRE, reference=reference, reference_auto=auto)


def _window(frames, tmp_path=None):
    return SimpleNamespace(
        runtime_data=SimpleNamespace(
            frame_data_dct=frames,
            metadata={
                'resolution': RESOLUTION,
                'dimension': DIMENSION,
                'pullback_length': np.arange(1, len(frames) + 1) * 0.5,
            },
            images=np.zeros((len(frames), DIMENSION, DIMENSION)),
        ),
        config=SimpleNamespace(intravascular=SimpleNamespace(branch_reference_min_diameter_mm=1.0)),
        display=SimpleNamespace(n_points_contour=200),
        file_name=str(tmp_path / 'case') if tmp_path else '',
    )


def _by(annotations, kind, role):
    return [a for a in annotations if a.kind == kind and a.role == role]


def test_a_branch_run_shares_one_group_and_its_reference_frame_carries_the_sweep_points():
    frames = {0: _frame(30), 1: _frame(40), 2: _frame(35), 3: _frame(), 4: _frame(20)}
    annotations = pullback_annotations(_window(frames), sorted(frames), (DIMENSION, DIMENSION))

    regions = _by(annotations, 'sidebranch', 'region')
    run = {a.frame: a.group for a in regions if a.frame in (0, 1, 2)}
    assert len(set(run.values())) == 1  # one run
    lone = [a for a in regions if a.frame == 4]
    assert len(lone) == 1 and lone[0].group not in run.values()  # narrow, its own group

    for role, count in (('landmark', 1), ('axis_tip', 1), ('diameter_end', 2)):
        (helper,) = _by(annotations, 'sidebranch', role)
        assert helper.frame == 1 and helper.group == run[1]  # the widest frame
        assert len(helper.points) == count
    start, end = _by(annotations, 'sidebranch', 'diameter_end')[0].points
    assert np.hypot(start[0] - end[0], start[1] - end[1]) * RESOLUTION == pytest.approx(1.6, abs=0.05)


def test_a_plaque_is_exported_as_its_region_outline_under_its_multimodars_kind():
    frames = {0: _frame(calcium=True)}
    (outline,) = _by(pullback_annotations(_window(frames), [0], (DIMENSION, DIMENSION)), 'calcification', 'region')
    radii = [np.hypot(x - 180, y - 180) for x, y in outline.points]
    assert min(radii) > 12 and max(radii) < 17  # traced around the drawn circle of radius 15


def test_reference_points_tell_hand_placed_from_automatic():
    frames = {0: _frame(reference=(10.0, 20.0)), 1: _frame(reference=(30.0, 40.0), auto=True)}
    refs = _by(pullback_annotations(_window(frames), [0, 1], (DIMENSION, DIMENSION)), 'reference', 'point')
    assert {(a.frame, a.group, a.points) for a in refs} == {(0, 0, ((10.0, 20.0),)), (1, 1, ((30.0, 40.0),))}


def test_the_csv_writes_every_point_like_the_contours_file(tmp_path):
    frames = {0: _frame(40, reference=(100.0, 50.0)), 1: _frame(calcium=True)}
    window = _window(frames, tmp_path)
    lumen = [list(LUMEN), list(LUMEN)]
    save_csv_files(window, [c[0] for c in lumen], [c[1] for c in lumen], name='diastolic', frames=[0, 1])

    with open(tmp_path / 'case_csv_files' / 'diastolic_annotations.csv', newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    assert {(r[1], r[2]) for r in rows} >= {
        ('sidebranch', 'region'),
        ('sidebranch', 'landmark'),
        ('sidebranch', 'axis_tip'),
        ('sidebranch', 'diameter_end'),
        ('calcification', 'region'),
        ('reference', 'point'),
    }
    (ref,) = [r for r in rows if r[1] == 'reference']
    frame, x, y, z = int(ref[0]), float(ref[4]), float(ref[5]), float(ref[6])
    assert frame == 1  # 1-based, as the contours file
    assert (x, y) == pytest.approx((100 * RESOLUTION, abs(50 * RESOLUTION - DIMENSION * RESOLUTION)))
    assert z == pytest.approx(0.0)  # measured from the first frame
