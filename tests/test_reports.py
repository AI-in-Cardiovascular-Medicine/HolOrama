"""Tests for the per-frame report (input_output.output.reports).

One row per contoured frame, holding everything drawn on it: the lumen and EEM metrics,
each plaque as an area and an angle, blood as one combined angle, the two hand
measurements, and the pullback's own parameters as the last columns.
"""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from domain.io_types import Contour, FrameData, Measure, set_sector_points
from input_output.output.reports import report
from tools.angle import points_for_sector

DIM = 200
CENTRE = DIM / 2
RESOLUTION = 0.02  # mm/pixel
N_FRAMES = 4
LUMEN_R, PLAQUE_R, EEM_R = 25.0, 45.0, 70.0


def _circle(radius, n=40):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (
        [float(CENTRE + radius * math.cos(a)) for a in angles],
        [float(CENTRE + radius * math.sin(a)) for a in angles],
    )


def _arc(radius, from_deg, to_deg, n=20):
    angles = np.linspace(math.radians(from_deg), math.radians(to_deg), n)
    return (
        [float(CENTRE + radius * math.cos(a)) for a in angles],
        [float(CENTRE + radius * math.sin(a)) for a in angles],
    )


def _sector(from_deg, sweep_deg):
    """One angular sector stored the way the display stores it."""
    return points_for_sector((CENTRE, CENTRE), 60.0, math.radians(from_deg), math.radians(sweep_deg))


@pytest.fixture
def main_window(tmp_path):
    """A stub window with a lumen and an EEM on every frame, and nothing else."""
    frames = {}
    for i in range(N_FRAMES):
        frame_data = FrameData(phase='D' if i % 2 else 'S')
        frame_data.lumen = Contour(contours=[_circle(LUMEN_R)], closed=[True])
        frame_data.eem = Contour(contours=[_circle(EEM_R)], closed=[True])
        frames[i] = frame_data

    runtime = SimpleNamespace(
        frame_data_dct=frames,
        images=np.zeros((N_FRAMES, DIM, DIM), dtype=np.uint8),
        metadata={
            'num_frames': N_FRAMES,
            'resolution': RESOLUTION,
            'dimension': DIM,
            'modality': 'OCT',
            'pullback_speed': 1.0,
            'pullback_start_frame': 0,
            'frame_rate': 30.0,
            'pullback_length': 20.0,
        },
        gated_frames_dia=[],
        gated_frames_sys=[],
        tagged_frames=[],
    )

    def get_full_contour_list(contour_type):
        """Stands in for Display.get_full_contour_list: the frame's contour per frame."""
        out = []
        for i in range(N_FRAMES):
            contour_obj = getattr(frames[i], contour_type.value)
            entry = contour_obj.contours[0] if contour_obj.contours else None
            out.append((np.array(entry[0]), np.array(entry[1])) if entry else None)
        return out

    return SimpleNamespace(
        image_displayed=True,
        runtime_data=runtime,
        display=SimpleNamespace(get_full_contour_list=get_full_contour_list),
        config=SimpleNamespace(report=SimpleNamespace(save_as_csv=False, plot=False)),
        file_name=str(tmp_path / 'case'),
    )


def _report(main_window, **kwargs):
    kwargs.setdefault('suppress_messages', True)
    kwargs.setdefault('write_files', False)
    data = report(main_window, **kwargs)
    assert data is not None
    return data


class TestItRunsAtAll:
    """_full_list was called with two of its three arguments, so every report raised."""

    def test_a_frame_per_contoured_frame(self, main_window):
        data = _report(main_window)
        assert list(data['frame']) == [1, 2, 3, 4]

    def test_the_lumen_metrics_are_there(self, main_window):
        data = _report(main_window)
        expected_area = math.pi * (LUMEN_R * RESOLUTION) ** 2
        assert data['lumen_area'].iloc[0] == pytest.approx(expected_area, rel=0.02)
        assert data['eem_area'].iloc[0] == pytest.approx(math.pi * (EEM_R * RESOLUTION) ** 2, rel=0.02)


class TestColumns:
    def test_every_plaque_is_reported_as_an_area_and_an_angle(self, main_window):
        data = _report(main_window)
        for plaque in ('calcium', 'lipid', 'macrophage'):
            assert f'{plaque}_area' in data.columns
            assert f'{plaque}_angle' in data.columns

    def test_blood_is_reported_but_the_wire_is_not(self, main_window):
        data = _report(main_window)
        assert 'blood_angle' in data.columns
        assert not [column for column in data.columns if 'wire' in column]

    def test_the_measurements_and_the_pullback_parameters_come_last(self, main_window):
        data = _report(main_window)
        assert list(data.columns[-7:]) == [
            'measurement_1',
            'measurement_2',
            'modality',
            'pullback_speed',
            'pullback_start_frame',
            'frame_rate',
            'resolution',
        ]

    def test_the_plaque_columns_sit_before_the_measurements(self, main_window):
        columns = list(_report(main_window).columns)
        assert columns.index('calcium_angle') < columns.index('measurement_1')
        assert columns.index('blood_angle') < columns.index('measurement_1')


class TestPlaqueAreaAndAngle:
    def test_an_open_arc_spans_its_own_angle(self, main_window):
        main_window.runtime_data.frame_data_dct[0].calcium = Contour(contours=[_arc(PLAQUE_R, 0, 90)], closed=[False])
        data = _report(main_window)

        assert data['calcium_angle'].iloc[0] == pytest.approx(90, abs=4)
        quarter_annulus = math.pi * (EEM_R**2 - PLAQUE_R**2) / 4
        assert data['calcium_area'].iloc[0] == pytest.approx(quarter_annulus * RESOLUTION**2, rel=0.15)

    def test_a_ring_around_the_lumen_spans_the_whole_circle(self, main_window):
        main_window.runtime_data.frame_data_dct[0].calcium = Contour(contours=[_circle(PLAQUE_R)], closed=[True])
        data = _report(main_window)

        assert data['calcium_angle'].iloc[0] == pytest.approx(360, abs=2)

    def test_two_arcs_of_the_same_type_count_their_overlap_once(self, main_window):
        main_window.runtime_data.frame_data_dct[0].lipid = Contour(
            contours=[_arc(PLAQUE_R, 0, 90), _arc(PLAQUE_R, 45, 135)], closed=[False, False]
        )
        data = _report(main_window)

        assert data['lipid_angle'].iloc[0] == pytest.approx(135, abs=5)

    def test_each_plaque_type_is_measured_on_its_own(self, main_window):
        frame_data = main_window.runtime_data.frame_data_dct[1]
        frame_data.calcium = Contour(contours=[_arc(PLAQUE_R, 0, 60)], closed=[False])
        frame_data.macrophage = Contour(contours=[_arc(PLAQUE_R, 180, 300)], closed=[False])
        data = _report(main_window)

        assert data['calcium_angle'].iloc[1] == pytest.approx(60, abs=4)
        assert data['macrophage_angle'].iloc[1] == pytest.approx(120, abs=4)
        assert data['lipid_angle'].iloc[1] == 0.0

    def test_a_frame_without_plaques_reports_zeros(self, main_window):
        data = _report(main_window)
        for plaque in ('calcium', 'lipid', 'macrophage'):
            assert list(data[f'{plaque}_area']) == [0.0] * N_FRAMES
            assert list(data[f'{plaque}_angle']) == [0.0] * N_FRAMES


class TestBloodAngle:
    def test_one_sector_is_its_own_opening(self, main_window):
        set_sector_points(main_window.runtime_data.frame_data_dct[0].blood, 0, _sector(10, 120))
        data = _report(main_window)

        assert data['blood_angle'].iloc[0] == pytest.approx(120, abs=0.5)

    def test_two_sectors_are_combined(self, main_window):
        blood = main_window.runtime_data.frame_data_dct[0].blood
        set_sector_points(blood, 0, _sector(0, 60))
        set_sector_points(blood, 1, _sector(180, 90))
        data = _report(main_window)

        assert data['blood_angle'].iloc[0] == pytest.approx(150, abs=0.5)

    def test_overlapping_sectors_are_counted_once(self, main_window):
        blood = main_window.runtime_data.frame_data_dct[0].blood
        set_sector_points(blood, 0, _sector(0, 90))
        set_sector_points(blood, 1, _sector(45, 90))
        data = _report(main_window)

        assert data['blood_angle'].iloc[0] == pytest.approx(135, abs=0.5)

    def test_a_wire_on_the_frame_changes_nothing(self, main_window):
        set_sector_points(main_window.runtime_data.frame_data_dct[0].wire, 0, _sector(0, 120))
        data = _report(main_window)

        assert data['blood_angle'].iloc[0] == 0.0

    def test_a_frame_without_blood_reports_zero(self, main_window):
        assert list(_report(main_window)['blood_angle']) == [0.0] * N_FRAMES


class TestMeasurements:
    def test_both_measurements_are_reported(self, main_window):
        frame_data = main_window.runtime_data.frame_data_dct[2]
        frame_data.measurement_1 = Measure(points=((1.0, 2.0), (3.0, 4.0)), length=1.25)
        frame_data.measurement_2 = Measure(points=((5.0, 6.0), (7.0, 8.0)), length=2.5)
        data = _report(main_window)

        assert data['measurement_1'].iloc[2] == 1.25
        assert data['measurement_2'].iloc[2] == 2.5

    def test_a_frame_without_them_reports_nothing(self, main_window):
        data = _report(main_window)
        assert data['measurement_1'].isna().all()


class TestPullbackParameters:
    def test_they_are_on_every_row(self, main_window):
        data = _report(main_window)
        assert list(data['pullback_speed']) == [1.0] * N_FRAMES
        assert list(data['frame_rate']) == [30.0] * N_FRAMES
        assert list(data['modality']) == ['OCT'] * N_FRAMES

    def test_a_missing_one_does_not_stop_the_report(self, main_window):
        main_window.runtime_data.metadata['frame_rate'] = None
        data = _report(main_window)
        assert data['frame_rate'].isna().all()


class TestWrittenFile:
    def test_it_writes_a_csv_and_no_txt(self, main_window):
        _report(main_window, write_files=True)

        assert os.path.exists(main_window.file_name + '_report.csv')
        assert not os.path.exists(main_window.file_name + '_report.txt')

    def test_the_csv_is_comma_separated_with_the_columns_as_its_header(self, main_window):
        data = _report(main_window, write_files=True)

        with open(main_window.file_name + '_report.csv', encoding='utf-8') as report_file:
            header = report_file.readline().strip()
        assert header.split(',') == list(data.columns)
