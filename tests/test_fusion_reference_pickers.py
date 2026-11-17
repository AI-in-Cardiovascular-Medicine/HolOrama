"""The fusion column lets the user pick which reference anchors loading and centerline alignment."""

import pytest

from fusion_pipeline.annotations import CaseReference
from pages.fusion.right_half.intravascular_column import IntravascularColumn


@pytest.fixture
def column(qt_app, tmp_path):
    (tmp_path / 'diastolic_reference_points.csv').write_text('408\t1\t2\t3\t0\n')
    (tmp_path / 'systolic_reference_points.csv').write_text('89\t1\t2\t3\t1\n368\t4\t5\t6\t0\n')
    widget = IntravascularColumn()
    widget.set_input_path(str(tmp_path))
    return widget


def test_each_phase_lists_its_references_and_defaults_to_multimodars_most_proximal(column):
    plan = column.load_plan()
    assert (plan['dia_reference_frame'], plan['sys_reference_frame']) == (None, None)  # multimodars decides
    combo = column._sys_ref_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        'Most proximal (default)',
        'Frame 89 (auto)',
        'Frame 368 (by hand)',
    ]

    combo.setCurrentIndex(1)
    assert column.load_plan()['sys_reference_frame'] == 89


def test_a_restored_plan_selects_its_anchor(column):
    column.set_load_plan({'sys_reference_frame': 89})
    assert column.load_plan()['sys_reference_frame'] == 89


def test_the_centerline_reference_is_only_passed_when_it_differs_from_the_load_anchor(column):
    assert column.align_kwargs()['reference_frame'] is None  # nothing loaded yet
    column.set_pullback_references([CaseReference(48, True), CaseReference(408, False)], anchor=408)
    assert column.align_kwargs()['reference_frame'] is None  # the anchor itself, aligned as before
    column._iv_ref_combo.setCurrentIndex(0)
    assert column.align_kwargs()['reference_frame'] == 48
    assert column.manual_align_kwargs()['reference_frame'] == 48
