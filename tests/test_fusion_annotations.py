"""The structures and references of a loaded pullback, and the case files they come from."""

from types import SimpleNamespace

import numpy as np

from fusion_pipeline import annotations as iv


def _point(x, y, z):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def _frame(original, structures=(), reference=None, eem=None):
    return SimpleNamespace(
        lumen=SimpleNamespace(original_frame=original),
        structures=[SimpleNamespace(kind=k, closed=c, points=[_point(*p) for p in pts]) for k, c, pts in structures],
        reference_point=_point(*reference) if reference else None,
        extras={'Eem': SimpleNamespace(points=[_point(*p) for p in eem])} if eem else {},
    )


def _geometry():
    frames = [
        _frame(10, [('calcification', True, [(0, 0, 0), (1, 0, 0)])], reference=(5, 5, 0), eem=[(9, 0, 0)]),
        _frame(11, [('calcification', True, [(0, 1, 0.5)]), ('sidebranch', False, [(3, 3, 0.5)])]),
        _frame(12, reference=(6, 6, 1.0)),
    ]
    return SimpleNamespace(frames=frames, master_reference=12, reference_frames=lambda: [12, 10])


def test_structures_are_pooled_by_kind_over_all_frames():
    points = iv.structure_points(_geometry())
    assert set(points) == {'calcification', 'sidebranch'}
    assert points['calcification'].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0.5]]
    assert iv.structure_points(SimpleNamespace(frames=[SimpleNamespace()])) == {}  # a multimodars without them


def test_references_tell_the_master_from_automatic_and_hand_placed_ones():
    points, status = iv.reference_points(_geometry(), auto_frames={10})
    assert points.tolist() == [[5, 5, 0], [6, 6, 1.0]]
    assert status == ['auto', 'master']
    assert iv.reference_points(_geometry(), auto_frames=set())[1] == ['hand', 'master']
    assert iv.geometry_references(_geometry(), {10}) == [iv.CaseReference(10, True), iv.CaseReference(12, False)]


def test_eem_comes_from_its_extra():
    assert iv.contour_points(_geometry(), 'Eem').tolist() == [[9, 0, 0]]
    assert iv.contour_points(_geometry(), 'Wall').shape == (0, 3)


def test_case_references_with_their_auto_flag(tmp_path):
    (tmp_path / 'systolic_reference_points.csv').write_text('89\t1\t2\t3\t1\n368\t4\t5\t6\t0\n')
    (tmp_path / 'tagged_reference_points.csv').write_text('7\t1\t2\t3\n')  # no auto column, as before
    systolic = iv.read_case_references(str(tmp_path), 'systolic')
    assert systolic == [iv.CaseReference(89, True), iv.CaseReference(368, False)]
    assert iv.read_case_references(str(tmp_path), 'tagged') == [iv.CaseReference(7, False)]
    assert iv.read_case_references(str(tmp_path), 'diastolic') == []


def test_case_structures_skip_pipes_and_other_phases(tmp_path):
    rows = '3\t1.0\t2.0\t0.5\t0\t1\n3\t1.5\t2.5\t0.5\t0\t1\n'
    for name in ('calcium_tagged', 'branch_tagged', 'eem_tagged', 'calcium_diastolic', 'tagged'):
        (tmp_path / f'{name}_contours.csv').write_text(rows)
    structures = iv.read_case_structures(str(tmp_path), 'tagged')
    assert set(structures) == {'calcium', 'branch'}  # not the lumen, the EEM or another phase
    assert np.array_equal(structures['calcium'], [[3, 1.0, 2.0, 0.5, 0, 1], [3, 1.5, 2.5, 0.5, 0, 1]])
    assert iv.read_case_structures(str(tmp_path), 'systolic') is None
