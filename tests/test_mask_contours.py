"""Tests for reading a mask back into contours (input_output.input.mask_contours).

The bar is the round trip: contours drawn, painted into a mask, read back and painted again
must give the same mask, label by label (intersection over union of at least MIN_IOU) —
and every contour has to come back as what it was drawn as: an open arc as an open arc,
a sector as a sector, the EEM as a smooth vessel wall rather than one dented by the wire.
"""

import math

import numpy as np
import pytest

from domain.intravascular.types import ContourType
from domain.intravascular.contour_presets import DEFAULT_PRESET_FILE, ContourPreset, load_preset
from domain.intravascular.io_types import Contour, FrameData, iter_sectors, set_sector_points
from input_output.input.mask_contours import frame_contours
from input_output.output.imgs_masks import contours_to_mask
from tools.angle import points_for_sector, sector_from_points

DIM = 400
CX, CY = 196.0, 204.0  # the lumen sits a little off the image centre, as it does
CENTRE = (DIM / 2, DIM / 2)
LUMEN_R, EEM_R = 50.0, 110.0
MIN_IOU = 0.95

LABELS = {'lumen': 1, 'eem': 2, 'calcium': 3, 'lipid': 4, 'macrophage': 5, 'branch': 7, 'wire': 9, 'blood': 10}


def _circle(radius, cx=CX, cy=CY, n=60):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return ([float(cx + radius * math.cos(a)) for a in angles], [float(cy + radius * math.sin(a)) for a in angles])


def _arc(radius, from_deg, to_deg, n=20):
    angles = np.radians(np.linspace(from_deg, to_deg, n))
    return ([float(CX + radius * math.cos(a)) for a in angles], [float(CY + radius * math.sin(a)) for a in angles])


def _sector(from_deg, sweep_deg):
    return points_for_sector(CENTRE, 100.0, math.radians(from_deg), math.radians(sweep_deg))


def _vessel(**contours) -> FrameData:
    frame_data = FrameData(centroid=(CX, CY))
    frame_data.lumen = Contour(contours=[_circle(LUMEN_R)], closed=[True])
    frame_data.eem = Contour(contours=[_circle(EEM_R)], closed=[True])
    for key, contour in contours.items():
        frame_data.contours[key] = contour
    return frame_data


def _closed(*entries):
    return Contour(contours=list(entries), closed=[True] * len(entries))


def _open(*entries):
    return Contour(contours=list(entries), closed=[False] * len(entries))


def _sectors(*sectors):
    contour = Contour()
    for index, points in enumerate(sectors):
        set_sector_points(contour, index, points)
    return contour


def _paint(frame_data, preset):
    return contours_to_mask(np.zeros((1, DIM, DIM), dtype=np.uint8), [0], {0: frame_data}, preset)[0]


N_KNOTS = 20  # n_interactive_points, which every type read off a mask gets


def _knots_for(defn):
    return N_KNOTS


def _read(mask, preset):
    contours, centroid = frame_contours(mask, preset, _knots_for, handle_radius=100.0)
    return FrameData(contours=contours, centroid=centroid)


def _round_trip(frame_data, preset=None):
    preset = preset or load_preset(DEFAULT_PRESET_FILE)
    first = _paint(frame_data, preset)
    read = _read(first, preset)
    return first, read, _paint(read, preset)


def _iou(first, second, label):
    a, b = first == label, second == label
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def _assert_same_mask(first, second):
    for label in sorted(set(np.unique(first)) | set(np.unique(second)) - {0}):
        assert _iou(first, second, label) >= MIN_IOU, f'label {label}: IoU {_iou(first, second, label):.3f}'


SCENES = {
    'lumen and eem': {},
    'wire shadow': {'wire': _sectors(_sector(10, 30))},
    'two blood sectors and a wire': {
        'blood': _sectors(_sector(100, 60), _sector(250, 40)),
        'wire': _sectors(_sector(-20, 25)),
    },
    'a sector wider than half a turn': {'blood': _sectors(_sector(30, 250))},
    'an open calcium arc': {'calcium': _open(_arc(LUMEN_R + 8, 40, 120))},
    'a closed lipid pool in the wall': {
        'lipid': _closed(_circle(14, cx=CX + 80 * math.cos(3.6), cy=CY + 80 * math.sin(3.6)))
    },
    'a ring of calcium round the lumen': {'calcium': _closed(_circle(LUMEN_R + 12))},
    'a calcium arc the wire cuts in two': {
        'calcium': _open(_arc(LUMEN_R + 6, -40, 70)),
        'wire': _sectors(_sector(5, 20)),
    },
    'a side branch under the lumen': {'branch': _closed(_circle(35, cx=CX + 70, cy=CY))},
    'everything at once': {
        'calcium': _open(_arc(LUMEN_R + 8, 200, 260)),
        'lipid': _closed(_circle(12, cx=CX + 80 * math.cos(1.2), cy=CY + 80 * math.sin(1.2))),
        'macrophage': _open(_arc(LUMEN_R + 10, 120, 160)),
        'branch': _closed(_circle(30, cx=CX - 130, cy=CY + 20)),
        'wire': _sectors(_sector(300, 25)),
        'blood': _sectors(_sector(20, 50)),
    },
}


@pytest.mark.parametrize('scene', list(SCENES))
def test_the_mask_survives_the_round_trip(scene):
    first, _, second = _round_trip(_vessel(**SCENES[scene]))
    _assert_same_mask(first, second)


class TestWhatComesBack:
    def test_the_eem_is_not_dented_by_the_wire(self):
        _, read, _ = _round_trip(_vessel(wire=_sectors(_sector(10, 40))))
        xs, ys = read.eem.contours[0]
        radii = np.hypot(np.array(xs) - CX, np.array(ys) - CY)
        assert radii.min() > EEM_R - 4  # every knot on the vessel wall, none down at the lumen

    def test_an_arc_comes_back_open_with_its_ends(self):
        _, read, _ = _round_trip(_vessel(calcium=_open(_arc(LUMEN_R + 8, 40, 120))))
        calcium = read.contour('calcium')
        assert calcium.closed == [False]
        xs, ys = calcium.contours[0]
        ends = [math.degrees(math.atan2(ys[i] - CY, xs[i] - CX)) for i in (0, -1)]
        assert sorted(ends) == pytest.approx([40, 120], abs=3)

    def test_a_pool_in_the_wall_comes_back_closed(self):
        lipid = _closed(_circle(14, cx=CX + 80, cy=CY))
        _, read, _ = _round_trip(_vessel(lipid=lipid))
        assert read.contour('lipid').closed == [True]

    def test_a_ring_comes_back_as_one_closed_ring(self):
        _, read, _ = _round_trip(_vessel(calcium=_closed(_circle(LUMEN_R + 12))))
        assert read.contour('calcium').closed == [True]
        xs, ys = read.contour('calcium').contours[0]
        assert np.hypot(np.array(xs) - CX, np.array(ys) - CY).mean() == pytest.approx(LUMEN_R + 12, abs=3)

    def test_an_arc_the_wire_cuts_stays_one_arc(self):
        _, read, _ = _round_trip(_vessel(calcium=_open(_arc(LUMEN_R + 6, -40, 70)), wire=_sectors(_sector(5, 20))))
        assert read.contour('calcium').closed == [False]

    def test_sectors_come_back_with_their_angles(self):
        _, read, _ = _round_trip(_vessel(blood=_sectors(_sector(100, 60), _sector(30 + 180, 250 - 180))))
        found = sorted(
            (round(math.degrees(start)) % 360, round(math.degrees(sweep)))
            for start, sweep in (sector_from_points(points, CENTRE) for points in iter_sectors(read.contour('blood')))
        )
        assert [start for start, _ in found] == pytest.approx([100, 210], abs=2)
        assert [sweep for _, sweep in found] == pytest.approx([60, 70], abs=2)

    def test_a_wide_sector_keeps_its_side(self):
        _, read, _ = _round_trip(_vessel(blood=_sectors(_sector(30, 250))))
        (points,) = iter_sectors(read.contour('blood'))
        assert math.degrees(sector_from_points(points, CENTRE)[1]) == pytest.approx(250, abs=2)

    def test_noise_is_ignored(self):
        preset = load_preset(DEFAULT_PRESET_FILE)
        mask = _paint(_vessel(), preset)
        mask[10:13, 10:13] = LABELS['lipid']  # 9 stray pixels
        assert 'lipid' not in _read(mask, preset).contours

    def test_without_an_eem_only_the_lumen_comes_back(self):
        frame_data = FrameData(centroid=(CX, CY))
        frame_data.lumen = Contour(contours=[_circle(LUMEN_R)], closed=[True])
        _, read, second = _round_trip(frame_data)
        assert {key for key, contour in read.contours.items() if contour.contours} == {'lumen'}


def _lobed(radius, lobes, depth, n=200):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = radius * (1 + depth * np.sin(lobes * angles))
    return (
        [float(CX + r * math.cos(a)) for r, a in zip(radii, angles)],
        [float(CY + r * math.sin(a)) for r, a in zip(radii, angles)],
    )


@pytest.mark.parametrize('lobes, depth', [(3, 0.2), (4, 0.3), (7, 0.12)])
def test_a_ragged_contour_comes_back_with_n_interactive_points_knots(lobes, depth):
    """However much the outline bends, a contour read off a mask edits like a drawn one:
    as many knots as its type gets, the spline fitted by where they go."""
    frame_data = _vessel(lipid=_closed(_circle(14, cx=CX + 80, cy=CY)))
    frame_data.lumen = Contour(contours=[_lobed(LUMEN_R, lobes, depth)], closed=[True])
    first, read, second = _round_trip(frame_data)
    assert len(read.lumen.contours[0][0]) == N_KNOTS
    assert len(read.eem.contours[0][0]) == N_KNOTS
    assert len(read.contour('lipid').contours[0][0]) == N_KNOTS
    assert _iou(first, second, LABELS['lumen']) >= MIN_IOU


def test_a_thrombus_in_the_lumen_of_a_custom_preset():
    raw = load_preset(DEFAULT_PRESET_FILE).to_dict()
    raw['types'].append(
        {
            'id': 'thrombus',
            'name': 'Thrombus',
            'label': 13,
            'color': 'red',
            'tools': 'closed',
            'layer': 8,
            'inside': 'lumen',
        }
    )
    preset = ContourPreset.from_dict(raw)
    first, read, second = _round_trip(_vessel(thrombus=_closed(_circle(15, cx=CX + 20, cy=CY))), preset)
    _assert_same_mask(first, second)
    assert read.contour('thrombus').closed == [True]
    assert ContourType.LUMEN.value in read.contours


class TestLabelsThePresetLacks:
    def test_the_draft_adds_a_free_row_per_missing_label(self):
        from domain.intravascular.contour_presets import ToolSet, with_labels

        preset = load_preset(DEFAULT_PRESET_FILE)
        draft = with_labels(preset, [1, 2, 15, 16], 'Default + 2 labels')
        added = draft.types[len(preset.types) :]
        assert [(d.label, d.name, d.tools) for d in added] == [
            (15, 'Label 15', ToolSet.CLOSED),
            (16, 'Label 16', ToolSet.CLOSED),
        ]
        assert len({d.color for d in draft.types}) == len(draft.types)  # every colour its own

    def test_in_the_dialog_the_added_rows_can_still_be_anything(self, qt_app):
        from domain.intravascular.contour_presets import with_labels
        from pages.intravascular.popup_windows.contour_settings_dialog import _COL_TOOLS, ContourSettingsDialog

        draft = with_labels(load_preset(DEFAULT_PRESET_FILE), [15], 'Mask')
        dialog = ContourSettingsDialog(active_name='Default', draft=draft, library=[])
        rows = dialog._current.rows
        assert rows[-1].id is None and rows[2].id == 'calcium'
        assert dialog._table.cellWidget(len(rows) - 1, _COL_TOOLS).count() == 3  # closed, open, angle

    def _ask(self, monkeypatch, answer, values):
        from input_output.input import image

        asked = []
        monkeypatch.setattr(image.QMessageBox, 'question', lambda *args, **kwargs: asked.append(args[2]) or answer)
        mask = np.zeros((2, 8, 8), dtype=np.int32)
        mask.flat[: len(values)] = values
        return image._preset_for_mask(None, mask), asked

    def test_a_mask_the_preset_covers_asks_nothing(self, monkeypatch):
        preset, asked = self._ask(monkeypatch, None, [1, 2, 9])
        assert preset is not None and asked == []

    def test_declining_reads_with_the_active_preset(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        preset, asked = self._ask(monkeypatch, QMessageBox.StandardButton.No, [1, 2, 15])
        assert preset.name == 'Default' and '15' in asked[0]

    def test_cancel_gives_up_the_import(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        preset, _ = self._ask(monkeypatch, QMessageBox.StandardButton.Cancel, [1, 15])
        assert preset is None


def test_the_whole_import_is_one_undo_step():
    from domain.intravascular.runtime_types import RuntimeData
    from domain.intravascular.undo import PullbackContoursSnapshot, push_pullback_contours_snapshot

    runtime = RuntimeData()
    runtime.frame_data_dct = {0: _vessel(), 1: _vessel()}
    push_pullback_contours_snapshot(runtime, 0)
    runtime.frame_data_dct[1].lumen.contours = []
    snapshot = runtime.contour_undo.pop()
    assert isinstance(snapshot, PullbackContoursSnapshot)
    assert snapshot.frames[1][0]['lumen'].contours  # the copy kept what was there


def test_a_volume_stored_with_a_trailing_axis_of_one_reads_as_frames(tmp_path):
    """Some converters write a 3-D volume as (x, y, z, 1) — a one-component vector."""
    import nibabel as nib

    from input_output.input.image import _read_nifti

    volume = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6, 1)
    path = tmp_path / 'volume.nii.gz'
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))
    pixels, _ = _read_nifti(str(path))
    assert pixels.shape == (6, 5, 4)  # frames, rows, columns


class TestFibrousCap:
    """OCT masks label the fibrous cap (6) on its own, between the lumen and the lipid."""

    def _capped_lipid_mask(self, preset):
        mask = _paint(_vessel(lipid=_open(_arc(LUMEN_R - 2, 40, 120))), preset)
        rows, cols = np.mgrid[0:DIM, 0:DIM]
        r = np.hypot(cols - CX, rows - CY)
        mask[(mask == LABELS['lipid']) & (r < LUMEN_R + 6)] = 6  # a 6 px cap over the pool
        return mask

    def test_the_lipid_arc_runs_along_the_far_side_of_the_cap(self):
        preset = load_preset(DEFAULT_PRESET_FILE)
        read = _read(self._capped_lipid_mask(preset), preset)
        lipid = read.contour('lipid')
        assert lipid.closed == [False]
        xs, ys = lipid.contours[0]
        assert np.hypot(np.array(xs) - CX, np.array(ys) - CY).mean() == pytest.approx(LUMEN_R + 6, abs=1.5)

    def test_the_cap_is_read_as_vessel_wall(self):
        preset = load_preset(DEFAULT_PRESET_FILE)
        mask = self._capped_lipid_mask(preset)
        again = _paint(_read(mask, preset), preset)
        cap = mask == 6
        assert np.mean(again[cap] == LABELS['eem']) > 0.9
        assert _iou(np.where(cap, LABELS['eem'], mask), again, LABELS['lipid']) >= MIN_IOU

    def test_a_preset_with_a_type_on_label_6_reads_it_as_that(self):
        from domain.intravascular.contour_presets import with_labels

        preset = with_labels(load_preset(DEFAULT_PRESET_FILE), [6], 'With a cap type')
        read = _read(self._capped_lipid_mask(preset), preset)
        assert read.contour('label_6').contours  # its own type, not part of the lipid

    def test_the_cap_is_no_label_without_a_contour_type(self, monkeypatch):
        from input_output.input import image

        monkeypatch.setattr(image.QMessageBox, 'question', lambda *a, **k: pytest.fail('asked about the cap'))
        mask = self._capped_lipid_mask(load_preset(DEFAULT_PRESET_FILE))[np.newaxis]
        assert image._preset_for_mask(None, mask).name == 'Default'
