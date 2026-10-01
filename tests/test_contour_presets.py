"""Tests for the contour presets (domain.contour_presets): which contour types exist, how
each is layered into the mask and read back out of it, and what a preset has to satisfy.

The default preset has to reproduce what the types used to be hardcoded as — the same
mask labels and keyboard shortcuts — so every file and mask made before keeps its meaning.
"""

import copy
import json
import math

import numpy as np
import pytest

from domain.all_types import ContourType, SegmentationTool
from domain.contour_presets import (
    DEFAULT_PRESET_FILE,
    ContourPreset,
    PresetError,
    active_preset,
    load_preset,
    set_active_preset,
)
from domain.io_types import Contour, FrameData, frame_to_dict
from input_output.input.contours import _build_frame_data, _sector_keys
from input_output.output.imgs_masks import contours_to_mask, frame_region_metrics

DIM = 240
CENTRE = DIM / 2
RESOLUTION = 0.02  # mm/pixel
LUMEN_R, EEM_R = 40.0, 80.0


def _circle(radius, cx=CENTRE, cy=CENTRE, n=40):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (
        [float(cx + radius * math.cos(a)) for a in angles],
        [float(cy + radius * math.sin(a)) for a in angles],
    )


def _default_raw() -> dict:
    with open(DEFAULT_PRESET_FILE, encoding='utf-8') as f:
        return json.load(f)


def _with_row(raw: dict, row: dict) -> dict:
    raw = copy.deepcopy(raw)
    raw['types'].append(row)
    return raw


@pytest.fixture
def restore_active_preset():
    before = active_preset()
    yield
    set_active_preset(before)


class TestDefaultPreset:
    """The default preset is what the contour types were before they became data."""

    def test_it_loads_and_is_active_by_default(self):
        assert load_preset(DEFAULT_PRESET_FILE) == active_preset()

    def test_the_mask_labels_are_unchanged(self):
        preset = active_preset()
        labels = {defn.type.value: defn.label for defn in preset.types}
        assert labels == {
            'lumen': 1,
            'eem': 2,
            'calcium': 3,
            'lipid': 4,
            'macrophage': 5,
            'branch': 7,
            'wire': 9,
            'blood': 10,
        }

    def test_the_keyboard_shortcuts_are_unchanged(self):
        preset = active_preset()
        shortcuts = {defn.type.value: preset.shortcuts(defn.type) for defn in preset.types}
        assert shortcuts == {
            'lumen': ('E', None),
            'eem': ('Q', None),
            'calcium': ('7', 'Ctrl+7'),
            'branch': ('8', 'Ctrl+8'),
            'lipid': ('9', 'Ctrl+9'),
            'macrophage': ('0', 'Ctrl+0'),
            'wire': ('3', 'Ctrl+3'),
            'blood': ('B', 'Ctrl+B'),
        }

    def test_the_tools_are_unchanged(self):
        preset = active_preset()
        spline = {SegmentationTool.CLOSED_SPLINE, SegmentationTool.BRUSH}
        assert preset.allowed_tools(ContourType.LUMEN) == spline
        assert preset.allowed_tools('branch') == spline
        assert preset.allowed_tools('calcium') == spline | {SegmentationTool.OPEN_SPLINE}
        assert preset.allowed_tools('wire') == {SegmentationTool.ANGLE}

    def test_blood_is_the_bottom_and_the_wire_just_under_the_lumen(self):
        order = [defn.type.value for defn in active_preset().paint_order()]
        assert order[:2] == ['blood', 'eem']
        assert order[-2:] == ['wire', 'lumen']

    def test_the_plaques_lie_in_the_eem(self):
        preset = active_preset()
        assert [defn.type.value for defn in preset.contents(ContourType.EEM)] == ['calcium', 'lipid', 'macrophage']

    def test_an_eem_is_read_back_with_everything_painted_over_it(self):
        assert active_preset().mask_labels(ContourType.EEM) == {1, 2, 3, 4, 5}
        assert active_preset().mask_labels('calcium') == {3}

    def test_it_survives_a_round_trip(self):
        preset = active_preset()
        assert ContourPreset.from_dict(json.loads(json.dumps(preset.to_dict()))) == preset


class TestValidation:
    @pytest.mark.parametrize(
        'mutate, message',
        [
            (lambda raw: raw['types'].reverse(), 'lumen has to be the first'),
            (lambda raw: raw['types'].pop(1), 'EEM has to be the second'),
            (lambda raw: raw['types'][2].update(label=2), 'mask label'),
            (lambda raw: raw['types'][2].update(layer=1), 'layer'),
            (lambda raw: raw['types'][2].update(inside=None), 'needs a type to lie in'),
            (lambda raw: raw['types'][2].update(layer=-5), 'layered above'),
            (lambda raw: raw['types'][0].update(layer=-3), 'Lumen has to be layered above EEM'),
            (lambda raw: raw['types'][0].update(tools='open_closed'), 'closed contour'),
            (lambda raw: raw['types'][2].update(inside='wire'), 'angle'),
            (lambda raw: raw['types'][2].update(inside='thrombus'), 'lacks'),
            (lambda raw: raw['types'][2].update(id='phase'), 'reserved'),
            (lambda raw: raw['types'][2].update(id='Calcium'), 'valid id'),
            (lambda raw: raw['types'][2].update(label=300), 'between 1 and 255'),
            (lambda raw: raw['types'][6].update(inside='eem'), 'angle'),
        ],
    )
    def test_a_broken_rule_is_named(self, mutate, message):
        raw = _default_raw()
        mutate(raw)
        with pytest.raises(PresetError, match=message):
            ContourPreset.from_dict(raw)

    def test_two_types_cannot_lie_inside_each_other(self):
        raw = _with_row(
            _default_raw(),
            {'id': 'a', 'name': 'A', 'label': 20, 'color': 'red', 'tools': 'closed', 'layer': 20, 'inside': 'b'},
        )
        raw = _with_row(
            raw, {'id': 'b', 'name': 'B', 'label': 21, 'color': 'red', 'tools': 'closed', 'layer': 21, 'inside': 'a'}
        )
        raw['types'][0]['layer'] = 99
        with pytest.raises(PresetError, match='inside each other|layered above'):
            ContourPreset.from_dict(raw)

    def test_a_newer_format_is_refused(self):
        raw = _default_raw()
        raw['format'] = 99
        with pytest.raises(PresetError, match='newer'):
            ContourPreset.from_dict(raw)


# A plaque type a user added, and a thrombus that lies in the lumen rather than the wall.
_PLAQUE = {'id': 'fibrous', 'name': 'Fibrous', 'label': 12, 'color': 'pink', 'tools': 'open_closed', 'layer': 8}
_THROMBUS = {'id': 'thrombus', 'name': 'Thrombus', 'label': 13, 'color': 'red', 'tools': 'closed', 'layer': 9}


def _custom_preset() -> ContourPreset:
    raw = _with_row(_default_raw(), {**_PLAQUE, 'inside': 'eem'})
    raw = _with_row(raw, {**_THROMBUS, 'inside': 'lumen'})  # layered above the lumen (7)
    return ContourPreset.from_dict(raw)


def _vessel(**contours) -> FrameData:
    frame_data = FrameData(centroid=(CENTRE, CENTRE))
    frame_data.lumen = Contour(contours=[_circle(LUMEN_R)], closed=[True])
    frame_data.eem = Contour(contours=[_circle(EEM_R)], closed=[True])
    for key, contour in contours.items():
        frame_data.contours[key] = contour
    return frame_data


def _mask(frame_data, preset):
    return contours_to_mask(np.zeros((1, DIM, DIM), dtype=np.uint8), [0], {0: frame_data}, preset)[0]


class TestCustomTypes:
    def test_a_type_inside_the_lumen_is_painted_over_it(self):
        preset = _custom_preset()
        order = [defn.type.value for defn in preset.paint_order()]
        assert order.index('thrombus') > order.index('lumen')

        thrombus = Contour(contours=[_circle(15.0)], closed=[True])
        mask = _mask(_vessel(thrombus=thrombus), preset)
        assert (mask == 13).sum() == pytest.approx(math.pi * 15.0**2, rel=0.05)
        assert (mask == 1).sum() == pytest.approx(math.pi * (LUMEN_R**2 - 15.0**2), rel=0.05)

    def test_a_type_inside_the_lumen_has_to_be_layered_above_it(self):
        raw = _with_row(_default_raw(), {**_THROMBUS, 'inside': 'lumen', 'layer': -2})
        with pytest.raises(PresetError, match='Thrombus has to be layered above Lumen'):
            ContourPreset.from_dict(raw)

    def test_a_type_above_the_lumen_covers_it_even_when_it_lies_in_nothing(self):
        stent = {'id': 'stent', 'name': 'Stent', 'label': 14, 'color': 'grey', 'tools': 'closed', 'layer': 8}
        preset = ContourPreset.from_dict(_with_row(_default_raw(), stent))
        disc = Contour(contours=[_circle(10.0)], closed=[True])
        mask = _mask(_vessel(stent=disc), preset)
        assert (mask == 14).sum() == pytest.approx(math.pi * 10.0**2, rel=0.05)

    def test_a_type_inside_the_lumen_stays_in_it(self):
        thrombus = Contour(contours=[_circle(15.0, cx=CENTRE + 35.0)], closed=[True])  # half out of the lumen
        mask = _mask(_vessel(thrombus=thrombus), _custom_preset())
        ys, xs = np.nonzero(mask == 13)
        assert np.hypot(xs - CENTRE, ys - CENTRE).max() <= LUMEN_R + 1

    def test_a_new_plaque_is_clipped_to_the_wall_like_the_built_in_ones(self):
        blob = Contour(contours=[_circle(20.0, cx=CENTRE + 60.0)], closed=[True])  # straddles the EEM
        preset = _custom_preset()
        frame_data = _vessel(fibrous=blob)
        mask = _mask(frame_data, preset)
        ys, xs = np.nonzero(mask == 12)
        radii = np.hypot(xs - CENTRE, ys - CENTRE)
        assert radii.min() >= LUMEN_R - 1 and radii.max() <= EEM_R + 1

        areas = frame_region_metrics(frame_data, (DIM, DIM), RESOLUTION, preset=preset)
        assert 0 < areas['fibrous'] < areas['wall']
        assert areas['fibrous'] == pytest.approx((mask == 12).sum() * RESOLUTION**2, rel=0.02)

    def test_an_eem_read_back_includes_the_new_plaque(self):
        assert 12 in _custom_preset().mask_labels(ContourType.EEM)
        assert 13 not in _custom_preset().mask_labels(ContourType.EEM) - _custom_preset().mask_labels('lumen')


class TestContourFiles:
    def test_a_custom_type_survives_saving_and_loading(self, restore_active_preset):
        preset = _custom_preset()
        set_active_preset(preset)
        frame_data = _vessel(fibrous=Contour(contours=[_circle(50.0)], closed=[False]))
        raw = {'0': frame_to_dict(frame_data), 'contour_types': preset.to_dict()}

        raw = json.loads(json.dumps(raw))
        loaded = _build_frame_data(raw, pre_flags=False, sector_keys=_sector_keys(raw))[0]

        assert loaded.contour('fibrous').contours == [tuple(_circle(50.0))]
        assert loaded.contour('fibrous').closed == [False]

    def test_contours_are_saved_next_to_the_frame_fields(self):
        raw = frame_to_dict(_vessel())
        assert 'contours' not in raw
        assert {'lumen', 'eem', 'phase', 'reference'} <= set(raw)

    def test_a_file_s_own_sector_types_load_as_sectors(self):
        raw = _with_row(
            _default_raw(),
            {'id': 'shadow', 'name': 'Shadow', 'label': 30, 'color': 'grey', 'tools': 'angle', 'layer': -1},
        )
        assert 'shadow' in _sector_keys({'contour_types': raw})

    def test_a_file_without_a_preset_has_the_old_sector_types(self):
        assert {'wire', 'blood'} <= _sector_keys({})

    def test_a_contour_of_a_type_the_preset_lacks_is_kept(self):
        raw = {'0': {'phase': '-', 'thrombus': {'contours': [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], 'closed': [True]}}}
        loaded = _build_frame_data(raw, pre_flags=False, sector_keys=_sector_keys(raw))[0]
        assert loaded.contour('thrombus').contours == [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]
        assert 'thrombus' in frame_to_dict(loaded)


class TestFrameData:
    def test_a_type_not_drawn_yet_hands_out_an_empty_contour_that_sticks(self):
        frame_data = FrameData()
        frame_data.contour('calcium').contours.append(([1.0], [2.0]))
        assert frame_data.contours['calcium'].contours == [([1.0], [2.0])]

    def test_the_measurements_and_the_reference_are_no_contours(self):
        with pytest.raises(KeyError):
            FrameData().contour(ContourType.REFERENCE)
