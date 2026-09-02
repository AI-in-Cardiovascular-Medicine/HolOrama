"""Tests for what a plaque contour means once rasterized
(input_output.output.imgs_masks._plaque_mask, and the two callers that read it).

Every plaque contour marks the luminal side of the plaque, which then fills outwards to
the EEM. An open arc can only ever mean that; a closed contour is normally the plaque
itself, filled in, *except* when it was drawn right around the lumen — a circumferential
calcification — where filling the disc would read as plaque from the lumen out to the
ring, the opposite of what was drawn.

The frames here are concentric circles, so every expected region is an annulus whose area
is known in closed form.
"""

import math

import numpy as np
import pytest

from domain.all_types import ContourType
from domain.io_types import Contour, FrameData
from input_output.output.imgs_masks import contours_to_mask, frame_region_metrics

DIM = 240
CENTRE = DIM / 2
RESOLUTION = 0.02  # mm/pixel
LUMEN_R, RING_R, EEM_R = 25.0, 45.0, 70.0

LABELS = {'lumen': 1, 'eem': 2, 'calcium': 3, 'branch': 7}


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


def _frame(calcium=None, closed=True, lumen_r=LUMEN_R, eem_r=EEM_R):
    frame_data = FrameData()
    frame_data.lumen = Contour(contours=[_circle(lumen_r)], closed=[True])
    if eem_r is not None:
        frame_data.eem = Contour(contours=[_circle(eem_r)], closed=[True])
    if calcium is not None:
        frame_data.calcium = Contour(contours=[calcium], closed=[closed])
    frame_data.centroid = (CENTRE, CENTRE)
    return frame_data


def _mask(frame_data):
    images = np.zeros((1, DIM, DIM), dtype=np.uint8)
    return contours_to_mask(images, [0], {0: frame_data})[0]


def _annulus_px(inner_r, outer_r):
    return math.pi * (outer_r**2 - inner_r**2)


class TestClosedPlaqueAroundTheLumen:
    """A 360° closed calcium contour: the wall outside it, not the annulus inside it."""

    def test_it_fills_outwards_to_the_eem(self):
        mask = _mask(_frame(calcium=_circle(RING_R)))
        assert (mask == LABELS['calcium']).sum() == pytest.approx(_annulus_px(RING_R, EEM_R), rel=0.05)

    def test_it_does_not_fill_inwards_to_the_lumen(self):
        # The annulus between the lumen and the ring is what filling the disc produced.
        mask = _mask(_frame(calcium=_circle(RING_R)))
        yy, xx = np.nonzero(mask == LABELS['calcium'])
        assert np.hypot(yy - CENTRE, xx - CENTRE).min() >= RING_R - 2

    def test_the_wall_left_over_is_eem(self):
        mask = _mask(_frame(calcium=_circle(RING_R)))
        assert (mask == LABELS['eem']).sum() == pytest.approx(_annulus_px(LUMEN_R, RING_R), rel=0.05)

    def test_the_lumen_is_untouched(self):
        mask = _mask(_frame(calcium=_circle(RING_R)))
        assert (mask == LABELS['lumen']).sum() == pytest.approx(math.pi * LUMEN_R**2, rel=0.05)

    def test_frame_region_metrics_agrees_with_the_mask(self):
        frame_data = _frame(calcium=_circle(RING_R))
        areas = frame_region_metrics(frame_data, (DIM, DIM), RESOLUTION)
        expected = _annulus_px(RING_R, EEM_R) * RESOLUTION**2
        assert areas['calcium'] == pytest.approx(expected, rel=0.05)
        assert areas['calcium'] <= areas['wall']

    def test_a_ring_that_cuts_inside_the_lumen_here_and_there_still_counts(self):
        """Drawn by hand along the lumen border, so it wanders in and out of it."""
        angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        wobble = [LUMEN_R - 4 if i % 4 == 0 else LUMEN_R + 8 for i in range(len(angles))]
        ring = (
            [float(CENTRE + r * math.cos(a)) for r, a in zip(wobble, angles)],
            [float(CENTRE + r * math.sin(a)) for r, a in zip(wobble, angles)],
        )
        mask = _mask(_frame(calcium=ring))
        # Nearly the whole wall is calcium, and the leftover EEM backdrop is a thin rim.
        wall = _annulus_px(LUMEN_R, EEM_R)
        assert (mask == LABELS['calcium']).sum() > 0.8 * wall


class TestClosedPlaqueInTheWall:
    """The ordinary case is unchanged: a closed contour drawn in the wall is the plaque."""

    def _blob(self):
        # A small closed contour sitting in the wall, clear of the lumen.
        blob_centre = CENTRE + (LUMEN_R + EEM_R) / 2
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return (
            [float(blob_centre + 10 * math.cos(a)) for a in angles],
            [float(CENTRE + 10 * math.sin(a)) for a in angles],
        )

    def test_the_disc_is_filled(self):
        mask = _mask(_frame(calcium=self._blob()))
        assert (mask == LABELS['calcium']).sum() == pytest.approx(math.pi * 10**2, rel=0.1)

    def test_it_stays_inside_the_wall(self):
        mask = _mask(_frame(calcium=self._blob()))
        calcium = mask == LABELS['calcium']
        yy, xx = np.nonzero(calcium)
        radii = np.hypot(yy - CENTRE, xx - CENTRE)
        assert radii.min() >= LUMEN_R - 1
        assert radii.max() <= EEM_R + 1


class TestOpenPlaque:
    """An open arc has always meant the wall outside it; that is the reading being shared."""

    def test_the_sector_outside_the_arc_is_filled(self):
        mask = _mask(_frame(calcium=_arc(RING_R, 0, 90), closed=False))
        quarter = _annulus_px(RING_R, EEM_R) / 4
        assert (mask == LABELS['calcium']).sum() == pytest.approx(quarter, rel=0.15)

    def test_nothing_lands_between_the_arc_and_the_lumen(self):
        mask = _mask(_frame(calcium=_arc(RING_R, 0, 90), closed=False))
        yy, xx = np.nonzero(mask == LABELS['calcium'])
        assert np.hypot(yy - CENTRE, xx - CENTRE).min() >= RING_R - 2


class TestWithoutAnEem:
    """No EEM means nothing to fill outwards to, so the disc is used either way."""

    def test_a_ring_falls_back_to_the_annulus_inside_it(self):
        mask = _mask(_frame(calcium=_circle(RING_R), eem_r=None))
        assert (mask == LABELS['calcium']).sum() == pytest.approx(_annulus_px(LUMEN_R, RING_R), rel=0.05)

    def test_areas_fall_back_the_same_way(self):
        frame_data = _frame(calcium=_circle(RING_R), eem_r=None)
        areas = frame_region_metrics(frame_data, (DIM, DIM), RESOLUTION)
        assert areas['calcium'] == pytest.approx(_annulus_px(LUMEN_R, RING_R) * RESOLUTION**2, rel=0.05)


class TestWithoutALumen:
    def test_a_ring_is_filled_as_drawn(self):
        frame_data = FrameData()
        frame_data.eem = Contour(contours=[_circle(EEM_R)], closed=[True])
        frame_data.calcium = Contour(contours=[_circle(RING_R)], closed=[True])
        frame_data.centroid = (CENTRE, CENTRE)
        mask = _mask(frame_data)
        # Nothing says the lumen is inside it, so the contour is taken at face value.
        assert (mask == LABELS['calcium']).sum() == pytest.approx(math.pi * RING_R**2, rel=0.05)


class TestSeveralContours:
    def test_a_ring_and_a_wall_blob_are_read_one_by_one(self):
        frame_data = _frame()
        frame_data.calcium = Contour(contours=[_circle(RING_R), _arc(RING_R, 180, 260)], closed=[True, False])
        mask = _mask(frame_data)
        calcium = mask == LABELS['calcium']
        # The ring alone already covers the wall outside it; the arc adds nothing new
        # there, so the total stays that annulus rather than doubling up.
        assert calcium.sum() == pytest.approx(_annulus_px(RING_R, EEM_R), rel=0.05)
        assert ContourType.CALCIUM.value in frame_region_metrics(frame_data, (DIM, DIM), RESOLUTION)


class TestPlaqueAngle:
    """How much of the circle a plaque covers, measured off the same mask as its area."""

    def _angle(self, frame_data):
        return frame_region_metrics(frame_data, (DIM, DIM), RESOLUTION)['calcium_angle']

    def test_no_plaque_spans_nothing(self):
        assert self._angle(_frame()) == 0.0

    @pytest.mark.parametrize('opening', [30, 90, 180, 270])
    def test_an_open_arc_spans_its_own_opening(self, opening):
        angle = self._angle(_frame(calcium=_arc(RING_R, 0, opening), closed=False))
        assert angle == pytest.approx(opening, abs=4)

    def test_a_ring_around_the_lumen_spans_the_whole_circle(self):
        assert self._angle(_frame(calcium=_circle(RING_R))) == pytest.approx(360, abs=2)

    def test_a_wall_blob_spans_only_what_it_covers(self):
        blob_centre = CENTRE + (LUMEN_R + EEM_R) / 2
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        blob = (
            [float(blob_centre + 10 * math.cos(a)) for a in angles],
            [float(CENTRE + 10 * math.sin(a)) for a in angles],
        )
        # A 10 px disc centred 47 px out subtends roughly 2*asin(10/47) about the centre.
        expected = math.degrees(2 * math.asin(10 / (blob_centre - CENTRE)))
        assert self._angle(_frame(calcium=blob)) == pytest.approx(expected, abs=6)

    def test_two_arcs_count_their_shared_degrees_once(self):
        frame_data = _frame()
        frame_data.calcium = Contour(contours=[_arc(RING_R, 0, 90), _arc(RING_R, 45, 135)], closed=[False, False])
        assert self._angle(frame_data) == pytest.approx(135, abs=5)

    def test_an_arc_across_the_wrap_point_is_not_split(self):
        angle = self._angle(_frame(calcium=_arc(RING_R, -45, 45), closed=False))
        assert angle == pytest.approx(90, abs=4)
