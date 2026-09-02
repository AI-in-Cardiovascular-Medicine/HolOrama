"""Tests for the angular-sector geometry shared by the wire and blood contour types
(tools.angle).

The point of the interior marker is that a sector can open past 180 degrees at all: two
boundary points on their own only ever describe the smaller of the two arcs between them,
which is what the app was limited to before. These tests pin both readings down, since
contour files written either way have to keep meaning what they said.
"""

import math

import pytest

from tools.angle import (
    MAX_SWEEP,
    MIN_SWEEP,
    TWO_PI,
    accumulate_sweep,
    angle_of,
    clamp_sweep,
    contains_angle,
    continue_sweep,
    edge_point,
    point_at,
    points_for_sector,
    sector_from_points,
    signed_to_sector,
    sweep_between,
)

CENTRE = (100.0, 100.0)
RADIUS = 50.0


def _at(degrees: float, radius: float = RADIUS):
    return point_at(CENTRE, radius, math.radians(degrees))


class TestAngles:
    def test_angle_of_matches_the_rasteriser_convention(self):
        # atan2(y - cy, x - cx): +x is 0, and growing angle turns towards +y.
        assert angle_of((150.0, 100.0), CENTRE) == pytest.approx(0.0)
        assert angle_of((100.0, 150.0), CENTRE) == pytest.approx(math.pi / 2)
        assert angle_of((100.0, 50.0), CENTRE) == pytest.approx(-math.pi / 2)

    def test_point_at_round_trips_through_angle_of(self):
        for degrees in (0, 37, 90, 179, 180, 270, 359):
            read_back = angle_of(_at(degrees), CENTRE)
            assert math.cos(read_back - math.radians(degrees)) == pytest.approx(1.0)  # equal modulo a full turn

    def test_sweep_between_is_one_specific_arc(self):
        assert sweep_between(0.0, math.pi / 2) == pytest.approx(math.pi / 2)
        # The other way round is the complement, not the same smaller arc.
        assert sweep_between(math.pi / 2, 0.0) == pytest.approx(1.5 * math.pi)

    def test_contains_angle(self):
        start, sweep = math.radians(350), math.radians(30)  # wraps past zero
        assert contains_angle(math.radians(355), start, sweep)
        assert contains_angle(math.radians(10), start, sweep)
        assert not contains_angle(math.radians(30), start, sweep)


class TestSectorFromPoints:
    def test_fewer_than_two_points_is_no_sector(self):
        assert sector_from_points([], CENTRE) is None
        assert sector_from_points([_at(0)], CENTRE) is None

    def test_two_points_give_the_smaller_arc(self):
        # The legacy shape: whichever order they are stored in, the sector is the 90
        # degrees between them, never the 270 on the other side.
        for points in ([_at(0), _at(90)], [_at(90), _at(0)]):
            start, sweep = sector_from_points(points, CENTRE)
            assert sweep == pytest.approx(math.pi / 2)
            assert contains_angle(math.radians(45), start, sweep)

    def test_two_points_never_exceed_half_a_turn(self):
        _, sweep = sector_from_points([_at(0), _at(300)], CENTRE)
        assert sweep == pytest.approx(math.radians(60))

    def test_marker_picks_the_arc_it_sits_in(self):
        small = sector_from_points([_at(0), _at(90), _at(45)], CENTRE)
        assert small[1] == pytest.approx(math.pi / 2)
        assert contains_angle(math.radians(45), *small)

        # Same two boundaries, marker on the other side: the sector is the big arc.
        big = sector_from_points([_at(0), _at(90), _at(200)], CENTRE)
        assert big[1] == pytest.approx(math.radians(270))
        assert contains_angle(math.radians(200), *big)
        assert not contains_angle(math.radians(45), *big)

    def test_marker_radius_is_irrelevant(self):
        near = sector_from_points([_at(0), _at(90), _at(200, radius=5.0)], CENTRE)
        far = sector_from_points([_at(0), _at(90), _at(200, radius=380.0)], CENTRE)
        assert near[0] == pytest.approx(far[0])
        assert near[1] == pytest.approx(far[1])

    @pytest.mark.parametrize('degrees', [1, 45, 90, 179, 181, 270, 355])
    def test_points_for_sector_round_trip(self, degrees):
        start, sweep = math.radians(20), math.radians(degrees)
        points = points_for_sector(CENTRE, RADIUS, start, sweep)
        read_start, read_sweep = sector_from_points(points, CENTRE)
        assert read_sweep == pytest.approx(sweep)
        assert math.cos(read_start - start) == pytest.approx(1.0)  # equal modulo a full turn

    def test_points_for_sector_puts_everything_on_the_handle_circle(self):
        for x, y in points_for_sector(CENTRE, RADIUS, 0.3, math.radians(250)):
            assert math.hypot(x - CENTRE[0], y - CENTRE[1]) == pytest.approx(RADIUS)


class TestSweepContinuity:
    def test_clamp_sweep_keeps_clear_of_both_seams(self):
        assert clamp_sweep(0.0) == MIN_SWEEP
        assert clamp_sweep(TWO_PI) == MAX_SWEEP
        assert clamp_sweep(1.0) == 1.0

    def test_ordinary_drag_passes_through(self):
        assert continue_sweep(1.0, 1.2) == pytest.approx(1.2)
        assert continue_sweep(1.0, 0.8) == pytest.approx(0.8)

    def test_opening_past_half_a_turn_is_not_a_seam_crossing(self):
        opened = continue_sweep(math.radians(178), math.radians(184))
        assert opened == pytest.approx(math.radians(184))

    def test_dragging_a_boundary_past_the_other_stops_at_the_seam(self):
        # A 5 degree sector whose end handle is pulled back past the start would
        # otherwise read as 355 degrees.
        assert continue_sweep(math.radians(5), math.radians(355)) == MIN_SWEEP
        # And the reverse, for a sector that is nearly a full turn.
        assert continue_sweep(math.radians(355), math.radians(5)) == MAX_SWEEP


class TestPlacementAccumulation:
    def test_turning_one_way_keeps_adding_up(self):
        sweep = 0.0
        angle = 0.0
        for degrees in range(0, 250, 10):  # the pointer turns a long way round
            sweep = accumulate_sweep(sweep, angle, math.radians(degrees))
            angle = math.radians(degrees)
        assert sweep == pytest.approx(math.radians(240))

    def test_crossing_the_wrap_point_does_not_jump(self):
        # 350 -> 10 degrees is 20 degrees of movement, not 340 backwards.
        assert accumulate_sweep(0.0, math.radians(350), math.radians(10)) == pytest.approx(math.radians(20))

    def test_turning_back_reverses_the_opening(self):
        sweep = accumulate_sweep(0.0, 0.0, math.radians(30))
        sweep = accumulate_sweep(sweep, math.radians(30), math.radians(-20))
        assert sweep == pytest.approx(math.radians(-20))

    def test_never_opens_beyond_a_full_turn(self):
        sweep = 0.0
        angle = 0.0
        for degrees in range(0, 720, 10):
            sweep = accumulate_sweep(sweep, angle, math.radians(degrees))
            angle = math.radians(degrees)
        assert sweep == pytest.approx(MAX_SWEEP)

    def test_signed_to_sector_flips_the_start_for_the_other_direction(self):
        assert signed_to_sector(1.0, 0.5) == (1.0, 0.5)
        start, sweep = signed_to_sector(1.0, -0.5)
        assert (start, sweep) == pytest.approx((0.5, 0.5))

    def test_a_reversed_placement_stores_the_same_sector(self):
        # Turning 60 degrees anticlockwise from 200 must describe the same wedge as
        # turning 60 clockwise from 140.
        forward = points_for_sector(CENTRE, RADIUS, *signed_to_sector(math.radians(140), math.radians(60)))
        backward = points_for_sector(CENTRE, RADIUS, *signed_to_sector(math.radians(200), math.radians(-60)))
        assert sector_from_points(forward, CENTRE) == pytest.approx(sector_from_points(backward, CENTRE))


class TestEdgePoint:
    def test_ray_stops_on_the_border_of_the_square(self):
        assert edge_point((400.0, 400.0), 0.0, 400.0) == pytest.approx((800.0, 400.0))
        assert edge_point((400.0, 400.0), math.pi / 2, 400.0) == pytest.approx((400.0, 800.0))

    def test_diagonal_ray_stops_on_the_nearer_side(self):
        x, y = edge_point((400.0, 400.0), math.radians(45), 400.0)
        assert (x, y) == pytest.approx((800.0, 800.0))
