"""Angular sectors: the shape behind every ContourType in ANGLE_TYPES (the guide-wire
shadow, the blood artefact). A sector is the wedge between two radial lines through the
image centre, so only the *angles* of its points carry meaning — their distance from the
centre is free, and the app keeps every handle on one circle (see
Display._angle_handle_radius) purely so a sector looks the same in every pullback.

How the stored points describe it
---------------------------------
A sector lives in one entry of a Contour (see FrameData.wire / .blood) as 2-3 points in
original image pixel coordinates:

    [p_start, p_end]            legacy shape: the sector is the *smaller* of the two arcs
                                between the boundaries, which is all the app could draw
                                before sectors could open past 180 degrees
    [p_start, p_end, p_inside]  the sector is whichever arc between p_start and p_end
                                contains p_inside — so any opening from ~0 to ~360
                                degrees is unambiguous

`p_inside` is derived state (the bisector of the sector being drawn), written only so the
opening survives a save/load round trip. Nothing reads it as a boundary. Files written
before it existed keep their original meaning, which is why the legacy shape stays
supported instead of being migrated.

Nothing here touches Qt: the mask rasteriser reads sectors through the same helpers as
the display (see AngleSector in tools.geometry for their drawing).

Angles follow the convention of the rasteriser in input_output.output.imgs_masks:
`atan2(y - centre_y, x - centre_x)`, and a sweep runs in the direction of increasing
angle. Image y grows downwards, so that direction is clockwise on screen; nothing here
depends on which way it looks, only that the whole app agrees.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

TWO_PI = 2 * math.pi

# A sector may be opened up to, but never through, a full turn: at exactly 0 or 2*pi the
# two boundaries coincide and the sector could equally be read as empty or as the whole
# image. Keeping half a degree of clearance on both ends means a sweep never has to be
# stored, drawn or rasterised in that ambiguous state.
MIN_SWEEP = math.radians(0.5)
MAX_SWEEP = TWO_PI - MIN_SWEEP

Coord = Tuple[float, float]


def angle_of(point: Coord, centre: Coord) -> float:
    """Angle of `point` seen from `centre`, in (-pi, pi]."""
    return math.atan2(point[1] - centre[1], point[0] - centre[0])


def point_at(centre: Coord, radius: float, angle: float) -> Coord:
    """The point `radius` away from `centre` in direction `angle`."""
    return (centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle))


def sweep_between(start: float, end: float) -> float:
    """How far `start` has to open, in the direction of increasing angle, to reach `end`.

    Always in [0, 2*pi), so this is the size of one specific arc rather than the smaller
    of the two.
    """
    return (end - start) % TWO_PI


def clamp_sweep(sweep: float) -> float:
    """A sweep pulled inside [MIN_SWEEP, MAX_SWEEP]."""
    return min(max(sweep, MIN_SWEEP), MAX_SWEEP)


def contains_angle(angle: Any, start: float, sweep: float) -> Any:
    """Whether direction `angle` falls inside the sector (`start`, `sweep`).

    `angle` may equally be a numpy array of angles, which is how the mask rasteriser
    tests every pixel of a frame in one go.
    """
    return sweep_between(start, angle) <= sweep


def sector_from_points(points: Sequence[Coord], centre: Coord) -> Tuple[float, float] | None:
    """The (start angle, sweep) the stored `points` describe, or None for fewer than two.

    Handles both shapes documented at the top of this module: with an interior marker the
    sector is the arc that contains it, without one the smaller arc.
    """
    if len(points) < 2:
        return None

    first = angle_of(points[0], centre)
    second = angle_of(points[1], centre)
    forward = sweep_between(first, second)

    if len(points) > 2:
        marker = sweep_between(first, angle_of(points[2], centre))
        if marker <= forward:
            return first, forward
        return second, TWO_PI - forward

    return (first, forward) if forward <= math.pi else (second, TWO_PI - forward)


def points_for_sector(centre: Coord, radius: float, start: float, sweep: float) -> List[Coord]:
    """The three points that store the sector (`start`, `sweep`) unambiguously.

    Both boundaries and the interior marker land on the circle of radius `radius`, so a
    sector edited through the GUI always comes back with its handles where the user
    grabbed them.
    """
    return [
        point_at(centre, radius, start),
        point_at(centre, radius, start + sweep),
        point_at(centre, radius, start + sweep / 2),
    ]


def continue_sweep(previous: float, target: float) -> float:
    """`target` as the next value of a sweep that was `previous`, held back at the seam.

    A boundary dragged past the other one sends the raw sweep the long way round the
    circle — a 5 degree sector would snap to 355 degrees on a pixel of mouse movement, or
    the reverse. Reading a jump of more than half a turn as the user pushing against the
    closed (or fully open) end keeps the drag on the arc they were already editing.
    """
    if previous <= math.pi and target > previous + math.pi:
        return MIN_SWEEP
    if previous > math.pi and target < previous - math.pi:
        return MAX_SWEEP
    return clamp_sweep(target)


def accumulate_sweep(signed_sweep: float, previous_angle: float, angle: float) -> float:
    """The signed sweep after the pointer turned from `previous_angle` to `angle`.

    Summing the short way round on every move is what lets a sector being placed open
    past 180 degrees: the total is tracked rather than re-derived from the two
    boundaries, which on their own can only ever describe the smaller arc. The sign is
    the direction the user turned in, so turning back through the first boundary opens
    the sector the other way instead of jumping to nearly a full turn.
    """
    delta = (angle - previous_angle + math.pi) % TWO_PI - math.pi
    return max(-MAX_SWEEP, min(MAX_SWEEP, signed_sweep + delta))


def signed_to_sector(start: float, signed_sweep: float) -> Tuple[float, float]:
    """A signed sweep about `start` as the equivalent (start angle, positive sweep)."""
    if signed_sweep >= 0:
        return start, signed_sweep
    return start + signed_sweep, -signed_sweep


def edge_point(centre: Coord, angle: float, half_size: float) -> Coord:
    """Where the ray from `centre` in direction `angle` leaves a square of that half-size.

    The boundaries of a sector are radial lines with no end of their own; they are drawn
    out to the image border, as the wire shadow always has been.
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    scale_x = abs(half_size / dx) if dx else math.inf
    scale_y = abs(half_size / dy) if dy else math.inf
    scale = min(scale_x, scale_y)
    if not math.isfinite(scale):
        return centre
    return (centre[0] + scale * dx, centre[1] + scale * dy)
