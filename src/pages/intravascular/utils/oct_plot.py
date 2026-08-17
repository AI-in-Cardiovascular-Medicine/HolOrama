"""Schematic pullback overview for OCT.

Fills the slot the interactive gating plot occupies for IVUS (see
right_half._build_oct): OCT has no cardiac gating to plot there, so instead of a
per-frame signal this condenses the whole pullback into one idealized vessel seen
from the side, letting the user tell at a glance where the vessel narrows and what
the wall is made of.

What is drawn, from the centre outwards:

* the imaging catheter, as two grey lines at +/- its radius (config
  `display.catheter_diameter`, default 0.9 mm = a 2.7 F OCT catheter),
* the lumen, as two lines at +/- half the lumen's *shortest distance* (the
  narrowest diameter, `lumen.measurements.minor_axis`),
* the EEM, as two lines at +/- the radius of a circle of the same area as the EEM
  contour, with the space between lumen and EEM filled in tissue red -> that band
  is the plaque,
* a calcium strip along the top and a lipid strip along the bottom, coloured by
  each frame's mask area as a fraction of the plaque area (EEM - lumen): whiter =
  more calcium, yellower = more lipid.

Both radii are idealized: the vessel is drawn as if it were a circular tube, so a
frame's eccentricity is not visible here (that is what the cross-section on the
left is for). Because the two radii come from different measurements (shortest
distance vs. equal-area), a very eccentric lumen can compute a larger radius than
its EEM; the EEM radius is floored at the lumen radius so the wall band never
turns inside out.

Frames the user has not contoured carry no measurement, so their values are
linearly interpolated from the nearest contoured frames on either side, and each
boundary line is dotted (with a faded fill) wherever it is interpolated: solid
means measured on that frame, dotted means inferred, and the dots on the lumen
line mark the frames the numbers actually come from. Nothing is drawn before the
first or after the last contoured frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from input_output.output.imgs_masks import frame_region_areas
from tools.geometry import SplineGeometry

CATHETER_DIAMETER_MM = 0.9  # 2.7 F imaging catheter; overridden by config.display.catheter_diameter

BACKGROUND = QColor('#0c0e11')
LUMEN_FILL = QColor('#1d0f12')  # the "hole" down the middle: blood, near black but not the background
TISSUE_OUTER = QColor('#b0564a')  # adventitial side of the plaque band
TISSUE_INNER = QColor('#6b2a24')  # luminal side of the plaque band
CATHETER_GREY = QColor('#9aa0a6')
AXIS_GREY = QColor('#5f656d')
LABEL_GREY = QColor('#aab0b8')
MARKER_COLOR = QColor('#ffd24d')
PILL_BACKDROP = QColor(0, 0, 0, 150)  # behind text that may land on a bright strip
INTERPOLATED_ALPHA = 150  # plaque fill alpha where the stretch was interpolated

# fraction of plaque area -> strip colour
CALCIUM_RAMP = ((0.0, QColor('#24272c')), (0.55, QColor('#8f959d')), (1.0, QColor('#ffffff')))
LIPID_RAMP = ((0.0, QColor('#2a2418')), (0.55, QColor('#ab8220')), (1.0, QColor('#ffe066')))
STRIP_MIN_SCALE = 0.25  # smallest full-scale value of a strip, so faint plaque still shows
MASK_GRID_PX = 250  # grid the plaque masks are measured on, see _downsample
MASK_BUDGET_S = 0.15  # mask work per event-loop slice (see refresh)
MASK_RESUME_MS = 15  # pause between slices, to keep the UI responsive

LEFT_MARGIN = 34
RIGHT_MARGIN = 10
TOP_MARGIN = 3
BOTTOM_MARGIN = 15
READOUT_HEIGHT = 14
STRIP_HEIGHT = 13
STRIP_GAP = 3


@dataclass
class _FrameMetrics:
    """One contoured frame's idealized radii (mm) and plaque composition (fractions)."""

    lumen_r: float
    eem_r: float | None
    calcium: float  # NaN when unknown (no EEM contour -> no plaque area to relate to)
    lipid: float
    complete: bool = True  # False while this frame's plaque masks are still outstanding


@dataclass
class _Profile:
    """Per-frame arrays over the whole pullback; NaN outside the contoured span."""

    n_frames: int
    lumen_r: np.ndarray
    eem_r: np.ndarray
    calcium: np.ndarray
    lipid: np.ndarray
    lumen_measured: np.ndarray  # frame has its own lumen contour
    eem_measured: np.ndarray  # frame has its own EEM contour
    strip_measured: np.ndarray  # frame has its own plaque composition
    r_max: float
    calcium_scale: float
    lipid_scale: float


def _contour_signature(frame_data) -> tuple:
    """Cheap fingerprint of the contours this plot reads, to skip unchanged frames.

    Coordinate sums catch knot drags that leave the point count untouched.
    """
    signature: list = []
    for key in ('lumen', 'eem', 'calcium', 'lipid'):
        for entry in getattr(frame_data, key).contours:
            xs = entry[0] if entry else []
            ys = entry[1] if len(entry) > 1 else []
            signature.append((len(xs), round(float(sum(xs)), 3), round(float(sum(ys)), 3)))
        signature.append(key)  # separator, so two calcium contours != one calcium + one lipid
    measurements = frame_data.lumen.measurements
    signature += [measurements.minor_axis, measurements.area]
    return tuple(signature)


def _spline_area_mm2(contour_obj, resolution: float, n_points: int) -> float | None:
    """Area (mm²) of a closed contour's first entry, interpolated as the display draws it."""
    if not contour_obj.contours or not contour_obj.contours[0]:
        return None
    entry = contour_obj.contours[0]
    xs = list(entry[0])
    ys = list(entry[1]) if len(entry) > 1 else []
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    try:
        x_full, y_full = SplineGeometry(xs, ys, n_points, None, None).interpolate()
        # shoelace formula on the closed polygon
        area_px = 0.5 * abs(float(np.dot(x_full, np.roll(y_full, -1)) - np.dot(y_full, np.roll(x_full, -1))))
    except Exception as exc:
        logger.debug(f'OCT plot: contour area failed: {exc}')
        return None
    return area_px * resolution**2


def _ramp_color(ramp, position: float) -> QColor:
    """Colour at *position* (0..1) along a ramp of (position, QColor) stops."""
    position = min(max(position, 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(ramp, ramp[1:]):
        if position <= p1:
            f = 0.0 if p1 == p0 else (position - p0) / (p1 - p0)
            return QColor(
                round(c0.red() + f * (c1.red() - c0.red())),
                round(c0.green() + f * (c1.green() - c0.green())),
                round(c0.blue() + f * (c1.blue() - c0.blue())),
            )
    return QColor(ramp[-1][1])


def _interpolate(points: dict[int, float], n_frames: int) -> np.ndarray:
    """Spread sparse per-frame values over all frames; NaN outside their span.

    Values are bridged only *between* known frames, never extrapolated, so the
    schematic stops where the contours stop instead of inventing a vessel there.
    """
    out = np.full(n_frames, np.nan)
    frames = np.array([f for f, v in sorted(points.items()) if v is not None and not math.isnan(v)], dtype=int)
    if frames.size == 0:
        return out
    values = np.array([points[int(f)] for f in frames], dtype=float)
    x = np.arange(n_frames)
    inside = (x >= frames[0]) & (x <= frames[-1])
    out[inside] = np.interp(x[inside], frames, values)
    return out


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) spans of consecutive True entries."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(flags) - 1))
    return spans


def _segments(measured: np.ndarray, values: np.ndarray) -> list[tuple[list[int], bool]]:
    """Stretches to draw: runs of consecutive measured frames, and the bridges between.

    Each entry is (frames, is_measured); bridges join the last frame of one run to
    the first of the next and are drawn dotted.
    """
    spans = _runs(measured & ~np.isnan(values))
    segments: list[tuple[list[int], bool]] = []
    for index, (start, end) in enumerate(spans):
        segments.append((list(range(start, end + 1)), True))
        if index + 1 < len(spans):
            segments.append(([end, spans[index + 1][0]], False))
    return segments


def _downsample(image_shape) -> int:
    """Rasterize the plaque masks on a ~MASK_GRID_PX grid instead of the full frame.

    Only the plaque/wall *ratio* is plotted, and rasterizing a 1024² frame per
    contoured frame costs a few hundred ms — two orders of magnitude more than
    everything else this widget does. A ~250 px grid keeps the fractions within a
    few tenths of a percent (measured against full-resolution rasterization) while
    costing roughly an order of magnitude less.
    """
    return max(1, round(min(image_shape) / MASK_GRID_PX))


def _strip_scale(values: np.ndarray) -> float:
    """Full-scale value of a strip: the largest fraction seen, at least STRIP_MIN_SCALE.

    Scaling to the pullback keeps a mildly diseased vessel readable instead of
    uniformly dark; each strip prints its own range so the colours stay quantitative.
    """
    if values.size == 0 or np.all(np.isnan(values)):
        return STRIP_MIN_SCALE
    return max(STRIP_MIN_SCALE, math.ceil(float(np.nanmax(values)) * 20) / 20)


class OCTPlot(QWidget):
    """Idealized longitudinal schematic of a whole OCT pullback (see module docstring)."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(110)
        self.setToolTip(
            'Idealized vessel over the pullback: lumen and EEM as circular tubes, '
            'plaque in between, calcium (top) and lipid (bottom) as a fraction of the '
            'plaque area. Dotted = interpolated between contoured frames. Click to jump '
            'to a frame.'
        )

        self._profile: _Profile | None = None
        self._frame: int = 0
        self._cache: dict[int, tuple[tuple, _FrameMetrics | None]] = {}
        self._deadline: float = 0.0
        self._incomplete: bool = False
        self._resume_scheduled: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the profile from the current contours and repaint.

        Cheap to call repeatedly: per-frame results are cached against a contour
        fingerprint, so unchanged frames are never measured twice.

        Rasterizing the plaque masks is the one expensive part (tens of ms per
        contoured frame), so it runs under a time budget: the vessel itself is always
        drawn immediately and the calcium/lipid strips fill in over as many
        event-loop slices as it takes, leaving the UI responsive on a pullback with
        plaque labelled on hundreds of frames.
        """
        self._deadline = time.perf_counter() + MASK_BUDGET_S
        try:
            self._profile = self._build_profile()
        except Exception as exc:
            logger.warning(f'OCT plot: could not build pullback profile: {exc}')
            self._profile = None
            self._incomplete = False
        self.update()

        if self._incomplete and not self._resume_scheduled:
            self._resume_scheduled = True
            QTimer.singleShot(MASK_RESUME_MS, self._resume_build)

    def _resume_build(self) -> None:
        self._resume_scheduled = False
        if self._incomplete:  # reset() clears the flag and so cancels a running build
            self.refresh()

    def set_frame(self, frame: int) -> None:
        """Move the current-frame marker (driven by the display slider)."""
        if int(frame) != self._frame:
            self._frame = int(frame)
            self.update()

    def reset(self) -> None:
        """Forget everything and cancel any build still in progress."""
        self._profile = None
        self._cache.clear()
        self._incomplete = False
        self.update()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _catheter_radius(self) -> float:
        diameter = getattr(self.main_window.config.display, 'catheter_diameter', CATHETER_DIAMETER_MM)
        return float(diameter) / 2.0

    def _frame_metrics(self, frame: int, frame_data, image_shape, resolution: float) -> _FrameMetrics | None:
        signature = _contour_signature(frame_data)
        cached = self._cache.get(frame)
        if cached is not None and cached[0] == signature:
            return cached[1]
        # Radii are cheap and always computed; the plaque masks only while there is
        # budget left in this slice (refresh() comes back for the rest).
        metrics = self._compute_frame_metrics(
            frame_data, image_shape, resolution, allow_masks=time.perf_counter() < self._deadline
        )
        if metrics is None or metrics.complete:
            self._cache[frame] = (signature, metrics)
        else:
            self._incomplete = True
        return metrics

    def _compute_frame_metrics(
        self, frame_data, image_shape, resolution: float, allow_masks: bool = True
    ) -> _FrameMetrics | None:
        measurements = frame_data.lumen.measurements
        if measurements.minor_axis:
            lumen_r = measurements.minor_axis / 2.0
        elif measurements.area:
            lumen_r = math.sqrt(measurements.area / math.pi)  # shortest distance not computed yet
        else:
            return None
        if lumen_r <= 0:
            return None

        eem_area = _spline_area_mm2(frame_data.eem, resolution, self.main_window.display.n_points_contour)
        eem_r = max(math.sqrt(eem_area / math.pi), lumen_r) if eem_area else None

        calcium = lipid = float('nan')
        if eem_r is not None:
            if frame_data.calcium.contours or frame_data.lipid.contours:
                if not allow_masks:
                    return _FrameMetrics(lumen_r=lumen_r, eem_r=eem_r, calcium=calcium, lipid=lipid, complete=False)
                areas = frame_region_areas(frame_data, image_shape, resolution, downsample=_downsample(image_shape))
                wall = areas['wall']
                if wall > 0:
                    calcium = min(areas['calcium'] / wall, 1.0)
                    lipid = min(areas['lipid'] / wall, 1.0)
            else:
                calcium = lipid = 0.0  # EEM drawn but no plaque labelled -> genuinely zero

        return _FrameMetrics(lumen_r=lumen_r, eem_r=eem_r, calcium=calcium, lipid=lipid)

    def _build_profile(self) -> _Profile | None:
        runtime = self.main_window.runtime_data
        frame_data_dct = runtime.frame_data_dct or {}
        images = runtime.images
        if not frame_data_dct or images is None:
            return None

        n_frames = runtime.metadata.get('num_frames') or images.shape[0]
        resolution = runtime.metadata.get('resolution')
        if not n_frames or not resolution:
            return None
        image_shape = images.shape[1:3]

        lumen_points: dict[int, float] = {}
        eem_points: dict[int, float] = {}
        calcium_points: dict[int, float] = {}
        lipid_points: dict[int, float] = {}
        lumen_measured = np.zeros(n_frames, dtype=bool)
        eem_measured = np.zeros(n_frames, dtype=bool)
        strip_measured = np.zeros(n_frames, dtype=bool)
        self._incomplete = False

        for frame, frame_data in frame_data_dct.items():
            if frame_data is None or not frame_data.lumen.contours or not 0 <= frame < n_frames:
                continue
            metrics = self._frame_metrics(frame, frame_data, image_shape, resolution)
            if metrics is None:
                continue
            lumen_measured[frame] = True
            lumen_points[frame] = metrics.lumen_r
            if metrics.eem_r is not None:
                eem_measured[frame] = True
                eem_points[frame] = metrics.eem_r
            if not math.isnan(metrics.calcium):
                strip_measured[frame] = True
                calcium_points[frame] = metrics.calcium
                lipid_points[frame] = metrics.lipid

        if not lumen_points:
            return None

        # Drop measurements of frames that no longer exist (a new file was loaded).
        for stale in set(self._cache) - set(frame_data_dct):
            del self._cache[stale]

        lumen_r = _interpolate(lumen_points, n_frames)
        eem_r = _interpolate(eem_points, n_frames)
        calcium = _interpolate(calcium_points, n_frames)
        lipid = _interpolate(lipid_points, n_frames)

        r_max = float(np.nanmax(np.concatenate([lumen_r, eem_r, [self._catheter_radius()]])))
        return _Profile(
            n_frames=int(n_frames),
            lumen_r=lumen_r,
            eem_r=eem_r,
            calcium=calcium,
            lipid=lipid,
            lumen_measured=lumen_measured,
            eem_measured=eem_measured,
            strip_measured=strip_measured,
            r_max=r_max if r_max > 0 else 1.0,
            calcium_scale=_strip_scale(calcium),
            lipid_scale=_strip_scale(lipid),
        )

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)

        profile = self._profile
        if profile is None:
            painter.setPen(QPen(LABEL_GREY))
            painter.drawText(
                self.rect(),
                int(Qt.AlignmentFlag.AlignCenter),
                'Vessel overview — draw or segment a lumen contour to populate',
            )
            painter.end()
            return

        x0 = float(LEFT_MARGIN)
        x1 = float(max(self.width() - RIGHT_MARGIN, LEFT_MARGIN + 1))
        top = float(TOP_MARGIN + READOUT_HEIGHT)
        bottom = float(max(self.height() - BOTTOM_MARGIN, top + 1))
        half = (bottom - top) / 2.0
        y_center = top + half
        vessel_half = max(half - STRIP_HEIGHT - STRIP_GAP, 4.0)
        scale = vessel_half / (profile.r_max * 1.04)

        span = max(profile.n_frames - 1, 1)

        def x_of(frame: float) -> float:
            return x0 + (x1 - x0) * frame / span

        def y_of(radius: float) -> float:
            return y_center - radius * scale

        self._paint_vessel(painter, profile, x_of, y_of)
        self._paint_catheter(painter, x0, x1, y_of)
        self._paint_strips(painter, profile, x_of, top, bottom)
        self._paint_axes(painter, profile, x_of, y_of, x0, x1, bottom)
        self._paint_marker(painter, profile, x_of, top, bottom)
        self._paint_readout(painter, profile, x0)
        painter.end()

    def _tissue_brush(self, y_top: float, y_bottom: float, alpha: int) -> QLinearGradient:
        """Plaque fill: lighter towards the adventitia, darker towards the lumen."""
        gradient = QLinearGradient(QPointF(0.0, y_top), QPointF(0.0, y_bottom))
        outer, inner = QColor(TISSUE_OUTER), QColor(TISSUE_INNER)
        outer.setAlpha(alpha)
        inner.setAlpha(alpha)
        gradient.setColorAt(0.0, outer)
        gradient.setColorAt(0.5, inner)
        gradient.setColorAt(1.0, outer)
        return gradient

    def _tube_path(self, frames, radii, x_of, y_of) -> QPainterPath:
        """Closed path around the tube of radius *radii*: +r out, -r back."""
        path = QPainterPath()
        path.moveTo(x_of(frames[0]), y_of(radii[0]))
        for frame, radius in zip(frames[1:], radii[1:]):
            path.lineTo(x_of(frame), y_of(radius))
        for frame, radius in zip(reversed(frames), reversed(radii)):
            path.lineTo(x_of(frame), y_of(-radius))
        path.closeSubpath()
        return path

    def _paint_vessel(self, painter: QPainter, profile: _Profile, x_of, y_of) -> None:
        lumen_r, eem_r = profile.lumen_r, profile.eem_r
        y_top, y_bottom = y_of(profile.r_max), y_of(-profile.r_max)
        painter.setPen(Qt.PenStyle.NoPen)

        # 1. plaque across the whole EEM span, faded: everything here is at least partly
        #    interpolated until proven otherwise by the measured pass below.
        eem_frames = [int(f) for f in np.flatnonzero(~np.isnan(eem_r))]
        if len(eem_frames) > 1:
            painter.setBrush(self._tissue_brush(y_top, y_bottom, INTERPOLATED_ALPHA))
            painter.drawPath(self._tube_path(eem_frames, [eem_r[f] for f in eem_frames], x_of, y_of))

        # 2. full-strength plaque wherever consecutive frames carry both contours
        painter.setBrush(self._tissue_brush(y_top, y_bottom, 255))
        for frames, is_measured in _segments(profile.lumen_measured & profile.eem_measured, eem_r):
            if is_measured and len(frames) > 1:
                painter.drawPath(self._tube_path(frames, [eem_r[f] for f in frames], x_of, y_of))

        # 3. the lumen itself, punched out of the plaque band
        lumen_frames = [int(f) for f in np.flatnonzero(~np.isnan(lumen_r))]
        if len(lumen_frames) > 1:
            painter.setBrush(LUMEN_FILL)
            painter.drawPath(self._tube_path(lumen_frames, [lumen_r[f] for f in lumen_frames], x_of, y_of))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # 4. boundary lines, each dotted wherever its own contour was interpolated
        for measured, radii, color in (
            (profile.eem_measured, eem_r, self._eem_color()),
            (profile.lumen_measured, lumen_r, self._lumen_color()),
        ):
            for frames, is_measured in _segments(measured, radii):
                if len(frames) < 2:
                    continue
                style = Qt.PenStyle.SolidLine if is_measured else Qt.PenStyle.DotLine
                pen = QPen(color, 1.4, style)
                pen.setCosmetic(True)
                painter.setPen(pen)
                for sign in (1, -1):
                    path = QPainterPath()
                    path.moveTo(x_of(frames[0]), y_of(sign * radii[frames[0]]))
                    for frame in frames[1:]:
                        path.lineTo(x_of(frame), y_of(sign * radii[frame]))
                    painter.drawPath(path)

        # 5. dots on the frames the numbers actually come from
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._lumen_color())
        for frame in np.flatnonzero(profile.lumen_measured):
            radius = lumen_r[frame]
            if math.isnan(radius):
                continue
            for sign in (1, -1):
                painter.drawEllipse(QPointF(x_of(int(frame)), y_of(sign * radius)), 1.5, 1.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_catheter(self, painter: QPainter, x0: float, x1: float, y_of) -> None:
        pen = QPen(CATHETER_GREY, 1.2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        radius = self._catheter_radius()
        for sign in (1, -1):
            y = y_of(sign * radius)
            painter.drawLine(QPointF(x0, y), QPointF(x1, y))

    def _paint_strips(self, painter: QPainter, profile: _Profile, x_of, top: float, bottom: float) -> None:
        known = np.flatnonzero(profile.strip_measured)
        if known.size == 0:
            return
        x_start, x_end = x_of(int(known[0])), x_of(int(known[-1]))
        width = max(x_end - x_start, 1.0)
        metrics = QFontMetrics(painter.font())

        strips = (
            ('Ca', profile.calcium, profile.calcium_scale, CALCIUM_RAMP, top),
            ('Lipid', profile.lipid, profile.lipid_scale, LIPID_RAMP, bottom - STRIP_HEIGHT),
        )
        for label, values, full_scale, ramp, y in strips:
            rect = QRectF(x_start, y, width, STRIP_HEIGHT)
            gradient = QLinearGradient(QPointF(rect.left(), 0.0), QPointF(rect.right(), 0.0))
            for frame in known:
                position = (x_of(int(frame)) - x_start) / width
                gradient.setColorAt(min(max(position, 0.0), 1.0), _ramp_color(ramp, float(values[frame]) / full_scale))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRect(rect)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            baseline = rect.center().y() + metrics.capHeight() / 2.0
            # label in the free left margin, full-scale value on the strip itself
            painter.setPen(QPen(LABEL_GREY))
            painter.drawText(QPointF(max(LEFT_MARGIN - 3 - metrics.horizontalAdvance(label), 1), baseline), label)
            scale_text = f'0–{full_scale * 100:.0f}%'
            self._draw_pill_text(painter, scale_text, rect.right() - 3, baseline, align_right=True)

    def _draw_pill_text(self, painter: QPainter, text: str, x: float, baseline: float, align_right=False) -> None:
        """Draw *text* over a translucent backdrop, so a bright strip cannot swallow it."""
        metrics = QFontMetrics(painter.font())
        width = metrics.horizontalAdvance(text)
        left = x - width if align_right else x
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PILL_BACKDROP)
        painter.drawRect(QRectF(left - 2, baseline - metrics.capHeight() - 2, width + 4, metrics.capHeight() + 5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(LABEL_GREY))
        painter.drawText(QPointF(left, baseline), text)

    def _paint_axes(self, painter: QPainter, profile: _Profile, x_of, y_of, x0: float, x1: float, bottom: float):
        metrics = QFontMetrics(painter.font())
        axis_pen = QPen(AXIS_GREY, 1.0)
        axis_pen.setCosmetic(True)

        # radius scale, mirrored above and below the centre line
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(x0, y_of(profile.r_max)), QPointF(x0, y_of(-profile.r_max)))
        step = 1.0 if profile.r_max >= 1.5 else 0.5
        radii = [0.0] + [r * step for r in range(1, int(profile.r_max / step) + 1)]
        for radius in radii:
            for sign in (1,) if radius == 0.0 else (1, -1):
                y = y_of(sign * radius)
                painter.setPen(axis_pen)
                painter.drawLine(QPointF(x0 - 3, y), QPointF(x0, y))
                text = f'{radius:g}'
                painter.setPen(QPen(LABEL_GREY))
                painter.drawText(QPointF(x0 - 5 - metrics.horizontalAdvance(text), y + metrics.capHeight() / 2.0), text)
        painter.setPen(QPen(LABEL_GREY))
        painter.drawText(QPointF(1, y_of(profile.r_max) - 2), 'mm')

        # frame axis
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(x0, bottom), QPointF(x1, bottom))
        n_ticks = max(2, min(8, int((x1 - x0) // 90)))
        for i in range(n_ticks + 1):
            frame = round(i * (profile.n_frames - 1) / n_ticks)
            x = x_of(frame)
            painter.setPen(axis_pen)
            painter.drawLine(QPointF(x, bottom), QPointF(x, bottom + 3))
            text = str(frame)
            width = metrics.horizontalAdvance(text)
            painter.setPen(QPen(LABEL_GREY))
            painter.drawText(
                QPointF(min(max(x - width / 2.0, 0.0), self.width() - width), bottom + 3 + metrics.capHeight() + 2),
                text,
            )

    def _paint_marker(self, painter: QPainter, profile: _Profile, x_of, top: float, bottom: float) -> None:
        pen = QPen(MARKER_COLOR, 1.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = x_of(min(max(self._frame, 0), profile.n_frames - 1))
        painter.drawLine(QPointF(x, top), QPointF(x, bottom))

    def _paint_readout(self, painter: QPainter, profile: _Profile, x0: float) -> None:
        frame = min(max(self._frame, 0), profile.n_frames - 1)
        metrics = QFontMetrics(painter.font())
        baseline = TOP_MARGIN + metrics.capHeight() + 2

        lumen_r = profile.lumen_r[frame]
        if math.isnan(lumen_r):
            painter.setPen(QPen(LABEL_GREY))
            painter.drawText(QPointF(x0, baseline), f'Frame {frame}   (outside the contoured range)')
            return

        prefix = '' if profile.lumen_measured[frame] else '~'  # ~ = interpolated, not measured here
        parts = [f'Frame {frame}', f'{prefix}lumen Ø {2 * lumen_r:.2f} mm']
        eem_r = profile.eem_r[frame]
        if not math.isnan(eem_r):
            parts.append(f'{"" if profile.eem_measured[frame] else "~"}EEM Ø {2 * eem_r:.2f} mm')
            if eem_r > 0:
                burden = max(0.0, 1.0 - (lumen_r / eem_r) ** 2)
                parts.append(f'plaque {burden * 100:.0f}%')
        for label, values in (('Ca', profile.calcium), ('lipid', profile.lipid)):
            value = values[frame]
            if not math.isnan(value):
                strip_prefix = '' if profile.strip_measured[frame] else '~'
                parts.append(f'{strip_prefix}{label} {value * 100:.0f}%')
        if self._incomplete:
            parts.append('measuring plaque…')

        painter.setPen(QPen(LABEL_GREY))
        painter.drawText(QPointF(x0, baseline), '   '.join(parts))

    def _lumen_color(self) -> QColor:
        return QColor(getattr(self.main_window.config.display, 'color_contour', 'green'))

    def _eem_color(self) -> QColor:
        return QColor(getattr(self.main_window.config.display, 'color_eem', '#03b1fc'))

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        self._jump_to_frame(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._jump_to_frame(event)

    def _jump_to_frame(self, event) -> None:
        profile = self._profile
        if profile is None or not self.main_window.image_displayed:
            return
        x0 = float(LEFT_MARGIN)
        x1 = float(max(self.width() - RIGHT_MARGIN, LEFT_MARGIN + 1))
        fraction = (event.position().x() - x0) / (x1 - x0)
        frame = round(min(max(fraction, 0.0), 1.0) * (profile.n_frames - 1))
        self.main_window.display_slider.set_value(int(frame))
