"""Tests for the OCT pullback schematic (pages.intravascular.utils.oct_plot).

Covers the data layer — how sparse per-frame contours become the continuous radius
and plaque-composition profile the widget paints — plus one paint smoke test.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from domain.io_types import FrameData
from pages.intravascular.utils.oct_plot import (
    OCTPlot,
    _interpolate,
    _segments,
    _strip_scale,
)

RESOLUTION = 0.02  # mm/pixel
DIM = 300  # frame is DIM x DIM pixels
N_FRAMES = 60
CENTER = DIM / 2.0


def _ring(radius_mm, n=20, arc=None):
    """Knot points of a circle (or arc) of *radius_mm* about the frame centre, in pixels."""
    radius_px = radius_mm / RESOLUTION
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) if arc is None else np.linspace(arc[0], arc[1], n)
    return (
        [float(CENTER + radius_px * np.cos(a)) for a in angles],
        [float(CENTER + radius_px * np.sin(a)) for a in angles],
    )


def _frame(lumen_mm=1.0, eem_mm=None, calcium_arc=None, lipid_arc=None) -> FrameData:
    """A frame carrying a circular lumen, optionally an EEM and plaque arcs between them."""
    frame_data = FrameData()
    frame_data.lumen.contours = [_ring(lumen_mm)]
    frame_data.lumen.closed = [True]
    frame_data.lumen.measurements.minor_axis = 2 * lumen_mm
    frame_data.lumen.measurements.area = float(np.pi * lumen_mm**2)
    frame_data.centroid = (CENTER, CENTER)
    if eem_mm is not None:
        frame_data.eem.contours = [_ring(eem_mm)]
        frame_data.eem.closed = [True]
        mid = (lumen_mm + eem_mm) / 2
        if calcium_arc is not None:
            frame_data.calcium.contours = [_ring(mid, n=12, arc=calcium_arc)]
            frame_data.calcium.closed = [False]
        if lipid_arc is not None:
            frame_data.lipid.contours = [_ring(mid, n=12, arc=lipid_arc)]
            frame_data.lipid.closed = [False]
    return frame_data


@pytest.fixture
def plot(qt_app):
    """OCTPlot on a stub main_window: only config/runtime_data/display are read."""

    def _make(frames: dict[int, FrameData], n_frames: int = N_FRAMES):
        full = {i: FrameData() for i in range(n_frames)}
        full.update(frames)
        main_window = SimpleNamespace(
            config=SimpleNamespace(intravascular=SimpleNamespace(catheter_diameter=0.9)),
            runtime_data=SimpleNamespace(
                frame_data_dct=full,
                metadata={'num_frames': n_frames, 'resolution': RESOLUTION, 'modality': 'OCT'},
                images=np.zeros((n_frames, DIM, DIM), dtype=np.uint8),
            ),
            display=SimpleNamespace(n_points_contour=200),
            image_displayed=True,
            display_slider=SimpleNamespace(set_value=lambda value: None),
        )
        widget = OCTPlot(main_window)
        widget.resize(600, 200)
        return widget

    return _make


class TestHelpers:
    def test_interpolate_bridges_gaps_but_does_not_extrapolate(self):
        out = _interpolate({10: 1.0, 20: 2.0}, 30)
        assert np.isnan(out[9]) and np.isnan(out[21])  # nothing outside the known span
        assert out[10] == pytest.approx(1.0)
        assert out[15] == pytest.approx(1.5)  # linearly bridged
        assert out[20] == pytest.approx(2.0)

    def test_interpolate_skips_nan_values(self):
        out = _interpolate({0: float('nan'), 5: 1.0, 9: 3.0}, 10)
        assert np.isnan(out[4])
        assert out[7] == pytest.approx(2.0)

    def test_segments_marks_runs_solid_and_bridges_dotted(self):
        measured = np.array([False, True, True, False, False, True, False])
        values = np.zeros(7)
        segments = _segments(measured, values)
        assert segments == [([1, 2], True), ([2, 5], False), ([5], True)]

    def test_segments_ignores_frames_without_a_value(self):
        measured = np.array([True, True, True])
        values = np.array([1.0, float('nan'), 1.0])
        assert _segments(measured, values) == [([0], True), ([0, 2], False), ([2], True)]

    def test_strip_scale_never_below_minimum(self):
        assert _strip_scale(np.array([0.0, 0.01])) == pytest.approx(0.25)
        assert _strip_scale(np.full(3, np.nan)) == pytest.approx(0.25)

    def test_strip_scale_follows_the_largest_fraction(self):
        assert _strip_scale(np.array([0.1, 0.62, np.nan])) == pytest.approx(0.65)


class TestProfile:
    def test_no_contours_yields_no_profile(self, plot):
        widget = plot({})
        widget.refresh()
        assert widget._profile is None

    def test_radii_come_from_shortest_distance_and_eem_area(self, plot):
        widget = plot({20: _frame(lumen_mm=1.0, eem_mm=1.5)})
        widget.refresh()
        profile = widget._profile
        assert profile.lumen_r[20] == pytest.approx(1.0, abs=1e-6)  # minor_axis / 2
        assert profile.eem_r[20] == pytest.approx(1.5, rel=0.01)  # equal-area radius

    def test_uncontoured_frames_are_interpolated_not_measured(self, plot):
        widget = plot({10: _frame(lumen_mm=1.0, eem_mm=1.5), 20: _frame(lumen_mm=2.0, eem_mm=2.5)})
        widget.refresh()
        profile = widget._profile
        assert profile.lumen_measured[10] and profile.lumen_measured[20]
        assert not profile.lumen_measured[15]
        assert profile.lumen_r[15] == pytest.approx(1.5, abs=1e-6)
        assert np.isnan(profile.lumen_r[9]) and np.isnan(profile.lumen_r[21])

    def test_eem_radius_floored_at_lumen_radius(self, plot):
        # An eccentric lumen: shortest distance says 1 mm radius, EEM area says 0.5 mm.
        frame_data = _frame(lumen_mm=1.0, eem_mm=0.5)
        widget = plot({5: frame_data})
        widget.refresh()
        profile = widget._profile
        assert profile.eem_r[5] == pytest.approx(profile.lumen_r[5])

    def test_lumen_without_eem_has_no_plaque_data(self, plot):
        widget = plot({5: _frame(lumen_mm=1.0)})
        widget.refresh()
        profile = widget._profile
        assert profile.lumen_measured[5]
        assert not profile.eem_measured[5]
        assert not profile.strip_measured.any()
        assert np.isnan(profile.calcium).all()

    def test_eem_without_plaque_reads_as_zero_not_unknown(self, plot):
        widget = plot({5: _frame(lumen_mm=1.0, eem_mm=1.5)})
        widget.refresh()
        profile = widget._profile
        assert profile.strip_measured[5]
        assert profile.calcium[5] == 0.0 and profile.lipid[5] == 0.0

    def test_plaque_fractions_measured_against_the_wall(self, plot):
        """An open arc encloses the wall from the arc *outwards*, and that is what counts.

        Calcium is drawn on the lumen boundary over a quarter turn, so it should fill a
        quarter of the wall; lipid sits mid-wall over an eighth of a turn, so only the
        outer part of that sector counts.
        """
        lumen_mm, eem_mm = 1.0, 1.6
        frame_data = _frame(lumen_mm=lumen_mm, eem_mm=eem_mm)
        frame_data.calcium.contours = [_ring(lumen_mm, n=12, arc=(0.0, np.pi / 2))]
        frame_data.calcium.closed = [False]
        lipid_r = (lumen_mm + eem_mm) / 2
        frame_data.lipid.contours = [_ring(lipid_r, n=12, arc=(np.pi, 1.25 * np.pi))]
        frame_data.lipid.closed = [False]

        widget = plot({8: frame_data})
        widget.refresh()
        profile = widget._profile

        wall = eem_mm**2 - lumen_mm**2
        assert profile.calcium[8] == pytest.approx(0.25 * (eem_mm**2 - lumen_mm**2) / wall, abs=0.02)
        assert profile.lipid[8] == pytest.approx(0.125 * (eem_mm**2 - lipid_r**2) / wall, abs=0.02)
        assert profile.calcium_scale >= profile.calcium[8]

    def test_frames_missing_measurements_are_skipped(self, plot):
        frame_data = _frame(lumen_mm=1.0, eem_mm=1.5)
        frame_data.lumen.measurements.minor_axis = None
        frame_data.lumen.measurements.area = None
        widget = plot({5: frame_data})
        widget.refresh()
        assert widget._profile is None

    def test_area_fallback_when_shortest_distance_missing(self, plot):
        frame_data = _frame(lumen_mm=1.0, eem_mm=1.5)
        frame_data.lumen.measurements.minor_axis = None  # not computed yet
        widget = plot({5: frame_data})
        widget.refresh()
        assert widget._profile.lumen_r[5] == pytest.approx(1.0, rel=0.01)  # equal-area radius

    def test_r_max_covers_the_catheter_even_without_plaque(self, plot):
        widget = plot({5: _frame(lumen_mm=0.2)})
        widget.refresh()
        assert widget._profile.r_max == pytest.approx(0.45)  # catheter_diameter / 2


class TestCaching:
    def test_unchanged_frames_are_not_measured_twice(self, plot):
        widget = plot({5: _frame(lumen_mm=1.0, eem_mm=1.6, calcium_arc=(0.0, 1.0))})
        widget.refresh()
        first = widget._cache[5]
        widget.refresh()
        assert widget._cache[5] is first  # same tuple object: nothing recomputed

    def test_edited_contour_invalidates_its_frame(self, plot):
        frame_data = _frame(lumen_mm=1.0, eem_mm=1.6)
        widget = plot({5: frame_data})
        widget.refresh()
        before = widget._profile.lumen_r[5]

        frame_data.lumen.contours = [_ring(2.0)]
        frame_data.lumen.measurements.minor_axis = 4.0
        widget.refresh()
        assert widget._profile.lumen_r[5] == pytest.approx(4.0 / 2)
        assert widget._profile.lumen_r[5] != before

    def test_stale_frames_dropped_when_a_shorter_file_is_loaded(self, plot):
        widget = plot({5: _frame(lumen_mm=1.0), 40: _frame(lumen_mm=1.0)})
        widget.refresh()
        assert 40 in widget._cache

        runtime = widget.main_window.runtime_data
        runtime.frame_data_dct = {i: runtime.frame_data_dct[i] for i in range(10)}
        runtime.metadata['num_frames'] = 10
        runtime.images = np.zeros((10, DIM, DIM), dtype=np.uint8)
        widget.refresh()
        assert 40 not in widget._cache
        assert widget._profile.n_frames == 10


class TestIncrementalBuild:
    """The plaque masks are measured under a time budget, spread over event-loop slices."""

    @staticmethod
    def _plaque_pullback():
        return {i: _frame(lumen_mm=1.0, eem_mm=1.6, calcium_arc=(0.0, 1.2)) for i in range(N_FRAMES)}

    def test_vessel_is_complete_from_the_first_slice(self, plot, monkeypatch):
        import pages.intravascular.utils.oct_plot as module

        monkeypatch.setattr(module, 'MASK_BUDGET_S', 0.0)  # no budget: every mask deferred
        widget = plot(self._plaque_pullback())
        widget.refresh()
        profile = widget._profile
        assert widget._incomplete
        assert profile.lumen_measured.all() and profile.eem_measured.all()  # radii regardless
        assert not profile.strip_measured.any()  # plaque still outstanding
        widget.reset()

    def test_build_converges_over_slices(self, plot, qt_app):
        from PyQt6.QtCore import QEventLoop, QTimer

        widget = plot(self._plaque_pullback())
        widget.refresh()
        for _ in range(200):
            if not widget._incomplete:
                break
            loop = QEventLoop()
            QTimer.singleShot(5, loop.quit)
            loop.exec()
        assert not widget._incomplete
        assert widget._profile.strip_measured.all()
        assert widget._profile.calcium[10] > 0

    def test_reset_cancels_a_running_build(self, plot, monkeypatch):
        import pages.intravascular.utils.oct_plot as module

        monkeypatch.setattr(module, 'MASK_BUDGET_S', 0.0)
        widget = plot(self._plaque_pullback())
        widget.refresh()
        assert widget._incomplete
        widget.reset()
        assert not widget._incomplete
        assert widget._profile is None and not widget._cache


class TestPainting:
    @pytest.mark.parametrize(
        'frames',
        [
            {},  # placeholder text
            {5: _frame(lumen_mm=1.0)},  # lumen only
            {5: _frame(lumen_mm=1.0, eem_mm=1.6), 25: _frame(lumen_mm=1.2, eem_mm=1.8)},  # interpolated stretch
            {
                i: _frame(lumen_mm=1.0, eem_mm=1.6, calcium_arc=(0.0, 1.0), lipid_arc=(3.0, 4.0)) for i in range(5, 15)
            },  # measured stretch with both plaques
        ],
    )
    def test_paint_does_not_raise(self, plot, frames):
        from PyQt6.QtGui import QPixmap

        widget = plot(frames)
        widget.refresh()
        widget.set_frame(10)
        widget.render(QPixmap(widget.size()))

    def test_click_jumps_the_slider_to_that_frame(self, plot, qt_app):
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        from pages.intravascular.utils.oct_plot import LEFT_MARGIN, RIGHT_MARGIN

        widget = plot({5: _frame(lumen_mm=1.0), 50: _frame(lumen_mm=1.0)})
        widget.refresh()
        jumped = []
        widget.main_window.display_slider.set_value = jumped.append

        # halfway across the plot area -> halfway through the pullback
        x = LEFT_MARGIN + (widget.width() - RIGHT_MARGIN - LEFT_MARGIN) / 2
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(x, widget.height() / 2),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        widget.mousePressEvent(event)
        assert jumped == [round((N_FRAMES - 1) / 2)]
