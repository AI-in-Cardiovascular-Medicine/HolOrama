"""Tests that a frame's lumen measurements follow its contour.

The measurements (`lumen.measurements`, centroid, closest/farthest points) are what the
longitudinal view and the OCT schematic plot, and they are written as a side effect of
displaying a frame. These tests pin down that every way of changing a contour keeps them
in step: editing it, replacing it wholesale (copy from a neighbour, Ctrl+wheel scaling),
deleting it, and writing many frames at once (automatic segmentation, mask import).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from domain.io_types import FrameData
from domain.undo import UndoStack
from pages.intravascular.utils.metrics import clear_lumen_measurements

DIM = 200  # frame is DIM x DIM pixels
N_FRAMES = 8
RESOLUTION = 0.02  # mm/pixel
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'src' / 'config.yaml'


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _to_namespace(value) for key, value in obj.items()})
    return obj


def _ring(radius_px, n=20):
    """Knot points of a circle about the frame centre, in image pixels."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    center = DIM / 2
    return (
        [float(center + radius_px * np.cos(a)) for a in angles],
        [float(center + radius_px * np.sin(a)) for a in angles],
    )


def _put_contour(frame_data, radius_px):
    frame_data.lumen.contours = [_ring(radius_px)]
    frame_data.lumen.closed = [True]


@pytest.fixture
def display(qt_app):
    """A real Display on a stub main_window, plus its frame dict and a plot_areas spy."""
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = _to_namespace(yaml.safe_load(f))

    frames = {i: FrameData() for i in range(N_FRAMES)}
    plot_areas_calls = []
    runtime = SimpleNamespace(
        frame_data_dct=frames,
        metadata={'num_frames': N_FRAMES, 'resolution': RESOLUTION, 'modality': 'OCT', 'dimension': DIM},
        images=np.full((N_FRAMES, DIM, DIM), 100, dtype=np.uint8),
        images_rgb=None,
        contour_undo=UndoStack(),
    )
    longitudinal_view = SimpleNamespace(
        set_data=lambda images: None,
        plot_areas=lambda: plot_areas_calls.append(1),
        hide_lview_contours=lambda: None,
        show_lview_contours=lambda: None,
        update_marker=lambda frame: None,
    )
    main_window = SimpleNamespace(
        config=config,
        runtime_data=runtime,
        hide_contours=False,
        hide_special_points=True,  # skip the on-screen text; the computation must still happen
        colormap_enabled=False,
        mask_mode_box=None,
        longitudinal_view=longitudinal_view,
        image_displayed=True,
        file_name='test',
        status_bar=SimpleNamespace(showMessage=lambda *args: None),
    )

    from pages.intravascular.left_half.display import Display

    widget = Display(main_window)
    main_window.display = widget
    widget.set_data(runtime.images)
    return SimpleNamespace(widget=widget, frames=frames, plot_areas_calls=plot_areas_calls, main_window=main_window)


def _expected_area(radius_px):
    return np.pi * (radius_px * RESOLUTION) ** 2


class TestCurrentFrameMetrics:
    def test_contour_appearing_on_the_current_frame_is_measured(self, display):
        """Copying a contour from a neighbour onto an empty frame (Shift+A/D/W/S)."""
        display.widget.set_frame(2)
        _put_contour(display.frames[2], 30)
        display.widget.update_display()

        measurements = display.frames[2].lumen.measurements
        assert measurements.area == pytest.approx(_expected_area(30), rel=0.02)
        assert measurements.minor_axis is not None
        assert display.frames[2].centroid is not None

    def test_replacing_the_contour_replaces_the_measurements(self, display):
        """The stale-spline case: metrics must not keep the contour that was on screen."""
        _put_contour(display.frames[3], 20)
        display.widget.set_frame(3)
        before = display.frames[3].lumen.measurements.area

        _put_contour(display.frames[3], 40)  # 4x the area
        display.widget.update_display()

        after = display.frames[3].lumen.measurements.area
        assert before == pytest.approx(_expected_area(20), rel=0.02)
        assert after == pytest.approx(_expected_area(40), rel=0.02)

    def test_dragged_knots_are_measured_after_write_back(self, display):
        """Mirrors mouseReleaseEvent: knots go to frame_data, then the display redraws."""
        _put_contour(display.frames[6], 25)
        display.widget.set_frame(6)
        before = display.frames[6].lumen.measurements.area

        geometry = display.widget.finalized_splines['lumen'][0].geometry
        geometry.knot_points_x = [x * 1.5 for x in geometry.knot_points_x]
        geometry.knot_points_y = [y * 1.5 for y in geometry.knot_points_y]
        geometry.interpolate()
        sf = display.widget.scaling_factor
        display.frames[6].lumen.contours[0] = [
            [x / sf for x in geometry.knot_points_x],
            [y / sf for y in geometry.knot_points_y],
        ]
        display.widget.display_image(update_contours=True)

        assert display.frames[6].lumen.measurements.area == pytest.approx(before * 1.5**2, rel=0.02)

    def test_scaling_the_contour_updates_the_measurements(self, display):
        """Ctrl+wheel, which rewrites the stored contour and redraws."""
        _put_contour(display.frames[5], 25)
        display.widget.set_frame(5)
        before = display.frames[5].lumen.measurements.area

        display.widget._scale_active_contour(delta=120)  # one step outwards

        assert display.frames[5].lumen.measurements.area > before

    def test_deleted_contour_clears_its_measurements(self, display):
        _put_contour(display.frames[2], 30)
        display.widget.set_frame(2)
        assert display.frames[2].lumen.measurements.area is not None

        display.frames[2].lumen.contours = []
        display.widget.finalized_splines['lumen'] = []  # delete_contour drops the spline entry
        display.widget.display_image(update_contours=True)

        frame_data = display.frames[2]
        assert frame_data.lumen.measurements.area is None
        assert frame_data.centroid is None
        assert frame_data.closest_points is None and frame_data.farthest_points is None

    def test_uninterpolatable_contour_keeps_its_measurements(self, display):
        """A contour that is still there but could not be drawn is not a deletion."""
        _put_contour(display.frames[2], 30)
        display.widget.set_frame(2)
        area = display.frames[2].lumen.measurements.area

        display.widget.finalized_splines['lumen'] = []  # nothing drawable this pass
        display.widget.display_image(update_contours=True)

        assert display.frames[2].lumen.measurements.area == area


class TestBatchMetrics:
    def test_frames_written_in_bulk_are_measured_without_being_visited(self, display):
        """Automatic segmentation / mask import: contours for frames the user never opens."""
        for frame in (4, 5, 6):
            _put_contour(display.frames[frame], 20 + 5 * frame)
        assert all(display.frames[f].lumen.measurements.area is None for f in (4, 5, 6))

        before_calls = len(display.plot_areas_calls)
        display.widget.refresh_all_frame_metrics()

        for frame in (4, 5, 6):
            measurements = display.frames[frame].lumen.measurements
            assert measurements.area == pytest.approx(_expected_area(20 + 5 * frame), rel=0.02)
            assert measurements.minor_axis is not None  # the OCT schematic's lumen radius
        assert len(display.plot_areas_calls) > before_calls  # overviews redrawn

    def test_refresh_survives_a_failing_longitudinal_view(self, display):
        def boom():
            raise RuntimeError('no scene')

        display.main_window.longitudinal_view.plot_areas = boom
        _put_contour(display.frames[4], 30)
        display.widget.refresh_all_frame_metrics()  # must not propagate
        assert display.frames[4].lumen.measurements.area is not None

    def test_clear_lumen_measurements_drops_every_derived_value(self, display):
        _put_contour(display.frames[4], 30)
        display.widget.refresh_all_frame_metrics()
        frame_data = display.frames[4]
        assert frame_data.lumen.measurements.area is not None

        clear_lumen_measurements(frame_data)

        assert frame_data.lumen.measurements.area is None
        assert frame_data.lumen.measurements.minor_axis is None
        assert frame_data.centroid is None
        assert frame_data.closest_points is None and frame_data.farthest_points is None
