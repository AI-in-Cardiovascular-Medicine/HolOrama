"""Tests for placing and editing an angular sector on the display (the guide-wire
shadow and the blood artefact, see tools.angle).

What is worth pinning down here is the interaction rather than the maths (that lives in
test_tools_angle.py): one click sets the boundary the sector opens from, the pointer
opens it — past 180 degrees, which the two clicks alone could never express — the second
click fixes it, and both boundaries stay draggable afterwards.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PyQt6.QtCore import QPointF, Qt

from domain.all_types import ContourType
from domain.io_types import FrameData, sector_points
from domain.runtime_types import RuntimeData
from tools.angle import angle_of, sector_from_points

DIM = 200  # frame is DIM x DIM pixels
N_FRAMES = 4
RESOLUTION = 0.1  # mm/pixel — a 20 mm frame, so the 5 mm handle circle fits inside it
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'src' / 'config.yaml'


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _to_namespace(value) for key, value in obj.items()})
    return obj


@pytest.fixture
def display(qt_app):
    """A real Display on a stub main_window, sitting on frame 0 with nothing drawn."""
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = _to_namespace(yaml.safe_load(f))

    frames = {i: FrameData() for i in range(N_FRAMES)}
    runtime = RuntimeData()
    runtime.frame_data_dct = frames
    runtime.metadata = {'num_frames': N_FRAMES, 'resolution': RESOLUTION, 'modality': 'OCT', 'dimension': DIM}
    runtime.images = np.full((N_FRAMES, DIM, DIM), 100, dtype=np.uint8)
    longitudinal_view = SimpleNamespace(
        set_data=lambda images: None,
        plot_areas=lambda: None,
        hide_lview_contours=lambda: None,
        show_lview_contours=lambda: None,
        update_marker=lambda frame: None,
    )
    ui_syncs = []
    main_window = SimpleNamespace(
        config=config,
        runtime_data=runtime,
        hide_contours=False,
        hide_special_points=True,
        colormap_enabled=False,
        mask_mode_box=None,
        longitudinal_view=longitudinal_view,
        image_displayed=True,
        file_name='test',
        status_bar=SimpleNamespace(showMessage=lambda *args: None),
        left_half=SimpleNamespace(
            set_active_contour_type_ui=ui_syncs.append,
            closed_spline_btn=SimpleNamespace(setChecked=lambda checked: None),
            open_spline_btn=SimpleNamespace(setChecked=lambda checked: None),
            brush_btn=SimpleNamespace(setChecked=lambda checked: None),
        ),
    )
    main_window.save_contours_soon = runtime.mark_unsaved

    from pages.intravascular.left_half.display import Display

    widget = Display(main_window)
    main_window.display = widget
    widget.set_data(runtime.images)
    return SimpleNamespace(widget=widget, frames=frames, runtime=runtime, ui_syncs=ui_syncs)


def _at(widget, degrees, radius=120.0):
    """A scene position `degrees` round from the image centre."""
    centre = widget._scene_centre()
    angle = math.radians(degrees)
    return QPointF(centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle))


def _place(widget, from_deg, to_deg, step=10):
    """Place one sector: click, turn the pointer round to `to_deg`, click again.

    The pointer is moved in small steps, as a real one would be — that is what lets the
    opening pass 180 degrees.
    """
    widget._handle_angle_placement(_at(widget, from_deg))
    total = to_deg - from_deg
    steps = max(1, int(abs(total) / step))
    for i in range(1, steps + 1):
        widget._track_angle_opening(_at(widget, from_deg + total * i / steps))
    widget._handle_angle_placement(_at(widget, to_deg))


def _stored_sector(widget, frame_data, contour_type, index=0):
    """The (start, sweep) that was written for one sector, in scene angles."""
    contour_obj = getattr(frame_data, contour_type.value)
    points = [(x * widget.scaling_factor, y * widget.scaling_factor) for x, y in sector_points(contour_obj, index)]
    return sector_from_points(points, widget._scene_centre())


def _preview(widget):
    """The one sector currently drawn for the active type and index."""
    for contour_type, index, sector in widget._angle_sectors:
        if contour_type is widget.active_contour_type and index == widget.active_contour_index:
            return sector
    return None


class TestPlacement:
    def test_first_click_stores_one_point_and_previews_it_dotted(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()

        widget._handle_angle_placement(_at(widget, 0))

        assert len(sector_points(display.frames[0].wire, 0)) == 1
        preview = _preview(widget)
        assert preview is not None and preview.dotted
        assert preview._end_line.pen().style() == Qt.PenStyle.DotLine

    def test_the_opening_follows_the_pointer_before_the_second_click(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        widget._handle_angle_placement(_at(widget, 0))

        widget._track_angle_opening(_at(widget, 40))

        assert math.degrees(_preview(widget).sweep) == pytest.approx(40, abs=1)

    def test_second_click_stores_the_sector_and_draws_it_solid(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()

        _place(widget, 0, 60)

        assert not widget.angle_mode
        start, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(sweep) == pytest.approx(60, abs=1)
        drawn = _preview(widget)
        assert drawn is not None and not drawn.dotted
        assert drawn._end_line.pen().style() == Qt.PenStyle.SolidLine

    @pytest.mark.parametrize('opening', [45, 179, 181, 270, 350])
    def test_any_opening_can_be_placed(self, display, opening):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()

        _place(widget, 20, 20 + opening)

        start, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(start) % 360 == pytest.approx(20, abs=1)
        assert math.degrees(sweep) == pytest.approx(opening, abs=1)

    def test_turning_the_other_way_opens_the_other_side(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()

        _place(widget, 200, 200 - 240)  # a wide sector opened anticlockwise

        start, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(sweep) == pytest.approx(240, abs=1)
        assert math.degrees(start) % 360 == pytest.approx((200 - 240) % 360, abs=1)

    def test_a_second_click_that_opens_nothing_is_ignored(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()

        widget._handle_angle_placement(_at(widget, 30))
        widget._handle_angle_placement(_at(widget, 30))  # a double click on the first point

        assert widget.angle_mode  # still waiting for a click that opens the sector
        assert len(sector_points(display.frames[0].wire, 0)) == 1

    def test_leaving_angle_mode_after_one_click_drops_the_sector(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        widget._handle_angle_placement(_at(widget, 0))

        widget.cleanup_temporary_drawing()  # what Esc does

        assert display.frames[0].wire.contours == []
        assert not widget.angle_mode

    def test_handles_sit_on_the_configured_circle(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 90)

        expected = widget.angle_handle_radius_mm / RESOLUTION * widget.scaling_factor
        assert expected < 0.9 * widget.image_size / 2  # this frame is wide enough not to clamp
        centre = widget._scene_centre()
        for handle_x, handle_y in _preview(widget).handle_positions():
            assert math.hypot(handle_x - centre[0], handle_y - centre[1]) == pytest.approx(expected)

    def test_the_circle_is_pulled_inside_a_narrower_frame(self, display):
        widget = display.widget
        display.runtime.metadata['resolution'] = 0.01  # a 2 mm frame: 5 mm is off the image
        assert widget._angle_handle_radius() == pytest.approx(0.9 * widget.image_size / 2)


class TestSeveralSectors:
    def test_add_keeps_the_existing_ones(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 40)

        widget.start_angle(append=True)
        _place(widget, 180, 220)

        assert len(display.frames[0].wire.contours) == 2
        assert math.degrees(_stored_sector(widget, display.frames[0], ContourType.WIRE, 1)[0]) % 360 == pytest.approx(
            180, abs=1
        )

    def test_a_new_one_replaces_them(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 40)
        widget.start_angle(append=True)
        _place(widget, 180, 220)

        widget.start_angle()
        _place(widget, 90, 130)

        assert len(display.frames[0].wire.contours) == 1

    def test_blood_is_stored_apart_from_the_wire(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 40)

        widget.set_active_contour_type(ContourType.BLOOD)
        widget.start_angle()
        _place(widget, 100, 220)

        assert len(display.frames[0].wire.contours) == 1
        assert len(display.frames[0].blood.contours) == 1
        assert math.degrees(_stored_sector(widget, display.frames[0], ContourType.BLOOD)[1]) == pytest.approx(
            120, abs=1
        )

    def test_blood_is_drawn_in_its_own_colour(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.BLOOD)
        widget.start_angle()
        _place(widget, 0, 40)

        assert widget.contour_configs[ContourType.BLOOD].color == widget.color_blood
        assert widget.color_blood != widget.color_angle


class TestDragging:
    def _placed(self, display, from_deg=0, to_deg=60, contour_type=ContourType.WIRE):
        widget = display.widget
        widget.set_active_contour_type(contour_type)
        widget.start_angle()
        _place(widget, from_deg, to_deg)
        return widget

    def test_a_boundary_can_be_grabbed_and_turned(self, display):
        widget = self._placed(display)
        handle = _preview(widget).handle_positions()[1]  # the end boundary

        assert widget._grab_angle_handle(QPointF(*handle))
        widget._drag_angle_handle(_at(widget, 100))
        widget._release_angle_handle()

        start, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(start) % 360 == pytest.approx(0, abs=1)  # the other boundary stayed
        assert math.degrees(sweep) == pytest.approx(100, abs=1)

    def test_dragging_the_start_leaves_the_end_alone(self, display):
        widget = self._placed(display, 0, 60)
        handle = _preview(widget).handle_positions()[0]

        assert widget._grab_angle_handle(QPointF(*handle))
        widget._drag_angle_handle(_at(widget, -50))
        widget._release_angle_handle()

        start, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(start) % 360 == pytest.approx(310, abs=1)
        assert math.degrees(start + sweep) % 360 == pytest.approx(60, abs=1)

    def test_a_drag_past_the_other_boundary_does_not_flip_the_sector(self, display):
        widget = self._placed(display, 0, 20)  # a narrow sector
        handle = _preview(widget).handle_positions()[1]
        widget._grab_angle_handle(QPointF(*handle))

        for degrees in (15, 10, 5, 0, -5, -20):  # dragged back past the start boundary
            widget._drag_angle_handle(_at(widget, degrees))
        widget._release_angle_handle()

        _, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(sweep) < 1  # collapsed, rather than snapping round to ~340

    def test_a_click_away_from_every_handle_grabs_nothing(self, display):
        widget = self._placed(display)
        assert not widget._grab_angle_handle(QPointF(*widget._scene_centre()))

    def test_grabbing_another_type_makes_it_active(self, display):
        widget = self._placed(display, 0, 60, ContourType.WIRE)
        widget.set_active_contour_type(ContourType.BLOOD)
        widget.start_angle()
        _place(widget, 180, 240)
        widget.set_active_contour_type(ContourType.LUMEN)
        display.ui_syncs.clear()

        wire_handle = next(
            sector.handle_positions()[0]
            for contour_type, _, sector in widget._angle_sectors
            if contour_type is ContourType.WIRE
        )
        assert widget._grab_angle_handle(QPointF(*wire_handle))

        assert widget.active_contour_type is ContourType.WIRE
        assert display.ui_syncs == [ContourType.WIRE]
        widget._release_angle_handle()

    def test_a_drag_is_undoable(self, display):
        widget = self._placed(display)
        before = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        depth = len(display.runtime.contour_undo._stack)

        widget._grab_angle_handle(QPointF(*_preview(widget).handle_positions()[1]))
        widget._drag_angle_handle(_at(widget, 150))
        widget._release_angle_handle()

        assert len(display.runtime.contour_undo._stack) == depth + 1
        snapshot = display.runtime.contour_undo.pop()
        assert snapshot.key == ContourType.WIRE.value
        restored = [
            (x * widget.scaling_factor, y * widget.scaling_factor) for x, y in sector_points(snapshot.contour, 0)
        ]
        assert sector_from_points(restored, widget._scene_centre()) == pytest.approx(before)


class TestLegacyData:
    def test_a_two_point_wire_from_an_old_file_is_drawn_and_draggable(self, display):
        """Files written before the interior marker keep meaning the smaller wedge."""
        widget = display.widget
        centre = widget._image_centre()
        radius = 40.0
        display.frames[0].wire.contours = [
            (
                [centre[0] + radius, centre[0]],
                [centre[1], centre[1] + radius],
            )
        ]
        display.frames[0].wire.closed = [False]
        widget.display_image(update_contours=True)

        drawn = next(sector for contour_type, _, sector in widget._angle_sectors if contour_type is ContourType.WIRE)
        assert math.degrees(drawn.sweep) == pytest.approx(90, abs=1)

        # Dragging it rewrites it in the current shape, keeping the wedge it described.
        widget._grab_angle_handle(QPointF(*drawn.handle_positions()[1]))
        widget._drag_angle_handle(_at(widget, 120))
        widget._release_angle_handle()

        assert len(sector_points(display.frames[0].wire, 0)) == 3
        _, sweep = _stored_sector(widget, display.frames[0], ContourType.WIRE)
        assert math.degrees(sweep) == pytest.approx(120, abs=1)


class TestSectorsInTheMask:
    def test_both_types_reach_the_exported_mask(self, display):
        from input_output.output.imgs_masks import contours_to_mask

        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 90)
        widget.set_active_contour_type(ContourType.BLOOD)
        widget.start_angle()
        _place(widget, 180, 270)

        mask = contours_to_mask(display.runtime.images[:1], [0], display.frames)[0]
        assert (mask == 9).mean() == pytest.approx(0.25, abs=0.03)
        assert (mask == 10).mean() == pytest.approx(0.25, abs=0.03)

    def test_blood_sits_under_the_wire_shadow(self, display):
        from input_output.output.imgs_masks import contours_to_mask

        widget = display.widget
        widget.set_active_contour_type(ContourType.BLOOD)
        widget.start_angle()
        _place(widget, 0, 180)  # blood over half the frame
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 45, 90)  # a wire wholly inside it

        mask = contours_to_mask(display.runtime.images[:1], [0], display.frames)[0]
        assert (mask == 9).mean() == pytest.approx(45 / 360, abs=0.02)  # the wire survives in full
        assert (mask == 10).mean() == pytest.approx((180 - 45) / 360, abs=0.02)  # blood keeps the rest

    def test_a_wide_sector_reaches_it_too(self, display):
        from input_output.output.imgs_masks import contours_to_mask

        widget = display.widget
        widget.set_active_contour_type(ContourType.BLOOD)
        widget.start_angle()
        _place(widget, 0, 300)

        mask = contours_to_mask(display.runtime.images[:1], [0], display.frames)[0]
        assert (mask == 10).mean() == pytest.approx(300 / 360, abs=0.03)


def test_angle_points_are_measured_from_the_image_centre(display):
    """The stored points are directions, so only their angle about the centre matters."""
    widget = display.widget
    widget.set_active_contour_type(ContourType.WIRE)
    widget.start_angle()
    _place(widget, 30, 90)

    points = sector_points(display.frames[0].wire, 0)
    assert math.degrees(angle_of(points[0], widget._image_centre())) % 360 == pytest.approx(30, abs=1)


@pytest.fixture
def angle_controls(display):
    """The real LeftHalf on the same stub window, for the two angle controls it owns."""
    from PyQt6.QtWidgets import QApplication, QCheckBox, QSlider

    main_window = display.widget.main_window
    main_window.brush_settings_popup = SimpleNamespace(show_near=lambda widget: None, schedule_hide=lambda: None)
    main_window.hide_contours_box = QCheckBox()
    main_window.hide_special_points_box = QCheckBox()
    main_window.mask_mode_box = QCheckBox()
    main_window.display_slider = QSlider(Qt.Orientation.Horizontal)
    main_window.style = QApplication.instance().style
    main_window.contours_drawn = False

    from pages.intravascular.left_half.left_half import LeftHalf

    left_half = LeftHalf(main_window)
    main_window.left_half = left_half
    return SimpleNamespace(left_half=left_half, widget=display.widget, frames=display.frames)


class TestAngleControls:
    """The drop-down that picks the sector type, and the ➕📐 Add button that follows it."""

    def test_it_starts_on_the_wire(self, angle_controls):
        left_half = angle_controls.left_half
        assert left_half.angle_type is ContourType.WIRE
        assert left_half.angle_type_combo.currentText() == '📐 Angle Wire'
        assert left_half.add_angle_btn.text() == '➕📐 Add Wire'
        assert angle_controls.widget.color_angle in left_half.add_angle_btn.styleSheet()

    def test_choosing_blood_repoints_the_add_button(self, angle_controls):
        left_half = angle_controls.left_half

        left_half.angle_type_combo.setCurrentIndex(1)  # what a pick from the drop-down does first

        assert left_half.angle_type is ContourType.BLOOD
        assert left_half.add_angle_btn.text() == '➕📐 Add Blood'
        assert angle_controls.widget.color_blood in left_half.add_angle_btn.styleSheet()
        assert angle_controls.widget.color_blood in left_half.angle_type_combo.styleSheet()

    def test_choosing_a_type_starts_one_of_it(self, angle_controls):
        left_half = angle_controls.left_half

        left_half.angle_type_combo.setCurrentIndex(1)
        left_half._on_angle_type_activated(1)  # the drop-down entry is the action

        assert angle_controls.widget.angle_mode
        assert angle_controls.widget.active_contour_type is ContourType.BLOOD

    def test_add_appends_one_of_the_chosen_type(self, angle_controls):
        widget = angle_controls.widget
        left_half = angle_controls.left_half
        left_half.angle_type_combo.setCurrentIndex(1)
        left_half._on_angle_type_activated(1)
        _place(widget, 0, 40)

        left_half.add_angle_btn.click()
        _place(widget, 180, 220)

        assert len(angle_controls.frames[0].blood.contours) == 2
        assert angle_controls.frames[0].wire.contours == []

    def test_clicking_a_sector_on_the_image_moves_the_drop_down_to_it(self, angle_controls):
        widget = angle_controls.widget
        left_half = angle_controls.left_half
        left_half._on_angle_type_activated(0)
        _place(widget, 0, 60)
        left_half.angle_type_combo.setCurrentIndex(1)  # user wandered off to blood

        widget.set_active_contour_type(ContourType.WIRE)  # as grabbing the wire's handle does

        assert left_half.angle_type_combo.currentIndex() == 0
        assert left_half.add_angle_btn.text() == '➕📐 Add Wire'

    def test_the_contour_drop_down_still_owns_the_spline_types(self, angle_controls):
        left_half = angle_controls.left_half
        left_half.angle_type_combo.setCurrentIndex(1)

        left_half.set_active_contour_type_ui(ContourType.CALCIUM)

        assert left_half.contour_type_combo.currentText() == 'Calcium'
        assert left_half.angle_type is ContourType.BLOOD  # untouched


class TestNothingToGrab:
    def test_a_click_that_moves_nothing_is_not_an_undo_step(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 60)
        before = sector_points(display.frames[0].wire, 0)
        depth = len(display.runtime.contour_undo._stack)

        widget._grab_angle_handle(QPointF(*_preview(widget).handle_positions()[1]))
        widget._release_angle_handle()  # let go without moving

        assert sector_points(display.frames[0].wire, 0) == before
        assert len(display.runtime.contour_undo._stack) == depth

    def test_hidden_contours_leave_no_handles_behind(self, display):
        widget = display.widget
        widget.set_active_contour_type(ContourType.WIRE)
        widget.start_angle()
        _place(widget, 0, 60)
        handle = QPointF(*_preview(widget).handle_positions()[1])

        widget.main_window.hide_contours = True
        widget.display_image(update_contours=True)

        assert not widget._grab_angle_handle(handle)
