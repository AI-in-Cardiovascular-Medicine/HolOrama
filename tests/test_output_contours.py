import json
from dataclasses import asdict

import numpy as np

from domain.io_types import Contour, FrameData, iter_wires, set_wire_points
from input_output.input.contours import _build_wire
from input_output.output.contours import _to_serializable
from input_output.output.imgs_masks import _wire_shadow_mask


class TestToSerializable:
    def test_numpy_float_scalar_to_python_float(self):
        result = _to_serializable(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_int_scalar_to_python_int(self):
        result = _to_serializable(np.int32(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_array_to_list(self):
        result = _to_serializable(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_2d_array_to_nested_list(self):
        result = _to_serializable(np.array([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_unknown_type_falls_back_to_str(self):
        class Custom:
            def __str__(self):
                return 'custom_value'

        assert _to_serializable(Custom()) == 'custom_value'


def _wire(*wires):
    w = Contour()
    for i, points in enumerate(wires):
        set_wire_points(w, i, points)
    return w


class TestWireShadowMask:
    SHAPE = (100, 100)
    CENTER = (50.0, 50.0)  # (y, x)

    def _mask(self, wire):
        return _wire_shadow_mask(wire, self.SHAPE, *self.CENTER)

    def test_no_wire_gives_empty_mask(self):
        assert not self._mask(Contour()).any()
        assert not self._mask(None).any()

    def test_incomplete_wire_gives_empty_mask(self):
        # A single placed point defines no sector yet.
        assert not self._mask(_wire([(90.0, 50.0)])).any()

    def test_second_wire_adds_its_own_sector(self):
        one = self._mask(_wire([(90.0, 50.0), (80.0, 20.0)]))
        two = self._mask(_wire([(90.0, 50.0), (80.0, 20.0)], [(10.0, 50.0), (20.0, 80.0)]))
        assert one.any()
        assert two.sum() > one.sum()
        assert (two & one == one).all()  # the first wire's shadow is preserved

    def test_pre_migration_tuple_still_shadows(self):
        legacy = self._mask(((90.0, 50.0), (80.0, 20.0)))
        assert (legacy == self._mask(_wire([(90.0, 50.0), (80.0, 20.0)]))).all()


class TestWireRoundTrip:
    def test_survives_asdict_json_and_reload(self):
        fd = FrameData(wire=_build_wire([[316.0, 318.0], [372.0, 298.0]]))
        set_wire_points(fd.wire, 1, [(10.0, 20.0), (30.0, 40.0)])

        reloaded = _build_wire(json.loads(json.dumps(asdict(fd)))['wire'])
        assert iter_wires(reloaded) == [
            [(316.0, 318.0), (372.0, 298.0)],
            [(10.0, 20.0), (30.0, 40.0)],
        ]
