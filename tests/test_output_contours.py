import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from domain.io_types import Contour, FrameData, iter_sectors, set_sector_points
from input_output.input.contours import _build_sector_contour
from input_output.output.contours import _to_serializable
from input_output.output.imgs_masks import _angle_sector_mask
from tools.angle import points_for_sector


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


def _sectors(*sectors):
    contour = Contour()
    for i, points in enumerate(sectors):
        set_sector_points(contour, i, points)
    return contour


class TestAngleSectorMask:
    """The rasteriser behind both sector types (the wire shadow and blood).

    Every case is stated as a fraction of the image the wedge covers, which for a sector
    measured from the image centre is just its opening over a full turn.
    """

    SHAPE = (100, 100)
    CENTER = (50.0, 50.0)  # (y, x), as the rasteriser takes it
    CENTRE = (50.0, 50.0)  # (x, y), as tools.angle takes it

    def _mask(self, contour):
        return _angle_sector_mask(contour, self.SHAPE, *self.CENTER)

    def _stored(self, start_deg, sweep_deg):
        """One sector stored the way the display stores it, interior marker included."""
        return points_for_sector(self.CENTRE, 30.0, math.radians(start_deg), math.radians(sweep_deg))

    def test_no_sector_gives_empty_mask(self):
        assert not self._mask(Contour()).any()
        assert not self._mask(None).any()

    def test_incomplete_sector_gives_empty_mask(self):
        # A single placed point defines no wedge yet.
        assert not self._mask(_sectors([(90.0, 50.0)])).any()

    def test_second_sector_adds_its_own_wedge(self):
        one = self._mask(_sectors([(90.0, 50.0), (80.0, 20.0)]))
        two = self._mask(_sectors([(90.0, 50.0), (80.0, 20.0)], [(10.0, 50.0), (20.0, 80.0)]))
        assert one.any()
        assert two.sum() > one.sum()
        assert (two & one == one).all()  # the first sector is preserved

    def test_pre_migration_tuple_still_covers(self):
        legacy = self._mask(((90.0, 50.0), (80.0, 20.0)))
        assert (legacy == self._mask(_sectors([(90.0, 50.0), (80.0, 20.0)]))).all()

    def test_two_points_cover_the_smaller_wedge(self):
        # The legacy reading, unchanged: a quarter of the image rather than three quarters.
        assert self._mask(_sectors([(90.0, 50.0), (50.0, 90.0)])).mean() == pytest.approx(0.25, abs=0.02)

    @pytest.mark.parametrize('sweep_deg', [30, 90, 180, 270, 359])
    def test_the_stored_marker_carries_the_whole_opening(self, sweep_deg):
        # Including the openings past 180 degrees, which two points alone cannot express.
        assert self._mask(_sectors(self._stored(15, sweep_deg))).mean() == pytest.approx(sweep_deg / 360, abs=0.02)

    def test_a_wide_sector_is_the_complement_of_its_narrow_twin(self):
        # Same two boundaries; only the marker differs, so together they tile the image.
        wide = self._mask(_sectors(self._stored(0, 270)))
        narrow = self._mask(_sectors(self._stored(270, 90)))
        assert (wide | narrow).all()
        assert (wide & narrow).mean() < 0.02  # they meet on the boundaries and nowhere else


class TestSectorRoundTrip:
    def test_survives_asdict_json_and_reload(self):
        fd = FrameData(wire=_build_sector_contour([[316.0, 318.0], [372.0, 298.0]]))
        set_sector_points(fd.wire, 1, [(10.0, 20.0), (30.0, 40.0), (20.0, 30.0)])
        set_sector_points(fd.blood, 0, [(1.0, 2.0), (3.0, 4.0), (2.0, 3.0)])

        raw = json.loads(json.dumps(asdict(fd)))
        assert iter_sectors(_build_sector_contour(raw['wire'])) == [
            [(316.0, 318.0), (372.0, 298.0)],
            [(10.0, 20.0), (30.0, 40.0), (20.0, 30.0)],
        ]
        assert iter_sectors(_build_sector_contour(raw['blood'])) == [[(1.0, 2.0), (3.0, 4.0), (2.0, 3.0)]]
