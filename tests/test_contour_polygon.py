"""contour_polygon: lumen metrics need a single valid polygon, also from a self-intersecting spline."""

from shapely.geometry import Polygon

from pages.intravascular.utils.metrics import contour_polygon


def test_self_intersecting_contour_keeps_its_largest_loop():
    # Figure 8: a 2x2 loop and a 1x1 loop crossing at the origin. Repairing it gives a MultiPolygon.
    x = [0, 2, 2, 0, 0, -1, -1, 0]
    y = [0, 0, 2, 2, 0, 0, -1, -1]
    assert not Polygon(zip(x, y)).is_valid
    poly = contour_polygon(x, y)
    assert isinstance(poly, Polygon) and poly.area == 4.0
    assert len(poly.exterior.coords) > 3


def test_zero_area_contour_gives_none():
    assert contour_polygon([0, 1, 2], [0, 0, 0]) is None
