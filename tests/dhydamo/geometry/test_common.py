from hydrolib.dhydamo.geometry import common
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box


def test_interp_linestring_multilinestring() -> None:
    # each part is interpolated as a single LineString; this used to fail with
    # "'MultiLineString' object is not iterable"
    lines = [LineString([(0, 0), (10, 0)]), LineString([(0, 5), (0, 15)])]

    result = common.interp_linestring(MultiLineString(lines), 2.0)

    assert isinstance(result, MultiLineString)
    assert [part.equals(common.interp_linestring(line, 2.0)) for part, line in zip(result.geoms, lines)] == [True, True]
    assert len(result.geoms[0].coords) == 5


def test_interp_polygon_multipolygon() -> None:
    polygons = [box(0, 0, 10, 10), box(20, 0, 30, 10)]

    result = common.interp_polygon(MultiPolygon(polygons), 2.0)

    assert isinstance(result, MultiPolygon)
    assert [part.equals(common.interp_polygon(polygon, 2.0)) for part, polygon in zip(result.geoms, polygons)] == [True, True]
