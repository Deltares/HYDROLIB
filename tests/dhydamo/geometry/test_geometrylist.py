import pytest
from hydrolib.dhydamo.geometry.models import GeometryList
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    box,
)
from shapely.geometry.base import BaseGeometry


@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param(Point(1, 2), id="point"),
        pytest.param(MultiPoint([(1, 2), (3, 4)]), id="multipoint"),
        pytest.param(LineString([(0, 0), (1, 1)]), id="linestring"),
        # first and last x are equal, but it is not closed: used to come back as a polygon
        pytest.param(LineString([(0, 0), (1, 1), (0, 2)]), id="linestring_same_x_at_ends"),
        pytest.param(MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]), id="multilinestring"),
        pytest.param(box(0, 0, 1, 1), id="polygon"),
        pytest.param(box(0, 0, 4, 4).difference(box(1, 1, 2, 2)), id="polygon_with_hole"),
        pytest.param(MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), id="multipolygon"),
    ],
)
def test_geometrylist_round_trip(geometry: BaseGeometry) -> None:
    result = GeometryList.from_geometry(geometry).to_geometry()
    assert result.geom_type == geometry.geom_type
    assert result.equals(geometry)
