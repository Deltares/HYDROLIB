from typing import List, Union
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)
import numpy as np


def _as_geometry_list(geometry, singletype, multitype):
    """Convenience method to return a list with one or more

    Polygons/LineString/Point from a given Polygon/LineString/Point
    or MultiPolygon/MultiLineString/MultiPoint.

    Parameters
    ----------
    polygon : list or Polygon or MultiPolygon
        Object to be converted

    Returns
    -------
    list
        list of Polygons
    """
    if isinstance(geometry, singletype):
        return [geometry]
    elif isinstance(geometry, multitype):
        return [p for p in geometry.geoms]
    elif isinstance(geometry, list):
        lst = []
        for item in geometry:
            lst.extend(_as_geometry_list(item, singletype, multitype))
        return lst
    else:
        raise TypeError(f'Expected {singletype} or {multitype}. Got "{type(geometry)}"')


def as_linestring_list(
    linestring: Union[
        LineString, MultiLineString, List[Union[LineString, MultiLineString]]
    ]
) -> List[LineString]:
    """Returns a list of LineStrings from a given LineString or MultiLineString. Useful for
    iterating over varying multi of single type geometries.

    Args:
        linestring (Union[LineString, MultiLineString]): LineString of MultiLineString shapely geometry object

    Returns:
        List[LineString]: list of shapely geometry LineString objects
    """
    return _as_geometry_list(linestring, LineString, MultiLineString)


def as_polygon_list(
    polygon: Union[Polygon, MultiPolygon, List[Union[Polygon, MultiPolygon]]]
) -> List[Polygon]:
    """Returns a list of Polygons from a given Polygon or MultiPolygon. Useful for
    iterating over varying multi of single type geometries.

    Args:
        linestring (Union[Polygon, MultiPolygon]): Polygon of MultiPolygon shapely geometry object

    Returns:
        List[Polygon]: list of shapely geometry Polygon objects
    """
    return _as_geometry_list(polygon, Polygon, MultiPolygon)


def as_point_list(
    point: Union[Point, MultiPoint, List[Union[Point, MultiPoint]]]
) -> List[Point]:
    """Returns a list of Point from a given Point or MultiPoint. Useful for
    iterating over varying multi of single type geometries.

    Args:
        linestring (Union[Point, MultiPoint]): Point of MultiPoint shapely geometry object

    Returns:
        List[Point]: list of shapely geometry Point objects
    """
    return _as_geometry_list(point, Point, MultiPoint)


def interp_linestring(linestring, dist):
    # In case of multilinestring
    if isinstance(linestring, MultiLineString):
        return MultiLineString([interp_linestring(l, dist) for l in linestring])

    return LineString(
        [
            linestring.interpolate(x)
            for x in np.linspace(
                0, linestring.length, max(4, round(linestring.length / dist))
            )
        ]
    )


def interp_polygon(polygon, dist):
    # In case of multipolygon
    if isinstance(polygon, MultiPolygon):
        return MultiPolygon([interp_polygon(p, dist) for p in polygon])

    simplified = polygon.simplify(dist)
    polygon = Polygon(
        shell=interp_linestring(simplified.exterior, dist),
        holes=[interp_linestring(intr, dist) for intr in simplified.interiors],
    )
    return polygon
