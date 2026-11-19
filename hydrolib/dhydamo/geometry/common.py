from typing import overload

import numpy as np
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def as_linestring_list(
    linestring: LineString | MultiLineString | list[LineString | MultiLineString]
) -> list[LineString]:
    """Returns a list of LineStrings from a given LineString or MultiLineString. Useful for
    iterating over varying multi of single type geometries.

    Args:
        linestring (Union[LineString, MultiLineString]): LineString of MultiLineString shapely geometry object

    Returns:
        List[LineString]: list of shapely geometry LineString objects
    """
    if isinstance(linestring, LineString):
        return [linestring]
    if isinstance(linestring, MultiLineString):
        return list(linestring.geoms)
    if not isinstance(linestring, list):
        raise TypeError(f'Expected LineString or MultiLineString. Got "{type(linestring)}"')
    linestrings: list[LineString] = []
    for item in linestring:
        linestrings.extend(as_linestring_list(item))
    return linestrings


def as_polygon_list(
    polygon: Polygon | MultiPolygon | list[Polygon | MultiPolygon]
) -> list[Polygon]:
    """Returns a list of Polygons from a given Polygon or MultiPolygon. Useful for
    iterating over varying multi of single type geometries.

    Args:
        linestring (Union[Polygon, MultiPolygon]): Polygon of MultiPolygon shapely geometry object

    Returns:
        List[Polygon]: list of shapely geometry Polygon objects
    """
    if isinstance(polygon, Polygon):
        return [polygon]
    if isinstance(polygon, MultiPolygon):
        return list(polygon.geoms)
    if not isinstance(polygon, list):
        raise TypeError(f'Expected Polygon or MultiPolygon. Got "{type(polygon)}"')
    polygons: list[Polygon] = []
    for item in polygon:
        polygons.extend(as_polygon_list(item))
    return polygons


def as_point_list(
    point: Point | MultiPoint | list[Point | MultiPoint]
) -> list[Point]:
    """Returns a list of Point from a given Point or MultiPoint. Useful for
    iterating over varying multi of single type geometries.

    Args:
        linestring (Union[Point, MultiPoint]): Point of MultiPoint shapely geometry object

    Returns:
        List[Point]: list of shapely geometry Point objects
    """
    if isinstance(point, Point):
        return [point]
    if isinstance(point, MultiPoint):
        return list(point.geoms)
    if not isinstance(point, list):
        raise TypeError(f'Expected Point or MultiPoint. Got "{type(point)}"')
    points: list[Point] = []
    for item in point:
        points.extend(as_point_list(item))
    return points


@overload
def interp_linestring(linestring: LineString, dist: float) -> LineString: ...
@overload
def interp_linestring(linestring: MultiLineString, dist: float) -> MultiLineString: ...
def interp_linestring(linestring: LineString | MultiLineString, dist: float) -> LineString | MultiLineString:
    # In case of multilinestring
    if isinstance(linestring, MultiLineString):
        return MultiLineString([interp_linestring(ls, dist) for ls in linestring.geoms])

    n_points = max(4, round(linestring.length / dist))
    distances = np.linspace(0, linestring.length, n_points)
    points = shapely.line_interpolate_point(linestring, distances)
    return LineString(points)


@overload
def interp_polygon(polygon: Polygon, dist: float) -> Polygon: ...
@overload
def interp_polygon(polygon: MultiPolygon, dist: float) -> MultiPolygon: ...
def interp_polygon(polygon: Polygon | MultiPolygon, dist: float) -> Polygon | MultiPolygon:
    # In case of multipolygon
    if isinstance(polygon, MultiPolygon):
        return MultiPolygon([interp_polygon(p, dist) for p in polygon.geoms])

    simplified = polygon.simplify(dist)
    if not isinstance(simplified, Polygon):
        raise TypeError(f"Simplifying a Polygon gave a {simplified.geom_type}")
    polygon = Polygon(
        shell=interp_linestring(simplified.exterior, dist),
        holes=[interp_linestring(intr, dist) for intr in simplified.interiors],
    )
    return polygon
