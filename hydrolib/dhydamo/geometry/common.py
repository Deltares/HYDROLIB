"""Split multi-geometries into lists and add vertices to lines and polygons."""

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
    """Return the LineStrings in a LineString, a MultiLineString or a list of those.

    Useful to iterate over geometries that can be single or multi.

    Parameters
    ----------
    linestring : LineString, MultiLineString or list of those
        Geometry, or geometries, to split.

    Returns
    -------
    list[LineString]
        The single LineStrings, in order.

    Raises
    ------
    TypeError
        If a geometry is not a LineString or MultiLineString.
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
    """Return the Polygons in a Polygon, a MultiPolygon or a list of those.

    Useful to iterate over geometries that can be single or multi.

    Parameters
    ----------
    polygon : Polygon, MultiPolygon or list of those
        Geometry, or geometries, to split.

    Returns
    -------
    list[Polygon]
        The single Polygons, in order.

    Raises
    ------
    TypeError
        If a geometry is not a Polygon or MultiPolygon.
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
    """Return the Points in a Point, a MultiPoint or a list of those.

    Useful to iterate over geometries that can be single or multi.

    Parameters
    ----------
    point : Point, MultiPoint or list of those
        Geometry, or geometries, to split.

    Returns
    -------
    list[Point]
        The single Points, in order.

    Raises
    ------
    TypeError
        If a geometry is not a Point or MultiPoint.
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
    """Resample a line at regular distances along it.

    Parameters
    ----------
    linestring : LineString or MultiLineString
        Line to resample. The parts of a MultiLineString are resampled one by one.
    dist : float
        Desired distance between the vertices. The line gets ``round(length / dist)``
        vertices, at least 4, evenly spread from start to end.

    Returns
    -------
    LineString or MultiLineString
        The resampled line, of the same type as the input.
    """
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
    """Simplify a polygon, then resample its exterior and holes at regular distances.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Polygon to resample. The parts of a MultiPolygon are resampled one by one.
    dist : float
        Tolerance for the simplification, and desired distance between the vertices,
        see `interp_linestring`.

    Returns
    -------
    Polygon or MultiPolygon
        The resampled polygon, of the same type as the input.

    Raises
    ------
    TypeError
        If simplifying the polygon does not give a Polygon.
    """
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
