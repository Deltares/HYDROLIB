"""Conversion between shapely geometries and meshkernel's GeometryList."""

from collections.abc import Iterator
from typing import Self

import numpy as np
from meshkernel import GeometryList as GeometryListMK
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from hydrolib.core.dflowfm.net.models import split_by


class GeometryList(GeometryListMK):
    """meshkernel GeometryList that converts from and to shapely geometries.

    The coordinates of all parts are stored in one array. Parts of a multi-geometry
    are separated by ``geometry_separator`` (-999), the holes of a polygon by
    ``inner_outer_separator`` (-998).
    """

    @classmethod
    def from_geometry(cls, geometry: BaseGeometry) -> Self:
        """Create a GeometryList from a shapely geometry.

        Parameters
        ----------
        geometry : BaseGeometry
            A Point, LineString or Polygon, or a multi-geometry of those.

        Returns
        -------
        GeometryList
            The coordinates of the geometry.

        Raises
        ------
        TypeError
            If the geometry is of another type, e.g. a GeometryCollection.
        """
        if isinstance(geometry, Polygon):
            return cls.from_polygon(geometry)
        elif isinstance(geometry, MultiPolygon):
            return cls.from_multipolygon(geometry)
        elif isinstance(geometry, LineString):
            return cls.from_linestring(geometry)
        elif isinstance(geometry, MultiLineString):
            return cls.from_multilinestring(geometry)
        elif isinstance(geometry, Point):
            return cls.from_point(geometry)
        elif isinstance(geometry, MultiPoint):
            return cls.from_multipoint(geometry)
        else:
            raise TypeError(f"Geometry type {type(geometry)} not understood.")

    @classmethod
    def _from_simple(cls, geometry: LineString | Point) -> Self:
        # Extract coordinates from geometry
        x_crds, y_crds = np.array(geometry.coords[:]).T
        gl = cls(x_coordinates=x_crds, y_coordinates=y_crds)
        return gl

    @classmethod
    def from_linestring(cls, linestring: LineString) -> Self:
        """Create a GeometryList from a LineString."""
        return cls._from_simple(linestring)

    @classmethod
    def from_point(cls, point: Point) -> Self:
        """Create a GeometryList from a Point."""
        return cls._from_simple(point)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> Self:
        """Create a GeometryList from a Polygon: the exterior, then each hole after a separator."""
        # Create a list of coordinate lists
        # Add exterior
        x_ext, y_ext = np.array(polygon.exterior.coords[:]).T
        x_crds = [x_ext]
        y_crds = [y_ext]

        if not hasattr(cls, "inner_outer_separator"):
            cls.inner_outer_separator = -998

        # Add interiors, seperated by inner_outer_separator
        for interior in polygon.interiors:
            x_int, y_int = np.array(interior.coords[:]).T
            x_crds.append([cls.inner_outer_separator])
            x_crds.append(x_int)
            y_crds.append([cls.inner_outer_separator])
            y_crds.append(y_int)
        gl = cls(x_coordinates=np.concatenate(x_crds), y_coordinates=np.concatenate(y_crds))
        return gl

    @classmethod
    def _from_multigeometry(cls, multigeometry: MultiLineString | MultiPolygon | MultiPoint) -> Self:
        x_crds = []
        y_crds = []
        cls.inner_outer_separator = -998
        cls.geometry_separator = -999
        # Create a GeometryList for every polygon in the multipolygon
        for i, geometry in enumerate(multigeometry.geoms):
            gl = cls.from_geometry(geometry)
            # Add geometry seperator for every geometrylist, except the first one
            if i != 0:
                x_crds.append([cls.geometry_separator])
                y_crds.append([cls.geometry_separator])
            # Add coordinates of polygon GeometryList
            x_crds.append(gl.x_coordinates)
            y_crds.append(gl.y_coordinates)

        gl = cls(x_coordinates=np.concatenate(x_crds), y_coordinates=np.concatenate(y_crds))
        return gl

    @classmethod
    def from_multipolygon(cls, multipolygon: MultiPolygon) -> Self:
        """Create a GeometryList from a MultiPolygon, with a separator between the polygons."""
        return cls._from_multigeometry(multipolygon)

    @classmethod
    def from_multilinestring(cls, multilinestring: MultiLineString) -> Self:
        """Create a GeometryList from a MultiLineString, with a separator between the lines."""
        return cls._from_multigeometry(multilinestring)

    @classmethod
    def from_multipoint(cls, multipoint: MultiPoint) -> Self:
        """Create a GeometryList from a MultiPoint, with a separator between the points."""
        return cls._from_multigeometry(multipoint)

    def _to_polygon(self, geometries: list[GeometryListMK], is_multi: bool) -> Polygon | MultiPolygon:
        polygons = []
        for geometry in geometries:
            parts = [
                np.stack([p.x_coordinates, p.y_coordinates], axis=1)
                for p in split_by(geometry, self.inner_outer_separator)
            ]
            polygons.append(Polygon(shell=parts[0], holes=parts[1:]))
        if is_multi:
            return MultiPolygon(polygons)
        else:
            return polygons[0]

    def _to_linestring(self, geometries: list[GeometryListMK], is_multi: bool) -> LineString | MultiLineString:
        linestrings = [LineString(np.stack([p.x_coordinates, p.y_coordinates], axis=1)) for p in geometries]
        if is_multi:
            return MultiLineString(linestrings)
        else:
            return linestrings[0]

    def _to_points(self, geometries: list[GeometryListMK], is_multi: bool) -> Point | MultiPoint:
        points = [Point(np.stack([p.x_coordinates, p.y_coordinates], axis=1)) for p in geometries]
        if is_multi:
            return MultiPoint(points)
        else:
            return points[0]

    def to_geometry(self) -> Polygon | MultiPolygon | LineString | MultiLineString | Point | MultiPoint:
        """Convert the GeometryList to a shapely geometry.

        The first part determines the type: a closed ring of at least 4 coordinates
        is a polygon, more than one coordinate a line, otherwise a point. With more
        than one part, the result is a multi-geometry.

        Returns
        -------
        Polygon, MultiPolygon, LineString, MultiLineString, Point or MultiPoint
            The geometry.
        """
        geometries = [geo for geo in split_by(self, self.geometry_separator) if geo.x_coordinates.size > 0]
        is_multi = len(geometries) > 1

        # The first geometry determines the type
        exterior = split_by(geometries[0], self.inner_outer_separator)[0]
        x, y = exterior.x_coordinates, exterior.y_coordinates
        # A ring has at least 4 coordinates, and its first and last coordinate are equal
        if len(x) >= 4 and x[0] == x[-1] and y[0] == y[-1]:
            return self._to_polygon(geometries, is_multi)
        if len(x) > 1:
            return self._to_linestring(geometries, is_multi)
        return self._to_points(geometries, is_multi)

    @property
    def geoms(self) -> Iterator[Polygon | MultiPolygon | LineString | MultiLineString | Point | MultiPoint]:
        """Yield each part, split at the geometry separator, as a shapely geometry."""
        for geometrylist in split_by(self, self.geometry_separator):
            # yield as shapely geometry
            yield GeometryList(**geometrylist.__dict__).to_geometry()
