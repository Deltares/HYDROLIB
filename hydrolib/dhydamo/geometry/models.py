from meshkernel import GeometryList as GeometryListMK
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)
import numpy as np
from typing import Union, List
from hydrolib.core.io.net.models import split_by


class GeometryList(GeometryListMK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_geometry(cls, geometry):
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
    def _from_simple(cls, geometry: Union[LineString, Point]):
        # Extract coordinates from geometry
        x_crds, y_crds = np.array(geometry.coords[:]).T
        gl = cls(x_coordinates=x_crds, y_coordinates=y_crds)
        return gl

    @classmethod
    def from_linestring(cls, linestring: LineString):
        return cls._from_simple(linestring)

    @classmethod
    def from_point(cls, point: Point):
        return cls._from_simple(point)

    @classmethod
    def from_polygon(cls, polygon: Polygon):
        # Create a list of coordinate lists
        # Add exterior
        x_ext, y_ext = np.array(polygon.exterior.coords[:]).T
        x_crds = [x_ext]
        y_crds = [y_ext]
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
    def _from_multigeometry(cls, multigeometry: Union[MultiLineString, MultiPolygon]):
        x_crds = []
        y_crds = []
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
    def from_multipolygon(cls, multipolygon: MultiPolygon):
        return cls._from_multigeometry(multipolygon)

    @classmethod
    def from_multilinestring(cls, multilinestring: LineString):
        return cls._from_multigeometry(multilinestring)

    @classmethod
    def from_multipoint(cls, multipoint: Point):
        return cls._from_multigeometry(multipoint)

    def _to_polygon(self, geometries: List[GeometryListMK], is_multi: bool) -> Union[Polygon, MultiPolygon]:
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

    def _to_linestring(
        self, geometries: List[GeometryListMK], is_multi: bool
    ) -> Union[LineString, MultiLineString]:
        linestrings = [LineString(np.stack([p.x_coordinates, p.y_coordinates], axis=1)) for p in geometries]
        if is_multi:
            return MultiLineString(linestrings)
        else:
            return linestrings[0]

    def _to_points(self, geometries: List[GeometryListMK], is_multi: bool) -> Union[Point, MultiPoint]:
        points = [Point(np.stack([p.x_coordinates, p.y_coordinates], axis=1)) for p in geometries]
        if is_multi:
            return MultiPoint(points)
        else:
            return points[0]

    def to_geometry(self):
        geometries = [geo for geo in split_by(self, self.geometry_separator) if geo.x_coordinates.size > 0]
        is_multi = len(geometries) > 1

        for geometry_list in geometries:
            # Check if polygon, by comparing first and last coordinates
            exterior = split_by(geometry_list, self.inner_outer_separator)[0]
            is_polygon = exterior.x_coordinates[0] == exterior.x_coordinates[-1]
            # Check if linestring, by checking the length of coordinate sequences
            is_linestring = (len(geometry_list.x_coordinates) > 1) and (not is_polygon)

            if is_polygon:
                return self._to_polygon(geometries, is_multi)

            elif is_linestring:
                return self._to_linestring(geometries, is_multi)

            else:
                return self._to_point(geometries, is_multi)

    @property
    def geoms(self):
        """Returns GeometryList objects based on spliited by geometries"""
        for geometrylist in split_by(self, self.geometry_separator):
            # yield as shapely geometry
            yield GeometryList(**geometrylist.__dict__).to_geometry()
