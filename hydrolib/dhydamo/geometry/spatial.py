import logging

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely import affinity
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.prepared import prep

from hydrolib.dhydamo.geometry import common

logger = logging.getLogger(__name__)


def rotate_coordinates(origin, theta, xcrds, ycrds):
    """
    Rotate coordinates around origin (x0, y0) with a certain angle (radians)
    """
    x0, y0 = origin
    xcrds_rot = x0 + (xcrds - x0) * np.cos(theta) + (ycrds - y0) * np.sin(theta)
    ycrds_rot = y0 - (xcrds - x0) * np.sin(theta) + (ycrds - y0) * np.cos(theta)
    return xcrds_rot, ycrds_rot


def minimum_bounds_fixed_rotation(polygon, angle):
    """Get the minimum box for a polygon with a given axes rotation.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon that is rotated
    angle : int or float
        Rotation of the polygon in degrees

    Returns
    -------
    tuple
        Tuple with origin (x, y), xsize and ysize
    """
    # Determine spinning point
    spinpt = (polygon.envelope.bounds[0], polygon.envelope.bounds[1])

    # Rotate clip polygon with rotation, get envelope and rotate back.
    rotbox1 = affinity.rotate(polygon, angle=angle, origin=spinpt).envelope

    # Determine size of grid
    xsize = rotbox1.bounds[2] - rotbox1.bounds[0]
    ysize = rotbox1.bounds[3] - rotbox1.bounds[1]

    # Rotate again, and get origin
    rotbox2 = affinity.rotate(rotbox1, angle=-angle, origin=spinpt)
    origin = rotbox2.exterior.coords[0]

    return origin, xsize, ysize

def find_nearest_branch(branches, geometries, method='overal', maxdist=5):
    """
    Method to determine nearest branch for each geometry.
    The nearest branch can be found by finding t from both ends (ends) or the nearest branch from the geometry
    as a whole (overal), the centroid (centroid), or intersecting (intersect).

    Parameters
    ----------
    branches : geopandas.GeoDataFrame
        Geodataframe with branches
    geometries : geopandas.GeoDataFrame
        Geodataframe with geometries to snap
    method='overal' : str
        Method for determine branch
    maxdist=5 : int or float
        Maximum distance for finding nearest geometry
    minoffset : int or float
        Minimum offset from the end of the corresponding branch in case of method=equal
    """
    # Check if method is in allowed methods
    allowed_methods = ['intersecting', 'overal', 'centroid', 'ends']
    if method not in allowed_methods:
        raise NotImplementedError(f'Method "{method}" not implemented.')

    # Add columns if not present
    if 'branch_id' not in geometries.columns:
        geometries['branch_id'] = ''
    if 'branch_offset' not in geometries.columns:
        geometries['branch_offset'] = np.nan

    if method == 'intersecting':
        # Determine intersection geometries per branch
        sindex = geometries.sindex
        for branch in branches.itertuples():
            intersecting = geometries.iloc[sindex.query(branch.geometry, predicate="intersects")]

            # For each geometrie, determine offset along branch
            for geometry in intersecting.itertuples():
                # Determine distance of profile line along branch
                geometries.at[geometry.Index, 'branch_id'] = branch.Index

                # Calculate offset
                branchgeo = branch.geometry
                mindist = min(0.1, branchgeo.length / 2.)
                offset = round(branchgeo.project(branchgeo.intersection(geometry.geometry).centroid), 3)
                offset = max(mindist, min(branchgeo.length - mindist, offset))
                geometries.at[geometry.Index, 'branch_offset'] = offset

    else:
        sindex = branches.sindex
        # In case of looking for the nearest, it is easier to iteratie over the geometries instead of the branches
        for geometry in geometries.itertuples():
            # Find near branches
            nearidx = sindex.query(geometry.geometry, predicate="dwithin", distance=maxdist)
            selectie = branches.iloc[nearidx]

            if method == 'overal':
                # Determine distances to branches
                dist = selectie.distance(geometry.geometry)
            elif method == 'centroid':
                # Determine distances to branches
                dist = selectie.distance(geometry.geometry.centroid)
            elif method == 'ends':
                # Since a culvert can cross a channel, it is
                crds = geometry.geometry.coords[:]
                dist = (selectie.distance(Point(*crds[0])) + selectie.distance(Point(*crds[-1]))) * 0.5

            # Enforce a point
            geo = geometry.geometry
            if not isinstance(geo, Point):
                geo = geo.centroid

            # Log-friendly name with coordinates
            geom_name = f"at x={geo.x:.4f} and y={geo.y:.4f}"
            if hasattr(geometry, "code"):
                geom_name = f"for {geometry.code} {geom_name}"

            # Determine nearest
            if dist.min() < maxdist:
                branchidxmin = dist.idxmin()
                geometries.at[geometry.Index, 'branch_id'] = branchidxmin

                nbranches = (dist < maxdist).sum()
                if nbranches > 1:
                    logger.info(
                        "Centroid %s has %d branches with maxdist=%s and method=%s, chosing closest branch %s",
                        geom_name,
                        nbranches,
                        maxdist,
                        method,
                        branchidxmin,
                    )

                # Calculate offset
                branchgeo = branches.at[branchidxmin, 'geometry']
                mindist = min(0.1, branchgeo.length / 2.)
                offset = max(mindist, min(branchgeo.length - mindist, round(branchgeo.project(geo), 3)))
                geometries.at[geometry.Index, 'branch_offset'] = offset
            else:
                logger.info(
                    "Centroid %s has 0 branches with maxdist=%s and method=%s",
                    geom_name,
                    maxdist,
                    method,
                )

def orthogonal_line(line: LineString, offset: float, width: float=1.0) -> list[tuple[float]]:
    """
    Parameters
    ----------
    line : shapely.geometry.LineString
        Line geometry object on which the orthogonal line is drawn
    offset : float
        Offset of the orthogonal line along line
    width : float
        Width of the orthogonal line

    Returns
    -------
    line : list
        List with coordinate tuples
    """

    # Determine angle at offset
    angle = np.angle(complex(*np.diff([
        line.interpolate(offset - 0.001).coords[0][:2],
        line.interpolate(offset + 0.001).coords[0][:2]
    ], axis=0)[0])) + 0.5 * np.pi

    # Create new line
    pt = line.interpolate(offset).coords[0]

    f = 0.5 * width
    line = [(pt[0] + np.cos(angle) * f, pt[1] + np.sin(angle) * f), (pt[0] - np.cos(angle) * f, pt[1] - np.sin(angle) * f)]

    return line

def extend_linestring(line: LineString, near_pt: Point, length: float) -> LineString:

    # Get the nearest end
    nearest_end = (0, 1) if line.project(near_pt) < line.length / 2 else (-1, -2)

    # Extrapolate the end 1 meter, and create a perpendicular line
    coords = line.coords[:]
    x0, y0 = coords[nearest_end[0]]
    dx, dy = np.diff(np.vstack([coords[nearest_end[0]], coords[nearest_end[1]]]), axis=0)[0]
    segmentlength = (dx**2 + dy**2)**0.5
    dx /= (segmentlength * length)
    dy /= (segmentlength * length)

    return LineString([(x0, y0), (x0 - dx, y0 - dy)])

def get_voronoi_around_nodes(nodes: np.ndarray, facedata: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Creates voronoi polygons around face nodes.

    Args:
        nodes (np.ndarray): xy coordinates of face nodes
        facedata (gpd.GeoDataFrame): GeoDataFrame with face properties

    Returns:
        gpd.GeoDataFrame: Creating GeoDataFrame with created polygons and their properties
    """
    # Creat voronoi polygon
    # Add border to limit polygons
    border = box(nodes[:, 0].min(), nodes[:, 1].min(), nodes[:, 0].max(), nodes[:, 1].max()).buffer(1000).exterior
    borderpts = [border.interpolate(dist).coords[0] for dist in np.linspace(0, border.length, max(20, int(border.length / 100)))]
    vor = Voronoi(points=np.vstack([nodes, borderpts]))
    clippoly = facedata.union_all()
    # Get lines
    lines = []
    for poly in common.as_polygon_list(clippoly):
        lines.append(poly.exterior)
        lines.extend([line for line in poly.interiors])
    linesprep = prep(MultiLineString(lines))

    # Determine for all nodes at once whether they lie within the clip polygon
    inside = shapely.intersects(clippoly, shapely.points(nodes))

    # Collect polygons
    data = []
    for (pr, pt_inside) in zip(vor.point_region, inside):
        region = vor.regions[pr]
        if pr == -1:
            break
        region = [v for v in region if v != -1]
        if len(region) < 3:
            continue
        if pt_inside:
            crds = vor.vertices[region]
            poly = Polygon(crds)
            if linesprep.intersects(poly):
                poly = poly.intersection(clippoly)
                if isinstance(poly, MultiPolygon):
                    poly = poly.buffer(0.001)
                if isinstance(poly, MultiPolygon):
                    logger.warning('Got multipolygon when clipping voronoi polygon. Only adding coordinates for largest of the polygons.')
                    poly = poly[np.argmax([p.area for p in common.as_polygon_list(poly)])]
                crds = np.vstack(poly.exterior.coords[:])
            data.append({'geometry': poly, 'crds': crds})
            
    # Limit to model extend
    facedata = gpd.GeoDataFrame(data)
    facedata.index=np.arange(len(nodes), dtype=np.uint32) + 1

    return facedata
