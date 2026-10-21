import logging

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin
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


def rasterize_polygons(
    polygons: gpd.GeoDataFrame, column: str, cellsize: float, fill: float = -999.0
) -> tuple[np.ndarray, rasterio.Affine]:
    """Rasterize the values in a column of a polygon GeoDataFrame.

    The raster extent is aligned to a multiple of the cellsize. Where polygons
    overlap, the last polygon in the GeoDataFrame determines the value.

    Args:
        polygons (gpd.GeoDataFrame): (multi)polygons with values
        column (str): column with the values to burn into the raster
        cellsize (float): raster cell size
        fill (float, optional): value for cells outside the polygons. Defaults to -999.0.

    Returns:
        tuple[np.ndarray, rasterio.Affine]: float32 array and its transform
    """
    xmin, ymin, xmax, ymax = polygons.total_bounds
    xmin = np.floor(xmin / cellsize) * cellsize
    ymin = np.floor(ymin / cellsize) * cellsize
    xmax = np.ceil(xmax / cellsize) * cellsize
    ymax = np.ceil(ymax / cellsize) * cellsize
    ncols = max(int(round((xmax - xmin) / cellsize)), 1)
    nrows = max(int(round((ymax - ymin) / cellsize)), 1)

    transform = from_origin(xmin, ymax, cellsize, cellsize)
    array = features.rasterize(
        zip(polygons.geometry, polygons[column].astype(float)),
        out_shape=(nrows, ncols),
        transform=transform,
        fill=fill,
        dtype="float32",
    )
    return array, transform


def _line_parts(geometry) -> list[LineString]:
    """All LineStrings in a geometry, including those in a GeometryCollection; points are ignored."""
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for geom in geometry.geoms for part in _line_parts(geom)]
    return []

def polygons_to_branch_values(
    branches: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, column: str, step: float = 0.01
) -> dict[str, tuple[list[float], list[float]]]:
    """Convert polygon values to values per branch and chainage, e.g. for a 1dField file.

    Each part of a branch inside a polygon gets the value of that polygon. At polygon boundaries two
    chainages are written, `step` apart, so the value changes stepwise. Where polygons overlap, the last
    polygon in the GeoDataFrame determines the value. Parts of a branch outside all polygons get no
    chainage: D-Flow FM interpolates linearly between the given chainages and keeps the value constant
    beyond the first and last chainage of a branch. Branches that do not intersect any polygon are left out.

    Args:
        branches (gpd.GeoDataFrame): branches, with the branch id as index
        polygons (gpd.GeoDataFrame): (multi)polygons with values
        column (str): column with the values
        step (float, optional): distance between the two chainages at a polygon boundary. Defaults to 0.01.

    Returns:
        dict[str, tuple[list[float], list[float]]]: chainages and values per branch id
    """
    polygons = polygons.reset_index(drop=True)
    values = polygons[column].astype(float).to_numpy()
    result = {}
    for branch_id, line in zip(branches.index, branches.geometry):
        candidates = sorted(polygons.sindex.query(line, predicate="intersects"))
        if len(candidates) == 0:
            continue
        length = line.length

        # intervals (start, end) along the branch per polygon, in the order of the polygons
        intervals = []
        for i in candidates:
            for part in _line_parts(line.intersection(polygons.geometry.iloc[i])):
                if part.length < step:
                    continue
                chainages = [line.project(Point(c)) for c in part.coords]
                intervals.append((min(chainages), max(chainages), i))
        if not intervals:
            continue

        # value per sub-interval between breakpoints: the last polygon covering it
        breakpoints = sorted({0.0, length} | {s for s0, s1, _ in intervals for s in (s0, s1)})
        segments = []
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            if b - a < step:
                continue
            mid = 0.5 * (a + b)
            covering = [i for s0, s1, i in intervals if s0 <= mid <= s1]
            if not covering:
                continue
            value = values[max(covering)]
            if segments and np.isclose(segments[-1][1], a) and segments[-1][2] == value:
                segments[-1][1] = b
            else:
                segments.append([a, b, value])

        chainage, branch_values = [], []
        for a, b, value in segments:
            start = a if np.isclose(a, 0.0) else a + 0.5 * step
            end = b if np.isclose(b, length) else b - 0.5 * step
            chainage += [start, end]
            branch_values += [value, value]
        result[branch_id] = (chainage, branch_values)
    return result


def points_to_branch_values(
    branches: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    column: str,
    max_distance: float = 5.0,
    junction_tolerance: float = 0.1,
) -> dict[str, tuple[list[float], list[float]]]:
    """Convert point values to values per branch and chainage, e.g. for a 1dField file.

    If the points have columns 'branchid' and 'chainage' (e.g. from a model result on the same network),
    these are used. Otherwise each point is snapped to the nearest branch within `max_distance` and its
    chainage is the projection on that branch. D-Flow FM interpolates linearly between the chainages of a
    branch and keeps the value constant beyond the first and last chainage.

    Args:
        branches (gpd.GeoDataFrame): branches, with the branch id as index
        points (gpd.GeoDataFrame): points with values
        column (str): column with the values
        max_distance (float, optional): maximum distance to snap a point to a branch. Defaults to 5.0.
        junction_tolerance (float, optional): distance within which a point at a branch end is also used for the
            other branches at that junction. Defaults to 0.1.

    Returns:
        dict[str, tuple[list[float], list[float]]]: chainages and values per branch id
    """
    if {"branchid", "chainage"} <= set(points.columns):
        data = points[["branchid", "chainage", column]].copy()
        unknown = ~data["branchid"].isin(branches.index)
        if unknown.any():
            logger.warning(f"{unknown.sum()} points refer to branches that are not in the model and are left out.")
            data = data[~unknown]
    else:
        # first row: index of the point, second row: index of the nearest branch
        point_idx, branch_idx = branches.sindex.nearest(points.geometry, max_distance=max_distance, return_all=False)
        if len(point_idx) < len(points):
            logger.warning(
                f"{len(points) - len(point_idx)} points are further than {max_distance} m from a branch and are left out."
            )
        geometries = branches.geometry.to_numpy()
        data = points.iloc[point_idx][[column]].copy()
        data["branchid"] = branches.index.to_numpy()[branch_idx]
        data["chainage"] = [
            geometries[b].project(p) for b, p in zip(branch_idx, points.geometry.to_numpy()[point_idx])
        ]

    result = {}
    for branch_id, group in data.groupby("branchid", sort=False):
        group = group.sort_values("chainage").drop_duplicates("chainage", keep="last")
        result[branch_id] = (group["chainage"].astype(float).tolist(), group[column].astype(float).tolist())

    # A point at a junction belongs to one branch only. Give the other branches at that junction the same
    # value at their end, otherwise D-Flow FM extrapolates from their first inner point to the junction.
    geometries = branches.geometry
    for branch_id, (chainage, values) in result.items():
        line = geometries.loc[branch_id]
        for end_chainage, end_point in ((0.0, Point(line.coords[0])), (line.length, Point(line.coords[-1]))):
            if any(abs(c - end_chainage) < junction_tolerance for c in chainage):
                continue
            nearby = points.sindex.query(end_point.buffer(junction_tolerance), predicate="intersects")
            if len(nearby) == 0:
                continue
            position = int(np.searchsorted(chainage, end_chainage))
            chainage.insert(position, end_chainage)
            values.insert(position, float(points[column].iloc[nearby[0]]))
    return result
