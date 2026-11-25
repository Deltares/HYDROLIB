"""Spatial operations on branches, polygons and points."""

import itertools
import logging
from collections.abc import Hashable

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


def rotate_coordinates(
    origin: tuple[float, float], theta: float, xcrds: np.ndarray, ycrds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates clockwise around an origin.

    Parameters
    ----------
    origin : tuple[float, float]
        Point (x0, y0) to rotate around.
    theta : float
        Angle in radians.
    xcrds, ycrds : np.ndarray
        Coordinates to rotate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The rotated x and y coordinates.
    """
    x0, y0 = origin
    xcrds_rot = x0 + (xcrds - x0) * np.cos(theta) + (ycrds - y0) * np.sin(theta)
    ycrds_rot = y0 - (xcrds - x0) * np.sin(theta) + (ycrds - y0) * np.cos(theta)
    return xcrds_rot, ycrds_rot


def minimum_bounds_fixed_rotation(
    polygon: Polygon, angle: float
) -> tuple[tuple[float, float], float, float]:
    """Get the minimum box for a polygon with a given axes rotation.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon that is rotated.
    angle : float
        Rotation of the polygon in degrees.

    Returns
    -------
    tuple[tuple[float, float], float, float]
        Origin (x, y), xsize and ysize of the box.

    Raises
    ------
    TypeError
        If rotating the box back does not give a Polygon.
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
    if not isinstance(rotbox2, Polygon):
        raise TypeError(
            f"Expected a Polygon after rotating the envelope, got a {rotbox2.geom_type}"
        )
    x0, y0 = rotbox2.exterior.coords[0]

    return (x0, y0), xsize, ysize


def find_nearest_branch(
    branches: gpd.GeoDataFrame,
    geometries: gpd.GeoDataFrame,
    method: str = "overal",
    maxdist: float = 5,
) -> None:
    """Find the nearest branch for each geometry, in place.

    The columns ``branch_id`` and ``branch_offset`` of ``geometries`` are filled in, and
    added if missing. The offset is the chainage of the geometry along its branch, at
    least 0.1 m (or half the branch length) from both branch ends. Geometries without a
    branch within ``maxdist`` are logged and left as they are.

    Parameters
    ----------
    branches : geopandas.GeoDataFrame
        Branches, with the branch id as index.
    geometries : geopandas.GeoDataFrame
        Geometries to snap. With a ``code`` column, the code is used in the log messages.
    method : str, optional
        How the nearest branch is found. Defaults to ``"overal"``.

        - ``"overal"``: the nearest branch to the geometry as a whole;
        - ``"centroid"``: the nearest branch to the centroid;
        - ``"ends"``: the smallest mean distance to both ends of a line;
        - ``"intersecting"``: each branch that intersects the geometry; the offset is
          at the centroid of the intersection, and a geometry intersecting more
          branches gets the last one.
    maxdist : float, optional
        Maximum distance to a branch. Not used for ``"intersecting"``. Defaults to 5.

    Raises
    ------
    NotImplementedError
        If the method is not one of the above.
    """
    # Check if method is in allowed methods
    allowed_methods = ["intersecting", "overal", "centroid", "ends"]
    if method not in allowed_methods:
        raise NotImplementedError(f'Method "{method}" not implemented.')

    # Add columns if not present
    if "branch_id" not in geometries.columns:
        geometries["branch_id"] = ""
    if "branch_offset" not in geometries.columns:
        geometries["branch_offset"] = np.nan

    if method == "intersecting":
        # Determine intersection geometries per branch
        sindex = geometries.sindex
        for branch_id, branchgeo in zip(branches.index, branches.geometry):
            intersecting = geometries.iloc[
                sindex.query(branchgeo, predicate="intersects")
            ]

            # For each geometrie, determine offset along branch
            for geometry_id, geometry in zip(intersecting.index, intersecting.geometry):
                # Determine distance of profile line along branch
                geometries.at[geometry_id, "branch_id"] = branch_id

                # Calculate offset
                mindist = min(0.1, branchgeo.length / 2.0)
                offset = round(
                    branchgeo.project(branchgeo.intersection(geometry).centroid), 3
                )
                offset = max(mindist, min(branchgeo.length - mindist, offset))
                geometries.at[geometry_id, "branch_offset"] = offset

    else:
        sindex = branches.sindex
        # the code of each geometry, for the log messages
        has_code = "code" in geometries.columns
        codes = [None] * len(geometries)
        if has_code:
            codes = geometries["code"].tolist()
        # In case of looking for the nearest, it is easier to iteratie over the geometries instead of the branches
        for geometry_id, geometry, code in zip(
            geometries.index, geometries.geometry, codes
        ):
            # Find near branches
            nearidx = sindex.query(geometry, predicate="dwithin", distance=maxdist)
            selectie = branches.iloc[nearidx]

            if method == "overal":
                # Determine distances to branches
                dist = selectie.distance(geometry)
            elif method == "centroid":
                # Determine distances to branches
                dist = selectie.distance(geometry.centroid)
            elif method == "ends":
                # Since a culvert can cross a channel, it is
                crds = geometry.coords[:]
                dist = (
                    selectie.distance(Point(*crds[0]))
                    + selectie.distance(Point(*crds[-1]))
                ) * 0.5

            # Enforce a point
            geo = geometry
            if not isinstance(geo, Point):
                geo = geo.centroid

            # Log-friendly name with coordinates
            geom_name = f"at x={geo.x:.4f} and y={geo.y:.4f}"
            if has_code:
                geom_name = f"for {code} {geom_name}"

            # Determine nearest
            if dist.min() < maxdist:
                branchidxmin = dist.idxmin()
                geometries.at[geometry_id, "branch_id"] = branchidxmin

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
                branchgeo = branches.geometry[branchidxmin]
                mindist = min(0.1, branchgeo.length / 2.0)
                offset = max(
                    mindist,
                    min(branchgeo.length - mindist, round(branchgeo.project(geo), 3)),
                )
                geometries.at[geometry_id, "branch_offset"] = offset
            else:
                logger.info(
                    "Centroid %s has 0 branches with maxdist=%s and method=%s",
                    geom_name,
                    maxdist,
                    method,
                )


def orthogonal_line(
    line: LineString, offset: float, width: float = 1.0
) -> list[tuple[float, float]]:
    """Return a line perpendicular to a line, centered on a point along it.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Line on which the orthogonal line is drawn.
    offset : float
        Distance along the line of the center of the orthogonal line.
    width : float, optional
        Length of the orthogonal line. Defaults to 1.0.

    Returns
    -------
    list[tuple[float, float]]
        The two end points of the orthogonal line.
    """
    # Determine angle at offset
    angle = (
        np.angle(
            complex(
                *np.diff(
                    [
                        line.interpolate(offset - 0.001).coords[0][:2],
                        line.interpolate(offset + 0.001).coords[0][:2],
                    ],
                    axis=0,
                )[0]
            )
        )
        + 0.5 * np.pi
    )

    # Create new line
    pt = line.interpolate(offset).coords[0]

    f = 0.5 * width
    orthogonal = [
        (pt[0] + np.cos(angle) * f, pt[1] + np.sin(angle) * f),
        (pt[0] - np.cos(angle) * f, pt[1] - np.sin(angle) * f),
    ]

    return orthogonal


def extend_linestring(line: LineString, near_pt: Point, length: float) -> LineString:
    """Return a line that extends a line beyond the end nearest to a point.

    The extension starts at that end and continues in the direction of the end segment.

    Parameters
    ----------
    line : LineString
        Line to extend.
    near_pt : Point
        Point that determines which end is extended.
    length : float
        Length of the extension.

    Returns
    -------
    LineString
        The extension, from the end of the line outward.
    """
    # Get the nearest end
    nearest_end = (0, 1) if line.project(near_pt) < line.length / 2 else (-1, -2)

    # Extrapolate the end segment by the given length
    coords = line.coords[:]
    x0, y0 = coords[nearest_end[0]]
    dx, dy = np.diff(
        np.vstack([coords[nearest_end[0]], coords[nearest_end[1]]]), axis=0
    )[0]
    segmentlength = (dx**2 + dy**2) ** 0.5
    dx *= length / segmentlength
    dy *= length / segmentlength

    return LineString([(x0, y0), (x0 - dx, y0 - dy)])


def get_voronoi_around_nodes(
    nodes: np.ndarray, facedata: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Return the Voronoi cells around the nodes, clipped to the faces.

    Parameters
    ----------
    nodes : np.ndarray
        xy coordinates of the nodes, shape (n, 2).
    facedata : gpd.GeoDataFrame
        The faces, which together form the area to clip the cells to.

    Returns
    -------
    gpd.GeoDataFrame
        One cell per node, in the order of the nodes, with index 1..n. A cell that the
        faces split into parts is a MultiPolygon; a node outside the faces gets an empty
        cell.
    """
    # the model extent: the union of all faces, to which the cells are clipped
    clippoly = facedata.union_all()

    # GEOS needs unique coordinates; nodes are duplicated where meshes were added next to each other
    unique_nodes, inverse = np.unique(nodes, axis=0, return_inverse=True)

    # one cell per unique node, in the same order, covering at least the extent of the faces
    voronoi = shapely.voronoi_polygons(
        shapely.multipoints(unique_nodes), extend_to=clippoly, ordered=True
    )
    cells = shapely.get_parts(voronoi)

    # only the cells on the edge of the faces need clipping
    shapely.prepare(clippoly)
    on_edge = ~shapely.contains_properly(clippoly, cells)
    cells[on_edge] = shapely.intersection(cells[on_edge], clippoly)

    # back to one cell per node (duplicate nodes share a cell), numbered from 1 like the nodes
    return gpd.GeoDataFrame(
        geometry=cells[inverse.ravel()],
        index=np.arange(len(nodes), dtype=np.uint32) + 1,
    )


def rasterize_polygons(
    polygons: gpd.GeoDataFrame, column: str, cellsize: float, fill: float = -999.0
) -> tuple[np.ndarray, rasterio.Affine]:
    """Rasterize the values in a column of a polygon GeoDataFrame.

    The raster extent is aligned to a multiple of the cellsize. Where polygons
    overlap, the last polygon in the GeoDataFrame determines the value.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        (Multi)polygons with values.
    column : str
        Column with the values to burn into the raster.
    cellsize : float
        Raster cell size.
    fill : float, optional
        Value for cells outside the polygons. Defaults to -999.0.

    Returns
    -------
    tuple[np.ndarray, rasterio.Affine]
        float32 array and its transform.
    """
    xmin, ymin, xmax, ymax = polygons.total_bounds
    xmin = np.floor(xmin / cellsize) * cellsize
    ymin = np.floor(ymin / cellsize) * cellsize
    xmax = np.ceil(xmax / cellsize) * cellsize
    ymax = np.ceil(ymax / cellsize) * cellsize
    ncols = max(round((xmax - xmin) / cellsize), 1)
    nrows = max(round((ymax - ymin) / cellsize), 1)

    transform = from_origin(xmin, ymax, cellsize, cellsize)
    array = features.rasterize(
        zip(polygons.geometry, polygons[column].astype(float)),
        out_shape=(nrows, ncols),
        transform=transform,
        fill=fill,
        dtype="float32",
    )
    return array, transform


def line_parts(geometry: BaseGeometry) -> list[LineString]:
    """Return all LineStrings in a geometry, including those in a GeometryCollection.

    Points and polygons are ignored.

    Parameters
    ----------
    geometry : BaseGeometry
        Geometry to search, e.g. the result of an intersection.

    Returns
    -------
    list[LineString]
        The LineStrings, in order; empty if there are none.
    """
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for geom in geometry.geoms for part in line_parts(geom)]
    return []


def polygons_to_branch_values(
    branches: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    column: str,
    step: float = 0.01,
) -> dict[Hashable, tuple[list[float], list[float]]]:
    """Convert polygon values to values per branch and chainage, e.g. for a 1dField file.

    Each part of a branch inside a polygon gets the value of that polygon. At polygon boundaries two
    chainages are written, `step` apart, so the value changes stepwise. Where polygons overlap, the last
    polygon in the GeoDataFrame determines the value. Parts of a branch outside all polygons get no
    chainage: D-Flow FM interpolates linearly between the given chainages and keeps the value constant
    beyond the first and last chainage of a branch. Branches that do not intersect any polygon are left out.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        Branches, with the branch id as index.
    polygons : gpd.GeoDataFrame
        (Multi)polygons with values.
    column : str
        Column with the values.
    step : float, optional
        Distance between the two chainages at a polygon boundary. Defaults to 0.01.

    Returns
    -------
    dict[Hashable, tuple[list[float], list[float]]]
        Chainages and values per branch id.

    Raises
    ------
    TypeError
        If a branch is not a LineString.
    """
    polygons = polygons.reset_index(drop=True)
    values = polygons[column].astype(float).to_numpy()
    result: dict[Hashable, tuple[list[float], list[float]]] = {}
    for branch_id, line in zip(branches.index, branches.geometry):
        if not isinstance(line, LineString):
            raise TypeError(
                f"Expected branch {branch_id} to be a LineString, not a {line.geom_type}."
            )
        intervals = _polygon_intervals(line, polygons, step)
        if not intervals:
            continue
        segments = _covering_segments(intervals, values, line.length, step)
        result[branch_id] = _segments_to_chainages(segments, line.length, step)
    return result


def _polygon_intervals(
    line: LineString, polygons: gpd.GeoDataFrame, step: float
) -> list[tuple[float, float, int]]:
    """Chainage intervals of the parts of a branch inside each polygon.

    Parameters
    ----------
    line : LineString
        Branch geometry.
    polygons : gpd.GeoDataFrame
        Polygons with a default (0..n-1) index.
    step : float
        Parts of the branch shorter than this are ignored.

    Returns
    -------
    list[tuple[float, float, int]]
        (start chainage, end chainage, polygon index), in the order of the polygons.
    """
    intervals = []
    for i in sorted(polygons.sindex.query(line, predicate="intersects")):
        for part in line_parts(line.intersection(polygons.geometry.iloc[i])):
            if part.length < step:
                continue
            chainages = [line.project(Point(c)) for c in part.coords]
            intervals.append((min(chainages), max(chainages), i))
    return intervals


def _covering_segments(
    intervals: list[tuple[float, float, int]],
    values: np.ndarray,
    length: float,
    step: float,
) -> list[list[float]]:
    """Segments of a branch with the value of the last polygon that covers them.

    The branch is split at all interval ends. Each sub-interval gets the value of the last
    polygon covering its midpoint; sub-intervals shorter than `step` or outside all polygons
    are left out. Adjacent segments with the same value are merged.

    Parameters
    ----------
    intervals : list[tuple[float, float, int]]
        (start chainage, end chainage, polygon index) per part of the branch inside a polygon.
    values : np.ndarray
        Value per polygon index.
    length : float
        Length of the branch.
    step : float
        Minimum length of a segment.

    Returns
    -------
    list[list[float]]
        [start chainage, end chainage, value] per segment, along the branch.
    """
    breakpoints = sorted(
        {0.0, length} | {s for s0, s1, _ in intervals for s in (s0, s1)}
    )
    segments: list[list[float]] = []
    for a, b in itertools.pairwise(breakpoints):
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
    return segments


def _segments_to_chainages(
    segments: list[list[float]], length: float, step: float
) -> tuple[list[float], list[float]]:
    """Chainages and values for a 1dField file, with a stepwise change at segment ends.

    Segment ends inside the branch are moved half a `step` into the segment, so two
    neighbouring segments are `step` apart; ends at the start or end of the branch are kept.

    Parameters
    ----------
    segments : list[list[float]]
        [start chainage, end chainage, value] per segment.
    length : float
        Length of the branch.
    step : float
        Distance between the chainages at a boundary between two segments.

    Returns
    -------
    tuple[list[float], list[float]]
        Chainages and values, two per segment.
    """
    chainage, branch_values = [], []
    for a, b, value in segments:
        start = a if np.isclose(a, 0.0) else a + 0.5 * step
        end = b if np.isclose(b, length) else b - 0.5 * step
        chainage += [start, end]
        branch_values += [value, value]
    return chainage, branch_values


def points_to_branch_values(
    branches: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    column: str,
    max_distance: float = 5.0,
    junction_tolerance: float = 0.1,
) -> dict[Hashable, tuple[list[float], list[float]]]:
    """Convert point values to values per branch and chainage, e.g. for a 1dField file.

    If the points have columns 'branchid' and 'chainage' (e.g. from a model result on the same network),
    these are used. Otherwise each point is snapped to the nearest branch within `max_distance` and its
    chainage is the projection on that branch. D-Flow FM interpolates linearly between the chainages of a
    branch and keeps the value constant beyond the first and last chainage.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        Branches, with the branch id as index.
    points : gpd.GeoDataFrame
        Points with values.
    column : str
        Column with the values.
    max_distance : float, optional
        Maximum distance to snap a point to a branch. Defaults to 5.0.
    junction_tolerance : float, optional
        Distance within which a point at a branch end is also used for the other
        branches at that junction. Defaults to 0.1.

    Returns
    -------
    dict[Hashable, tuple[list[float], list[float]]]
        Chainages and values per branch id.
    """
    if {"branchid", "chainage"} <= set(points.columns):
        data = points[["branchid", "chainage", column]].copy()
        unknown = ~data["branchid"].isin(branches.index)
        if unknown.any():
            logger.warning(
                f"{unknown.sum()} points refer to branches that are not in the model and are left out."
            )
            data = data[~unknown]
    else:
        # first row: index of the point, second row: index of the nearest branch
        point_idx, branch_idx = branches.sindex.nearest(
            points.geometry, max_distance=max_distance, return_all=False
        )
        if len(point_idx) < len(points):
            logger.warning(
                f"{len(points) - len(point_idx)} points are further than {max_distance} m from a branch and are left out."
            )
        geometries = branches.geometry.to_numpy()
        data = points.iloc[point_idx][[column]].copy()
        data["branchid"] = branches.index.to_numpy()[branch_idx]
        data["chainage"] = [
            geometries[b].project(p)
            for b, p in zip(branch_idx, points.geometry.to_numpy()[point_idx])
        ]

    result: dict[Hashable, tuple[list[float], list[float]]] = {}
    for branch_id, group in data.groupby("branchid", sort=False):
        group = group.sort_values("chainage").drop_duplicates("chainage", keep="last")
        result[branch_id] = (
            group["chainage"].astype(float).tolist(),
            group[column].astype(float).tolist(),
        )

    _add_junction_end_values(result, branches, points, column, junction_tolerance)
    return result


def _add_junction_end_values(
    result: dict[Hashable, tuple[list[float], list[float]]],
    branches: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    column: str,
    junction_tolerance: float,
) -> None:
    """Give branch ends near a point at a junction the value of that point, in place.

    A point at a junction belongs to one branch only. The other branches at that junction get the same
    value at their end, otherwise D-Flow FM extrapolates from their first inner point to the junction.
    Branch ends that already have a chainage within `junction_tolerance` are left as they are.

    Parameters
    ----------
    result : dict[Hashable, tuple[list[float], list[float]]]
        Chainages and values per branch id; the lists are extended in place.
    branches : gpd.GeoDataFrame
        Branches, with the branch id as index.
    points : gpd.GeoDataFrame
        Points with values.
    column : str
        Column with the values.
    junction_tolerance : float
        Distance within which a point at a branch end is used for that end.
    """
    geometries = dict(zip(branches.index, branches.geometry))
    for branch_id, (chainage, values) in result.items():
        line = geometries[branch_id]
        for end_chainage, end_point in (
            (0.0, Point(line.coords[0])),
            (line.length, Point(line.coords[-1])),
        ):
            if any(abs(c - end_chainage) < junction_tolerance for c in chainage):
                continue
            nearby = points.sindex.query(
                end_point.buffer(junction_tolerance), predicate="intersects"
            )
            if len(nearby) == 0:
                continue
            position = int(np.searchsorted(chainage, end_chainage))
            chainage.insert(position, end_chainage)
            values.insert(position, float(points[column].iloc[nearby[0]]))
