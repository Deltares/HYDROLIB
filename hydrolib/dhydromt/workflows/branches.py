# -*- coding: utf-8 -*-

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from scipy.spatial import distance
import logging

from hydromt import config

logger = logging.getLogger(__name__)


__all__ = ["preprocess_branches"]


def reduce_gdf_precision(gdf: gpd.GeoDataFrame, rounding_precision=8):

    if isinstance(gdf.geometry[0], LineString):
        branches = gdf.copy()
        for i_branch, branch in enumerate(branches.itertuples()):
            points = shapely.wkt.loads(
                shapely.wkt.dumps(
                    branch.geometry, rounding_precision=rounding_precision
                )
            ).coords[:]
            branches.at[i_branch, "geometry"] = LineString(points)

    elif isinstance(gdf.geometry[0], Point):
        points = gdf.copy()
        for i_point, point in enumerate(points.itertuples()):
            new_point = shapely.wkt.loads(
                shapely.wkt.dumps(point.geometry, rounding_precision=rounding_precision)
            ).coords[:]
            points.at[i_point, "geometry"] = Point(new_point)

    else:
        raise NotImplementedError

    return gdf


# FIXME BMA: For HydroMT we will rewrite this snap_branches to snap_lines and take set_branches out of it
def snap_branch_ends(
    branches: gpd.GeoDataFrame,
    offset: float = 0.01,
    subsets=[],
    max_points: int = np.inf,
    id_col="BRANCH_ID",
    logger=logging,
):
    """
    Helper to snap branch ends to other branch ends within a given offset.


    Parameters
    ----------
    branches : gpd.GeoDataFrame
    offset : float [m]
        Maximum distance between end points. If the distance is larger, they are not snapped.
    subset : list
        A list of branch id subset to perform snapping (forced snapping)
    max_points: int
        maximum points allowed in a group.
        if snapping branch ends only, use max_points = 2
        if not specified, branch intersections will also be snapped
    Returns
    branches : gpd.GeoDataFrame
        Branches updated with snapped geometry
    """
    # Collect endpoints
    _endpoints = []
    for branch in branches.itertuples():
        _endpoints.append((branch.geometry.coords[0], branch.Index, 0))
        _endpoints.append((branch.geometry.coords[-1], branch.Index, -1))

    # determine which branches should be included
    if len(subsets) > 0:
        _endpoints = [[i for i in _endpoints if i[1] in subsets]]
    else:
        _endpoints = _endpoints

    # # group branch ends based on off set
    groups = {}
    coords = [i[0] for i in _endpoints]
    dist = distance.squareform(distance.pdist(coords))
    bdist = dist <= offset
    for row_i, row in enumerate(bdist):
        groups[_endpoints[row_i]] = []
        for col_i, col in enumerate(row):
            if col:
                groups[_endpoints[row_i]].append(_endpoints[col_i])

    # remove duplicated group, group that does not satisfy max_points in groups. Assign endpoints
    endpoints = {
        k: list(set(v))
        for k, v in groups.items()
        if (len(set(v)) >= 2) and (len(set(v)) <= max_points)
    }
    logger.debug(
        "Limit snapping to allow a max number of {max_points} contact points. If max number == 2, it means 1 to 1 snapping."
    )

    # Create a counter
    snapped = 0

    # snap each group (list) in endpoints together, by using the coords from the first point
    for point_reference, points_to_snap in endpoints.items():
        # get the point_reference coords as reference point
        ref_crd = point_reference[0]
        # for each of the rest
        for j, (endpoint, branchid, side) in enumerate(points_to_snap):
            # Change coordinates of branch
            crds = branches.at[branchid, "geometry"].coords[:]
            if crds[side] != ref_crd:
                crds[side] = ref_crd
                branches.at[branchid, "geometry"] = LineString(crds)
                snapped += 1
    logger.debug(f"Snapped {snapped} points.")

    return branches


def preprocess_branches(
    branches: gpd.GeoDataFrame,
    branches_ini_fn: str,
    snap_offset: float = 0.01,
    id_col: str = "BRANCH_ID",
    pipe_query: str = None,
    channel_query: str = None,
    logger=logging,
):

    """Function to (geo-) preprocess branches"""

    branches_ini = config.configread(branches_ini_fn, abs_path=False)

    # explode multiline string
    n = 0
    for branch_index, branch in branches.iterrows():
        if branch.geometry.type != "LineString":
            branches.at[branch_index, "geometry"] = LineString(
                [p for l in branch.geometry for p in l.coords]
            )
            n += 1
    logger.debug(f"Exploding {n} branches which have multipline geometry.")

    # remove duplicated geometry
    _branches = branches.copy()
    G = _branches["geometry"].apply(lambda geom: geom.wkb)
    n = len(G) - len(G.drop_duplicates().index)
    branches = _branches[_branches.index.isin(G.drop_duplicates().index)]
    logger.debug(f"Removing {n} branches which have duplicated geometry.")

    # remove branches that are too short
    n = np.sum(list(branches.geometry.length <= 0.1))
    branches = branches[branches.geometry.length >= 0.1]
    logger.debug(f"Removing {n} branches that are shorter than 0.1 meter.")

    # sort index
    if id_col in ["None", "NONE", "none", None, ""]:
        id_col = "BRANCH_ID"
        # regenerate ID based on ini # NOTE BMA: this is the step to ensure unique id cross the network
        id_prefix = branches_ini["global"]["id_prefix"]
        id_suffix = branches_ini["global"]["id_suffix"]
        branches[id_col] = [
            id_prefix + "_" + str(x) + "_" + id_suffix for x in range(len(branches))
        ]
        logger.warning(
            f"id_col is not specified. Branch id columns are read/generated using default: {id_col}."
        )

    # check duplicated id
    _branches = branches.copy()
    _branches.reset_index(drop=True, inplace=True)
    _branches["count"] = (
        _branches.groupby(id_col)[id_col].transform("cumcount").astype(int)
    )
    for bi, b in _branches.iterrows():
        if b["count"] >= 1:
            _branches.rename(
                index={bi: b[id_col] + "-" + str(b["count"])}, inplace=True
            )
        else:
            _branches.rename(index={bi: b[id_col]}, inplace=True)
    _branches[id_col] = _branches.index
    _branches.index.name = id_col
    branches = _branches.copy()
    n = sum(_branches["count"])
    logger.debug(
        f"Renaming {n} id_col duplicates. Convention: BRANCH_1, BRANCH_1 --> BRANCH_1, BRANCH_1-2."
    )

    # precision correction
    branches = reduce_gdf_precision(
        branches, rounding_precision=branches_ini["global"]["rounding_precision"]
    )  # recommned to be larger than e-8
    logger.debug(
        f"Reducing precision of the GeoDataFrame. Rounding precision (e-){branches_ini['global']['rounding_precision']} ."
    )

    # snap branches
    if branches_ini["global"]["allow_intersection_snapping"] is True:
        # snap points no matter it is at intersection or ends
        branches = snap_branch_ends(branches, offset=snap_offset, logger=logger)
        logger.debug(
            f"Performing snapping at all branch ends, including intersections (To avoid messy results, please use a lower snap_offset)."
        )

    else:
        # snap points at ends only
        branches = snap_branch_ends(
            branches, offset=snap_offset, max_points=2, logger=logger
        )
        logger.debug(
            f"Performing snapping at all branch ends, excluding intersections (To avoid messy results, please use a lower snap_offset).."
        )

    # setup channels (if needed)
    branches.loc[branches.query(channel_query).index, "branchType"] = "Channel"
    # setup pipes (if needed)
    branches.loc[branches.query(pipe_query).index, "branchType"] = "Pipe"

    # assign crs
    # branches.crs = branches_crs

    logger.info(str(len(branches)) + " branches are set up.")

    # validate pipe geometry
    if sum(branches.geometry.length <= 0) == 0:
        pass
    else:
        logger.error(
            f"Branches {branches.index[branches.geometry.length <= 0]} have length of 0 meter. "
            + f"Issue might have been caused by using a snap_offset that is too large. Please revise or modify the branches data layer. "
        )

    return branches
