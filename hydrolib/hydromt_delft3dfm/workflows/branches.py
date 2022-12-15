# -*- coding: utf-8 -*-

import configparser
import logging
from pydoc import doc
from typing import Union

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import snap, split

from .helper import cut_pieces, split_lines

logger = logging.getLogger(__name__)


__all__ = [
    "process_branches",
    "validate_branches",
    "update_data_columns_attributes",
    "update_data_columns_attribute_from_query",
    "snap_newbranches_to_branches_at_snapnodes",
]


def update_data_columns_attributes(
    branches: gpd.GeoDataFrame,
    attributes: pd.DataFrame,
    brtype: str = None,
):
    """
    Add or update columns in the branches geodataframe based on column and values in attributes
    (excluding 1st col branchType used for query).

    If brtype is set, only update the attributes of the specified type of branches.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        Branches.
    attribute : int or pd.DataFrame
        Values of the attribute. Either int of float for fixed value for all or a pd.DataFrame with values per
        "branchType", "shape" and "width" (column names of the DataFrame).
    attribute_name : str
        Name of the new attribute column in branches.

    Returns
    -------
    branches : gpd.GeoDataFrame
        Branches with new attribute values.
    """
    # If brtype is specified, only update attributes for this brtype
    if brtype:
        attributes = attributes[attributes["branchType"] == brtype]
    # Update attributes
    for i in range(len(attributes.index)):
        row = attributes.iloc[i, :]
        branch = row.loc["branchType"]
        for colname in row.index[1:]:
            # If attribute is not at all in branches, add a new column
            if colname not in branches.columns:
                branches[colname] = pd.Series(dtype=attributes[colname].dtype)
            # Then fill in empty or NaN values with defaults
            branches.loc[
                np.logical_and(
                    branches[colname].isna(), branches["branchType"] == branch
                ),
                colname,
            ] = row.loc[colname]

    return branches


def update_data_columns_attribute_from_query(
    branches: gpd.GeoDataFrame,
    attribute: pd.DataFrame,
    attribute_name: str,
    logger=logger,
):
    """
    Update an attribute column of branches based on query on "branchType", "shape" and "width"/"diameter"
    values specified in attribute DataFrame.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        Branches.
    attribute : pd.DataFrame
        pd.DataFrame with specific attribute values per
        "branchType", "shape" and "width"/"diameter" (column names of the DataFrame).
    attribute_name : str
        Name of the new attribute column in branches.

    Returns
    -------
    branches : gpd.GeoDataFrame
        Branches with new attribute values.
    """
    # Add a new empty attribute column to be filled in
    branches[attribute_name] = pd.Series(dtype=attribute[attribute_name].dtype)
    # Iterate over the attribute DataFrame lines
    for row in attribute.itertuples(index=False):
        # Width/diameter is not always mandatory
        if "width" in row._fields:
            if np.isnan(row.width):
                branches[attribute_name] = branches[attribute_name].where(
                    np.logical_and(
                        branches.branchType != row.branchType,
                        branches["shape"] != row.shape,  # shape is reserved
                    ),
                    getattr(row, attribute_name),
                )
            else:
                branches[attribute_name] = branches[attribute_name].where(
                    np.logical_and(
                        branches.branchType != row.branchType,
                        branches["shape"] != row.shape,  # shape is reserved
                        branches.width != row.width,
                    ),
                    getattr(row, attribute_name),
                )
        elif "diameter" in row._fields:
            if np.isnan(row.diameter):
                branches[attribute_name] = branches[attribute_name].where(
                    np.logical_and(
                        branches.branchType != row.branchType,
                        branches["shape"] != row.shape,  # shape is reserved
                    ),
                    getattr(row, attribute_name),
                )
            else:
                branches[attribute_name] = branches[attribute_name].where(
                    np.logical_and(
                        branches.branchType != row.branchType,
                        branches["shape"] != row.shape,  # shape is reserved
                        branches.diameter != row.diameter,
                    ),
                    getattr(row, attribute_name),
                )
        else:
            branches[attribute_name] = branches[attribute_name].where(
                np.logical_and(
                    branches.branchType != row.branchType,
                    branches["shape"] != row.shape,  # shape is reserved
                ),
                getattr(row, attribute_name),
            )

    return branches


# FIXME BMA: For HydroMT we will rewrite this snap_branches to snap_lines and take set_branches out of it
def process_branches(
    branches: gpd.GeoDataFrame,
    branch_nodes: gpd.GeoDataFrame,
    id_col: str = "branchId",
    snap_offset: float = 0.01,
    allow_intersection_snapping: bool = True,
    smooth_branches: bool = False,
    logger=logger,
):
    """Process the branches by cleaning up the branches, snapping them, splitting them and generating branchnodes.

    Parameters
    ----------
    branches: gpd.GeoDataFrame
        The branches to process.
    branch_nodes: gpd.GeoDataFrame
        Branch nodes.
    id_col: str, optional
        Defalt to branchId.
    snap_offset : float, optional
        Maximum distance in meters between end points. If the distance is larger, they are not snapped. Defaults to 0.01.
    allow_intersection_snapping : bool, optional
        Allow snapping at all branch ends, including intersections. Defaults to True.
    smooth_branches: bool, optional
        whether to return branches that are smoothed (straightend) , needed for pipes
        Default to False.
    logger
        The logger to log messages with.

    Returns
    -------
    branches : gpd.GeoDataFrame
        Preprocessed branches.
    branches_nodes : gpd.GeoDataFrame
        Preprocessed branches' nodes.
    """

    logger.debug(f"Cleaning up branches")
    # TODO: maybe add arguments,use branch cross sections
    # global_controls = branches_ini.get("global", None)

    branches = cleanup_branches(
        branches,
        id_col=id_col,
        snap_offset=snap_offset,
        allow_intersection_snapping=allow_intersection_snapping,
        logger=logger,
    )

    logger.debug(f"Splitting branches based on spacing")
    # TODO: add check, if spacing is used, then in branch cross section cannot be setup later
    branches = space_branches(branches, smooth_branches=smooth_branches, logger=logger)

    logger.debug(f"Generating branchnodes")
    branch_nodes = generate_branchnodes(branches, id_col, logger=logger)

    return branches, branch_nodes


def cleanup_branches(
    branches: gpd.GeoDataFrame,
    id_col: str = "branchId",
    snap_offset: float = 0.01,
    allow_intersection_snapping: bool = True,
    logger=logger,
):
    """Clean up the branches by:
    * Removing null geomtry
    * Exploding branches with multiline strings
    * Removing branches with duplicated geometry
    * Removing branches that are shorter than 0.1 meters
    * Renaming branches with duplicate IDs
    * Reducing the precision of the branch geometry to 6 digits.
    * Snapping the branches

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        The branches to clean up.
    id_col : str, optional
        The branch id column name. Defaults to 'BRANCH_ID'.
    snap_offset : float, optional
        Maximum distance in meters between end points. If the distance is larger, they are not snapped. Defaults to 0.01.
    allow_intersection_snapping : bool, optional
        Allow snapping at all branch ends, including intersections. Defaults to True.
    logger
        The logger to log messages with.

    Returns
    -------
    gpd.GeoDataFrame
        The cleanup branches.
    """

    # remove null geometry
    branches = branches.loc[~branches.geometry.isna(), :]

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
    if branches.crs.is_geographic:
        branches_length = branches.geometry.to_crs(3857).length
    else:
        branches_length = branches.geometry.length
    n = np.sum(list(branches_length <= 0.1))
    branches = branches[branches_length >= 0.1]
    logger.debug(f"Removing {n} branches that are shorter than 0.1 meter.")
    # remove branches with ring geometries
    branches = _remove_branches_with_ring_geometries(branches)

    # sort index
    if id_col in [
        "None",
        "NONE",
        "none",
        None,
        "",
    ]:  # TODO: id_column must be specified
        id_col = "BRANCH_ID"
        # regenerate ID based on ini # NOTE BMA: this is the step to ensure unique id cross the network
        # TODO could not find anay example of this or default
        id_prefix = 100  # branches_ini["global"]["id_prefix"]
        id_suffix = 100  # branches_ini["global"]["id_suffix"]
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

    # precision correction --> check if needs to be changed by the user, if so move that into a func argument
    branches = reduce_gdf_precision(
        branches, rounding_precision=6  # branches_ini["global"]["rounding_precision"]
    )  # recommned to be larger than e-8
    logger.debug(f"Reducing precision of the GeoDataFrame. Rounding precision (e-6) .")

    # snap branches
    if allow_intersection_snapping is True:
        # snap points no matter it is at intersection or ends
        branches = snap_branch_ends(branches, offset=snap_offset)
        logger.debug(
            f"Performing snapping at all branch ends, including intersections (To avoid messy results, please use a lower snap_offset)."
        )

    else:
        # snap points at ends only
        branches = snap_branch_ends(branches, offset=snap_offset, max_points=2)
        logger.debug(
            f"Performing snapping at all branch ends, excluding intersections (To avoid messy results, please use a lower snap_offset).."
        )

    # Drop count column
    if "count" in branches.columns:
        branches = branches.drop(columns=["count"])

    return branches


def space_branches(
    branches: gpd.GeoDataFrame,
    spacing_col: str = "spacing",  # TODO: seperate situation where interpolation is needed and interpolation is not needed
    smooth_branches: bool = False,
    logger=logger,
):
    """Space the branches based on the spacing_col on the branch.
    Removes the spacing column from the branches afterwards.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        The branches to clean up.
    spacing_col : str, optional
        The branch id column name. Defaults to 'spacing'.
    logger
        The logger to log messages with.

    Returns
    -------
    gpd.GeoDataFrame
        The split branches.
    """

    # split branches based on spacing
    branches_ = split_branches(
        branches, spacing_col=spacing_col, smooth_branches=smooth_branches
    )
    logger.debug(f"clipping branches into {len(branches_)} segments")

    # remove spacing column
    branches_ = branches_.drop(columns=[spacing_col])

    return branches_


def generate_branchnodes(
    branches: gpd.GeoDataFrame,
    id_col: str = None,
    logger=logger,
):
    """Generate branch nodes at the branch ends.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        The branches to generate the end nodes for.
    id_col : str, optional
        The branch id column name. Defaults to None.
    logger
        The logger to log messages with.

    Returns
    -------
    gpd.GeoDataFrame
        The branch nodes.
    """

    # generate node up and downstream
    nodes = pd.DataFrame(
        [Point(l.coords[0]) for li, l in branches["geometry"].iteritems()]
        + [Point(l.coords[-1]) for li, l in branches["geometry"].iteritems()],
        columns=["geometry"],
    )

    if id_col is None:
        id_col = branches.index.name

    nodes = []
    for bi, b in branches.iterrows():
        nodes.append([Point(b.geometry.coords[0]), b[id_col]])  # start
        nodes.append([Point(b.geometry.coords[-1]), b[id_col]])  # end
    nodes = pd.DataFrame(nodes, columns=["geometry", "branch_id"])
    nodes = pd.merge(
        nodes,
        branches.reset_index(drop=True),
        left_on="branch_id",
        right_on=branches.index.name,
        suffixes=("", "_b"),
    )
    nodes = nodes.drop(columns="geometry_b")
    # remove duplicated geometry
    _nodes = nodes.copy()
    G = _nodes["geometry"].apply(lambda geom: geom.wkb)
    n = len(G) - len(G.drop_duplicates().index)
    nodes = _nodes[_nodes.index.isin(G.drop_duplicates().index)]
    nodes = gpd.GeoDataFrame(nodes)
    nodes.crs = branches.crs
    # give index
    nodes.index = [f"branchnodes_{i}" for i in range(len(nodes))]
    nodes.index.name = "branchnodeid"
    # nodes[nodes.index.name] = nodes.index # creates a duplicate column
    return nodes


def validate_branches(
    branches: gpd.GeoDataFrame, logger=logger
):  # TODO: add more content and maybe make a seperate module
    """Validate the branches.
    Logs an error when one or more branches have a length of 0 meter.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        The branches to validate.
    logger
        The logger to log messages with.
    """
    # validate pipe geometry
    if sum(branches.geometry.length <= 0) == 0:
        logger.debug("Branches are valid.")
    else:
        logger.error(
            f"Branches {branches.index[branches.geometry.length <= 0]} have length of 0 meter. "
            + f"Issue might have been caused by using a snap_offset that is too large. Please revise or modify the branches data layer. "
        )


def split_branches(
    branches: gpd.GeoDataFrame,
    spacing_const: float = float("inf"),
    spacing_col: str = None,
    smooth_branches: bool = False,
    logger=logger,
):
    """
    Helper function to split branches based on a given spacing.
    If spacing_col is used (default), apply spacing as a categorical variable - distance used to split branches. priority issue --> Question to Rinske
    If spacing_const is used (overwrite), apply spacing as a constant -  distance used to split branches.
    Raise Error if neither exist.

    If ``smooth_branches``, split branches generated will be straight line.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
    spacing_const : float
        Constent spacing which will overwrite the spacing_col. Defaults to float("inf").
    spacing_col: str
        Name of the column in branchs that contains spacing information. Default to None.
    smooth_branches: bool, optional
        Switch to split branches into straight lines. By default False.
    logger
        The logger to log messages with.

    Returns
    -------
    split_branches : gpd.GeoDataFrame
        Branches after split, new ids will be overwritten for the branch index. Old ids are stored in "OLD_" + index.
    """

    id_col = branches.index.name
    if spacing_col is None:
        logger.info(f"Splitting branches with spacing of {spacing_const} [m]")
        split_branches = _split_branches_by_spacing_const(
            branches,
            spacing_const,
            id_col=id_col,
            smooth_branches=smooth_branches,
        )

    elif branches[spacing_col].astype(float).notna().any():
        logger.info(
            f"Splitting branches with spacing specifed in datamodel branches[{spacing_col}]"
        )
        split_branches = []
        for spacing_subset, branches_subset in branches.groupby(spacing_col):
            if spacing_subset:
                split_branches_subset = _split_branches_by_spacing_const(
                    branches_subset,
                    spacing_subset,
                    id_col=id_col,
                    smooth_branches=smooth_branches,
                )
            else:
                branches_subset.loc[:, f"ORIG_{id_col}"] = branches_subset[id_col]
                split_branches_subset = branches_subset
            split_branches.append(split_branches_subset)
        split_branches = pd.concat(split_branches)

    else:  # no spacing information specified anywhere, do not apply splitting
        branches.loc[:, f"ORIG_{id_col}"] = branches[id_col]
        split_branches = branches

    # reassign branch id for the generated
    split_branches.index = split_branches[id_col]
    return split_branches


def _split_branches_by_spacing_const(
    branches: gpd.GeoDataFrame,
    spacing_const: float,
    id_col: str = "BRANCH_ID",
    smooth_branches: bool = False,
):
    """
    Helper function to split branches based on a given spacing constant.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
    spacing_const : float
        Constant spacing which will overwrite the spacing_col.
    id_col: str
        Name of the column in branches that contains the id of the branches.
    smooth_branches: bool, optional
        Swith to split branches into straight lines. By default False.

    Returns
    -------
    split_branches : gpd.GeoDataFrame
        Branches after split, new ids will be stored in id_col. Original ids are stored in "ORIG_" + id_col.
    """

    if spacing_const == float("inf"):
        branches[f"ORIG_{id_col}"] = branches[id_col]
        branches.index = branches[id_col]
        return branches

    edge_geom = []
    edge_offset = []
    edge_invertup = []
    edge_invertdn = []
    edge_bedlevup = []
    edge_bedlevdn = []
    edge_index = []
    branch_index = []

    # Check for attributes
    interp_invlev = "invlev_up" and "invlev_dn" in branches.columns
    interp_bedlev = "bedlev_up" and "bedlev_dn" in branches.columns

    for bid, b in branches.iterrows():
        # prepare for splitting
        line = b.geometry
        num_new_lines = int(np.ceil(line.length / spacing_const))

        if num_new_lines <= 0:
            continue

        # interpolate geometry
        new_edges = split_lines(line, num_new_lines)
        if smooth_branches:
            for i in range(len(new_edges)):
                ed = new_edges[i]
                new_edges[i] = LineString([Point(ed.coords[0]), Point(ed.coords[-1])])
        offsets = np.linspace(0, line.length, num_new_lines + 1)

        # interpolate values
        edge_geom.extend(new_edges)
        edge_offset.extend(offsets[1:])
        if interp_invlev:
            edge_invertup.extend(
                np.interp(
                    offsets[:-1], [0, offsets[-1]], [b.invlev_up, b.invlev_dn]
                )  # TODO: renaming needed
            )
            edge_invertdn.extend(
                np.interp(offsets[1:], [0, offsets[-1]], [b.invlev_up, b.invlev_dn])
            )
        if interp_bedlev:
            edge_bedlevup.extend(
                np.interp(offsets[:-1], [0, offsets[-1]], [b.bedlev_up, b.bedlev_dn])
            )
            edge_bedlevdn.extend(
                np.interp(offsets[1:], [0, offsets[-1]], [b.bedlev_up, b.bedlev_dn])
            )
        edge_index.extend([bid + "_E" + str(i) for i in range(len(new_edges))])
        branch_index.extend([bid] * len(new_edges))

    edges = gpd.GeoDataFrame(
        {
            "EDGE_ID": edge_index,
            "geometry": edge_geom,
            id_col: branch_index,
            # "invlev_up": edge_invertup,
            # "invlev_dn": edge_invertdn,
            # "bedlev_up": edge_bedlevup,
            # "bedlev_dn": edge_bedlevdn,
        },
        crs=branches.crs,
    )
    if interp_invlev:
        edges["invlev_up"] = edge_invertup
        edges["invlev_dn"] = edge_invertdn
    if interp_bedlev:
        edges["bedlev_up"] = edge_bedlevup
        edges["bedlev_dn"] = edge_bedlevdn
    edges_attr = pd.concat(
        [branches.loc[idx, :] for idx in branch_index], axis=1
    ).transpose()
    edges = pd.concat(
        [
            edges,
            edges_attr.drop(
                columns=list(set(edges.columns) - set(["EDGE_ID"]))
            ).reset_index(),
        ],
        axis=1,
    )

    edges = edges.rename(columns={id_col: f"ORIG_{id_col}"})
    edges = edges.rename(columns={"EDGE_ID": id_col})
    edges.index = edges[id_col]
    split_branches = edges

    return split_branches


def reduce_gdf_precision(gdf: gpd.GeoDataFrame, rounding_precision: int = 8):
    """Reduce the geometry coordinate precision with the provided number of digits.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The geo data frame to reduce the precision for.
    rounding_precision : int, optional
        The number of digits to round the coordinates with. Defaults to 8.

    Returns
    -------
    gpd.GeoDataFrame
        The geo data frame with the rounded geometry.

    Raises
    ------
    NotImplementedError
        If the geometry is not a LineString or Point.
    """
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


def snap_branch_ends(
    branches: gpd.GeoDataFrame,
    offset: float = 0.01,
    subsets=[],
    max_points: int = np.inf,
    id_col: str = "BRANCH_ID",
):
    """
    Helper to snap branch ends to other branch ends within a given offset.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
    offset : float, optional
        Maximum distance in meters between end points. If the distance is larger, they are not snapped. Default to 0.01.
    subsets : list, optional
        A list of branch id subset to perform snapping (forced snapping). Default to an empty list.
    max_points : int, optional
        maximum points allowed in a group.
        if snapping branch ends only, use max_points = 2
        if not specified, branch intersections will also be snapped
        Defaults to np.inf.
    id_col : str, optional
        The branch id column name. Defaults to 'BRANCH_ID'.

    Returns
    -------
    branches : gpd.GeoDataFrame
        Branches updated with snapped geometry.
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


# TODO copied from dhydamo geometry.py, update when available in main
def possibly_intersecting(
    dataframebounds: np.ndarray, geometry: gpd.GeoDataFrame, buffer: int = 0
):
    """
    Finding intersecting profiles for each branch is a slow process in case of large datasets
    To speed this up, we first determine which profile intersect a square box around the branch
    With the selection, the interseting profiles can be determines much faster.

    Parameters
    ----------
    dataframebounds : numpy.array
    geometry : shapely.geometry.Polygon
    """

    geobounds = geometry.bounds
    idx = (
        (dataframebounds[0] - buffer < geobounds[2])
        & (dataframebounds[2] + buffer > geobounds[0])
        & (dataframebounds[1] - buffer < geobounds[3])
        & (dataframebounds[3] + buffer > geobounds[1])
    )
    # Get intersecting profiles
    return idx


# TODO copied from dhydamo geometry.py, update when available in main
def find_nearest_branch(
    branches: gpd.GeoDataFrame,
    geometries: gpd.GeoDataFrame,
    method: str = "overal",
    maxdist: int = 5,
):
    """
    Method to determine nearest branch for each geometry.
    The nearest branch can be found by finding t from both ends (ends) or the nearest branch from the geometry
    as a whole (overal), the centroid (centroid), or intersecting (intersect).

    Parameters
    ----------
    branches : geopandas.GeoDataFrame
        Geodataframe with branches.
    geometries : geopandas.GeoDataFrame
        Geodataframe with geometries to snap.
    method : {'overal','intersecting','centroid','ends'}
        Method for determine branch. Defaults to 'overal'.
    maxdist: int or float
        Maximum distance for finding nearest geometry. Defaults to 5.
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
        geobounds = geometries.bounds.values.T
        for branch in branches.itertuples():
            selectie = geometries.loc[
                possibly_intersecting(geobounds, branch.geometry)
            ].copy()
            intersecting = selectie.loc[selectie.intersects(branch.geometry).values]

            # For each geometrie, determine offset along branch
            for geometry in intersecting.itertuples():
                # Determine distance of profile line along branch
                geometries.at[geometry.Index, "branch_id"] = branch.Index

                # Calculate offset
                branchgeo = branch.geometry
                mindist = min(0.1, branchgeo.length / 2.0)
                offset = round(
                    branchgeo.project(
                        branchgeo.intersection(geometry.geometry).centroid
                    ),
                    3,
                )
                offset = max(mindist, min(branchgeo.length - mindist, offset))
                geometries.at[geometry.Index, "branch_offset"] = offset

    else:
        branch_bounds = branches.bounds.values.T
        # In case of looking for the nearest, it is easier to iteratie over the geometries instead of the branches
        for geometry in geometries.itertuples():
            # Find near branches
            nearidx = possibly_intersecting(
                branch_bounds, geometry.geometry, buffer=maxdist
            )
            selectie = branches.loc[nearidx]

            if method == "overal":
                # Determine distances to branches
                dist = selectie.distance(geometry.geometry)
            elif method == "centroid":
                # Determine distances to branches
                dist = selectie.distance(geometry.geometry.centroid)
            elif method == "ends":
                # Since a culvert can cross a channel, it is
                crds = geometry.geometry.coords[:]
                dist = (
                    selectie.distance(Point(*crds[0]))
                    + selectie.distance(Point(*crds[-1]))
                ) * 0.5

            # Determine nearest
            if dist.min() < maxdist:
                branchidxmin = dist.idxmin()
                geometries.at[geometry.Index, "branch_id"] = dist.idxmin()
                if isinstance(geometry.geometry, Point):
                    geo = geometry.geometry
                else:
                    geo = geometry.geometry.centroid

                # Calculate offset
                branchgeo = branches.at[branchidxmin, "geometry"]
                mindist = min(0.1, branchgeo.length / 2.0)
                offset = max(
                    mindist,
                    min(branchgeo.length - mindist, round(branchgeo.project(geo), 3)),
                )
                geometries.at[geometry.Index, "branch_offset"] = offset


def snap_newbranches_to_branches_at_snapnodes(
    new_branches: gpd.GeoDataFrame,
    branches: gpd.GeoDataFrame,
    snapnodes: gpd.GeoDataFrame,
):
    """function to snap new_branches to branches at snapnodes.
    snapnodes are located at branches. new branches will be snapped, and branches will be splitted.
    # NOTE: no interpolation of crosssection is needed because inter branch interpolation is turned on using branchorder

    Parameters
    ----------
    new_branches : geopandas.GeoDataFrame
        Geodataframe of new branches whose geometry will be modified: end nodes will be snapped to snapnodes
    branches : geopandas.GeoDataFrame
        Geodataframe who will be splitted at snapnodes to allow connection with the new_branches.
    snapnodes : geopandas.GeoDataFrame
        Geodataframe which contiains the spatial relation of the new_branches and branches.

    Returns
    -------
    new_branches_snapped : geopandas.GeoDataFrame
        Geodataframe of new branches with endnodes be snapped to snapnodes in branches_snapped.
    branches_snapped : geopandas.GeoDataFrame
        Geodataframe of branches splitted at snapnodes to allow connection with the new_branches_snapped.
    """

    new_branches.index = new_branches.branchId
    branches.index = branches.branchId

    # for each snapped endnodes
    new_branches_snapped = new_branches.copy()
    branches_snapped = branches.copy()

    # modify new branches
    for snapnode in snapnodes.itertuples():
        new_branch = new_branches.loc[snapnode.branchId]
        snapped_line = LineString(
            [
                snapnode.geometry_right
                if Point(xy).equals(snapnode.geometry_left)
                else Point(xy)
                for xy in new_branch.geometry.coords[:]
            ]
        )
        new_branches_snapped.at[snapnode.branchId, "geometry"] = snapped_line

    # modify old branches
    for branch_name in set(snapnodes.branch_name):
        branch = branches.loc[branch_name]
        distances = snapnodes[
            snapnodes.branch_name == branch_name
        ].branch_chainage.to_list()
        snapped_line = MultiLineString(cut_pieces(branch.geometry, distances))
        branches_snapped.at[branch_name, "geometry"] = snapped_line
        branches_snapped.at[branch_name, "branchOrder"] = (
            max(branches_snapped.branchOrder) + 1
        )  # allow interpolation on the snapped branch

    # explode multilinestring after snapping
    branches_snapped = branches_snapped.explode()

    # reset the idex
    branches_snapped = cleanup_branches(branches_snapped)

    # precision correction
    branches_snapped = reduce_gdf_precision(branches_snapped, rounding_precision=6)

    return new_branches_snapped, branches_snapped


def _remove_branches_with_ring_geometries(
    branches: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    first_nodes = [l.coords[0] for l in branches.geometry]
    last_nodes = [l.coords[-1] for l in branches.geometry]
    duplicate_ids = np.isclose(first_nodes, last_nodes)
    duplicate_ids = [
        branches.index[i] for i in range(len(branches)) if np.all(duplicate_ids[i])
    ]
    branches = branches.drop(duplicate_ids, axis=0)
    logger.debug("Removing branches with ring geometries.")

    return branches
