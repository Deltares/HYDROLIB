# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, Point

from .branches import find_nearest_branch

# from delft3dfmpy.core import geometry
from .helper import split_lines

logger = logging.getLogger(__name__)


__all__ = [
    "set_branch_crosssections",
    "set_xyz_crosssections",
]  # , "process_crosssections", "validate_crosssections"]


def set_branch_crosssections(
    branches: gpd.GeoDataFrame,
    midpoint: bool = True,
):
    """
    Function to set regular cross-sections for each branch.
    only support rectangle, trapezoid and circle.
    Crosssections are derived at branches mid points if ``midpoints`` is True,
    else at both upstream and downstream extremities of branches if False.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        The branches.

    Returns
    -------
    gpd.GeoDataFrame
        The cross sections.
    """
    # Get the crs at the midpoint of branches if midpoint
    if midpoint:
        crosssections = pd.DataFrame({}, index=branches.index)
        crosssections["geometry"] = [
            l.interpolate(0.5, normalized=True) for l in branches.geometry
        ]
        crosssections["crsloc_id"] = [f"crs_{bid}" for bid in branches["branchId"]]
        crosssections["crsloc_branchId"] = branches["branchId"]
        crosssections["crsloc_chainage"] = [l / 2 for l in branches["geometry"].length]
        crosssections["crsloc_shift"] = branches["bedlev"]
        crosssections = gpd.GeoDataFrame(crosssections, crs=branches.crs)
    # Else prepares crosssections at both upstream and dowsntream extremities
    else:
        # Upstream
        ids = [f"{i}_up" for i in branches.index]
        crosssections_up = gpd.GeoDataFrame({"geometry": [Point(l.coords[0]) for l in branches.geometry]}, index=ids, crs=branches.crs)

        crosssections_up["crsloc_id"] = [
            f"crs_up_{bid}" for bid in branches["branchId"]
        ]
        crosssections_up["crsloc_branchId"] = branches["branchId"].values
        crosssections_up["crsloc_chainage"] = [0.0 for l in branches.geometry]
        crosssections_up["crsloc_shift"] = branches["invlev_up"].values
        # Downstream
        ids = [f"{i}_dn" for i in branches.index]
        crosssections_dn = gpd.GeoDataFrame({"geometry": [Point(l.coords[0]) for l in branches.geometry]}, index=ids, crs=branches.crs)
        crosssections_dn["crsloc_id"] = [
            f"crs_dn_{bid}" for bid in branches["branchId"]
        ]
        crosssections_dn["crsloc_branchId"] = branches["branchId"].values
        crosssections_dn["crsloc_chainage"] = [l for l in branches["geometry"].length]
        crosssections_dn["crsloc_shift"] = branches["invlev_dn"].values
        # Merge
        crosssections = crosssections_up.append(crosssections_dn)

    # circle profile
    circle_indexes = branches.loc[branches["shape"] == "circle", :].index
    for bi in circle_indexes:
        if midpoint:
            bicrs = [bi]
        else:
            bicrs = [f"{bi}_up", f"{bi}_dn"]
        for b in bicrs:
            crosssections.at[b, "crsloc_definitionId"] = "circ_d{:,.3f}_{:s}".format(
                branches.loc[bi, "diameter"], "branch"
            )
            crosssections.at[b, "crsdef_id"] = crosssections.loc[
                b, "crsloc_definitionId"
            ]
            crosssections.at[b, "crsdef_type"] = branches.loc[bi, "shape"]
            crosssections.at[b, "crsdef_frictionId"] = branches.loc[bi, "frictionId"]
            crosssections.at[b, "crsdef_diameter"] = branches.loc[bi, "diameter"]
            crosssections.at[b, "crsdef_closed"] = branches.loc[bi, "closed"]

    # rectangle profile
    rectangle_indexes = branches.loc[branches["shape"] == "rectangle", :].index

    for bi in rectangle_indexes:
        if midpoint:
            bicrs = [bi]
        else:
            bicrs = [f"{bi}_up", f"{bi}_dn"]
        for b in bicrs:
            crosssections.at[
                b, "crsloc_definitionId"
            ] = "rect_h{:,.3f}_w{:,.3f}_c{:s}_{:s}".format(
                branches.loc[bi, "height"],
                branches.loc[bi, "width"],
                branches.loc[bi, "closed"],
                "branch",
            )
            crosssections.at[b, "crsdef_id"] = crosssections.loc[
                b, "crsloc_definitionId"
            ]
            crosssections.at[b, "crsdef_type"] = branches.loc[bi, "shape"]
            crosssections.at[b, "crsdef_frictionId"] = branches.loc[bi, "frictionId"]
            crosssections.at[b, "crsdef_height"] = branches.loc[bi, "height"]
            crosssections.at[b, "crsdef_width"] = branches.loc[bi, "width"]
            crosssections.at[b, "crsdef_closed"] = branches.loc[bi, "closed"]

    # trapezoid profile
    trapezoid_indexes = branches.loc[branches["shape"] == "trapezoid", :].index
    for bi in trapezoid_indexes:
        if midpoint:
            bicrs = [bi]
        else:
            bicrs = [f"{bi}_up", f"{bi}_dn"]
        for b in bicrs:
            crosssections.at[
                b, "crsloc_definitionId"
            ] = "trapz_h{:,.1f}_bw{:,.1f}_tw{:,.1f}_c{:s}_{:s}".format(
                branches.loc[bi, "height"],
                branches.loc[bi, "width"],
                branches.loc[bi, "t_width"],
                branches.loc[bi, "closed"],
                "branch",
            )
            crosssections.at[b, "crsdef_id"] = crosssections.loc[
                b, "crsloc_definitionId"
            ]
            crosssections.at[b, "crsdef_type"] = branches.loc[bi, "shape"]
            crosssections.at[b, "crsdef_frictionId"] = branches.loc[bi, "frictionId"]
            crosssections.at[b, "crsdef_height"] = branches.loc[bi, "height"]
            crosssections.at[b, "crsdef_width"] = branches.loc[bi, "width"]
            crosssections.at[b, "crsdef_t_width"] = branches.loc[bi, "t_width"]
            crosssections.at[b, "crsdef_closed"] = branches.loc[bi, "closed"]

    # setup thaiweg for GUI
    crosssections["crsdef_thalweg"] = 0.0

    return crosssections


def set_xyz_crosssections(
    branches: gpd.GeoDataFrame,
    crosssections: gpd.GeoDataFrame,
):
    """Set up xyz crosssections.
    xyz crosssections should be points gpd, column z and column order.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        The branches.
    crosssections : gpd.GeoDataFrame
        The crosssections.

    Returns
    -------
    pd.DataFrame
        The xyz cross sections.
    """
    # check if require columns exist
    required_columns = ["geometry", "crsId", "order", "z"]
    if set(required_columns).issubset(crosssections.columns):
        crosssections = gpd.GeoDataFrame(crosssections[required_columns])
    else:
        logger.error(
            f"Cannot setup crosssections from branch. Require columns {required_columns}."
        )

    # apply data type
    crosssections.loc[:, "x"] = crosssections.geometry.x
    crosssections.loc[:, "y"] = crosssections.geometry.y
    crosssections.loc[:, "z"] = crosssections.z
    crosssections.loc[:, "order"] = crosssections.loc[:, "order"].astype("int")

    # convert xyz crosssection into yz profile
    crosssections = crosssections.groupby(level=0).apply(xyzp2xyzl, (["order"]))

    # snap to branch
    # setup branch_id - snap bridges to branch (inplace of bridges, will add branch_id and branch_offset columns)
    find_nearest_branch(
        branches=branches, geometries=crosssections, method="intersecting"
    )  # FIXME: what if the line intersect with 2/wrong branches?

    # setup failed - drop based on branch_offset that are not snapped to branch (inplace of yz_crosssections) and issue warning
    _old_ids = crosssections.index.to_list()
    crosssections.dropna(axis=0, inplace=True, subset=["branch_offset"])
    _new_ids = crosssections.index.to_list()
    if len(_old_ids) != len(_new_ids):
        logger.warning(
            f"Crosssection with id: {list(set(_old_ids) - set(_new_ids))} are dropped: unable to find closest branch. "
        )

    # setup crsdef from xyz
    crsdefs = pd.DataFrame(
        {
            "crsdef_id": crosssections.index.to_list(),
            "crsdef_type": "xyz",
            "crsdef_branchId": crosssections.branch_id.to_list(),  # FIXME test if leave this out
            "crsdef_xyzCount": crosssections.x.map(len).to_list(),
            "crsdef_xCoordinates": [
                " ".join(["{:.1f}".format(i) for i in l])
                for l in crosssections.x.to_list()
            ],  # FIXME cannot use list in gpd
            "crsdef_yCoordinates": [
                " ".join(["{:.1f}".format(i) for i in l])
                for l in crosssections.y.to_list()
            ],
            "crsdef_zCoordinates": [
                " ".join(["{:.1f}".format(i) for i in l])
                for l in crosssections.z.to_list()
            ],
            # 'crsdef_xylength': ' '.join(['{:.1f}'.format(i) for i in crosssections.l.to_list()[0]]),
            # lower case key means temp keys (not written to file)
            "crsdef_frictionId": branches.loc[
                crosssections.branch_id.to_list(), "frictionId"
            ],
            # lower case key means temp keys (not written to file)
        }
    )

    # setup crsloc from xyz
    # delete generated ones # FIXME change to branchId everywhere
    crslocs = pd.DataFrame(
        {
            "crsloc_id": [
                f"{bid}_{bc:.2f}"
                for bid, bc in zip(
                    crosssections.branch_id.to_list(),
                    crosssections.branch_offset.to_list(),
                )
            ],
            "crsloc_branchId": crosssections.branch_id.to_list(),  # FIXME change to branchId everywhere
            "crsloc_chainage": crosssections.branch_offset.to_list(),
            "crsloc_shift": 0.0,
            "crsloc_definitionId": crosssections.index.to_list(),
            "geometry": crosssections.geometry.centroid.to_list()
            # FIXME: line to centroid? because could not be written to the same sdhp file
        }
    )
    crosssections_ = pd.merge(
        crslocs,
        crsdefs,
        how="left",
        left_on=["crsloc_definitionId"],
        right_on=["crsdef_id"],
    )
    return crosssections_


def set_point_crosssections(
    branches: gpd.GeoDataFrame,
    crosssections: gpd.GeoDataFrame,
    crs_type: str = "point",
):
    pass


def xyzp2xyzl(xyz: pd.DataFrame, sort_by: list = ["x", "y"]):
    """Convert xyz points to xyz lines.

    Parameters
    ----------
    xyz: pd.DataFrame
        The xyz points.
    sort_by: list, optional
        List of attributes to sort by. Defaults to ["x", "y"].

    Returns
    -------
    gpd.GeoSeries
        The xyz lines.
    """

    sort_by = [s.lower() for s in sort_by]

    if xyz is not None:
        yz_index = xyz.index.unique()
        xyz.columns = [c.lower() for c in xyz.columns]
        xyz.reset_index(drop=True, inplace=True)

        # sort
        xyz_sorted = xyz.sort_values(by=sort_by)

        new_z = xyz_sorted.z.to_list()
        # temporary
        # new_z[0] = 1.4
        # new_z[-1] = 1.4

        line = LineString([(px, py) for px, py in zip(xyz_sorted.x, xyz_sorted.y)])
        xyz_line = gpd.GeoSeries(
            {
                "geometry": line,
                "l": list(
                    np.r_[
                        0.0,
                        np.cumsum(
                            np.hypot(
                                np.diff(line.coords, axis=0)[:, 0],
                                np.diff(line.coords, axis=0)[:, 1],
                            )
                        ),
                    ]
                ),
                "index": yz_index.to_list()[0],
                "x": xyz_sorted.x.to_list(),
                "y": xyz_sorted.y.to_list(),
                "z": new_z,
            }
        )
    return xyz_line
