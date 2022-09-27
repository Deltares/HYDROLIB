# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, Point

from .branches import find_nearest_branch
from .helper import check_gpd_attributes

# from delft3dfmpy.core import geometry
from .helper import split_lines

logger = logging.getLogger(__name__)


__all__ = [
    "set_branch_crosssections",
    "set_xyz_crosssections",
    "set_point_crosssections"
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
        crosssections = gpd.GeoDataFrame({}, index=branches.index, crs=branches.crs)
        crosssections["geometry"] = [
            l.interpolate(0.5, normalized=True) for l in branches.geometry
        ]
        crosssections["crsloc_id"] = [f"crs_{bid}" for bid in branches["branchId"]]
        crosssections["crsloc_branchId"] = branches["branchId"]
        crosssections["crsloc_chainage"] = [l / 2 for l in branches["geometry"].length]
        crosssections["crsloc_shift"] = branches["bedlev"]
    # Else prepares crosssections at both upstream and dowsntream extremities
    else:
        # Upstream
        ids = [f"{i}_up" for i in branches.index]
        crosssections_up = gpd.GeoDataFrame({}, index=ids, crs=branches.crs)
        crosssections_up["geometry"] = [Point(l.coords[0]) for l in branches.geometry]
        crosssections_up["crsloc_id"] = [
            f"crs_up_{bid}" for bid in branches["branchId"]
        ]
        crosssections_up["crsloc_branchId"] = branches["branchId"].values
        crosssections_up["crsloc_chainage"] = [0.0 for l in branches.geometry]
        crosssections_up["crsloc_shift"] = branches["invlev_up"].values
        # Downstream
        ids = [f"{i}_dn" for i in branches.index]
        crosssections_dn = gpd.GeoDataFrame({}, index=ids, crs=branches.crs)
        crosssections_dn["geometry"] = [Point(l.coords[-1]) for l in branches.geometry]
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
    """ Set up xyz crosssections.
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
):
    """
    Function to set regular cross-sections from point.
    only support rectangle, trapezoid, circle and yz

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        Require index to contain the branch id
        The branches.
    crosssections : gpd.GeoDataFrame
        Required columns: shape,shift
        The crosssections.

    Returns
    -------
    gpd.GeoDataFrame
        The cross sections.
    """

    # check if crs mismatch
    if crosssections.crs != branches.crs:
        logger.error(f"mismatch crs between cross-sections and branches")
    # snap to branch
    # setup branch_id - snap bridges to branch (inplace of bridges, will add branch_id and branch_offset columns)
    find_nearest_branch(
        branches=branches, geometries=crosssections, method="overal"
    )
    # get branch friction
    crosssections = crosssections.merge(branches['frictionId'], left_on = 'branch_id', right_index = True)

    # setup failed - drop based on branch_offset that are not snapped to branch (inplace of yz_crosssections) and issue warning
    _old_ids = crosssections.index.to_list()
    crosssections.dropna(axis=0, inplace=True, subset=["branch_offset"])
    _new_ids = crosssections.index.to_list()
    if len(_old_ids) != len(_new_ids):
        logger.warning(
            f"Crosssection with id: {list(set(_old_ids) - set(_new_ids))} are dropped: unable to find closest branch. "
        )

    crosssections_ = pd.DataFrame()
    # loop through the shapes
    all_shapes = crosssections["shape"].unique().tolist()
    for shape in all_shapes:
        if shape == 'trapezoid':
            trapezoid_crs = crosssections.loc[crosssections["shape"] == shape, :]
            valid_attributes = check_gpd_attributes(
                trapezoid_crs, required_columns=["branch_id", "branch_offset", 'frictionId', "width", "height", "t_width", "closed"]
            )
            crosssections_ = pd.concat([crosssections_, _set_trapezoid_crs(trapezoid_crs)])
        elif shape == 'zw':
            zw_crs = crosssections.loc[crosssections["shape"] == shape, :]
            valid_attributes = check_gpd_attributes(
                trapezoid_crs, required_columns=["branch_id", "branch_offset", 'frictionId',"numlevels", "levels", "flowwidths","totalwidths", "closed"]
            )
            crosssections_ = pd.concat([crosssections_, _set_zw_crs(zw_crs)])
        elif shape == 'yz':
            yz_crs = crosssections.loc[crosssections["shape"] == shape, :]
            valid_attributes = check_gpd_attributes(
                trapezoid_crs, required_columns=["branch_id", "branch_offset", 'frictionId',"yzcount", "ycoordinates", "zcoordinates", "closed"]
            )
            crosssections_ = pd.concat([crosssections_, _set_yz_crs(yz_crs)])
        else:
            logger.error("crossection shape not supported. For now only support trapezoid, zw and yz")

    # setup thaiweg for GUI
    crosssections_["crsdef_thalweg"] = 0.0

    crosssections_ = gpd.GeoDataFrame(crosssections_, crs=branches.crs)

    return crosssections_

def _set_trapezoid_crs(crosssections: gpd.GeoDataFrame):
    """trapezoid need to be converted into zw type"""

    crsdefs = []
    crslocs = []
    for c in crosssections.itertuples():
        levels = f'0 {c.height:.6f}'
        flowwidths = f'{c.width:.6f} {c.t_width:.6f}'
        crsdefs.append({
            "crsdef_id": c.Index,
            "crsdef_type": "zw",
            "crsdef_branchId": c.branch_id,  # FIXME test if leave this out
            "crsdef_numlevels": 2,
            "crsdef_levels": levels,
            "crsdef_flowwidths": flowwidths,
            "crsdef_totalwidths": flowwidths,
            "crsdef_frictionId": c.frictionId,
        })
        crslocs.append({
            "crsloc_id": f"{c.branch_id}_{c.branch_offset:.2f}",
            "crsloc_branchId": c.branch_id,  # FIXME change to branchId everywhere
            "crsloc_chainage": c.branch_offset,
            "crsloc_shift": c.shift,
            "crsloc_definitionId": c.Index,
            "geometry": c.geometry
        })

    crosssections_ = pd.merge(
        pd.DataFrame.from_records(crslocs),
        pd.DataFrame.from_records(crsdefs),
        how="left",
        left_on=["crsloc_definitionId"],
        right_on=["crsdef_id"],
    )
    return crosssections_


def _set_zw_crs(crosssections: gpd.GeoDataFrame):
    """set zw profile"""

    crsdefs = []
    crslocs = []
    for c in crosssections.itertuples():
        crsdefs.append({
            "crsdef_id": c.Index,
            "crsdef_type": "zw",
            "crsdef_branchId": c.branch_id,  # FIXME test if leave this out
            "crsdef_numlevels": c.numlevels,
            "crsdef_levels": c.levels,
            "crsdef_flowwidths": c.flowwidths,
            "crsdef_totalwidths": c.totalwidths,
            "crsdef_frictionId": c.frictionId,
        })
        crslocs.append({
            "crsloc_id": f"{c.branch_id}_{c.branch_offset:.2f}",
            "crsloc_branchId": c.branch_id,  # FIXME change to branchId everywhere
            "crsloc_chainage": c.branch_offset,
            "crsloc_shift": c.shift,
            "crsloc_definitionId": c.Index,
            "geometry": c.geometry
        })

    crosssections_ = pd.merge(
        pd.DataFrame.from_records(crslocs),
        pd.DataFrame.from_records(crsdefs),
        how="left",
        left_on=["crsloc_definitionId"],
        right_on=["crsdef_id"],
    )
    return crosssections_


def _set_yz_crs(crosssections: gpd.GeoDataFrame):
    """set yz profile"""

    crsdefs = []
    crslocs = []
    for c in crosssections.itertuples():
        crsdefs.append({
            "crsdef_id": c.Index,
            "crsdef_type": "yz",
            "crsdef_branchId": c.branch_id,  # FIXME test if leave this out
            "crsdef_yzcount": c.yzcount,
            "crsdef_ycoordinates": c.ycoordinates,
            "crsdef_zcoordinates": c.zcoordinates,
            "crsdef_frictionId": c.frictionId,
        })
        crslocs.append({
            "crsloc_id": f"{c.branch_id}_{c.branch_offset:.2f}",
            "crsloc_branchId": c.branch_id,  # FIXME change to branchId everywhere
            "crsloc_chainage": c.branch_offset,
            "crsloc_shift": c.shift,
            "crsloc_definitionId": c.Index,
            "geometry": c.geometry
        })

    crosssections_ = pd.merge(
        pd.DataFrame.from_records(crslocs),
        pd.DataFrame.from_records(crsdefs),
        how="left",
        left_on=["crsloc_definitionId"],
        right_on=["crsdef_id"],
    )
    return crosssections_

def parse_sobek_crs(filename, logger = logger):
    """read sobek crosssection files as a dataframe. Include location and definition file.
    #TODO: include parsing geometry as well

    Parameters
    ----------
    filename : Path
        Path to the sobek crosssection files. supported format: .DAT amd .DEF
    logger : logger, Optional

    Raise
    -----
    NotImplementedError
        do not support other files than .dat and .def

    Returns
    -------
    df.DataFrame
        The data frame with each item as a row
    """
    import shlex
    from pathlib import Path
    import pandas as pd
    import numpy as np

    # check file
    if Path(filename).name.lower().endswith('.def'):
        logger.info("Parsing cross section definition")
        prefix = 'CRDS'
        suffix = 'crds'
    elif Path(filename).name.lower().endswith('.dat'):
        logger.info("Parsing cross section location")
        prefix = 'CRSN'
        suffix = 'crsn'
    else:
        raise NotImplementedError("do not support other files than .dat and .def")

    with open(filename) as myFile:
        text = myFile.read()
        raw_lines = text.split(suffix + '\n')

    lines = []
    for l in raw_lines:
        if l.startswith(prefix): # new item
            # preliminary handling
            l = l.removeprefix(prefix)
            t = None
            # parse zw profile
            if 'lt lw\nTBLE' in l:
                # the table contains height, total width en flowing width.
                l, t = l.split('lt lw\nTBLE')
                levels, totalwidths, flowwidths = np.array(
                    [shlex.split(r, posix=False) for r in t.split('<')][:-1]).T  # last element is the suffix of tble
                table_dict = {}
                table_dict['numlevels'] = len(levels)
                table_dict['levels'] = ' '.join(str(n) for n in levels)
                table_dict['totalwidths'] = ' '.join(str(n) for n in totalwidths)
                table_dict['flowwidths'] = ' '.join(str(n) for n in flowwidths)
            # parse yz profile
            if 'lt yz\nTBLE' in l:
                # Y horizontal distance increasing from the left to right,
                # Z vertical distance increasing from bottom to top in m.
                # In other words, use a coordinate system to define the Y-Z profile.
                l, t = l.split('lt yz\nTBLE')
                yCoordinates, zCoordinates = np.array(
                    [shlex.split(r, posix=False) for r in t.split('<')][:-1]).T
                table_dict['yzcount'] = len(yCoordinates)
                table_dict['ycoordinates'] = ' '.join(str(n) for n in yCoordinates)
                table_dict['zcoordinates'] = ' '.join(str(n) for n in zCoordinates)
                # storage width on surface in m
                if 'lt sw 0' in l:
                    l.replace('lt lw 0', 'lt_lw_0') # remove space
                else:
                    logger.error('storage width function is not supported. Check lt sw field')
            # parse line
            line = shlex.split(l, posix=False)
            line_dict = {line[i]: line[i + 1] for i in range(0, len(line), 2)}
            # add table
            if t is not None:
                line_dict.update(table_dict)
            lines.append(line_dict)

    df = pd.DataFrame.from_records(lines)
    df['id'] = df['id'].str.strip("'")
    df.set_index('id', inplace=True)

    return df

def xyzp2xyzl(xyz: pd.DataFrame, sort_by: list = ["x", "y"]):
    """ Convert xyz points to xyz lines.
    
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
