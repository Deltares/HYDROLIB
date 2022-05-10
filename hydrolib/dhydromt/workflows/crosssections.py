# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from scipy.spatial import distance
import configparser
import logging
from delft3dfmpy.core.geometry import find_nearest_branch
from hydromt import config
import hydromt.io

from .helper import split_lines

logger = logging.getLogger(__name__)


__all__ = [
    "set_branch_crosssections", "set_xyz_crosssections"
]  # , "process_crosssections", "validate_crosssections"]


def set_branch_crosssections(branches:gpd.GeoDataFrame,
                             crosssections: gpd.GeoDataFrame,
                                crs_type: str = 'branch'):
    """
    Function to set (inplcae) cross-section definition ids following the convention used in delft3dfmpy
    # FIXME BMA: closed profile not yet supported (as can be seem that the definitionId convention does not convey any info on whether profile is closed or not)
    """

    # 2. check if branches have the required columns and select only required columns
    required_columns = ["geometry", "branchId", "branchType", "friction_id", "frictionType", "frictionValue",
                        "shape", "width", "height", "t_width", "bedlev", "closed"]
    if set(required_columns).issubset(crosssections.columns):
        crosssections = crosssections[required_columns]
    else:
        self.logger.error(f"Cannto setup crosssections from branch. Require columns {required_columns}.")

    # 3. get the crs at the midpoint of branches
    crslocs = crosssections.copy()
    crslocs["crsloc_id"] = [f"crs_{bid}" for bid in crosssections["branchId"]]
    crslocs["crsloc_branchId"] = crosssections["branchId"]
    crslocs["crsloc_chainage"] = [l / 2 for l in crosssections["geometry"].length]
    crslocs["crsloc_shift"] = crosssections["bedlev"]

    if crsdefid_col not in crslocs.columns:
        crslocs["definitionId"] = None

    circle_indexes = crslocs.loc[crslocs["shape"] == 'circle', :].index
    for bi in circle_indexes:
        crslocs.at[bi, "crsloc_definitionId"] = 'circ_d{:,.3f}_{:s}'.format(crslocs.loc[bi, diameter_col], crs_type)

    rectangle_indexes = crslocs.loc[crslocs["shape"] == 'rectangle', :].index

    # if rectangle_indexes.has_duplicates:
    #     logger.error('Duplicate id is found. Please solve this. ')

    # FIXME BMA: duplicate indexes result in problems below.
    for bi in rectangle_indexes:
        crslocs.at[bi, "crsloc_definitionId"] = 'rect_h{:,.3f}_w{:,.3f}_c{:s}_{:s}'.format(
                crslocs.loc[bi, "height"], crslocs.loc[bi, "width"], crslocs.loc[bi, "closed"], crs_type)

    # FIXME BMA: review the trapezoid when available
    trapezoid_indexes = crslocs.loc[crslocs["shape"] == 'trapezoid', :].index
    for bi in trapezoid_indexes:
        crslocs.at[bi, "crsloc_definitionId"] = 'trapz_h{:,.1f}_bw{:,.1f}_tw{:,.1f}_c{:s}_{:s}'.format(
            crslocs.loc[bi, "height"], crslocs.loc[bi, "width"], crslocs.loc[bi, "t_width"], crslocs.loc[bi, "closed"], crs_type)

    return crslocs

def set_xyz_crosssections(branches:gpd.GeoDataFrame, crosssections:gpd.GeoDataFrame,crs_type: str = 'xyz'
                            ):
    """setup xyz crosssections
    xyz crosssections should be points gpd, column z and column order.
    """
    # check if require columns exist
    required_columns = ["geometry", "crsId", "order", "z"]
    if set(required_columns).issubset(crosssections.columns):
        crosssections = gpd.GeoDataFrame(crosssections[required_columns])
    else:
        self.logger.error(f"Cannto setup crosssections from branch. Require columns {required_columns}.")

    # apply data type
    crosssections.loc[:, 'x'] = crosssections.geometry.x
    crosssections.loc[:, 'y'] = crosssections.geometry.y
    crosssections.loc[:, 'z'] = crosssections.z
    crosssections.loc[:, 'order'] = crosssections.loc[:, 'order'].astype('int')

    # convert xyz crosssection into yz profile
    crosssections = crosssections.groupby(level=0).apply(xyzp2xyzl, (['order']))

    # snap to branch
    # setup branch_id - snap bridges to branch (inplace of bridges, will add branch_id and branch_offset columns)
    find_nearest_branch(branches=branches, geometries=crosssections,
                                 method="intersecting")  # FIXME: what if the line intersect with 2/wrong branches?

    # setup failed - drop based on branch_offset that are not snapped to branch (inplace of yz_crosssections) and issue warning
    _old_ids = crosssections.index.to_list()
    crosssections.dropna(axis=0, inplace=True, subset=['branch_offset'])
    _new_ids = crosssections.index.to_list()
    if len(_old_ids) != len(_new_ids):
        logger.warning(
            f'Crosssection with id: {list(set(_old_ids) - set(_new_ids))} are dropped: unable to find closest branch. ')

    # setup crsdef from xyz
    crsdefs = pd.DataFrame(
                             {'crsdef_id': crosssections.index.to_list(),
                              'crsdef_type': "xyz",
                              'crsdef_branchId': crosssections.branch_id.to_list(),  # FIXME test if leave this out
                              'crsdef_xCoordinates': ' '.join(['{:.1f}'.format(i) for i in crosssections.x.to_list()[0]]), # FIXME cannot use list in gpd
                              'crsdef_yCoordinates': ' '.join(['{:.1f}'.format(i) for i in crosssections.y.to_list()[0]]),
                              'crsdef_zCoordinates': ' '.join(['{:.1f}'.format(i) for i in crosssections.z.to_list()[0]]),
                              'crsdef_xylength': ' '.join(['{:.1f}'.format(i) for i in crosssections.l.to_list()[0]]),
                              # lower case key means temp keys (not written to file)
                              'crsdef_frictionId': branches.loc[crosssections.branch_id.to_list(), 'friction_id'],
                              # lower case key means temp keys (not written to file)
                              })

    # setup crsloc from xyz
    # delete generated ones # FIXME change to branchId everywhere
    crslocs = pd.DataFrame({
                             'crsloc_id': [f'{bid}_{bc:.2f}' for bid, bc in
                                    zip(crosssections.branch_id.to_list(), crosssections.branch_offset.to_list())],
                             'crsloc_branchId': crosssections.branch_id.to_list(),  # FIXME change to branchId everywhere
                             'crsloc_chainage': crosssections.branch_offset.to_list(),
                             'crsloc_shift': 0.0,
                             'crsloc_definitionId': crosssections.index.to_list(),
                             'geometry': crosssections.geometry.centroid.to_list()
                             # FIXME: line to centroid? because could not be written to the same sdhp file
                         })
    crosssections_ = pd.merge(crslocs, crsdefs, how='left', left_on=['crsloc_definitionId'], right_on=['crsdef_id'])
    return crosssections_


def set_point_crosssections(branches:gpd.GeoDataFrame, crosssections:gpd.GeoDataFrame, crs_type: str = 'point'):
    pass


def xyzp2xyzl(xyz: pd.DataFrame, sort_by: list = ['x', 'y']):

    """xyz point to xyz line"""
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
        xyz_line = gpd.GeoSeries({'geometry': line,
                                  'l': list(np.r_[0., np.cumsum(
                                      np.hypot(np.diff(line.coords, axis=0)[:, 0], np.diff(line.coords, axis=0)[:, 1]))]),
                                  'index': yz_index.to_list()[0],
                                  'x': xyz_sorted.x.to_list(),
                                  'y': xyz_sorted.y.to_list(),
                                  'z': new_z,
                                  })
    return xyz_line

