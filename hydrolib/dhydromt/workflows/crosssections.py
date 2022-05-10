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


# FIXME BMA: For HydroMT we will rewrite this snap_branches to snap_lines and take set_branches out of it



def set_branch_crosssections(branches:gpd.GeoDataFrame,
                             crosssections: gpd.GeoDataFrame,
                                shape_col: str = 'shape',
                                diameter_col: str = 'diameter',
                                height_col: str = 'height',
                                width_col: str = 'width',
                                t_width_col: str = 't_width',
                                closed_col: str = 'closed',
                                crsdefid_col: str = 'definitionId',
                                frictionid_col: str = 'frictionId',
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
        crslocs.at[bi, "definitionId"] = 'circ_d{:,.3f}_{:s}'.format(crslocs.loc[bi, diameter_col], crs_type)

    rectangle_indexes = crslocs.loc[crslocs["shape"] == 'rectangle', :].index

    # if rectangle_indexes.has_duplicates:
    #     logger.error('Duplicate id is found. Please solve this. ')

    # FIXME BMA: duplicate indexes result in problems below.
    for bi in rectangle_indexes:
        crslocs.at[bi, "crsloc_definitionId"] = 'rect_h{:,.3f}_w{:,.3f}_c{:s}_{:s}'.format(
                crslocs.loc[bi, height_col], crslocs.loc[bi, width_col], crslocs.loc[bi, closed_col], crs_type)

    # FIXME BMA: review the trapezoid when available
    trapezoid_indexes = crslocs.loc[crslocs["shape"] == 'trapezoid', :].index
    for bi in trapezoid_indexes:
        crslocs.at[bi, "crsloc_definitionId"] = 'trapz_h{:,.1f}_bw{:,.1f}_tw{:,.1f}_c{:s}_{:s}'.format(
            crslocs.loc[bi, height_col], crslocs.loc[bi, width_col], crslocs.loc[bi, t_width_col], crslocs.loc[bi, closed_col], crs_type)

    return crslocs



def set_crsdefid_for_branches(crslocs: gpd.GeoDataFrame,
                                shape_col: str = 'shape',
                                diameter_col: str = 'diameter',
                                height_col: str = 'height',
                                width_col: str = 'width',
                                t_width_col: str = 't_width',
                                closed_col: str = 'closed',
                                crsdefid_col: str = 'definitionId',
                                frictionid_col: str = 'frictionId',
                                crs_type: str = 'branch'):
    """
    Function to set (inplcae) cross-section definition ids following the convention used in delft3dfmpy
    # FIXME BMA: closed profile not yet supported (as can be seem that the definitionId convention does not convey any info on whether profile is closed or not)
    """

    if crsdefid_col not in crslocs.columns:
        crslocs[crsdefid_col] = None

    circle_indexes = crslocs.loc[crslocs[shape_col] == 'circle', :].index
    for bi in circle_indexes:
        crslocs.at[bi, crsdefid_col] = 'circ_d{:,.3f}_{:s}'.format(crslocs.loc[bi, diameter_col], crs_type)

    rectangle_indexes = crslocs.loc[crslocs[shape_col] == 'rectangle', :].index

    # if rectangle_indexes.has_duplicates:
    #     logger.error('Duplicate id is found. Please solve this. ')

    # FIXME BMA: duplicate indexes result in problems below.
    for bi in rectangle_indexes:
        crslocs.at[bi, crsdefid_col] = 'rect_h{:,.3f}_w{:,.3f}_c{:s}_{:s}'.format(
                crslocs.loc[bi, height_col], crslocs.loc[bi, width_col], crslocs.loc[bi, closed_col], crs_type)

    # FIXME BMA: review the trapezoid when available
    trapezoid_indexes = crslocs.loc[crslocs[shape_col] == 'trapezoid', :].index
    for bi in trapezoid_indexes:
        crslocs.at[bi, crsdefid_col] = 'trapz_h{:,.1f}_bw{:,.1f}_tw{:,.1f}_c{:s}_{:s}'.format(
            crslocs.loc[bi, height_col], crslocs.loc[bi, width_col], crslocs.loc[bi, t_width_col], crslocs.loc[bi, closed_col], crs_type)

    return crslocs



def set_xyz_crosssections(branches:gpd.GeoDataFrame, crosssections:gpd.GeoDataFrame,
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
                              'crsdef_xCoordinates': crosssections.x.to_list(),
                              'crsdef_yCoordinates': crosssections.y.to_list(),
                              'crsdef_zCoordinates': crosssections.z.to_list(),
                              'crsdef_xylength': crosssections.l.to_list(),
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


def generate_crosssections(
    crosssections: gpd.GeoDataFrame,
    crosssections_ini: configparser,
    branch_query: str = None,
    logger=logging,
):
    """
    Setup crosssections (crsdefs and crslocs) based on branches
    # TODO BMA: add other type of cross sections e.g. point, xyz
    """

    logger.info("Setting up crosssections (crsdefs and crslocs).")
    crsdefs = None
    crslocs = None

    # setup column names
    shape_col = "type"
    diameter_col = "diameter"
    height_col = "height"
    width_col = "width"
    t_width_col = "t_width"
    closed_col = "closed"

    # setup helper
    none2str = lambda x: "" if x is None else x
    upstream_prefix = none2str(crosssections_ini["global"]["upstream_prefix"])
    upstream_suffix = none2str(crosssections_ini["global"]["upstream_suffix"])
    downstream_prefix = none2str(crosssections_ini["global"]["downstream_prefix"])
    downstream_suffix = none2str(crosssections_ini["global"]["downstream_suffix"])

    # prepare one shift columns from multiple branch types
    shift_col = "shift"
    upstream_shift_col = "start" + shift_col
    downstream_shift_col = "end" + shift_col
    crosssections[upstream_shift_col] = None
    crosssections[downstream_shift_col] = None
    for c in crosssections_ini["global"]["shift_col"]:
        upstream_c_col = upstream_prefix + c + upstream_suffix
        downstream_c_col = downstream_prefix + c + downstream_suffix
        crosssections[upstream_shift_col] = crosssections[
            upstream_shift_col
        ].combine_first(crosssections[upstream_c_col].rename({}))
        crosssections[downstream_shift_col] = crosssections[
            downstream_shift_col
        ].combine_first(crosssections[downstream_c_col])

    # setup crsdef from branches - upstream and downstream
    crsdefs = pd.concat(
        [
            _setup_crsdefs_from_branches_at(
                crosssections,
                at="start",
                col_prefix=upstream_prefix,
                col_suffix=upstream_suffix,
                shape_col=shape_col,
                diameter_col=diameter_col,
                height_col=height_col,
                width_col=width_col,
                t_width_col=t_width_col,
                closed_col=closed_col,
                frictionid_col="frictionId",  # convention in Delft3D FM
                crsdefid_col="definitionId",  # convention in Delft3D FM
                crs_type="branch",
                is_shared="True",
                logger=logger,
            ),
            _setup_crsdefs_from_branches_at(
                crosssections,
                at="end",
                col_prefix=downstream_prefix,
                col_suffix=downstream_suffix,
                shape_col=shape_col,
                diameter_col=diameter_col,
                height_col=height_col,
                width_col=width_col,
                t_width_col=t_width_col,
                closed_col=closed_col,
                frictionid_col="frictionId",  # convention in Delft3D FM
                crsdefid_col="definitionId",  # convention in Delft3D FM
                crs_type="branch",
                is_shared="True",
                logger=logger,
            ),
        ]
    ).drop_duplicates()

    # setup crslocs from branches - upstream and downstream
    crslocs = pd.concat(
        [
            _setup_crslocs_from_branches_at(
                crosssections,
                at="start",
                col_prefix=upstream_prefix,
                col_suffix=upstream_suffix,
                shift_col=upstream_shift_col,
                crsdefid_col="definitionId",  # convention in Delft3D FM
                logger=logger,
            ),
            _setup_crslocs_from_branches_at(
                crosssections,
                at="end",
                col_prefix=downstream_prefix,
                col_suffix=downstream_suffix,
                shift_col=downstream_shift_col,
                crsdefid_col="definitionId",  # convention in Delft3D FM
                logger=logger,
            ),
        ]
    ).drop_duplicates()

    # validate crsdefs geometry (#TODO add a real geometry shape to each def?)
    for ri, r in crsdefs.iterrows():
        if r["type"] == "rectangle":
            if np.all(r[["width", "height"]] > 0):
                pass
            else:
                logger.error(
                    f"Rectangle crosssection definition of {r['id']} have 0 m width/height."
                    + "Issue might have been caused due to incorrect values in branches data layer. Please revise. "
                )

        if r["type"] == "trapezoid":
            if np.all(r[["t_width", "height"]] > 0):
                pass
            else:
                logger.error(
                    f"Trapezoid crosssection definition of {r['id']} have 0 m t_width/height. "
                    + "Issue might have been caused due to incorrect values in branches data layer. Please revise. "
                )

        if r["type"] == "circle":
            if np.all(r[["diameter"]] > 0):
                pass
            else:
                logger.error(
                    f"Circle crosssection definition of {r['id']} have 0m diameter. "
                    + "Issue might have been caused due to incorrect values in branches data layer. Please revise. "
                )

    return crsdefs, crslocs, crosssections


def setup_crosssections(
    branches: gpd.GeoDataFrame,
    crosssections_ini_fn: str = None,
    crosssections_fn: str = None,
    generate_crosssections_from_branches: str = True,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    required_query: str = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "intersect",
    snap_offset: float = 1,
    pilot: gpd.GeoDataFrame = None,
    clip_predicate="contains",
    clip_buffer: float = 10,
    logger=logging,
):
    """
    Setup crosssections (crsdefs and crslocs) based on branches
    # TODO BMA: add other type of cross sections e.g. point, xyz
    """

    logger.info("Setting up crosssections (crsdefs and crslocs).")
    crsdefs = None
    crslocs = None
    use_infile_order = True
    drop_snap_failed = True

    if crosssections_ini_fn is not None:
        crosssections_ini = parse_ini(crosssections_ini_fn)

        # setup funcs
        crosssection_type = crosssections_ini["global"]["crosssection_type"]
        use_infile_order = crosssections_ini["global"]["use_infile_order"]
        drop_snap_failed = crosssections_ini["global"]["drop_snap_failed"]

        # setup column names
        shape_col = crosssections_ini["global"]["shape_col"]
        diameter_col = crosssections_ini["global"]["diameter_col"]
        height_col = crosssections_ini["global"]["height_col"]
        width_col = crosssections_ini["global"]["width_col"]
        t_width_col = crosssections_ini["global"]["t_width_col"]
        closed_col = crosssections_ini["global"]["closed_col"]

        # setup helper
        none2str = lambda x: "" if x is None else x
        upstream_prefix = none2str(crosssections_ini["global"]["upstream_prefix"])
        upstream_suffix = none2str(crosssections_ini["global"]["upstream_suffix"])
        downstream_prefix = none2str(crosssections_ini["global"]["downstream_prefix"])
        downstream_suffix = none2str(crosssections_ini["global"]["downstream_suffix"])

    # use branches
    if generate_crosssections_from_branches is True and branches is not None:
        branches_ = read_gpd(branches, logger=logger)

        # prepare default cross-sections
        branches_ = append_data_columns_based_on_ini_query(branches_, crosssections_ini)

        # prepare one shift columns from multiple branch types
        shift_col = "shift"
        upstream_shift_col = "start" + shift_col
        downstream_shift_col = "end" + shift_col
        branches_[upstream_shift_col] = None
        branches_[downstream_shift_col] = None
        for c in crosssections_ini["global"]["shift_col"]:
            upstream_c_col = upstream_prefix + c + upstream_suffix
            downstream_c_col = downstream_prefix + c + downstream_suffix
            branches_[upstream_shift_col] = branches_[upstream_shift_col].combine_first(
                branches_[upstream_c_col].rename({})
            )
            branches_[downstream_shift_col] = branches_[
                downstream_shift_col
            ].combine_first(branches_[downstream_c_col])

        # setup crsdef from branches - upstream and downstream
        crsdefs = pd.concat(
            [
                _setup_crsdefs_from_branches_at(
                    branches_,
                    at="start",
                    col_prefix=upstream_prefix,
                    col_suffix=upstream_suffix,
                    shape_col=shape_col,
                    diameter_col=diameter_col,
                    height_col=height_col,
                    width_col=width_col,
                    t_width_col=t_width_col,
                    closed_col=closed_col,
                    frictionid_col="frictionId",  # convention in Delft3D FM
                    crsdefid_col="definitionId",  # convention in Delft3D FM
                    crs_type="branch",
                    is_shared="True",
                    logger=logger,
                ),
                _setup_crsdefs_from_branches_at(
                    branches_,
                    at="end",
                    col_prefix=downstream_prefix,
                    col_suffix=downstream_suffix,
                    shape_col=shape_col,
                    diameter_col=diameter_col,
                    height_col=height_col,
                    width_col=width_col,
                    t_width_col=t_width_col,
                    closed_col=closed_col,
                    frictionid_col="frictionId",  # convention in Delft3D FM
                    crsdefid_col="definitionId",  # convention in Delft3D FM
                    crs_type="branch",
                    is_shared="True",
                    logger=logger,
                ),
            ]
        ).drop_duplicates()

        # setup crslocs from branches - upstream and downstream
        crslocs = pd.concat(
            [
                _setup_crslocs_from_branches_at(
                    branches_,
                    at="start",
                    col_prefix=upstream_prefix,
                    col_suffix=upstream_suffix,
                    shift_col=upstream_shift_col,
                    crsdefid_col="definitionId",  # convention in Delft3D FM
                    logger=logger,
                ),
                _setup_crslocs_from_branches_at(
                    branches_,
                    at="end",
                    col_prefix=downstream_prefix,
                    col_suffix=downstream_suffix,
                    shift_col=downstream_shift_col,
                    crsdefid_col="definitionId",  # convention in Delft3D FM
                    logger=logger,
                ),
            ]
        ).drop_duplicates()

    # use xyz
    if crosssections_fn is not None and crosssection_type == "xyz":
        crosssections = read_gpd(
            crosssections_fn,
            id_col,
            rename_map=rename_map,
            required_columns=required_columns,
            required_dtypes=required_dtypes,
            required_query=required_query,
            clip_geom=pilot.buffer(clip_buffer),
            clip_predicate=clip_predicate,
            logger=logger,
        )

        # check if cross sections exsit
        if len(crosssections) > 0:
            # additional filter based on id instead of geom
            crosssection_ids = crosssections.CRS_ID.unique()
            crosssections = read_gpd(
                crosssections_fn,
                id_col,
                rename_map=rename_map,
                required_columns=required_columns,
                required_dtypes=required_dtypes,
                required_query=required_query,
                clip_geom=pilot.buffer(clip_buffer + 1000),
                clip_predicate=clip_predicate,
                logger=logger,
            )

            crosssections = crosssections[crosssections.index.isin(crosssection_ids)]

            logger.info("Crosssections are added from: " + crosssections_fn)

            # convert xyz to yz
            # FIXME: now the xy are read from the columns instead of the geometry. It is better to read from geometry otherwise might be confusing.
            if use_infile_order is True:
                crosssections = crosssections.groupby(level=0).apply(
                    geometry.xyzp2xyzl, (["ORDER"])
                )
            else:
                crosssections = crosssections.groupby(level=0).apply(geometry.xyzp2xyzl)
            logger.info(f"{len(crosssections)} xyz cross sections are setup.")

            # subset branch
            if branch_query is not None:
                branches_ = branches_.query(branch_query)

            # snap to branch
            # setup branch_id - snap bridges to branch (inplace of bridges, will add branch_id and branch_offset columns)
            logger.info(
                "Crosssections are snapped to nearest branch using "
                + str(snap_method)
                + " method."
            )
            geometry.find_nearest_branch(
                branches=branches_, geometries=crosssections, method=snap_method
            )  # FIXME: what if the line intersect with 2/wrong branches?

            # setup failed - drop based on branch_offset that are not snapped to branch (inplace of yz_crosssections) and issue warning
            if drop_snap_failed == True:
                _old_ids = crosssections.index.to_list()
                crosssections.dropna(axis=0, inplace=True, subset=["branch_offset"])
                _new_ids = crosssections.index.to_list()
                if len(_old_ids) != len(_new_ids):
                    logger.warning(
                        f"Crosssection with id: {list(set(_old_ids) - set(_new_ids)) } are dropped: unable to find closest branch. "
                    )

            # setup crsdef from xyz
            crsdefs = pd.concat(
                [
                    crsdefs,
                    pd.DataFrame(
                        {
                            "id": crosssections.index.to_list(),
                            "type": crosssection_type,  # FIXME read from file
                            "branchId": crosssections.branch_id.to_list(),  # FIXME test if leave this out
                            "xCoordinates": crosssections.x.to_list(),
                            "yCoordinates": crosssections.y.to_list(),
                            "zCoordinates": crosssections.z.to_list(),
                            "xylength": crosssections.l.to_list(),  # lower case key means temp keys (not written to file)
                            "frictionid": branches_.loc[
                                crosssections.branch_id.to_list(), "frictionId"
                            ],  # lower case key means temp keys (not written to file)
                        }
                    ),
                ]
            )  # type as xyz but will be written as yz in dfmreader and dfmcore and dfmwriter

            # setup crsloc from xyz
            # delete generated ones
            crslocs = crslocs[
                ~crslocs.branchid.isin(crosssections.branch_id.unique())
            ]  # FIXME change to branchId everywhere
            crslocs = pd.concat(
                [
                    crslocs,
                    pd.DataFrame(
                        {
                            "id": [
                                f"{bid}_{bc:.2f}"
                                for bid, bc in zip(
                                    crosssections.branch_id.to_list(),
                                    crosssections.branch_offset.to_list(),
                                )
                            ],
                            "branchid": crosssections.branch_id.to_list(),  # FIXME change to branchId everywhere
                            "chainage": crosssections.branch_offset.to_list(),
                            "shift": 0.0,
                            "definitionId": crosssections.index.to_list(),
                            "geometry": crosssections.geometry.centroid.to_list(),  # FIXME: line to centroid? because could not be written to the same sdhp file
                        }
                    ),
                ]
            )

        else:
            logger.warning("no xyz cross sections are setup.")

    # validate crsdefs geometry (#TODO add a real geometry shape to each def?)
    for ri, r in crsdefs.iterrows():
        if r["type"] == "rectangle":
            if np.all(r[["width", "height"]] > 0):
                pass
            else:
                logger.error(
                    f"Rectangle crosssection definition of {r['id']} have 0 m width/height."
                    + "Issue might have been caused due to incorrect values in branches data layer. Please revise. "
                )

        if r["type"] == "trapezoid":
            if np.all(r[["t_width", "height"]] > 0):
                pass
            else:
                logger.error(
                    f"Trapezoid crosssection definition of {r['id']} have 0 m t_width/height. "
                    + "Issue might have been caused due to incorrect values in branches data layer. Please revise. "
                )

        if r["type"] == "circle":
            if np.all(r[["diameter"]] > 0):
                pass
            else:
                logger.error(
                    f"Circle crosssection definition of {r['id']} have 0m diameter. "
                    + "Issue might have been caused due to incorrect values in branches data layer. Please revise. "
                )

    return crsdefs, crslocs, branches_


def _setup_crsdefs_from_branches_at(
    branches: gpd.GeoDataFrame,
    at: str = "",
    col_prefix: str = "",
    col_suffix: str = "",
    shape_col: str = "SHAPE",
    diameter_col: str = "DIAMETER",
    height_col: str = "HEIGHT",
    width_col: str = "WIDTH",
    t_width_col: str = "T_WIDTH",
    closed_col: str = "CLOSED",
    frictionid_col: str = "frictionId",
    crsdefid_col: str = "definitionId",
    crs_type: str = "branch",
    is_shared: str = "False",
    logger=logging,
):
    """
    Function to setup crsdefs from given branches.
    possible to apply col_prefix and col_suffix on height_col, width_col, t_width_col, crsdefid_col
    """

    crsdefid_col = at + crsdefid_col

    if col_prefix is None:
        col_prefix = ""
    if col_suffix is None:
        col_suffix = ""

    height_col = col_prefix + height_col + col_suffix
    width_col = col_prefix + width_col + col_suffix
    t_width_col = col_prefix + t_width_col + col_suffix

    logger.debug(
        f"Generating crsdefs using the following columns in GeoDataFrame: "
        + f"{shape_col}, {diameter_col}, {height_col}, {width_col}, {t_width_col}, {frictionid_col}"
    )

    # set crossection definition id for branch
    branches = __set_crsdefid_for_branches(
        branches,
        shape_col=shape_col,
        diameter_col=diameter_col,
        height_col=height_col,
        width_col=width_col,
        t_width_col=t_width_col,
        closed_col=closed_col,
        crsdefid_col=crsdefid_col,
        crs_type=crs_type,
    )
    #  get crosssection definition
    crsdefs = __get_crsdef_from_branches(
        branches,
        shape_col=shape_col,
        diameter_col=diameter_col,
        height_col=height_col,
        width_col=width_col,
        t_width_col=t_width_col,
        closed_col=closed_col,
        crsdefid_col=crsdefid_col,
        frictionid_col=frictionid_col,
        crs_type=crs_type,
        is_shared=is_shared,
    )

    return crsdefs


def _setup_crslocs_from_branches_at(
    branches: gpd.GeoDataFrame,
    at: str = "start",
    col_prefix: str = "",
    col_suffix: str = "",
    shift_col: str = "shift",
    crsdefid_col: str = "definitionId",
    logger=logging,
):
    """
    Function to setup crslocs from given branches at start/end.
    possible to apply col_prefix and col_suffix on shift_col, crsdef_col
    """

    crsdefid_col = at + crsdefid_col

    if col_prefix is None:
        col_prefix = ""
    if col_suffix is None:
        col_suffix = ""

    logger.debug(
        f"Generating crslocs at {at} of the branch using the following columns in GeoDataFrame: "
        + f"{shift_col}, {crsdefid_col}"
    )

    #  get crosssection locations
    crslocs = __get_crsloc_from_branches(
        branches, at=at, shift_col=shift_col, crsdefid_col=crsdefid_col
    )
    return crslocs


def __set_crsdefid_for_branches(
    branches: gpd.GeoDataFrame,
    shape_col: str = "SHAPE",
    diameter_col: str = "DIAMETER",
    height_col: str = "HEIGHT",
    width_col: str = "WIDTH",
    t_width_col: str = "T_WIDTH",
    closed_col: str = "CLOSED",
    crsdefid_col: str = "definitionId",
    frictionid_col: str = "frictionid",
    crs_type: str = "branch",
):
    """
    Function to set (inplcae) cross-section definition ids following the convention used in delft3dfmpy
    # FIXME BMA: closed profile not yet supported (as can be seem that the definitionId convention does not convey any info on whether profile is closed or not)
    """

    if crsdefid_col not in branches.columns:
        branches[crsdefid_col] = None

    circle_indexes = branches.loc[branches[shape_col] == "circle", :].index
    for bi in circle_indexes:
        branches.at[bi, crsdefid_col] = "circ_d{:,.3f}_{:s}".format(
            branches.loc[bi, diameter_col], crs_type
        )

    rectangle_indexes = branches.loc[branches[shape_col] == "rectangle", :].index

    # if rectangle_indexes.has_duplicates:
    #     logger.error('Duplicate id is found. Please solve this. ')

    # FIXME BMA: duplicate indexes result in problems below.
    for bi in rectangle_indexes:
        branches.at[bi, crsdefid_col] = "rect_h{:,.3f}_w{:,.3f}_c{:s}_{:s}".format(
            branches.loc[bi, height_col],
            branches.loc[bi, width_col],
            branches.loc[bi, closed_col],
            crs_type,
        )

    # FIXME BMA: review the trapezoid when available
    trapezoid_indexes = branches.loc[branches[shape_col] == "trapezoid", :].index
    for bi in trapezoid_indexes:
        slope = (
            (branches.loc[bi, t_width_col] - branches.loc[bi, width_col])
            / 2
            / branches.loc[bi, height_col]
        )
        branches.at[
            bi, crsdefid_col
        ] = "trapz_s{:,.1f}_bw{:,.1f}_bw{:,.1f}_c{:s}_{:s}".format(
            slope,
            branches.loc[bi, width_col],
            branches.loc[bi, t_width_col],
            branches.loc[bi, closed_col],
            crs_type,
        )

    return branches


def __get_crsdef_from_branches(
    branches: gpd.GeoDataFrame,
    shape_col: str = "SHAPE",
    diameter_col: str = "DIAMETER",
    height_col: str = "HEIGHT",
    width_col: str = "WIDTH",
    t_width_col: str = "T_WIDTH",
    closed_col: str = "CLOSED",
    frictionid_col: str = "frictionId",
    crsdefid_col: str = "definitionId",
    crs_type: str = "branch",
    is_shared: str = "False",
):
    """
    Function to get cross-section definition based on the definitionId following the convention used in delft3dfmpy
    # FIXME BMA: avoid the below function by add data preprocessing when reading
    """

    crsdef = []
    for i in branches[crsdefid_col].unique():
        b = (
            branches.loc[
                branches[crsdefid_col] == i,
                [
                    shape_col,
                    diameter_col,
                    height_col,
                    width_col,
                    t_width_col,
                    closed_col,
                    frictionid_col,
                ],
            ]
            .drop_duplicates()
            .to_dict("index")
        )
        bi = list(b.keys())[0]
        crsdef.append(
            {
                "id": i,
                "type": b[bi][shape_col],
                "thalweg": 0.0,
                "height": b[bi][height_col],
                "width": b[bi][width_col],
                "t_width": b[bi][t_width_col],
                "closed": b[bi][closed_col],
                "diameter": b[bi][diameter_col],
                "frictionid": b[bi][frictionid_col],
                "crs_type": crs_type,
                "is_shared": is_shared,
            }
        )

    crsdef = pd.DataFrame(crsdef).drop_duplicates()
    return crsdef


def __get_crsloc_from_branches(
    branches: gpd.GeoDataFrame,
    shift_col: str = "shift",
    crsdefid_col: str = "definitionId",
    at: str = "start",
    offset: float = 0.0,
):
    """Function to obtain the crsloc from branches

    parameters
    --------------------
    branches: branch geodataframe with index as the branchid
    shift_col:str = 'shift': column that register the shift
    crsdefid_col:str = 'definitionId': column that register the crossection definination (default is definitionId)
    at:str = 'start':  get crsloc from branches at start/end of the branches
    offset:float = 0.0: get crsloc from branches with an offset
    # FIXME BMA: for now offset != 0.0 is not supported. Adding offset means interpolations of profiles also need to be done
    """

    # determine the chainage of the crosssection
    chainage_options = {"start": "0.0 + offset", "end": "b.geometry.length -  offset"}
    chainage = chainage_options[at]

    # get crsloc data model
    crsloc = []
    for bi, b in branches[["geometry", shift_col, crsdefid_col]].iterrows():
        crsloc.append(
            {
                "id": "%s_%.2f" % (bi, eval(chainage)),
                "branchid": bi,
                "chainage": eval(chainage),
                "shift": b[shift_col],
                "definitionId": b[crsdefid_col],
                "geometry": b["geometry"].interpolate(eval(chainage)),
            }
        )

    crsloc = gpd.GeoDataFrame(crsloc, geometry="geometry").drop_duplicates()
    return crsloc
