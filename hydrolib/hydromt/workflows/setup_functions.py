# -*- coding: utf-8 -*-
import configparser
import json
import logging
import os
import pathlib
import random
import shutil
import subprocess
import zipfile

import contextily as ctx
import geopandas as gpd
import hydromt

# Plotting
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from delft3dfmpy import (
    OSM,
    DFlowFMModel,
    DFlowFMWriter,
    DFlowRRModel,
    DFlowRRWriter,
    HyDAMO,
    Rectangular,
)
from delft3dfmpy.core.geometry import sample_raster_values
from delft3dfmpy.core.logging import initialize_logger

# from delft3dfmpy.core.preprocess import *
# FIXME: use above-mentioned preprocess from hydromt workflows
from delft3dfmpy.datamodels.common import ExtendedGeoDataFrame
from delft3dfmpy.datamodels.datamodel import datamodel
from delft3dfmpy.io import dfmreader
from delft3dfmpy.io.UgridReader import UgridReader
from matplotlib.collections import LineCollection

# Geometries
from shapely.geometry import LineString, Point, Polygon

# TODO BMA: maybe all the geometry functions can be replaced by hydroMT functions


# Import csv


# TODO BMA: make class and use self (hydroMT related)

# ===============================================================
#                  Helper
# ===============================================================
# TODO BMA: integrate data checks such as below with HydroMT config parser


def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False


def isint(x):
    try:
        int(x)
        return True
    except:
        return False


def parse_arg(arg, dtype=None):
    if dtype is None:
        if arg in ["None", "none", "NONE", None, "", "NULL"]:
            arg_ = None
        elif arg in ["FALSE", "False", "false", False]:
            arg_ = False
        elif arg in ["TRUE", "True", "true", True]:
            arg_ = True
        elif isint(arg):
            arg_ = int(arg)
        elif isfloat(arg):
            arg_ = float(arg)
        elif isinstance(arg, pathlib.Path):
            if arg.is_file() == False:
                arg_ = None  # path
            else:
                arg_ = str(arg)  # filespath
        elif isinstance(arg, dict):
            arg_ = arg
        elif "{" in arg:
            arg_ = eval(arg)  # dict
        elif "," in arg:
            arg_ = arg.split(",")
            arg_ = [a.strip() for a in arg_]  # list
        elif arg.startswith("'") & arg.endswith("'"):
            arg_ = arg[1:-1]  # reserve string
        else:
            arg_ = str(arg)  # str

    else:
        if dtype in [bool, int, float, str]:
            arg_ = dtype(arg)
        else:
            raise Warning("Not supported dtype")

    return arg_


def parse_ini(ini_fn):
    config = configparser.ConfigParser(inline_comment_prefixes=[";", "#"])
    config.optionxform = str  # case sensitive parsing
    _ = config.read(ini_fn)
    config_ = {}
    for section in config.sections():
        config_[section] = {}
        for k, v in config[section].items():
            config_[section][k] = parse_arg(v)
    return config_


def read_gpd(
    gpdfile,
    id_col: str = None,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    clip_geom: gpd.GeoDataFrame = None,
    clip_predicate: str = "contains",
):
    """Function to read gpd.GeoDataFrame with preprocessing: rename, slice, convert type and set index"""

    # read & clip
    if isinstance(gpdfile, str) or isinstance(gpdfile, pathlib.Path):
        _data = hydromt.io.open_vector(
            fn=gpdfile,
            predicate=clip_predicate,
            geom=clip_geom,
            mode="r",
        )
        logging.debug(f"Reading file: {gpdfile}")
    elif isinstance(gpdfile, gpd.GeoDataFrame):
        _data = gpdfile
        logging.debug(f"Reading data: {gpdfile}")

    # get subset
    if required_columns is None and required_dtypes is None:
        data = _data.copy()

    else:
        if required_columns is not None and required_dtypes is None:
            try:
                logging.debug(f"Slicing data with required_columns: {required_columns}")
                data = gpd.GeoDataFrame(geometry=_data["geometry"])
                for c in required_columns:
                    data[c] = _data[c]
            except Exception as e:
                logging.error(e)

        elif required_columns is None and required_dtypes is not None:
            logging.error(
                "Required_dtypes must be used together with required_columns. No type conversion is performed."
            )

        elif len(required_dtypes) != len(required_columns):
            logging.error(
                "Required_dtypes must be of the same length with required_columns. No type conversion is performed."
            )

        else:
            try:
                logging.debug(
                    f"Ensuring data types with required_dtypes: {required_columns}"
                )
                data = gpd.GeoDataFrame(geometry=_data["geometry"])
                for c, dc in zip(required_columns, required_dtypes):
                    _data[c] = _data[c].replace(
                        {"NULL": None, "None": None, "none": None}
                    )
                    if dc == "bool":
                        _data[c] = _data[c].replace(
                            {
                                "True": True,
                                "True": True,
                                "true": True,
                                "FALSE": False,
                                "False": False,
                                "false": False,
                            }
                        )
                    data[c] = _data[c].astype(eval(dc))

            except Exception as e:
                logging.error(e)

    # rename
    if rename_map is not None:
        data.rename(inplace=True, columns=rename_map)
        logging.debug(f"Renaming data with map: {rename_map}")

    # assign index
    if id_col is not None:
        logging.debug(f"Indexing data with id_col: {id_col}")
        data.index = data[id_col]
        data.index.name = id_col

    return data


def read_raster(
    raster_fn: str, id_col: str = None, clip_geom: gpd.GeoDataFrame = None, nodata=-999
):

    da = hydromt.io.open_mfraster(str(raster_fn))
    da.attrs.update(_FillValue=nodata)

    if clip_geom is not None:
        da = da.raster.clip_geom(clip_geom)

    if id_col is not None:
        rename = {pathlib.Path(raster_fn).name.split(".")[0]: id_col}
        da = da.rename(rename)

    return da


# ===============================================================
#                  IO
# ===============================================================
# TODO BMA: change column names to meet 10 char length limits
# TODO BMA: limit precision
def write_shp(data: gpd.GeoDataFrame, filename: str, columns: list = None):
    if data is not None:
        # convert to numerical
        data = data.apply(pd.to_numeric, errors="ignore")
        # convert list to strings
        for c in data.columns:
            if isinstance(data[c][0], list):
                data[c] = data[c].apply(";".join)
        if columns is not None:
            if "geometry" not in columns:
                columns = columns + ["geometry"]
            gpd.GeoDataFrame(data[columns]).to_file(filename, index=False)
        else:
            gpd.GeoDataFrame(data).to_file(filename, index=False)


def export_to_fews(
    exportpath,
    pilotname,
    pilotcrs,
    dfmmodel,
    drrmodel,
    staticgeoms,
    guipath=None,
    logger=logging,
):
    """ "Function to prepare a model following FEWS conventions"""

    # prepare exportpath
    if not exportpath.exists():
        exportpath.mkdir()
    # write fm in fews format
    dfmmodel.mdu_parameters["Wrimap_flow_analysis"] = "0"
    fm_writer = DFlowFMWriter(dfmmodel, output_dir=exportpath, name=pilotname)
    fm_writer.write_all()
    # write rr in fews format
    rr_writer = DFlowRRWriter(drrmodel, output_dir=exportpath, name=None)
    rr_writer.d3b_parameters.update(
        {
            "OutputOptions": {"OutputRRUnpaved": 0, "OutputRRNWRW": 0},
            "Options": {
                "GenerateNetCdfOutput": -1,
                "GenerateHisOutput": 0,
                "MeteoNetCdfInput": -1,
                "PrecipitationNetCdfSeriesId": "precipitation",
                "EvaporationNetCdfSeriesId": "evaporation",
                "TemperatureNetCdfSeriesId": "precipitation",
                "ControlModule": -1,
                "RestartIn": 0,
                # FIXME BMA: need to change to -1 when starting to use restart
                "RestartOut": 0,
            },
        }
    )  # FIXME BMA: need to change to -1 when starting to use restart
    rr_writer.copyRRFiles()
    rr_writer.update_config()
    rr_writer.write_nwrw()
    rr_writer.write_topology()
    rr_writer.write_dimr(DFM_comp_name=pilotname)

    # prepare an input folder (empty)
    input_folder = exportpath.joinpath("input")
    if not input_folder.exists():
        input_folder.mkdir()

    # zip to moduledataset
    zname = exportpath.joinpath(exportpath.name + ".zip")
    zout = zipfile.ZipFile(
        zname, "w", zipfile.ZIP_DEFLATED
    )  # <--- this is the change you need to make
    for f in exportpath.glob("**/*"):
        fname = f.relative_to(exportpath)
        if fname.name != zname.name:
            zout.write(f, fname)
    zout.close()
    logger.info(f"ModuleDataset prepared as {exportpath.name}.")

    # prepare locationsets

    # rainfall catchments
    # TODO: perform dissolve based on METEO_ID is hard coded
    try:
        rainfall_catchments = staticgeoms["precipitation_region"]
        rainfall_catchments = rainfall_catchments.dissolve(level=0)
        rainfall_catchments = rainfall_catchments.assign(MODEL_ID=pilotname)
        write_shp(rainfall_catchments, exportpath.joinpath("catchments.shp"))
        logger.info("LocationSet for rainfall catchment prepared.")
    except Exception as e:
        logger.error(f"Could not prepare LocationSet for rainfall catchment: {e}")

    # boundary
    # FIXME: boundary should exact from dfm model with node Id and xy
    try:
        boundaries = staticgeoms["boundaries"]
        boundaries = boundaries.assign(MODEL_ID=pilotname)
        write_shp(boundaries, exportpath.joinpath("boundaries.shp"))
        logger.info("LocationSet for boundaries prepared.")
    except Exception as e:
        logger.error(f"Could not prepare LocationSet for boundaries: {e}")

    # mesh1d as nodes
    fews_csv = pd.DataFrame(
        [
            dfmmodel.network.mesh1d.description1d["mesh1d_node_ids"],
            dfmmodel.network.mesh1d.get_values("nodex"),
            dfmmodel.network.mesh1d.get_values("nodey"),
        ]
    ).T
    fews_csv.columns = ["ID", "X", "Y"]
    # fews_csv.to_csv(exportpath.joinpath(pilotname + '_mesh1d.csv'), index=False)

    # fews_xml = exportpath.joinpath(pilotname + '_mesh1d.xml')
    # with open(fews_xml, 'w') as f:
    #     f.write('\t<irregular locationId="' + pilotname + '">\n')
    #     f.write('\t\t<geoDatum>' + pilotcrs + '</geoDatum>\n')
    #     for p in fews_csv.itertuples():
    #         f.write('\t<cellCentre>\n')
    #         f.write('\t\t<x>' + str(p.X) + '</x>\n')
    #         f.write('\t\t<y>' + str(p.Y) + '</y>\n')
    #         f.write('\t</cellCentre>\n')
    #     f.write('\t</irregular>\n\n')
    #     f.write('\t</irregular>\n\n')

    fews_gdf = gpd.GeoDataFrame(
        fews_csv, geometry=gpd.points_from_xy(fews_csv.X, fews_csv.Y)
    )
    fews_gdf.to_file(exportpath.joinpath(pilotname + "-mesh1d.shp"), index=False)
    logger.info("LocationSet for mesh1d prepared.")

    # mesh2d run dimr to generate clean mesh2d
    if guipath is not None:
        dimrpath = guipath.joinpath(
            r"plugins\DeltaShell.Dimr\kernels\x64\dimr\scripts\run_dimr.bat"
        )
        run_dimr(dimrpath, cwd=exportpath, timeout=60, logger=logger)
        # test if flow geom is generated
        mesh2d_file = exportpath.joinpath("dflowfm/FEWS_netgeom.nc")
        if mesh2d_file.is_file():
            shutil.move(mesh2d_file, exportpath.joinpath(pilotname + "-mesh2d.nc"))
    logger.info("Grids for mesh2d prepared.")

    # delete files that are not useful
    shutil.rmtree(exportpath.joinpath("dflowfm"))
    shutil.rmtree(exportpath.joinpath("rr"))
    shutil.rmtree(exportpath.joinpath("input"))
    os.remove(exportpath.joinpath("rr2flow.nc"))
    os.remove(exportpath.joinpath("dimr_config.xml"))


def run_dimr(dimrpath, cwd, timeout=300, logger=logging):
    """run dimr for the fews model to generate mesh2d needed (will be killed after timeout)"""
    logger.info(f"Running dimr in {cwd}, timeout = {timeout}")
    proc = subprocess.Popen(
        dimrpath,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        universal_newlines=True,
    )
    try:
        outs, errs = proc.communicate(timeout=timeout)
    except:
        pass


# ===============================================================
#                  General
# ===============================================================


def setup_datamodel(
    data: gpd.GeoDataFrame, datamodel_type: str, datamodel_map: dict = None
) -> ExtendedGeoDataFrame:
    """funciton to setup datamodel with renamed columns from data"""

    if datamodel_type == "branches":
        datamodel_map = {
            "id": "BRANCH_ID",
            "name": "BRANCH_ID",
            "branchType": {
                'branchType == "Channel"': 0,
                'branchType == "SewerConnection"': 1,
                'branchType == "Pipe"': 2,
            },
            "material": {
                'MATERIAL == "concrete"': 1,
                'MATERIAL == "clay"': 0,
                'MATERIAL == "CastIron"': 2,
                'MATERIAL == "StoneWare"': 3,
                'MATERIAL == "Hdpe"': 4,
                'MATERIAL == "Masonry"': 5,
                'MATERIAL == "SheetMetal"': 6,
                'MATERIAL == "Polyester"': 7,
                'MATERIAL == "Polyvinylchlorid"': 8,
                'MATERIAL == "Steel"': 9,
            },
            "sourceCompartmentName": "MANHOLE_UP",
            "targetCompartmentName": "MANHOLE_DN",
        }
    elif datamodel_type == "manholes":
        datamodel_map = {
            "Id": "MANHOLE_ID",
            "Name": "MANHOLE_ID",
            "NodeId": "MANHOLE_ID",
            "ManholeId": "MANHOLE_ID",
            "UseTable": "usetable",
            "BedLevel": "bedlev",
            "Area": "area",
            "StreetLevel": "streetlev",
            "StorageType": "storetype",
            "StreetStorageArea": "streetarea",
            "useStreetStorage": "usestreetstorage",
        }
    elif datamodel_type == "bridges":
        datamodel_map = {
            "id": "STRUC_ID",
            "branchid": "branch_id",
            "chainage": "branch_offset",
            "length": "LENGTH",
            "shift": "BEDLEV",
            "crosssection": "definitionId",
            "inletlosscoeff": "IN_LOSS",
            "outletlosscoeff": "OUT_LOSS",
            "frictionid": "frictionId",
        }
    elif datamodel_type == "gates":
        datamodel_map = {
            "id": "STRUC_ID",
            "branchId": "branch_id",
            "chainage": "branch_offset",
            "type": "type",
            "allowedFlowdir": "FLOW_DIR",
            "upstream1Width": "upstream1Width",
            "upstream1Level": "upstream1Level",
            "upstream2Width": "upstream2Width",
            "upstream2Level": "upstream2Level",
            "crestWidth": "CREST_W",
            "crestLevel": "CREST_LEV",
            "crestLength": "crestLength",
            "downstream1Width": "downstream1Width",
            "downstream1Level": "downstream1Level",
            "downstream2Width": "downstream2Width",
            "downstream2Level": "downstream2Level",
            "gateLowerEdgeLevel": "GATE_LEV",
            "gateHeight": "GATE_H",
            "gateOpeningWidth": "OPEN_W",
            "useVelocityHeight": "useVelocityHeight",
        }
    elif datamodel_type == "pumps":
        datamodel_map = {
            "id": "STRUC_ID",
            "branchId": "branch_id",
            "chainage": "branch_offset",
            "type": "STRUC_TYPE",
            "orientation": "PUMP_DIR",
            "controlSide": "CTRL_SIDE",
            "numStages": "NUM_STAGES",
            "capacity": "CAPACITY",
            "startLevelSuctionSide": "START_L_SS",
            "stopLevelSuctionSide": "STOP_L_SS",
            "startLevelDeliverySide": "START_L_DS",
            "stopLevelDeliverySide": "STOP_L_DS",
        }
    elif datamodel_type == "culverts":
        datamodel_map = {
            "id": "STRUC_ID",
            "branchId": "branch_id",
            "chainage": "branch_offset",
            "type": "STRUC_TYPE",
            "allowedFlowDir": "FLOW_DIR",
            "leftLevel": "INVLEV_UP",
            "rightLevel": "INVLEV_DN",
            "csDefId": "definitionId",
            "length": "LENGTH",
            "inletLossCoeff": "IN_LOSS",
            "outletLossCoeff": "OUT_LOSS",
            "valveOnOff": "VALVE_ON",
            "numLossCoeff": "numLossCoeff",
            "bedFrictionType": "ROUGH_TYPE",
            "bedFriction": "ROUGH_VAL",
        }
    elif datamodel_type == "compounds":
        datamodel_map = {
            "id": "STRUC_ID",
            "branchId": "branchId",
            "chainage": "chainage",
            "type": "type",
            "numStructures": "numStructures",
            "structureIds": "structureIds",
        }
    elif datamodel_type == "subcatchments":
        datamodel_map = {
            "ManholeId": "MANHOLE_ID",
            "cl_slope": "cl_slope",
            "cl_flat": "cl_flat",
            "cl_stretch": "cl_stretch",
            "op_slope": "op_slope",
            "op_flat": "op_flat",
            "op_stretch": "op_stretch",
            "rf_slope": "rf_slope",
            "rf_flat": "rf_flat",
            "rf_stretch": "rf_stretch",
            "up_slope": "up_slope",
            "up_flat": "up_flat",
            "up_stretch": "up_stretch",
            "inhabitant": "inhabitant",
            "dwf_def": "dwf_def",
            "meteo_id": "meteo_id",
            "px": "px",
            "py": "py",
        }
    else:
        datamodel_map = None
    if (datamodel_map is not None) & (data is not None):
        _data = data.copy()
        if isinstance(data, gpd.GeoDataFrame):
            datamodel = gpd.GeoDataFrame(data["geometry"])
        else:
            datamodel = pd.DataFrame()
        for model_col, data_col in datamodel_map.items():
            if isinstance(data_col, str):  # rename
                datamodel.loc[:, model_col] = _data.loc[:, data_col]
            elif isinstance(data_col, dict):  # map
                for criteria, val in data_col.items():
                    datamodel.loc[_data.eval(criteria), model_col] = val

        if isinstance(data, gpd.GeoDataFrame):
            datamodel_ = ExtendedGeoDataFrame(geotype=datamodel.geometry[0].type)
            datamodel_.set_data(datamodel, check_geotype=False)
        else:
            datamodel_ = data
    else:
        datamodel_ = data
    return datamodel_


def setup_pilot(
    pilot_fn: str = None,
    pilot_name: str = "pilot_name",
    pilot_crs: str = None,
    logger=logging,
):
    """setup pilot using pilot_fn (used for 1d)
    pilot is used to clip model features and rename the model"""

    pilot = None
    if pilot_fn is not None:
        pilot = read_gpd(pilot_fn)

        if len(pilot) > 1:
            pilot_geom = pilot.unary_union
            pilot = gpd.GeoDataFrame(geometry=gpd.GeoSeries(pilot_geom))
            logger.warning(f"Perform union for pilot: {pilot_fn}. ")

        if pilot_name is not None:
            pilot.index = [pilot_name]
            logger.info(f"Pilot is setup for {pilot_name}.")

        if pilot_crs is not None:

            if pilot_crs != pilot.crs:
                pilot.crs = pilot_crs
                logger.warning(f"Pilot crs is overwritten as: {pilot_crs}")

    return pilot


# ===============================================================
#                  BRANCHES
# ===============================================================

# FIXME BMA: use pipe_query and channel_query in ini to replace hard coded queries
# def setup_branches
def setup_branches(
    branches: gpd.GeoDataFrame = None,
    branches_fn: str = None,
    branches_ini_fn: str = None,
    snap_offset: float = 0.0,
    id_col: str = None,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    branch_query: str = None,
    pipe_query: str = None,
    channel_query: str = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
    **kwargs,
):
    """setup branches using branches_fn"""

    if branches_ini_fn is not None:
        branches_ini = parse_ini(branches_ini_fn)

    if branches is None and branches_fn is not None:

        branches = read_gpd(
            branches_fn,
            id_col,
            rename_map,
            required_columns,
            required_dtypes,
            clip_geom=pilot.buffer(10),
            clip_predicate="intersects",
        )  # use clip_predicate="intersects" for reduced domain - more branches with boundary nodes
        branches = append_data_columns_based_on_ini_query(branches, branches_ini)
        branches_crs = branches.crs

        if branch_query is not None:
            branches = branches.query(branch_query)
            logging.debug(f"query branches for {branch_query}")

        min_length = branches.geometry.length.min()
        logger.info(f"Reading branches from {branches_fn} for given pilot")

    if branches is not None:
        # preprocess
        branches = _preprocess_branches(
            branches,
            branches_ini,
            snap_offset=snap_offset,
            id_col=id_col,
            logger=logger,
        )
        branches = _space_branches(branches)
        branch_nodes = _generate_nodes(branches)

        # setup channels (if needed)
        branches.loc[branches.query(channel_query).index, "branchType"] = "Channel"
        # setup pipes (if needed)
        branches.loc[branches.query(pipe_query).index, "branchType"] = "Pipe"

        # assign crs
        branches.crs = branches_crs
        logger.info(str(len(branches)) + " branches are set up.")

    # validate pipe geometry
    if sum(branches.geometry.length <= 0) == 0:
        pass
    else:
        logger.error(
            f"Branches {branches.index[branches.geometry.length <= 0]} have length of 0 meter. "
            + f"Issue might have been caused by using a snap_offset that is too large. Please revise or modify the branches data layer. "
        )

    return branches, branch_nodes


def _preprocess_branches(
    branches: gpd.GeoDataFrame,
    branches_ini: configparser,
    snap_offset: float = 0.01,
    id_col: str = "BRANCH_ID",
    logger=logging,
):

    """Function to (geo-) preprocess branches"""

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

    return branches


def _space_branches(branches: gpd.GeoDataFrame, spacing_col="spacing", logger=logging):
    """space branches based on spacing_col on branch"""

    # split branches based on spacing
    branches_ = split_branches(branches, spacing_col=spacing_col)

    # remove spacing column
    branches_ = branches_.drop(columns=[spacing_col])

    return branches_


def _generate_nodes(branches: gpd.GeoDataFrame, logger=logging):
    """generate branch nodes at branch ends"""
    # generate node up and downstream
    nodes = pd.DataFrame(
        [Point(l.coords[0]) for li, l in branches["geometry"].iteritems()]
        + [Point(l.coords[-1]) for li, l in branches["geometry"].iteritems()],
        columns=["geometry"],
    )

    # remove duplicated geometry
    _nodes = nodes.copy()
    G = _nodes["geometry"].apply(lambda geom: geom.wkb)
    n = len(G) - len(G.drop_duplicates().index)
    nodes = _nodes[_nodes.index.isin(G.drop_duplicates().index)]
    nodes = gpd.GeoDataFrame(nodes)
    nodes.crs = branches.crs
    return nodes


# FIXME BMA: below is not used
def setup_channels(branches: gpd.GeoDataFrame = None):
    """set up channels
    channel should all connect - break lines at intersect point and reconnect
    """
    _channels = branches.query('branchType == "Channel"')

    non_channels = branches.query('branchType != "Channel"')

    # snap channel ends to channels
    start_points = gpd.GeoDataFrame(
        [
            {_channels.index.name: idx, "geometry": Point(line.coords[0])}
            for idx, line in _channels.geometry.iteritems()
        ],
        crs=_channels.crs,
    )
    start_points_ = snap_nodes_to_lines(_channels, start_points, offset, logger)
    end_points = gpd.GeoDataFrame(
        [
            {_channels.index.name: idx, "geometry": Point(line.coords[-1])}
            for idx, line in _channels.geometry.iteritems()
        ],
        crs=_channels.crs,
    )
    end_points_ = snap_nodes_to_lines(_channels, end_points, offset, logger)
    fig, ax = plt.subplots()
    branches.plot(ax=ax)
    end_points.plot(ax=ax, color="g")
    end_points_.plot(ax=ax, color="r")
    plt.show()

    # extend channel ends by replacing LineString Geometry first and last points
    for channel_id in _channels.index:
        channel_line_coords = list(channel_line.loc[channel_id, "geometry"].coords)
        channel_line_coords[0] = (
            start_points_.loc[channel_id, "geometry"].x,
            start_points_.loc[channel_id, "geometry"].y,
        )
        channel_line_coords[-1] = (
            end_points_.loc[channel_id, "geometry"].x,
            end_points_.loc[channel_id, "geometry"].y,
        )
        channel_line_ = LineString(channel_line_coords)
        _channels.at[channel_id, "geometry"] = channel_line_

    # break channels at intersections --> get new geometry

    # left join channels --> get have the same fields as branch columns

    # concat everything
    branches = pd.concat([channels, non_channels])
    return branches


def setup_pipes(branches: gpd.GeoDataFrame = None):
    """set up pipes
    not implemented"""
    return branches


# ===============================================================
#                  MANHOLES
# ===============================================================


def setup_manholes(
    branches: gpd.GeoDataFrame = None,
    branch_nodes: gpd.GeoDataFrame = None,
    manholes_fn: str = None,
    manholes_ini_fn: str = None,
    generate_manholes: bool = False,
    snap_offset: float = 0.0,
    id_col: str = "MANHOLE_ID",
    id_prefix: str = "",
    id_suffix: str = "",
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    branch_query: str = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """setup manholes using manholes_ini_fn and manholes_fn"""
    if branches is None:
        logger.info(
            f"No branch data is available meaning that no manholes will be created."
        )
        manholes_ = None
    else:
        branches_ = read_gpd(branches)

        if branch_query is not None:
            branches_ = branches_.query(branch_query)
            logging.debug(f"query branches: {branch_query}")

        if manholes_ini_fn is None and manholes_fn is None:
            logger.info(
                f"Manholes are not created. Both manholes_fn and manholes_ini_fn are required to create manholes."
            )
            manholes_ = None
        elif manholes_ini_fn is None and manholes_fn is not None:
            # use user manholes only
            logger.info(f"Manhole are read from {manholes_fn}.")
            manholes_ = read_gpd(
                manholes_fn, id_col, rename_map, required_columns, required_dtypes
            )
        elif manholes_ini_fn is not None and manholes_fn is None:
            # use ini manholes only
            manholes_ini = parse_ini(manholes_ini_fn)
            if not generate_manholes:
                logger.info(f"Manholes are not created.")
                manholes_ = None
            else:
                logger.info(f"Manhole are generated based on ini .")
                manholes_, branches_ = _generate_default_manholes_on_branches(
                    branches, manholes_ini, id_col, id_prefix, id_suffix, logger
                )
        else:
            manholes_ini = parse_ini(manholes_ini_fn)
            if not generate_manholes:
                # use user manholes only
                logger.info(f"Manhole are read from {manholes_fn}.")
                manholes_ = read_gpd(
                    manholes_fn, id_col, rename_map, required_columns, required_dtypes
                )
            else:
                # use both user and ini manholes
                logger.info(f"Manhole are read from {manholes_fn}.")
                manholes_user = read_gpd(
                    manholes_fn, id_col, rename_map, required_columns, required_dtypes
                )
                # FIXME BMA: manholes user will not have the query information specified in manholes_ini
                manholes_user = __fill_manholes(manholes_user, manholes_ini)
                # generate manholes with default values specifed in ini
                logger.info(f"Manhole are generated based on ini .")
                manholes_, branches_ = _generate_default_manholes_on_branches(
                    branches, manholes_ini, id_col, id_prefix, id_suffix, logger
                )
                # overwrite manholes_generated using manholes_user
                logger.info(f"Manholes read and generated are merged.")
                manholes_ = _merge_user_manholes_to_generated_manholes(
                    manholes_, manholes_user, snap_offset, logger
                )
                branches_ = __update_pipes_from_manholes(manholes_, branches_)

    manholes_.index.name = id_col

    manholes_.crs = branches.crs
    branches_.crs = branches.crs

    logger.info(str(len(branches_)) + " branches are set up.")
    logger.info(str(len(manholes_)) + " manholes are set up.")

    # update branch_nodes
    branch_nodes_ = filter_geometry_and_assign(branch_nodes, manholes_, is_manhole=True)

    return manholes_, branches_, branch_nodes_


def _generate_default_manholes_on_branches(
    branches: gpd.GeoDataFrame,
    manholes_ini: configparser.ConfigParser,
    id_col: str = "MANHOLE_ID",
    id_prefix: str = "",
    id_suffix: str = "",
    logger=logging,
):
    """generate default manholes with the id_col as index, based on spacing specified manholes_ini"""
    # generate
    logger.info("Generating manhole on pipes based on spacing")
    manholes_generated, branches_generated = __generate_manholes(
        branches, manholes_ini, id_col, id_prefix, id_suffix
    )
    # fill
    logger.info(
        "Filling generated manhole based on defaults specified in manholes_ini_fn"
    )
    fill_method = "default"
    if manholes_ini["global"]["fill_method"] not in [None, "", "None"]:
        fill_method = manholes_ini["global"]["fill_method"]
    manholes_filled = __fill_manholes(manholes_generated, manholes_ini, fill_method)

    # apply additional shifts for bedlevel and streetlevel
    if manholes_ini["global"]["bedlevel_source"] not in [None, "default"]:
        logger.info(
            "bedlevel interpolated from raster source: "
            + manholes_ini["global"]["bedlevel_source"]
        )
    if manholes_ini["global"]["bedlevel_shift"] != 0.0:
        logger.info(
            "bedlevel shifted for: "
            + str(manholes_ini["global"]["bedlevel_shift"])
            + "m"
        )
        manholes_filled["bedlev"] = (
            manholes_filled["bedlev"] + manholes_ini["global"]["bedlevel_shift"]
        )
    if manholes_ini["global"]["streetlevel_source"] not in [None, "default"]:
        logger.info(
            "streetlevel interpolated from raster source: "
            + manholes_ini["global"]["streetlevel_source"]
        )
        temp = sample_raster_values(
            points=manholes_filled,
            rasterpath=manholes_ini["global"]["streetlevel_source"],
        )
        manholes_filled["streetlev"] = temp["raster_value"]
    if manholes_ini["global"]["streetlevel_shift"] != 0.0:
        logger.info(
            "streetlevel shifted for: "
            + str(manholes_ini["global"]["streetlevel_shift"])
            + "m"
        )
        manholes_filled["streetlev"] = (
            manholes_filled["streetlev"] + manholes_ini["global"]["streetlevel_shift"]
        )
    return manholes_filled, branches_generated


def __generate_manholes(
    branches: gpd.GeoDataFrame,
    manholes_ini: configparser.ConfigParser,
    id_col: str = "MANHOLE_ID",
    id_prefix="MANHOLE_",
    id_suffix="",
):
    """generate manhole locations !Do not split anymore"""

    # prepare branches
    _pipes = branches.query('branchType == "Pipe"')
    _channels = branches.query('branchType == "Channel"')

    # split branches to generate manhole
    pipes = _pipes
    channels = _channels

    # generate nodes on pipes
    # TODO: add INVLEV_UP and INVLEV_DN to manhole_settings.ini file
    # generate nodes upstream and downstream for every pipe
    _nodes_pipes_up = pd.DataFrame(
        [
            (Point(l.geometry.coords[0]), l.INVLEV_UP, li)
            for li, l in pipes[["geometry", "INVLEV_UP"]].iterrows()
        ],
        columns=["geometry", "bedlev", pipes.index.name],
    )
    _nodes_pipes_dn = pd.DataFrame(
        [
            (Point(l.geometry.coords[-1]), l.INVLEV_DN, li)
            for li, l in pipes[["geometry", "INVLEV_DN"]].iterrows()
        ],
        columns=["geometry", "bedlev", pipes.index.name],
    )
    _nodes_pipes = pd.concat([_nodes_pipes_up, _nodes_pipes_dn])
    nodes_pipes = (
        _nodes_pipes.set_index(pipes.index.name)
        .merge(
            pd.DataFrame(
                pipes,
            ).drop(columns=[pipes.index.name, "geometry"]),
            on=pipes.index.name,
        )
        .reset_index()
    )

    # nodes on pipe are manholes
    nodes_pipes = append_data_columns_based_on_ini_query(nodes_pipes, manholes_ini)

    # get connecting pipes characteristics based on statistical functions
    # TODO BMA: add this to function name for control in manhole_ini file
    nodes_pipes["where"] = nodes_pipes["geometry"].apply(lambda geom: geom.wkb)
    nodes_pipes = __get_pipe_stats_for_manholes(nodes_pipes, "where", "DIAMETER", "max")
    nodes_pipes = __get_pipe_stats_for_manholes(nodes_pipes, "where", "bedlev", "min")
    nodes_pipes = __get_pipe_stats_for_manholes(
        nodes_pipes, "where", "ORIG_BRANCH_ID", ";".join
    )

    # drop duplicated nodes
    nodes_pipes = nodes_pipes.loc[nodes_pipes["where"].drop_duplicates().index, :]
    nodes_pipes.drop(columns="where", inplace=True)

    # remove pipes nodes on channels
    # generate nodes on channels
    _nodes_channels = pd.DataFrame(
        [(Point(l.coords[0]), li) for li, l in channels["geometry"].iteritems()]
        + [(Point(l.coords[-1]), li) for li, l in channels["geometry"].iteritems()],
        columns=["geometry", channels.index.name],
    )
    nodes_channels = _nodes_channels.set_index(channels.index.name).merge(
        pd.DataFrame(channels).drop(columns=[channels.index.name, "geometry"]),
        on=channels.index.name,
    )
    # snap channel nodes to pipe nodes
    nodes_pipes = snap_nodes_to_nodes(nodes_channels, nodes_pipes, 0.1)

    # if snapped, meaning there should be a channel nodes, therefore remove pipe nodes
    mask = ~nodes_pipes.index.isin(nodes_channels.index)
    nodes_pipes = nodes_pipes.loc[mask].reset_index(drop=True)

    # manhole generated
    manholes_generated = gpd.GeoDataFrame(nodes_pipes)
    manholes_generated.loc[:, id_col] = [
        "%s_%s_%s" % (id_prefix, str(x), id_suffix)
        for x in range(len(manholes_generated))
    ]
    manholes_generated.index = [
        "%s_%s_%s" % (id_prefix, str(x), id_suffix)
        for x in range(len(manholes_generated))
    ]

    # update manholes generated to pipes
    pipes = __update_pipes_from_manholes(manholes_generated, pipes)

    # merge pipe and channels
    branches_generated = pd.concat([pipes, channels], join="inner")
    branches_generated.index = branches_generated[branches.index.name]

    return manholes_generated, branches_generated


def __update_pipes_from_manholes(manholes: gpd.GeoDataFrame, pipes: gpd.GeoDataFrame):
    manholes_dict = {(m.geometry.x, m.geometry.y): mi for mi, m in manholes.iterrows()}
    for pi, p in pipes.iterrows():
        cs = p.geometry.coords
        try:
            pipes.at[pi, "MANHOLE_UP"] = manholes_dict[
                cs[0]
            ]  # FIXME BMA: generalize or use fixed datamodel name in HydroMT
        except:  # no manholes at outlets
            pipes.at[pi, "MANHOLE_UP"] = ""
        try:
            pipes.at[pi, "MANHOLE_DN"] = manholes_dict[
                cs[-1]
            ]  # FIXME BMA: generalize or use fixed datamodel name in HydroMT
        except:  # no manholes at outlets
            pipes.at[pi, "MANHOLE_DN"] = ""

    return pipes


def __get_pipe_stats_for_manholes(
    manholes: gpd.GeoDataFrame, pipes_col: str, stats_col: str, method: str
):
    """get the stats from all pipes connecting a single manholes

    parameters
    --------------------
    pipes_col: used to identify pipes connected to the manhole (multiple rows of pipes for a single manhole), e.g. BRANCH_ID.
    stats_col: the column used to obtain the stats, e.g. DIAMETER
    method: method used to obtain the stats: e.g. max
    """
    manholes.loc[:, stats_col] = manholes.groupby(pipes_col)[stats_col].transform(
        method
    )
    return manholes


def __fill_manholes(
    manholes: gpd.GeoDataFrame,
    manholes_ini: configparser.ConfigParser,
    fill_method: str = "default",
):
    """fill manholes attributes using fill method"""
    if fill_method == "default":
        manholes_filled = append_data_columns_based_on_ini_query(manholes, manholes_ini)
    else:
        raise NotImplementedError(
            "Fill method is not recognized Please use only default"
        )  # TODO BMA: add other fill method, e.g. landuse (In HydroMT)
    return manholes_filled


# TODO BMA: __fill_manholes_with_DTM


def _merge_user_manholes_to_generated_manholes(
    manholes_generated: gpd.GeoDataFrame,
    manholes_user: gpd.GeoDataFrame,
    snap_offset: float = 0.0,
    logger=logging,
):
    """
    snap user manholes to generated manholes and overwrite generated manholes fields
    """
    manholes_merged = merge_nodes_with_nodes_prior_by_snapping(
        manholes_user, manholes_generated
    )
    return manholes_merged


# ===============================================================
#                   roughness
# ===============================================================
def setup_roughness(
    branches: gpd.GeoDataFrame,
    generate_roughness_from_branches: bool = True,
    roughness_ini_fn: str = None,
    logger=logging,
):
    """
    setup 1D roughness using frictionId following the convention used in delft3dfmpy
    """
    logger.info("Setup 1d roughness")
    branches_ = read_gpd(branches)
    if generate_roughness_from_branches is True and roughness_ini_fn is not None:
        roughness_ini = parse_ini(roughness_ini_fn)
        branches_ = _setup_rougnness_from_default(branches_, roughness_ini, logger)

    return branches_


def _setup_rougnness_from_default(
    branches: gpd.GeoDataFrame, roughness_ini: dict, logger=logging
):
    """setup rougness based on defaults specified in ini (query branch columns)"""

    logger.info(f"1D roughness are generated from ini.")
    branches = append_data_columns_based_on_ini_query(branches, roughness_ini)
    fricid_col = "frictionId"
    for bi, b in branches.iterrows():
        branches.loc[bi, fricid_col] = "%s_%s" % (b["rough_type"], b["rough_val"])
    return branches


# ===============================================================
#                   CROSS-SECTIONS
# ===============================================================


def setup_crosssections(
    branches: gpd.GeoDataFrame,
    crosssections_ini_fn: str = None,
    crosssections_fn: str = None,
    generate_crosssections_from_branches: str = True,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "intersect",
    snap_offset: float = 1,
    pilot: gpd.GeoDataFrame = None,
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
        crosssection_ini = parse_ini(crosssections_ini_fn)

        # setup funcs
        crosssection_type = crosssection_ini["global"]["crosssection_type"]
        use_infile_order = crosssection_ini["global"]["use_infile_order"]
        drop_snap_failed = crosssection_ini["global"]["drop_snap_failed"]

        # setup column names
        shape_col = crosssection_ini["global"]["shape_col"]
        diameter_col = crosssection_ini["global"]["diameter_col"]
        height_col = crosssection_ini["global"]["height_col"]
        width_col = crosssection_ini["global"]["width_col"]
        t_width_col = crosssection_ini["global"]["t_width_col"]
        closed_col = crosssection_ini["global"]["closed_col"]

        # setup helper
        none2str = lambda x: "" if x is None else x
        upstream_prefix = none2str(crosssection_ini["global"]["upstream_prefix"])
        upstream_suffix = none2str(crosssection_ini["global"]["upstream_suffix"])
        downstream_prefix = none2str(crosssection_ini["global"]["downstream_prefix"])
        downstream_suffix = none2str(crosssection_ini["global"]["downstream_suffix"])

    # use branches
    if generate_crosssections_from_branches is True and branches is not None:
        branches_ = read_gpd(branches)

        # prepare default cross-sections
        branches_ = append_data_columns_based_on_ini_query(branches_, crosssection_ini)

        # prepare one shift columns from multiple branch types
        shift_col = "shift"
        upstream_shift_col = "start" + shift_col
        downstream_shift_col = "end" + shift_col
        branches_[upstream_shift_col] = None
        branches_[downstream_shift_col] = None
        for c in crosssection_ini["global"]["shift_col"]:
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
            rename_map,
            required_columns,
            required_dtypes,
            clip_geom=pilot.buffer(100),
        )
        logger.info("Crosssections are added from: " + crosssections_fn)

        if branch_query is not None:
            branches_ = branches_.query(branch_query)

        # get length from xy
        # FIXME: now the xy are read from the columns instead of the geometry. It is better to read from geometry otherwise might be confusing.
        if use_infile_order is True:
            crosssections = crosssections.groupby(level=0).apply(
                geometry.xyzp2xyzl, (["ORDER"])
            )
        else:
            crosssections = crosssections.groupby(level=0).apply(geometry.xyzp2xyzl)

        # compare yzs
        # diff = gpd.overlay(crosssections1, crosssections2, how='difference')
        # if len(diff) > 0:
        #     logging.warning(f'xyz crosssection with id {diff.index.to_list()} does not match user specified order. Please check cross section data layer.')

        # snap to branch
        # setup branch_id - snap bridges to branch (inplace of bridges, will add branch_id and branch_offset columns)
        logger.info(
            "Crosssections are snapped to nearest branch using "
            + str(snap_method)
            + "method."
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
                logging.warning(
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


# ===============================================================
#                  STRUCTURES
# ===============================================================


def setup_bridges(
    branches: gpd.GeoDataFrame,
    crsdefs: gpd.GeoDataFrame = None,
    crslocs: gpd.GeoDataFrame = None,
    roughness_ini_fn: str = None,
    bridges_ini_fn: str = None,
    bridges_fn: str = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "overall",
    snap_offset: float = 1,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """
    setup bridges
    """
    logger.info("Setting up bridges.")
    bridges = None

    # prepare structures ini
    if bridges_ini_fn is not None:
        bridges_ini = parse_ini(bridges_ini_fn)

    # setup structure defaults
    if branches is not None:
        bridges = _setup_structures_defaults(
            branches,
            bridges_ini,
            bridges_fn,
            "bridge",
            id_col,
            branch_query,
            snap_method,
            snap_offset,
            rename_map,
            required_columns,
            required_dtypes,
            pilot=pilot,
            logger=logger,
        )
    # prepare roughness ini
    if roughness_ini_fn is not None:
        roughness_ini = parse_ini(roughness_ini_fn)

    # setup structure crsdefs
    if not any(x is None for x in [branches, bridges, crsdefs]):
        bridges, crsdefs = _setup_structures_crsdefs(
            branches=branches,
            structures=bridges,
            crsdefs=crsdefs,
            roughness_ini=roughness_ini,
            crs_type="bridge",
            structures_ini=bridges_ini,
            logger=logger,
        )

    logger.info(str(len(bridges)) + " bridges are set up.")
    return bridges, crsdefs


def setup_gates(
    branches: gpd.GeoDataFrame,
    crsdefs: gpd.GeoDataFrame = None,
    crslocs: gpd.GeoDataFrame = None,
    roughness_ini_fn: str = None,
    gates_ini_fn: str = None,
    gates_fn: str = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "overall",
    snap_offset: float = 1,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """
    setup gates

    """
    logger.info("Setting up gates.")
    gates = None

    # prepare structures ini
    if gates_ini_fn is not None:
        gates_ini = parse_ini(gates_ini_fn)

    # setup structure defaults
    if branches is not None:
        gates = _setup_structures_defaults(
            branches,
            gates_ini,
            gates_fn,
            "gate",
            id_col,
            branch_query,
            snap_method,
            snap_offset,
            rename_map,
            required_columns,
            required_dtypes,
            pilot=pilot,
            logger=logger,
        )

    # setup velocity height
    if gates_ini["global"]["disable_velocityheight_on_pipes"]:
        gates = _disable_velocityheight_on_pipes(gates, logger)

    # setup crosssections
    if not any(x is None for x in [branches, gates, crsdefs, crslocs]) and (
        gates_ini["global"]["interpolate_crosssections"] == True
    ):

        gates_crsdefs, gates_crslocs = _interpolate_branch_crosssections_at_gates(
            branches, gates, crsdefs, crslocs, logger
        )

        # only applied when useVelocityHeight = True
        for gi, g in gates[gates.useVelocityHeight == "true"].iterrows():
            gates.at[gi, "upstream1Width"] = gates_crsdefs.loc[gi, "width"]
            gates.at[gi, "upstream2Width"] = gates_crsdefs.loc[gi, "width"]
            gates.at[gi, "downstream1Width"] = gates_crsdefs.loc[gi, "width"]
            gates.at[gi, "downstream2Width"] = gates_crsdefs.loc[gi, "width"]
            gates.at[gi, "upstream1Level"] = gates_crslocs.loc[gi, "shift"]
            gates.at[gi, "upstream2Level"] = gates_crslocs.loc[gi, "shift"]
            gates.at[gi, "downstream1Level"] = gates_crslocs.loc[gi, "shift"]
            gates.at[gi, "downstream2Level"] = gates_crslocs.loc[gi, "shift"]

    logger.info(str(len(gates)) + " gates are set up.")
    return gates


def setup_pumps(
    branches: gpd.GeoDataFrame,
    crsdefs: gpd.GeoDataFrame = None,
    crslocs: gpd.GeoDataFrame = None,
    roughness_ini_fn: str = None,
    pumps_ini_fn: str = None,
    pumps_fn: str = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "overall",
    snap_offset: float = 1,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """
    setup pumps
    """
    logger.info("Setting up pumps.")
    pumps = None

    # prepare structures ini
    if pumps_ini_fn is not None:
        pumps_ini = parse_ini(pumps_ini_fn)

    # setup structure defaults
    if branches is not None:
        pumps = _setup_structures_defaults(
            branches,
            pumps_ini,
            pumps_fn,
            "pump",
            id_col,
            branch_query,
            snap_method,
            snap_offset,
            rename_map,
            required_columns,
            required_dtypes,
            pilot=pilot,
            logger=logger,
        )

    logger.info(str(len(pumps)) + " pumps are set up.")
    return pumps


def setup_culverts(
    branches: gpd.GeoDataFrame,
    crsdefs: gpd.GeoDataFrame = None,
    crslocs: gpd.GeoDataFrame = None,
    roughness_ini_fn: str = None,
    culverts_ini_fn: str = None,
    culverts_fn: str = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "overall",
    snap_offset: float = 1,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """setup culverts"""
    # FIXME: why have to use ROUGH_TYPE ROUGH_VAL
    logger.info("Setting up culverts.")
    culverts = None

    # prepare structures ini
    if culverts_ini_fn is not None:
        culverts_ini = parse_ini(culverts_ini_fn)

    # setup structure defaults
    if branches is not None:
        culverts = _setup_structures_defaults(
            branches,
            culverts_ini,
            culverts_fn,
            "culvert",
            id_col,
            branch_query,
            snap_method,
            snap_offset,
            rename_map,
            required_columns,
            required_dtypes,
            pilot=pilot,
            logger=logger,
        )

    # prepare roughness ini
    if roughness_ini_fn is not None:
        roughness_ini = parse_ini(roughness_ini_fn)

    # setup structure crsdefs
    if not any(x is None for x in [branches, culverts, crsdefs]):
        culverts, crsdefs = _setup_structures_crsdefs(
            branches=branches,
            structures=culverts,
            crsdefs=crsdefs,
            roughness_ini=roughness_ini,
            crs_type="culvert",
            structures_ini=culverts_ini,
            logger=logger,
        )

    logger.info(str(len(culverts)) + " culverts are set up.")
    return culverts, crsdefs


def setup_compounds(
    structures: list = [],
    roughness_ini_fn: str = None,
    compounds_ini_fn: str = None,
    compounds_fn: str = None,
    generate_compounds=True,
    id_col: str = None,
    id_prefix: str = "",
    id_suffix: str = "",
    branch_query: str = None,
    snap_method: str = "overall",
    snap_offset: float = 0.1,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """setup compound structures"""
    logger.info("Setting up compounds.")
    compounds = None

    # setup valid structures with unified id
    structures_ = []
    if len(structures) > 0:
        if id_col is None:
            id_col = "STRUCT_ID"

        for ss in structures:
            if ss is not None:
                ss.index.name = id_col
                structures_.append(
                    ss[["branch_id", "branch_offset", "branch_type", "geometry"]]
                )

    # setup structures only for branch_query
    if branch_query is not None:
        try:
            structures_ = [ss.query(branch_query) for ss in structures_]
        except:
            # FIXME BMA only allows branch_type in query
            logger.error(
                "branch_query specified for bridges is invalid (only allows branch_type in query).  Please check structure section in ini."
            )

    # check if there are structures to add
    structures_ = pd.concat(structures_)
    if len(structures_) > 0:

        # prepare ini
        if compounds_ini_fn is not None:
            compounds_ini = parse_ini(compounds_ini_fn)

            # generate compound structures
            compounds_generated = None
            if generate_compounds is True:
                logger.info("Compounds are generated based on ini.")
                compounds_generated = _setup_generated_compounds(
                    structures_,
                    compounds_ini,
                    id_col,
                    id_prefix,
                    id_suffix,
                    snap_offset,
                    logger,
                )
                # FIXME BMA: prefix and suffix must be used together

            # read compound structures
            compounds_user = None
            if compounds_fn is not None:
                logger.info("Compounds are read from user input.")
                compounds_user = _setup_user_compounds(
                    structures_,
                    compounds_fn,
                    id_col,
                    rename_map,
                    required_columns,
                    required_dtypes,
                    pilot,
                    logger,
                )

            # compare generated compound structures with user
            if (compounds_user is not None) and (compounds_generated is not None):
                logger.info("comparing compounds generated with compounds from user.")
                _compare_compounds(compounds_generated, compounds_user, logger)
                logger.info("Use generated compounds.")
                compounds = compounds_generated

            elif (compounds_user is not None) and (compounds_generated is None):
                logger.info("Use user specified compounds.")
                compounds = compounds_user

            elif (compounds_user is None) and (compounds_generated is not None):
                logger.info("Use generated compounds.")
                compounds = compounds_generated

            else:
                compounds = None

    if compounds is not None:
        logger.info(str(len(compounds)) + " compounds are set up.")

    return compounds


def _setup_generated_compounds(
    structures: pd.DataFrame,
    compounds_ini: configparser,
    id_col: str = "STRUCT_ID",
    id_prefix: str = "",
    id_suffix: str = "",
    snap_offset: float = 1,
    logger=logging,
):
    """generate compounds at each structures

    1 structure on branch/pipe --> compound with 1 structure
    >=2 structures on branch within snap_offset --> compound with >=2 structures
    >= 2 structures on pipe --> #TODO BMA:  maybe compound on parallel pipes --> give NotImpementedError

    """

    compounds_ = []

    # prepare structures - group structures as structureIds

    # group structures based on branch id
    groups_by_branchid = structures.groupby("branch_id").groups
    for bi in groups_by_branchid:
        bi_group = structures.loc[groups_by_branchid[bi].to_list(), :]

        # group decimal numbers by their rounded value, branch_offset with snap_offset used as divident
        groups_by_branchid_chainage = bi_group.groupby(
            (bi_group["branch_offset"] / snap_offset).round(0)
        ).groups
        for bi_c in groups_by_branchid_chainage:
            bi_c_group = bi_group.loc[groups_by_branchid_chainage[bi_c].to_list(), :]

            # setup generated compounds
            compound = {
                id_col: id_prefix + str(len(compounds_)) + id_suffix,
                "branchId": bi,
                "chainage": bi_c_group["branch_offset"].mean(),
                "type": "compound",
                "numStructures": len(bi_c_group),
                "structureIds": bi_c_group.index.to_list(),
                "geometry": bi_c_group["geometry"][0],
            }
            compounds_.append(compound)

            # check if compound with multiple structures on pipe
            if np.any(bi_c_group["branch_type"] == "Pipe") and len(bi_c_group) > 1:
                logger.warning(
                    f'Compound added: Pipe {compound["branchId"]} does not accept more than one structures. Please check: '
                    + f'{compound[id_col]} : {";".join(compound["structureIds"])} .'
                )
                pass

    compounds_ = gpd.GeoDataFrame(compounds_)
    if compounds_.index.name != id_col:
        compounds_ = compounds_.set_index(id_col)
    compounds_[id_col] = compounds_.index

    # sort stuctureIds
    compounds_["structureIds"] = compounds_.structureIds.sort_values().apply(
        lambda x: sorted(x)
    )
    compounds_["structureIds"] = compounds_["structureIds"].str.join(";")

    return compounds_


def _setup_user_compounds(
    structures: pd.DataFrame,
    compounds_fn: str = None,
    id_col: str = None,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """read user compounds"""

    # read user compounds
    compounds = read_gpd(
        compounds_fn,
        id_col,
        rename_map,
        required_columns,
        required_dtypes,
        clip_geom=pilot,
    )

    # prepare user compounds - group structures as structureIds
    for k, v in compounds.groupby(axis=1, level=0, dropna=True).groups.items():
        if len(v) > 1:
            compounds[k + "s"] = (
                compounds[v].replace({"None": None}).values.tolist()
            )  # concat duplicated columns to list
            compounds[k + "s"] = compounds[k + "s"].apply(
                lambda c: list(set(x for x in c if x is not None))
            )  # remove None and duplicates

    # setup user compounds
    if "structureIds" not in compounds.columns:
        logger.error(
            "compounds must have structureIds. Please specify structureId in ini."
        )
    else:
        compounds_ = pd.DataFrame(
            columns=["branchId", "chainage", "type", "structureIds", "geometry"],
            index=compounds[id_col],
        )

        for i, cmp in compounds[["branch_id", "structureIds"]].iterrows():

            # check if all structures in a compound are previously setup
            if not set(cmp["structureIds"]).issubset(set(structures.index.to_list())):
                logger.error(
                    "Compound not added: not all structures in compound have been setup successfully. Please check: "
                    + f'{i} : {";".join(cmp["structureIds"])}'
                )
                continue
            else:
                pass

            # check if all structures in a compound have the same branch_id
            if len(structures.loc[cmp["structureIds"], "branch_id"].unique()) > 1:
                logger.error(
                    "Compound not added: structures in the compounds are on not the same branch. Please check: "
                    + f'{i} : {";".join(cmp["structureIds"])}'
                )
                continue
            # check if all structures in a compound have the correct branch_id
            elif not np.all(
                structures.loc[cmp["structureIds"], "branch_id"].unique()
                == cmp["branch_id"]
            ):
                logger.warning(
                    f'Compound added: structures in the compounds are located on {structures.loc[cmp["structureIds"], "branch_id"].unique()[0]}, '
                    + f'which is different from user specified {cmp["branch_id"]}. Please check: '
                    + f'{i} : {";".join(cmp["structureIds"])}'
                )
                pass
            else:
                pass

            # check if compound on pipe --> Not Implemented Error
            if (
                np.any(structures.loc[cmp["structureIds"], "branch_type"] == "Pipe")
                and len(cmp["structureIds"]) > 1
            ):
                logger.warning(
                    f'Compound added: Pipe {cmp["branch_id"]} does not accept more than one structures. Please check: '
                    + f'{i} : {";".join(cmp["structureIds"])}'
                )
                pass

            # setup successful --> get chainage
            compounds_.at[i, "branchId"] = structures.loc[
                cmp["structureIds"], "branch_offset"
            ].mean()
            compounds_.at[i, "chainage"] = structures.loc[
                cmp["structureIds"], "branch_offset"
            ].mean()
            compounds_.at[i, "type"] = "compound"
            compounds_.at[i, "structureIds"] = cmp["structureIds"]
            compounds_.at[i, "geometry"] = structures.loc[
                cmp["structureIds"], "geometry"
            ][
                0
            ]  # estimate locations

        compounds_ = gpd.GeoDataFrame(compounds_.dropna())
        if compounds_.index.name != id_col:
            compounds_ = compounds_.set_index(id_col)
        compounds_[id_col] = compounds_.index

        # sort stuctureIds
        compounds_["structureIds"] = compounds_.structureIds.sort_values().apply(
            lambda x: sorted(x)
        )
        compounds_["structureIds"] = compounds_["structureIds"].str.join(";")

    return compounds_


def _compare_compounds(compounds_generated, compounds_user, logger=logging):
    """Compare compounds between generated and user specified for those more than 1 structures"""

    # compare based on structureIds
    combined_dfs = pd.concat(
        [compounds_generated["structureIds"], compounds_user["structureIds"]]
    )

    # count the strucureIds, should be 2 if the compound structure exist in both compounds_generated and compounds_user
    combined_dfs_counts = combined_dfs.value_counts()

    # compare compounds with more than 1 structure that do not exist in both
    for cmp_i, cmp_n in combined_dfs_counts.items():
        if len(cmp_i.split(";")) > 1 and cmp_n < 2:
            if cmp_i in compounds_generated["structureIds"].values:
                logger.warning(
                    f"Compounds: generated compound structure of {cmp_i} could not be found in user specified compounds."
                    + "Please check previous warnings on user specified compounds "
                    + "and user specified compounds data layer. "
                )

    return None


def _setup_structures_defaults(
    branches: gpd.GeoDataFrame,
    structures_ini: dict = None,
    structures_fn: str = None,
    structures_type: str = None,
    id_col: str = None,
    branch_query: str = None,
    snap_method: str = "overall",
    snap_offset: float = 1,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):
    """
    setup structure default values from ini
    """
    # get branches - subset from branches based on branch_query
    if branch_query is not None:
        try:
            branches_ = read_gpd(branches)
            branches_ = branches_.query(branch_query)
        except:
            branches_ = None
            logger.error(
                "branch_query specified for bridges is invalid.  Please check structure section in ini."
            )
    else:
        branches_ = branches
    if len(branches_) == 0:
        logger.warning(
            "Structures are not added: there are no branches to add. Please check structure section in ini."
        )

    # prepare structures - read from shp file
    if structures_fn is None:
        structures = None
        logger.warning(
            "Structures are not added: structures filename are not specified. Please check structure section in ini."
        )
    else:
        try:
            structures = read_gpd(
                structures_fn,
                None,
                rename_map,
                required_columns,
                required_dtypes,
                clip_geom=pilot,
            )
            logger.info("Structures are added from: " + structures_fn)
        except:
            structures = None
            logger.error(
                "Structures filename specified is invalid . Please check structure section in ini."
            )

    # add structures to branch
    if structures is not None:

        # setup struct_id
        if id_col is not None:
            structures.index = structures[id_col]
            structures[
                id_col
            ] = (
                structures.index
            )  # retain the id_col as columns for setting up the model
            logger.info("Structures ids are identified from: " + id_col)
        else:
            id_col = "STRUC_ID"
            structures[id_col] = [
                structures_type + "_" + str(x) for x in range(len(structures))
            ]
            logger.info("Structures ids are identified from: " + id_col)

        # setup branch_id - snap bridges to branch (inplace of bridges, will add branch_id and branch_offset columns)
        logger.info(
            "Structures are snapped to nearest branch within a snap_offset = "
            + str(snap_offset)
        )
        geometry.find_nearest_branch(
            branches=branches_,
            geometries=structures,
            method=snap_method,
            maxdist=snap_offset,
            move_geometries=True,
        )

        # setup failed - drop structures based on branch_offset that are not snapped to branch (inplace of structures) and issue warning
        if structures_ini["global"]["drop_snap_failed"] == True:
            _old_ids = structures.index.to_list()
            structures.dropna(axis=0, inplace=True, subset=["branch_offset"])
            _new_ids = structures.index.to_list()
            if len(_old_ids) != len(_new_ids):
                logging.warning(
                    "Structure with id:"
                    + ",".join(list(set(_old_ids) - set(_new_ids)))
                    + " are dropped: unable to find closes branch. "
                    + "Please try changing branch_query or snap_offset in structure section in ini."
                )

        # setup successful - compare snapped with user specified branchid
        if structures_ini["global"]["compare_snap_with_column"] == True:
            if branches.index.name not in structures.columns:
                logging.error(
                    "Failed to execute: compare_snap_with_column = True. branch index could not be found in structures: "
                    + branches.index.name
                )
            else:
                for i, struct in structures.iterrows():
                    if struct["branch_id"] != struct[branches.index.name]:
                        logger.warning(
                            f"Structure {i} is snapped to : "
                            + struct["branch_id"]
                            + ", which does not match user specified branch: "
                            + struct[branches.index.name]
                            + ". Please check."
                        )

        # setup successful - fill in values from default
        if structures_ini["global"]["append_data_columns_based_on_ini_query"] == True:
            structures = append_data_columns_based_on_ini_query(
                structures, structures_ini
            )
            logger.info(
                "Structures are filled with values specified in: " + str(structures_ini)
            )

        # update branchType from Pipe to SewerConnection (not changed if branchType == "Channel")
        for bi, b in structures.iterrows():
            if branches.loc[b["branch_id"], "branchType"] == "Pipe":
                branches.at[b["branch_id"], "branchType"] = "SewerConnection"
                logger.info(
                    "Structures are altering branchType from Pipe to SewerConnection at: "
                    + b["branch_id"]
                )

    return structures


def _setup_structures_crsdefs(
    branches: gpd.GeoDataFrame,
    structures: gpd.GeoDataFrame,
    crsdefs: gpd.GeoDataFrame,
    roughness_ini=None,
    crs_type: str = "culvert",
    structures_ini: dict = None,
    logger=logging,
):
    """Function to update crsdefs for structures (support bridges, culverts)
    # FIXME BMA: derive from crosssection definitions by interpolation
    """
    _structures = structures.copy()

    if structures_ini is not None:

        # setup columns from branch (not only friction)
        structures = pd.merge(
            _structures,
            branches.reset_index(drop=True),
            left_on="branch_id",
            right_on=branches.index.name,
            suffixes=("", "_b"),
        ).set_index(_structures.index.name)
        structures[structures.index.name] = structures.index

        # setup crsdefs from structures
        if structures_ini["global"]["setup_structures_crsdefs"] == True:
            logger.info("Structures require crosssection definition.")

        # setup crosssection definition firctions
        if structures_ini["global"]["use_branch_friction"] == True:
            logger.info("Structures friction uses branch frictions")
            # frictionId already in structures

        elif structures_ini["global"]["use_default_friction"] == True:
            logger.info(
                "Structures friction uses defaults based on material, check roughness section in ini."
            )
            assert {"material_col"}.issubset(
                structures_ini["global"].keys()
            ), "'material_col' must be specified in structure ini"
            # regenerate frictionId
            structures = _setup_rougnness_from_default(
                structures.drop(columns=["frictionId", "rough_type", "rough_val"]),
                roughness_ini,
                logger,
            )

        elif structures_ini["global"]["use_infile_friction"] == True:
            logger.info(
                "Structures crosssection definition uses frictions specified in file"
            )
            assert {"frictype_col", "fricval_col"}.issubset(
                structures_ini["global"].keys()
            ), "'frictype_col', 'fricval_col' must be specified in structure ini"

            # regenerate frictionId
            fricid_col = "frictionId"
            for si, s in structures.iterrows():
                structures.at[si, fricid_col] = "%s_%s" % (
                    s[structures_ini["global"]["frictype_col"]],
                    s[structures_ini["global"]["fricval_col"]],
                )

        else:
            logger.error(
                "No friction methods is chosen. Please check [global] section in ini."
            )

        # setup crosssection definition
        if {
            "shape_col",
            "diameter_col",
            "height_col",
            "width_col",
            "t_width_col",
            "closed_col",
        }.issubset(structures_ini["global"].keys()):
            structures_crsdefs = _setup_crsdefs_from_branches_at(
                structures,
                at="",
                shape_col=structures_ini["global"]["shape_col"],
                diameter_col=structures_ini["global"]["diameter_col"],
                height_col=structures_ini["global"]["height_col"],
                width_col=structures_ini["global"]["width_col"],
                t_width_col=structures_ini["global"]["t_width_col"],
                closed_col=structures_ini["global"]["closed_col"],
                frictionid_col="frictionId",
                crsdefid_col="definitionId",
                crs_type=crs_type,
                is_shared="False",
            )
        else:
            logger.error(
                "Failed to setup crosssection definition for structures. "
                + "'shape_col', 'diameter_col', 'height_col', 'width_col', 't_width_col', 'closed_col' "
                + " are not specified for structures.  Please check structure.ini."
            )

        # append to existing crsdefs
        if structures_crsdefs is not None:
            crsdefs = (
                pd.concat([crsdefs, structures_crsdefs])
                .drop_duplicates(subset="id")
                .reset_index(drop=True)
            )
            logger.info(
                "Succeeded to setup crosssection definition for structures. "
                + "Appending to exiting crosssection definitions"
            )

    return structures, crsdefs


def _interpolate_branch_crosssections_at_gates(
    branches: gpd.GeoDataFrame,
    structures: gpd.GeoDataFrame,
    crsdefs,
    crslocs,
    logger=logging,
):
    """function to interpolate crsdefs and crslocs at a certain chainge on branches"""

    logger.debug("Interpolating branch crosssections at structures.")

    # setup columns from branch
    branches["branch_length"] = branches["geometry"].length
    structures = pd.merge(
        structures,
        branches.reset_index(drop=True),
        left_on="branch_id",
        right_on=branches.index.name,
        suffixes=("", "_b"),
    ).set_index(structures.index.name)
    structures[structures.index.name] = structures.index

    # check if the required columns are in the structures
    assert {
        "branch_id",
        "branch_offset",
        "branch_length",
        "startdefinitionId",
        "enddefinitionId",
    }.issubset(
        structures.columns
    ), "Branches must have 'branch_id', 'branch_offset', 'branch_length', 'startdefinitionId', 'enddefinitionId' in the columns name."

    structures_crsdefs, structures_crslocs = __interpolate_crs_at_gates(
        structures, crsdefs, crslocs
    )

    return structures_crsdefs, structures_crslocs


def __interpolate_crs_at_gates(structures, crsdefs, crslocs):
    """function to interpolate crsdefs and crslocs at a certain chainge on branches"""

    structures_crslocs = pd.DataFrame(columns=crslocs.columns)
    structures_crsdefs = pd.DataFrame(columns=crsdefs.columns)

    if crslocs is not None:
        for bi, b in structures.iterrows():

            # get the start and end crs loc (meaning before and after the structure)
            try:
                start_crsloc = (
                    crslocs.query(
                        f'branchid == "{b.branch_id}" & chainage < {b.branch_offset}'
                    )
                    .sort_values("chainage")
                    .iloc[-1]
                )
            except:
                start_crsloc = None
            try:
                end_crsloc = (
                    crslocs.query(
                        f'branchid == "{b.branch_id}" & chainage > {b.branch_offset}'
                    )
                    .sort_values("chainage")
                    .iloc[0]
                )
            except:
                end_crsloc = None

            if all(v is None for v in [start_crsloc, end_crsloc]):
                # no crs available
                raise ValueError("no cross section on the branch for interpolation")
            elif any(v is None for v in [start_crsloc, end_crsloc]):
                # only 1 crs available for interpolation, use that one
                start_crsloc = end_crsloc = [
                    v for v in [start_crsloc, end_crsloc] if v is not None
                ][0]
            else:
                # both crs available
                pass

            # interpolate shift only
            structures_crslocs.loc[bi, "id"] = bi
            structures_crslocs.loc[bi, "branchid"] = b.branch_id
            structures_crslocs.loc[bi, "chainage"] = b.branch_offset
            structures_crslocs.loc[bi, "shift"] = np.interp(
                b["branch_offset"],
                [start_crsloc["chainage"], end_crsloc["chainage"]],
                [start_crsloc["shift"], end_crsloc["shift"]],
            )

            # get the definition id
            start_crsdef = crsdefs.query(f'id == "{start_crsloc.definitionId}"')
            end_crsdef = crsdefs.query(f'id == "{end_crsloc.definitionId}"')

            # interpolate width
            if start_crsdef.type.isin(["xyz", "yz"]).any():
                structures_crsdefs.loc[bi, "width"] = np.interp(
                    b["branch_offset"],
                    [0, b["branch_length"]],
                    [
                        max(start_crsdef["xylength"].values[0]),
                        max(end_crsdef["xylength"].values[0]),
                    ],
                )
                # add shift from profile to shift
                structures_crslocs.loc[bi, "shift"] = structures_crslocs.loc[
                    bi, "shift"
                ] + np.interp(
                    b["branch_offset"],
                    [0, b["branch_length"]],
                    [
                        min(start_crsdef["zCoordinates"].values[0]),
                        min(end_crsdef["zCoordinates"].values[0]),
                    ],
                )

            elif start_crsdef.type.isin(["rectangle"]).any():
                structures_crsdefs.loc[bi, "width"] = np.interp(
                    b["branch_offset"],
                    [0, b["branch_length"]],
                    [start_crsdef["width"].values[0], end_crsdef["width"].values[0]],
                )
                # no need to add shift from profile to shift

            else:
                raise NotImplementedError(
                    "cross section type does not support interpolation. use rectangle, xyz and yz."
                )

    return structures_crsdefs, structures_crslocs


# FIXME BMA: also needed on weirs (general structures) and orifices. --> not yet implemented
def _disable_velocityheight_on_pipes(structures, logger=logging):
    """Function to disable velocity height for structures on pipes"""
    logger.debug("Disable velocityheight for structures on Pipes/ SewerConnections")

    assert (
        "branch_type" in structures.columns
    ), "Structure must have 'branch_type' in columns"
    assert (
        "useVelocityHeight" in structures.columns
    ), "Structure must have 'useVelocityHeight' in columns"

    idxs = structures["branch_type"].isin(["Pipe", "SewerConnection"])
    structures.at[idxs, "useVelocityHeight"] = "false"

    return structures


def setup_fixedweirs(
    fixedweirs_fn: str = None,
    fixedweirs_fn_ini: str = None,
    id_col: str = None,
    split_offset: float = 100.0,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):

    # prepare fixedweirs - read from shp file
    if fixedweirs_fn is None:
        fixedweirs = None
        logger.warning(
            "Fixedweirs are not added: structures filename are not specified. Please check structure section in ini."
        )
    else:
        try:
            fixedweirs = read_gpd(
                fixedweirs_fn,
                None,
                rename_map,
                required_columns,
                required_dtypes,
                clip_geom=pilot,
            )
            logger.info("Fixedweirs are added from: " + fixedweirs_fn)
        except:
            fixedweirs = None
            logger.error(
                "Fixedweirs filename specified is invalid . Please check structure section in ini."
            )

    if fixedweirs is not None:

        # setup struct_id
        if id_col is not None:
            fixedweirs.index = fixedweirs[id_col]
            fixedweirs[
                id_col
            ] = (
                fixedweirs.index
            )  # retain the id_col as columns for setting up the model
            logger.info("Fixedweirs ids are identified from: " + id_col)
        else:
            id_col = "STRUC_ID"
            fixedweirs[id_col] = [
                "fixedweir" + "_" + str(x) for x in range(len(fixedweirs))
            ]
            logger.info("Structures ids are identified from: " + id_col)

        # prepare vertices based on split_offset
        for fi, f in fixedweirs.iterrows():
            old_line = f.geometry
            new_line = redistribute_vertices(old_line, split_offset)
            fixedweirs.at[fi, "geometry"] = new_line
            logger.info(
                f"Fixed weirs geometries are redistributed based on split_offset = {split_offset}"
            )

    if fixedweirs is not None:
        logger.info(str(len(fixedweirs)) + " fixedweirs are set up.")

    return fixedweirs


# ===============================================================
#                   BOUNDARIES
# ===============================================================
def setup_boundaries(
    boundaries_fn: str = None,
    boundaries_fn_ini: str = None,
    id_col: str = None,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot: gpd.GeoDataFrame = None,
    logger=logging,
):

    """Setup boundaries from shp file"""

    # prepare boundaries - read from shp file
    if boundaries_fn is None:
        boundaries = None
        logger.warning(
            "Boundaries are not added: filename are not specified. Please check ini."
        )
    else:
        try:
            boundaries = read_gpd(
                boundaries_fn, None, rename_map, required_columns, required_dtypes
            )  # do not clip boundaries
            logger.info("Boundaries are added from: " + boundaries_fn)
        except:
            boundaries = None
            logger.error("Boundaries filename specified is invalid. Please check ini.")

    if boundaries is not None:

        # fill in defaults
        if boundaries_fn_ini is not None:
            boundaries_ini = parse_ini(boundaries_fn_ini)
            boundaries = append_data_columns_based_on_ini_query(
                boundaries, boundaries_ini
            )

        # setup boundaries id
        if id_col is not None:
            boundaries.index = boundaries[id_col]
            boundaries[
                id_col
            ] = (
                boundaries.index
            )  # retain the id_col as columns for setting up the model
            logger.info("Boundaries ids are identified from: " + id_col)
        else:
            id_col = "BC_ID"
            boundaries[id_col] = [
                "Boundaries" + "_" + str(x) for x in range(len(boundaries))
            ]
            logger.info("Boundaries ids are identified from: " + id_col)

    if boundaries is not None:
        logger.info(str(len(boundaries)) + " boundaries are set up.")

    return boundaries


# ===============================================================
#                   SUBCATCHMENT
# ===============================================================
# continue in preprocess.py
# Question to Rinske: does geopandas savigng shp file also has limit on column length?
# Question to Rinske: is this only for NWRW concept? --> I think so because for rural, theissan polygon might not apply; also we use manholes
# Question to Rinske: do we allow setup subcatchmnet based on _fn being regined?


def setup_subcatchments(
    manholes: gpd.GeoDataFrame,
    subcatchments_ini_fn: str = None,
    subcatchments_fn: str = None,
    region_fn: str = None,
    barriers_fn: str = None,
    landuse_fn: str = None,
    snap_offset: float = 500.0,
    id_col: str = "SUBCAT_ID",
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    pilot=None,
    logger=logging,
):
    """setup rr catchments using NWRW urban rainfall runoff concept that have relation to manholes.

    Arguments
    ----------
    manholes:  gpd.GeoDataFrame
        The manholes that recives runoff from the subcatchments
    subcatchments_ini_fn: str
        The path to subcatchments_settings.ini: arguments that control the setting up of the subcatchments.
    subcatchments_fn: str, optional
        The path to existing subcatchments filename. Must have subcatchments and manholes ids relations, with columns names consistent with previously with the id_col.
        Default is None.
    region_fn: str, optional
        The path to polygon(s) shape file, which defines the extents of the subcatchments.
        Default is None.
    barriers_fn: str, optional
        The path to line(s) shape file, which will devide the drainage area.
        Default is None.
    landuse_fn: str, optional
        The path to landuse raster file, which contains 4 predefined categories [1,2,3,4] of the nwrw landuse types.
        Default is None.
    snap_offset: float, optional
        The maximum distance allowed to snap subcatchments to manholes. e.g. if a subcatchments is devided into seperate parts by the barriers.
        Default is 500.
    id_col: str, optional
        The index name of the subcatchments.
        Default is  "SUBCAT_ID"
    **rename_map, required_columns, required_dtypes
        Additional keyword arguments to pass into subcatchment_fn reader.
    """
    subcatchments = None

    if manholes is None:
        logger.error(
            f"Cannot setup subcatchments. NWRW urban rainfall runoff concept require manholes information."
        )

    elif subcatchments_fn is None and subcatchments_ini_fn is None:
        logger.info(
            f"Cannot setup subcatchments. Both subcatchments_fn and subcatchments_ini_fn are required to create manholes."
        )

    elif subcatchments_fn is not None and subcatchments_ini_fn is not None:
        logger.info(f"Subcatchments are directly read from subcatchments_fn.")

        # read
        # FIXME make clip buffer accessible from ini file
        subcatchments = read_gpd(
            subcatchments_fn,
            None,
            rename_map,
            required_columns,
            required_dtypes,
            clip_geom=pilot.buffer(100),
        )
        subcatchments.index = subcatchments[id_col]

        # check
        if {
            "SUBCAT_ID",
            "MANHOLE_ID",
            "cl_slope",
            "cl_flat",
            "cl_stretch",
            "op_slope",
            "op_flat",
            "op_stretch",
            "rf_slope",
            "rf_flat",
            "rf_stretch",
            "up_slope",
            "up_flat",
            "up_stretch",
            "inhabitant",
            "company",
            "dwf_def",
            "meteo_id",
        }.issubset(subcatchments.columns):
            needs_filling = False
        else:
            needs_filling = True

        # fill
        if needs_filling is True:
            logger.info(f"Subcatchments are filled from subcatchments_ini_fn.")
            subcatchments_ini = parse_ini(subcatchments_ini_fn)
            subcatchments = _fill_subcatchments(
                subcatchments, subcatchments_ini, landuse_fn, logger=logger
            )

    elif subcatchments_fn is None and subcatchments_ini_fn is not None:
        logger.info(f"Subcatchments are generated from subcatchments_ini_fn.")

        if id_col is None:
            id_col = "SUBCAT_ID"

        subcatchments_ini = parse_ini(subcatchments_ini_fn)

        # FIXME everything change to laterals; subcatchments --> laterals
        # generate subcatchment
        subcatchments = _generate_subcatchments(
            manholes,
            subcatchments_ini,
            region_fn,
            barriers_fn,
            id_col=id_col,
            snap_offset=snap_offset,
            logger=logger,
        )

        # fill subcatchment
        _subcatchments = subcatchments.copy()
        subcatchments = _fill_subcatchments(
            subcatchments, subcatchments_ini, landuse_fn=landuse_fn, logger=logger
        )

    # get xy coords of the subcatchments based on manholes
    if subcatchments is not None:
        _s_ = pd.merge(
            subcatchments.reset_index(drop=True),
            manholes.reset_index(drop=True),
            left_on=manholes.index.name,
            right_on=manholes.index.name,
            suffixes=("", "_m"),
        ).set_index(subcatchments.index.name)
        subcatchments["px"] = _s_["geometry_m"].x
        subcatchments["py"] = _s_["geometry_m"].y

    # TODO: fill in rain gauge information (maybe in setup precipitation)

    if subcatchments is not None:
        logger.info(f"{len(subcatchments)} subcatchments are setup")

    return subcatchments


def _generate_subcatchments(
    manholes: gpd.GeoDataFrame,
    subcatchments_ini: configparser.ConfigParser,
    region_fn: str = None,
    barriers_fn: str = None,
    snap_offset: float = 500.0,
    id_col: str = "SUBCAT_ID",
    logger=logging,
):
    """generate subcatchments based on subcatchments_ini"""

    subcatchments = None
    method = subcatchments_ini["global"]["generate_method"]
    logging.info(f"Generating subcatchment using method: {method}")

    if method == "thiessen":
        subcatchments = __generate_thiessen_subcatchments(
            manholes,
            region_fn,
            barriers_fn,
            how=subcatchments_ini["global"]["how"],
            join_by=subcatchments_ini["global"]["join_by"],
            snap_offset=snap_offset,
            id_col=id_col,
            logger=logger,
        )
    elif method == "equal":
        subcatchments = __generate_equal_subcatchments(
            manholes,
            region_fn,
            barriers_fn,
            how=subcatchments_ini["global"]["how"],
            join_by=subcatchments_ini["global"]["join_by"],
            snap_offset=snap_offset,
            id_col=id_col,
            logger=logger,
        )
    else:
        raise NotImplementedError(
            "method not recongnised. Please use one of the following: thiessen, equal"
        )

    return subcatchments


def __generate_equal_subcatchments(
    manholes: gpd.GeoDataFrame,
    region_fn: str = None,
    barriers_fn: str = None,
    how: str = "join",
    join_by: str = "BRANCH_ID",
    snap_offset: float = 500.0,
    id_col: str = "SUBCAT_ID",
    logger=logging,
):
    """Method to generate subcatchments area by distributing equally to manholes within region"""

    _manholes = manholes.copy()

    # generate region file
    _region = None
    if region_fn is not None:
        _region = read_gpd(region_fn)

        region = gpd.GeoDataFrame()
        for _, r in _region.iterrows():
            if r.geometry.type == "MultiPolygon":
                r_ = _region.explode()
                region = pd.concat([region, r_])
            else:
                region = pd.concat([region, r.to_frame().T])
    region = gpd.GeoDataFrame(region, crs=_region.crs)

    # remove duplicated geometry
    _region = region.drop_duplicates()
    G = _region["geometry"].apply(lambda geom: geom.wkb)
    n = len(G) - len(G.drop_duplicates().index)
    region = _region[_region.index.isin(G.drop_duplicates().index)]
    logger.debug(f"Removing {n} sub-regions which have duplicated geometry.")

    # manhole overlay
    if how == "intersection":
        raise ValueError(
            "method equal could only be used with how ==  join. Please specify how and join_by"
        )
    elif how == "join":
        region["region_index"] = region[join_by]
        region.index = region["region_index"]
        manhole_join_by = "ORIG_" + join_by
        if manhole_join_by not in _manholes.columns:
            raise NotImplementedError(
                f"Join_by {join_by} is not accepted. Only accepts branch index column name."
            )
        else:
            manholes_ = _manholes.assign(
                region_index=_manholes[manhole_join_by].str.split(";")
            )
            manholes_ = gpd.GeoDataFrame(
                pd.DataFrame(manholes_).explode("region_index", ignore_index=True)
            )
            manholes_.index = manholes_[_manholes.index.name]
    else:
        raise NotImplementedError(
            f"how: {how} is not accepted. Only accepts intersection/join."
        )

    # obtain subcatchment by looping through region
    i = 0
    subcatchments = gpd.GeoDataFrame()
    for region_index, subregion in region.iterrows():
        i += 1
        print(f"{i}/{len(region)} subcatchments developed")
        # get the subregion and manholes within subregion
        if not isinstance(subregion, gpd.GeoDataFrame):

            subregion = gpd.GeoDataFrame(
                subregion.to_frame().T, geometry="geometry", crs=region.crs
            )
            subregion = subregion.buffer(
                0
            )  # applying a zero distance buffer can fix the invalid geometries
            subregion = gpd.GeoDataFrame(
                subregion.to_frame().rename(columns={0: "geometry"}),
                geometry="geometry",
                crs=region.crs,
            )

        submanholes = gpd.GeoDataFrame(
            manholes_.loc[manholes_["region_index"] == region_index, :],
            geometry="geometry",
            crs=_manholes.crs,
        ).drop_duplicates()

        # if multiple manholes
        if len(submanholes) > 0:

            # note the resolution of 16 will make 99.8% loss of area, use higher might be better but slower
            _subcatchments = gpd.GeoDataFrame(
                geometry=[
                    p.buffer(
                        np.sqrt(subregion.area / len(submanholes) / np.pi),
                        resolution=16,
                    )
                    for p in submanholes.geometry.to_list()
                ],
                index=submanholes.index,
            )

            # combine all
            subcatchments = pd.concat([subcatchments, _subcatchments])

        else:

            logger.warn(f"Subcatchments {region_index} have no manholes. ")

    # set crs
    subcatchments.crs = region.crs
    subcatchments_ = subcatchments.copy()

    # link to manhole
    subcatchments_[manholes.index.name] = subcatchments_.index
    logger.info(
        f"Subcatchments are linked to manholes using manhole ids as {manholes.index.name}. "
    )

    # index
    subcatchments_.reset_index(drop=True, inplace=True)
    subcatchments_[id_col] = [f"subcat_{i}" for i in range(len(subcatchments_))]
    subcatchments_.index = subcatchments_[id_col]
    logger.info(
        f"Subcatchments are equally generated within region using ids as {id_col}. "
    )

    # check area
    if (
        region.area.sum() * 99.8 / 100 > subcatchments_.area.sum()
    ):  # allowed error due to precision
        logging.warning(
            "Total subcatchments area lower than the specified region area. Please check intermediate exports. "
        )

    return subcatchments_


def __generate_thiessen_subcatchments(
    manholes: gpd.GeoDataFrame,
    region_fn: str = None,
    barriers_fn: str = None,
    how: str = "intersection",
    join_by: str = "BRANCH_ID",
    snap_offset: float = 500.0,
    id_col: str = "SUBCAT_ID",
    logger=logging,
):
    """Method to generate subcatchments using thiessen polygon"""

    _manholes = manholes.copy()

    # generate region file
    _region = None
    if region_fn is not None:
        _region = gpd.read_file(region_fn)
        region = gpd.GeoDataFrame()
        for _, r in _region.iterrows():
            if r.geometry.type == "MultiPolygon":
                r_ = _region.explode()
                region = pd.concat([region, r_])
            else:
                region = pd.concat([region, r.to_frame().T])
    region = gpd.GeoDataFrame(region, crs=_region.crs)

    # remove duplicated geometry
    _region = region.drop_duplicates()
    G = _region["geometry"].apply(lambda geom: geom.wkb)
    n = len(G) - len(G.drop_duplicates().index)
    region = _region[_region.index.isin(G.drop_duplicates().index)]
    logger.debug(f"Removing {n} sub-regions which have duplicated geometry.")

    # generate barrier file
    barrier = None
    if barriers_fn is not None:
        _barrier = gpd.read_file(barriers_fn)

        barrier = gpd.GeoDataFrame()
        for _, r in _barrier.iterrows():
            if r.geometry.type == "MultiLineString":
                r_ = _barrier.explode()
                barrier = pd.concat([barrier, r_])
            else:
                barrier = pd.concat([barrier, r.to_frame().T])
        barrier = gpd.GeoDataFrame(barrier, crs=_region.crs).reset_index(drop=True)

    # manhole overlay
    if how == "intersection":
        region["region_index"] = region.index
        region.index = region["region_index"]
        manholes_ = gpd.overlay(_manholes, region, how="intersection")
        manholes_.index = manholes_[_manholes.index.name]
    elif how == "join":
        region["region_index"] = region[join_by]
        region.index = region["region_index"]
        manhole_join_by = "ORIG_" + join_by
        if manhole_join_by not in _manholes.columns:
            raise NotImplementedError(
                f"Join_by {join_by} is not accepted. Only accepts branch index column name."
            )
        else:
            manholes_ = _manholes.assign(
                region_index=_manholes[manhole_join_by].str.split(";")
            )
            manholes_ = gpd.GeoDataFrame(
                pd.DataFrame(manholes_).explode("region_index", ignore_index=True)
            )
            manholes_.index = manholes_[_manholes.index.name]
    else:
        raise NotImplementedError(
            f"how: {how} is not accepted. Only accepts intersection/join."
        )

    # obtain subcatchment by looping through region
    i = 0
    subcatchments = gpd.GeoDataFrame()
    for region_index, subregion in region.iterrows():
        i += 1
        print(f"{i}/{len(region)} subcatchments developed")

        # get the subregion and manholes within subregion
        if not isinstance(subregion, gpd.GeoDataFrame):

            subregion = gpd.GeoDataFrame(
                subregion.to_frame().T, geometry="geometry", crs=region.crs
            )
            subregion = subregion.buffer(
                0
            )  # applying a zero distance buffer can fix the invalid geometries
            subregion = gpd.GeoDataFrame(
                subregion.to_frame().rename(columns={0: "geometry"}),
                geometry="geometry",
                crs=region.crs,
            )

        submanholes = gpd.GeoDataFrame(
            manholes_.loc[manholes_["region_index"] == region_index, :],
            geometry="geometry",
            crs=_manholes.crs,
        ).drop_duplicates()

        # if multiple manholes
        if len(submanholes) > 0:

            if len(submanholes) <= 1:
                # use subregion entirely for only 1 point
                _subcatchments = gpd.GeoDataFrame(
                    geometry=subregion.geometry.to_list(), index=submanholes.index
                )
            else:
                # apply thiessen polygon for more than 2 points
                try:
                    _subcatchments = get_thiessen_polygons(
                        submanholes, subregion, barriers=None, logger=logger
                    )
                    _subcatchments = gpd.GeoDataFrame(
                        {
                            id_col: [
                                "subcat_%s" % (i) for i in range(len(_subcatchments))
                            ],
                            "geometry": _subcatchments.geometry,
                        }
                    )
                except:
                    logger.warn(
                        f"Subcatchment {subregion.index.to_list()} could not be used thus skipped. "
                    )
                    continue

            # combine all
            subcatchments = pd.concat([subcatchments, _subcatchments])

        else:

            logger.warn(f"Subcatchments {subregion.index.to_list()} have no manholes. ")

    subcatchments.crs = region.crs
    subcatchments = subcatchments.loc[~subcatchments.index.isna(), :]
    # subcatchments = subcatchments.explode() # explode multi polygon
    if how == "intersection":
        subcatchments_ = gpd.sjoin(
            subcatchments, manholes_, how="inner", op="intersects"
        )
        # link the remaining ones using nearest distance
        leftover_subcatchments = set(subcatchments.index) - set(subcatchments_.index)
        if len(leftover_subcatchments) > 0:
            leftover_subcatchments_ = subcatchments.loc[list(leftover_subcatchments), :]
            leftover_subcatchments_ = __link_subcatchments(
                leftover_subcatchments_, manholes_, barrier, snap_offset, logger=logger
            )
            leftover_subcatchments_[id_col] = [
                f"subcat_{i}"
                for i in range(
                    len(subcatchments_),
                    len(subcatchments_) + len(leftover_subcatchments_),
                )
            ]
            leftover_subcatchments_[manholes.index.name] = leftover_subcatchments_[
                "TO_INDEX"
            ]
            subcatchments_ = pd.concat(
                [subcatchments_, leftover_subcatchments_]
            ).drop_duplicates(keep=False)
            logger.info(
                "Isolated subcatchments are linked to manholes circumventing barriers. "
            )

    elif how == "join":
        subcatchments_ = subcatchments.copy()
        subcatchments_[manholes.index.name] = subcatchments_.index

    subcatchments_.reset_index(drop=True, inplace=True)
    subcatchments_[id_col] = [f"subcat_{i}" for i in range(len(subcatchments_))]
    logger.info("Subcatchments are delineated within region. ")

    # dissolve
    subcatchments_ = subcatchments_.dissolve(by=manholes.index.name)
    subcatchments_[manholes.index.name] = subcatchments_.index
    subcatchments_.index = subcatchments_[id_col]
    logger.info("Subcatchments are linked to manholes. ")

    # check area
    if (
        region.area.sum() * 99.8 / 100 > subcatchments_.area.sum()
    ):  # allowed error due to precision
        logging.warning(
            "Total subcatchments area lower than the specified region area. Please check intermediate exports. "
        )

    return subcatchments_


def _fill_subcatchments(
    subcatchments: gpd.GeoDataFrame,
    subcatchments_ini: configparser.ConfigParser,
    landuse_fn: str = None,
    logger=logging,
):
    """sill subcatchments based on subcatchments_ini"""

    _subcatchments = subcatchments.copy()

    method = subcatchments_ini["global"]["fill_method"]
    logger.info(f"Filling subcatchments using method {method}")

    # fill in default anyway
    subcatchments = __fill_subcatchments_with_defaults(
        subcatchments, subcatchments_ini, logger=logger
    )

    if method == "default":
        pass

    # overwrite defaults with landuse
    elif method == "landuse":
        subcatchments = __fill_subcatchments_with_landuse(
            subcatchments, landuse_fn, logger=logger
        )

    else:
        raise NotImplementedError(
            "method not recongnised. Please use one of the following: thiessen, equal"
        )

    if subcatchments is not None:
        logger.info("subcatchments is setup")

    return subcatchments


def __fill_subcatchments_with_defaults(
    subcatchments: gpd.GeoDataFrame,
    subcatchments_ini: configparser.ConfigParser,
    logger=logging,
):
    """fill subcatchments attributes using fill method (overwrite exisitng columns)"""

    # fill in defaults
    subcatchments = append_data_columns_based_on_ini_query(
        subcatchments, subcatchments_ini
    )

    # compute landuse categories
    if "forced_area" in subcatchments.columns:
        subcatchments["area"] = subcatchments["forced_area"]
    else:
        subcatchments["area"] = subcatchments.geometry.area

    # compute landuse type
    for landuse_col in ["cl", "op", "rf", "up"]:
        landuse_area = subcatchments["area"] * subcatchments[f"{landuse_col}_ratio"]

        # compute runoff type
        for runoff_col in ["flat", "stretch", "slope"]:
            subcatchments[f"{landuse_col}_{runoff_col}"] = (
                landuse_area * subcatchments[f"{runoff_col}_ratio"]
            )

    return subcatchments


def __fill_subcatchments_with_landuse(
    subcatchments: gpd.GeoDataFrame, landuse_fn: str, logger=logging
):
    """function to fill in nwrw catchments suing landuse classes
    (1. Closed paved; 2. Open paved; 3. Roof;4. Unpaved)
    """

    if landuse_fn is None:

        logger.error(
            f"Can not setup subcatchment from landuse raster. no landuse_fn is specified."
        )
        pass

    else:

        logger.info(
            f"Filling in subcatchment based on {landuse_fn}... might take a minute"
        )
        landuse_df = None

        try:

            # read raster
            landuse_da = read_raster(
                raster_fn=landuse_fn, id_col="landuse", nodata=-999
            )
            logger.debug("read landuse raster (nodata is recongnised as -999)")

            # parse raster to nwrw classes
            logger.debug(
                "parsing landuse classes (1. Closed paved; 2. Open paved; 3. Roof;4. Unpaved)"
            )
            landuse_da[f"cl"] = landuse_da["landuse"] == 1
            landuse_da[f"op"] = landuse_da["landuse"] == 2
            landuse_da[f"rf"] = landuse_da["landuse"] == 3
            landuse_da[f"up"] = landuse_da["landuse"] == 4
            landuse_da = landuse_da.drop_vars("landuse")

            # calculate states
            logger.debug("computing landuse area (m2) ")
            landuse_df = landuse_da.raster.zonal_stats(
                subcatchments, stats="sum"
            ).to_pandas()
            landuse_df = landuse_df.multiply(np.prod(landuse_da.raster.res))
            landuse_df = landuse_df.rename(
                columns={"cl_sum": "cl", "op_sum": "op", "rf_sum": "rf", "up_sum": "up"}
            )
            landuse_df.drop(columns="spatial_ref", inplace=True)

        except Exception as e:
            logger.error(e)

        if landuse_df is not None:

            for lu in ["cl", "op", "rf", "up"]:
                for rf in ["flat", "stretch", "slope"]:
                    subcatchments[f"{lu}_{rf}"] = (
                        landuse_df[lu] * subcatchments[f"{rf}_ratio"]
                    )

            precision_loss = 1 - landuse_df.sum().sum() / subcatchments.area.sum()
            if precision_loss > 0.01:
                logger.warning(f"area loss: {precision_loss} greater than 1%")

    logger.info("subcatchments landuse are set up.")

    return subcatchments


def __link_subcatchments(
    subcatchments: gpd.GeoDataFrame,
    manholes: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame = None,
    maxdist: float = float("inf"),
    max_area: float = float("inf"),
    ignore_disconnected: bool = True,
    logger=logging,
):
    """link subcatchments to manholes"""
    logger.debug("Link subcatchments to manholes within maximum distance.")
    if maxdist is None:
        maxdist = np.inf
        logger.debug("Max distance lis not specified. Will use inf. ")
    elif maxdist < 10:
        maxdist = 10
        logger.debug(
            "Max distance less than 10 meters is not recommended. Will be overruled by 10 m."
        )
    links = link_polygons_to_points(subcatchments, manholes, barriers, maxdist, logger)
    # assign manhole to subcatchments
    links.index = links["FROM_INDEX"]
    col_name = manholes.index.name
    subcatchments = subcatchments.join(links, rsuffix="_SUBCAT_TO_MANHOLE")
    # ignore disconnected
    if ignore_disconnected:
        logger.debug("Subcatchments obstructed by barrier are removed.")
        subcatchments = subcatchments.dropna(subset=["FROM_INDEX"])

    # make index type consisitent
    subcatchments["FROM_INDEX"] = subcatchments["FROM_INDEX"].astype(
        type(links["FROM_INDEX"][0])
    )
    subcatchments["TO_INDEX"] = subcatchments["TO_INDEX"].astype(
        type(links["TO_INDEX"][0])
    )

    # TODO BMA: afterwards, do we need to split based on a max_area?
    return subcatchments


def setup_precipitation(
    precipitation_fn: str = None,
    region_fn: str = None,
    id_col: str = None,
    rename_map: dict = None,
    required_columns: list = None,
    required_dtypes: list = None,
    forced_upon: str = "drr",
    subcatchments: gpd.GeoDataFrame = None,
    logger=logging,
):
    """setup precipitation as dataframe"""

    if not forced_upon == "drr":
        raise NotImplementedError(
            "Can not setup precipitation. Only support forced_upon = drr"
        )
    else:
        logger.info("setting up precipitation for drr")

    # setup default global precipitation
    if subcatchments is None:
        logger.error(
            "Can not setup precipitation. Subcatchments do not exist for setting up precipitation"
        )
    else:
        x1, y1, x2, y2 = subcatchments.unary_union.bounds
        precipitation_region = gpd.GeoDataFrame(
            {id_col: ["global"], "geometry": [box(x1, y1, x2, y2)]}
        )

    # read forcing regions
    if region_fn is not None:
        precipitation_region_ = read_gpd(
            region_fn, id_col, rename_map, required_columns, required_dtypes
        )
        if precipitation_region_.geom_type[0] == "Polygon":
            logging.info(
                f"precipitation is applied within polygon regions specified in: {region_fn}"
            )
        elif precipitation_region_.geom_type[0] == "Point":
            logging.info(
                f"precipitation is applied at station specified in: {region_fn}"
            )
            precipitation_region_ = get_thiessen_polygons(
                precipitation_region_,
                precipitation_region,
                barriers=None,
                logger=logger,
            )
    else:
        logging.info(
            f"precipitation is applied globally within the entire subcatchments domain"
        )
        precipitation_region_ = precipitation_region

    # TODO move to preprocess or read_gpd
    # remove duplicated geometry
    _precipitation_region_ = precipitation_region_.drop_duplicates()
    G = _precipitation_region_["geometry"].apply(lambda geom: geom.wkb)
    n = len(G) - len(G.drop_duplicates().index)
    precipitation_region_ = _precipitation_region_[
        _precipitation_region_.index.isin(G.drop_duplicates().index)
    ]
    logger.debug(f"Removing {n} sub-regions which have duplicated geometry.")

    # revise meteo_id in subcatchments based on precipitation_region
    # subcatchments_ = gpd.sjoin(subcatchments, precipitation_region_, how="inner", op='intersects')
    subcatchments_ = gpd.overlay(subcatchments, precipitation_region_)
    subcatchments_wo_meteo = subcatchments_.loc[subcatchments_["METEO_ID"].isna(), :]
    if len(subcatchments_wo_meteo) > 0:
        logger.warning(
            f"{len(subcatchments_wo_meteo)} subcatchments without meteo_id identified. These subcatchments will not be used."
            + f"To use all subcatchments, please make sure region_fn under precipitation covers all subcatchment area."
        )
        subcatchments_ = subcatchments_.loc[~subcatchments_["METEO_ID"].isna(), :]
        # TODO: link to precipitation_region using nearest distance, or change to use thiessen polygon as precipitation_region
    subcatchments_["meteo_id"] = subcatchments_["METEO_ID"]

    # read forcing time series
    precipitation = None
    if precipitation_fn is not None:
        precipitation = pd.read_csv(precipitation_fn, parse_dates=True, index_col=[0])
    else:
        logging.warning("no precipitation is specified")

    if len(precipitation.columns) == 1 and "global" in precipitation.columns[0].lower():

        logger.info("Broadcase global precipitation to all subcatchments")
        # apply global precipitation to all stations
        precipitation_ = pd.concat(
            [precipitation] * len(subcatchments_), axis=1, ignore_index=True
        )
        precipitation_.columns = subcatchments_.meteo_id

    elif precipitation.columns != subcatchments_.meteo_id.drop_duplicates().to_list():
        logger.error("precipitation time series do not match subcatchment meteo_ids")
        precipitation_ = precipitation
        raise ValueError

    else:
        logger.info("1-to-1 mapping of precipitation time series to subcatchments")

    logger.info("Precipitation is set up.")
    return precipitation_, precipitation_region_, subcatchments_


# ===============================================================
#                  model
# ===============================================================


def setup_dm(staticgeoms: dict = None, logger=logging):

    """
    setup datamodel, perform staticgeoms (shp + scripts) naming convention --> deflt3dfmpy naming convension
    """

    # initialise datamodel
    region = staticgeoms.pop("region", {})
    dm = datamodel()
    # then loop over and add other layers
    for layer_name, _layer in staticgeoms.items():
        # setup based on mapping
        layer = setup_datamodel(_layer, layer_name)
        setattr(dm, layer_name, layer)

    return dm


def setup_dflowfm(
    dm,
    mdu_fn=None,
    model_type: str = "1d",
    network1d_ini_fn: str = None,
    grid_2d_ini_fn: str = None,
    grid_polygon_fn: str = None,
    grid_fn: str = None,
    links1d2d_ini_fn: str = None,
    bedlevel_fn: str = None,
    roughness_fn: str = None,
    dflowfm_path: str = None,  # FIXME: use GUI path and derive relative path to dflowfm-cli.exe (so that later we could do the same for dimr)
    logger=logging,
):
    """
    setup dflowfm
    """

    logger.info(f"An {model_type} D-Flow FM model is generated")
    # initialise dflowfm model
    dfmmodel = DFlowFMModel()

    if mdu_fn is not None:
        dfmmodel.mdufile = mdu_fn
        logging.info(f"model config template read from {mdu_fn}")

    # generate 1D network including cross-section
    if model_type == "1d" or model_type == "1d2d":
        logger.info(f"The 1D network is added to the D-Flow FM model")

        # parse ini
        network1d_ini = parse_ini(network1d_ini_fn)

        # add branch geometry
        if "branches" not in vars(dm).keys():
            raise ValueError("DFlowFM 1D model must have 1D branches.")

        dfmmodel.network.set_branches(dm.__getattribute__("branches"), id_col="id")

        logger.info(f"Add cross-section to D-Flow FM model")
        # add cross-sections
        if {"crsdefs", "crslocs"}.issubset(set(vars(dm).keys())):
            dfmmodel.crosssections.io.from_datamodel(
                dm.__getattribute__("crsdefs"),
                dm.__getattribute__("crslocs"),
            )

        # add structures
        if "bridges" in vars(dm).keys():
            dfmmodel.structures.io.bridges_from_datamodel(
                dm.__getattribute__("bridges")
            )

        if "gates" in vars(dm).keys():
            dfmmodel.structures.io.gates_from_datamodel(dm.__getattribute__("gates"))

        if "pumps" in vars(dm).keys():
            dfmmodel.structures.io.pumps_from_datamodel(dm.__getattribute__("pumps"))

        if "culverts" in vars(dm).keys():
            dfmmodel.structures.io.culverts_from_datamodel(
                dm.__getattribute__("culverts")
            )

        if "compounds" in vars(dm).keys():
            dfmmodel.structures.io.compounds_from_datamodel(
                dm.__getattribute__("compounds")
            )

        # generate computational grid. required before manholes.
        dfmmodel.network.generate_1dnetwork(
            one_d_mesh_distance=network1d_ini["global"]["one_d_mesh_distance"],
            seperate_structures=network1d_ini["global"][
                "seperate_structures"
            ],  # FIXME: will seperate structures nonetheless, because the argument is not checked
            urban_branches=dm.__getattribute__("branches")
            .query("branchType == 2")
            .index.to_list(),
        )

        # add manholes as storage nodes (manholes need to be added after 1D grids)
        if "manholes" in vars(dm).keys():
            if model_type == "1d":
                dm.__getattribute__("manholes").useStreetStorage = 1
            elif model_type == "1d2d":
                dm.__getattribute__("manholes").useStreetStorage = 0
            dfmmodel.storage_nodes.add_manholes(dm.__getattribute__("manholes"))

        # add laterals at manholes
        # FIXME: setup lateral seperately, use configuration control discharge
        if "manholes" in vars(dm).keys():
            dfmmodel.external_forcings.add_lateral_at_manholes(
                dfmmodel.storage_nodes, discharge="realtime"
            )

    # generate 2D grid or add an existing 2D grid to model
    if model_type == "2d" or model_type == "1d2d":
        logger.info(f"Start of adding 2D to D-Flow FM model.")

        # Add or generate grid
        if grid_fn is None and grid_2d_ini_fn is None:
            raise ValueError(
                f"Model type {model_type} is specified but no 2D grid is generated or specified."
            )

        elif grid_fn is not None and grid_2d_ini_fn is not None:
            raise ValueError(
                f"Both grid generation option is selected and an 2D grid ({grid_fn}) is specified. Choose one of the options."
            )

        elif grid_fn is not None:
            # FIXME: (1) GUI uses different capital letters for netCDF variables then specified in UM. This will give problems for model generation.
            # FIXME: (2) The code uncommented is a step in the right direction.
            # _mesh = UgridReader(dfmmodel.network)
            # mesh = _mesh.read_ugrid_2d(grid_fn)
            raise ValueError(
                f"Use of an existing 2D grid is not yet implemented in the code"
            )

        else:
            logger.info(f"An 2D grid is generated")
            # parse ini
            grid_2d_ini = parse_ini(grid_2d_ini_fn)

            # generate grid based on polygon
            if grid_2d_ini["global"]["generate_grid"] == "polygon":

                # check polygon is present
                if grid_polygon_fn is None:
                    raise ValueError(
                        f"Grid generation based on polygon is selected but no polygon is provided "
                        f"(grid_polygon_fn)"
                    )

                # generate grid based on shape
                if grid_2d_ini["global"]["shape_cell"] == "square":
                    mesh = Rectangular()
                    clipgdf = gpd.read_file(grid_polygon_fn)
                    clipgeo = clipgdf.unary_union
                    mesh.generate_within_polygon(
                        polygon=clipgeo,
                        cellsize=grid_2d_ini["global"]["cellsize"],
                        rotation=grid_2d_ini["global"]["rotation"],
                    )
                    logger.info(
                        f"An {grid_2d_ini['global']['shape_cell']} grid with resolution of {grid_2d_ini['global']['cellsize']} within a polygon is created"
                    )

                elif grid_2d_ini["global"]["shape_cell"] == "triangle":
                    # TODO: implement grid generation with triangle cells
                    raise ValueError(
                        f"Use of triangular cells is not yet implemented in the code"
                    )
                else:
                    raise ValueError(
                        f"{grid_2d_ini['default']['shape_cell']} is not an existing option for the shape of "
                        f"an 2D grid cell. Please use rectangular or triangle"
                    )

            # generate grid via bounding box
            elif grid_2d_ini["global"]["generate_grid"] == "bounding box":
                # TODO: implement bounding box option for grid generation.
                raise ValueError(
                    f"Bounding box method  is not yet implemented in the code"
                )
            else:
                raise ValueError(
                    f"An unknwon grid generation method is specified. Please select grid generation based "
                    f"on polygon or bounding box"
                )

            if grid_2d_ini["global"]["refine_grid"] is not None:
                # TODO: Add different ways of refinement. Determine if this option should also be available for grids that are added.
                raise ValueError(f"Refinement of 2D grids is not yet implemented")

            # TODO: Add clipping of 2D grid to text. Determine if this option should also be available for grids that are added.

        # Add bedlevels to 2d grid
        if "bedlevel" not in grid_2d_ini["default"]:
            raise ValueError(
                f"No default bed level is specified in {grid_2d_ini_fn} under [default]. "
                f"Please specify a bedlevel = ... "
            )

        # Set uniform bedlevels to -999.0 in case nothing is specified
        if grid_2d_ini["default"]["bedlevel"] is None:
            logger.info(
                f"No default bed level is specified, so it will be set to -999.0"
            )
            grid_2d_ini["default"]["bedlevel"] = -999.0

        if bedlevel_fn is not None and grid_fn is not None:
            raise ValueError(
                f"The model generator cannot yet handle the adding of 2D grids with Z-values (bed levels)"
            )
            # TODO: Please add option to delete z-values and add new ones.
            # TODO: Add warning that old z-values will replaced by new ones. If the user does not want this, then now bedlevel_type should be specified.

        elif bedlevel_fn is not None:
            if bedlevel_fn.endswith(".pol"):
                raise ValueError(
                    f"Setting of 2D bedlevels via polygons is not yet implemented "
                )
            else:
                # FIXME: check if  missing should be -999.0 or default bedlevel. Discuss with Xiaohan
                mesh.altitude_from_raster(
                    bedlevel_fn,
                    where="face",
                    stat=grid_2d_ini["global"]["bedlevel_averagingType"],
                    missing=grid_2d_ini["default"]["bedlevel"],
                )
                logger.info(
                    f"Bed level file ({bedlevel_fn}) is interpolated on the grid using "
                    f"{grid_2d_ini['global']['bedlevel_averagingType']} averagingType"
                )
        else:
            logger.info(
                f"No bedlevel file is specified, so default bed level of {grid_2d_ini['default']['bedlevel']}"
                f" m AD will be uniformly applied by setting it in the MDU-file"
            )

        # Set default bed level in MDU-file
        logger.info(
            f"Uniform bed level of {grid_2d_ini['default']['bedlevel']} m AD is set in the MDU-file."
        )
        dfmmodel.mdu_parameters["BedlevUni"] = grid_2d_ini["default"]["bedlevel"]

        # Add grid to D-Flow FM model schematisation
        dfmmodel.network.add_mesh2d(mesh)

        # Add roughness to 2D

        # Give error message if no roughness value or attribute is not specified.
        if "roughness" not in grid_2d_ini["default"]:
            raise ValueError(
                f"No default roughness value is specified in {grid_2d_ini_fn} under [default]. "
                f"Please specify a roughness = ... under [default] "
            )
        # Give error message if no roughness_type value or attribute is specified.
        if "roughness_type" not in grid_2d_ini["global"]:
            raise ValueError(
                f"No roughness_type is specified for 2D. Please specify roughness_type in {grid_2d_ini_fn} under [global]. "
                f"Choose between Chezy, Manning, WallLawNikuradse or WhiteColebrook."
            )

        # Give warning for no default roughness
        if grid_2d_ini["default"]["roughness"] is None:
            logger.info(
                f"No default roughness is specified, so it will be set to -999.0"
            )
            grid_2d_ini["default"]["roughness"] = -999.0
        # Give error message if roughness type is not specified
        if grid_2d_ini["global"]["roughness_type"] is None:
            raise ValueError(
                f"No roughness_type is specified for 2D. Please specify roughness_type in {grid_2d_ini_fn} under [global]. "
                f"Choose between Chezy, Manning, WallLawNikuradse or WhiteColebrook."
            )

        if roughness_fn is not None:
            raise ValueError(
                f"Use of roughness layer is not implemented in the code. Only an uniform roughness can be applied"
            )
            # TODO: search for interpolation settings in ini-file
            # TODO: Create function to rewrite shapefiles to polygon files
            # TODO: create function initial and parameter field
            # TODO: required and optional options
            # TODO: location type, only roughness can be applied to 2D.

        if isinstance(grid_2d_ini["global"]["roughness_type"], str):
            roughness_type = _rewrite_roughness_type(
                grid_2d_ini["global"]["roughness_type"]
            )
            logger.info(
                f"The roughness type is rewritten from {grid_2d_ini['global']['roughness_type']} to {roughness_type}."
            )
        elif isinstance(grid_2d_ini["global"]["roughness_type"], int):
            roughness_type = grid_2d_ini["global"]["roughness_type"]
        elif isinstance(grid_2d_ini["global"]["roughness_type"], float):
            roughness_type = int(grid_2d_ini["global"]["roughness_type"])
        else:
            raise ValueError(
                f"Specified roughness_type in {grid_2d_ini_fn} under [global] is incorrect."
                f"Choose between Chezy, Manning, WallLawNikuradse or WhiteColebrook."
            )

        # Set default bed level in MDU-file
        logger.info(
            f"Uniform roughness of {grid_2d_ini['default']['roughness']}  is set in the MDU-file."
        )
        dfmmodel.mdu_parameters["UniFrictCoef"] = grid_2d_ini["default"]["roughness"]
        logger.info(
            f"Uniform roughness of {grid_2d_ini['global']['roughness_type']}  is set in the MDU-file."
        )
        dfmmodel.mdu_parameters["UnifFrictType"] = roughness_type

    # FIXME: The following part will be committed as soon as networkfile can be opened in GUI.
    # Generate 1d2d links for 1d2d models
    if model_type == "1d2d":
        print("1d2d")
        # dfmmodel.network.links1d2d.generate_2d_to_1d(max_distance=50)
        dfmmodel.network.links1d2d.generate_1d_to_2d(
            max_distance=50,
            branchid=dm.__getattribute__("branches")
            .query("branchType == 2")
            .index.to_list(),
        )

        # if links1d2d_ini_fn is None:
        #     raise ValueError(f'Modeltype {model_type} is specified but no information is given for 1D2D links generation')

        # parse ini
        # links1d2d_ini = parse_ini(links1d2d_ini_fn)

        # determine if 1D2D links should be generated for specific branches
        # if links1d2d_ini['']

        # if links1d2d_ini['global']['generate_1d2d_links'] == '1-to-1':
        #
        #     dfmmodel.network.links1d2d.generate_1d_to_2d(max_distance=50)
        # else:
        #     # TODO: Method for 1-to-n link generation is available but not available for users.
        #     raise ValueError(f'1-to-n 1d2s link generation is not yet implemented in code.')

        # # Plotting
        # import matplotlib.pyplot as plt
        # from matplotlib.collections import LineCollection
        # fig, ax = plt.subplots(figsize=(13, 10))
        # ax.set_aspect(1.0)
        #
        # segments = dfmmodel.network.mesh2d.get_segments()
        # ax.add_collection(LineCollection(segments, color='0.3', linewidths=0.5, label='2D-mesh'))

    # add forcing
    # add boundaries
    if "boundaries" in vars(dm).keys():
        dfmmodel.external_forcings.io.from_datamodel(dm.__getattribute__("boundaries"))
    # FIXME: add laterals here

    # FIXME: no fixed weirs yet because they apply to 2D model
    # dfmmodel.structures.io.fixedweirs_from_datamodel(datamodel.fixedweirs)

    # FIXME adjust mdu parameters
    # TODO
    dfmmodel.mdu_parameters["refdate"] = 20200101
    dfmmodel.mdu_parameters["TStop"] = 7200
    dfmmodel.mdu_parameters["Wrimap_flow_analysis"] = "0"

    # FIXME no adjust mdu parameters yet because they are setup separately

    return dfmmodel


def _rewrite_roughness_type(roughness_type):
    if roughness_type == "Chezy":
        roughness_type_value = 0
    elif roughness_type == "Manning":
        roughness_type_value = 1
    elif roughness_type == "WallLawNikuradse":
        roughness_type_value = 2
    elif roughness_type == "WhiteColebrook":
        roughness_type_value = 3
    else:
        raise ValueError(
            f"Specified roughness_type in {grid_2d_ini_fn} under [global] is incorrect."
            f"Choose between Chezy, Manning, WallLawNikuradse or WhiteColebrook."
        )
    return roughness_type_value


def setup_drr(dm, dfmmodel, rr_concept: str = "nwrw", logger=logging):
    """
    setup drr, dependent on dflowfmmodel, only for nwrw concept
    """

    # initialise drr model
    drrmodel = DFlowRRModel()

    # add precip
    if "precipitation" in vars(dm).keys():
        drrmodel.external_forcings.io.precip_from_datamodel(
            dm.__getattribute__("precipitation")
        )

    # for nwrw concept
    if rr_concept == "nwrw":
        # must-haves for nwrw concept

        # add subcatchments
        if "subcatchments" in vars(dm).keys():
            drrmodel.nwrw.io.nwrw_from_datamodel(
                catchments=dm.__getattribute__("subcatchments")
            )

        # update dfm model laterals (remove and overwrite as real time at only subcatchments locations
        old_laterals = dfmmodel.external_forcings.laterals
        old_laterals.at[:, "discharge"] = "realtime"
        new_laterals = old_laterals.loc[
            old_laterals.id.isin(
                dm.__getattribute__("subcatchments").ManholeId.tolist()
            )
        ]
        dfmmodel.external_forcings.laterals = new_laterals

    # change rr parameterds
    # TODO
    drrmodel.d3b_parameters = {
        "TimeSettings": {
            "TimestepSize": 60,
            "StartTime": "'2020/01/01;00:00:00'",
            "EndTime": "'2020/01/01;02:00:00'",
        }
    }

    return drrmodel, dfmmodel


# ===============================================================
#                  writers
# ===============================================================


def write_dfmmodel(dfmmodel, output_dir, name: str = "DFLOWFM", logger=logging):

    """write dflowfm model to output_dir"""

    if name != "DFLOWFM":
        raise NotImplementedError("Do not support customised name.")

    fm_writer = DFlowFMWriter(dfmmodel, output_dir=output_dir, name=name)
    fm_writer.write_all()
    logger.info(f"Write dflowfm model to {str(output_dir)} with name as {name}")


def write_drrmodel(drrmodel, output_dir, name: str = "DRR", logger=logging):

    """write drr model to output_dir"""

    if name != "DRR":
        raise NotImplementedError("Do not support customised name.")

    rr_writer = DFlowRRWriter(drrmodel, output_dir=output_dir, name=name)
    rr_writer.copyRRFiles()
    rr_writer.update_config()
    rr_writer.write_meteo()
    rr_writer.write_nwrw()
    rr_writer.write_topology()
    rr_writer.write_dimr()
    logger.info(f"Write drr model to {str(output_dir)} with name as {name}")


# ===============================================================
#                  validation
# ===============================================================


def setup_validation(
    branches: gpd.GeoDataFrame,
    validate_1dnetwork=True,
    plotit=True,
    exportpath=os.getcwd(),
    logger=logging,
):
    """ "Function to validate model, only 1dnetwork is implemented"""
    if validate_1dnetwork:
        validate_1dnetwork_connectivity(
            branches, plotit=plotit, exportpath=exportpath, logger=logger
        )
        validate_1dnetwork_flowpath(
            branches,
            branchType_col="branchType",
            plotit=plotit,
            exportpath=exportpath,
            logger=logger,
        )
    return None


def random_color():
    return tuple(
        [
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
        ]
    )


def gpd_to_digraph(branches: gpd.GeoDataFrame) -> nx.DiGraph():

    branches["from_node"] = [
        row.geometry.coords[0] for index, row in branches.iterrows()
    ]
    branches["to_node"] = [
        row.geometry.coords[-1] for index, row in branches.iterrows()
    ]
    G = nx.from_pandas_edgelist(
        branches,
        source="from_node",
        target="to_node",
        create_using=nx.DiGraph,
        edge_attr=True,
    )
    return G


def validate_branch_flowpath(branches: gpd.GeoDataFrame, ax=None):
    """function to validate flowpath (flowpath to outlet) connectivity for a given branch"""
    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create graph
    G = gpd_to_digraph(branches)

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # add basemap
    branches.plot(alpha=0.1, edgecolor="k", ax=ax)
    ctx.add_basemap(
        source="http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}",
        alpha=0.5,
        crs=branches.crs.to_epsg(),
        ax=ax,
    )

    # end points for flowpath
    network_outlets = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(
        f"{len(network_outlets)} branch outlet identified "
        + f"(The branches have {len(network_outlets)} fully connected components). "
        + f"Drawing a flow path delineation plot."
    )

    # make plot
    pos = {xy: xy for xy in G.nodes()}
    RG = G.reverse()
    for outlet in network_outlets:
        c = random_color()
        outlet_G = G.subgraph(
            list(dict(nx.bfs_predecessors(RG, outlet)).keys()) + [outlet]
        )
        nx.draw_networkx(
            outlet_G,
            pos,
            node_size=50,
            node_color=[c],
            width=2,
            edge_color=[c],
            with_labels=False,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            outlet_G,
            pos,
            nodelist=[outlet],
            node_size=200,
            node_color="none",
            edgecolors=c,
            ax=ax,
        )
    return network_outlets
