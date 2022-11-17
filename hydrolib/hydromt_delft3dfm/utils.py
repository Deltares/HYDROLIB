from pathlib import Path
from typing import Dict, Union

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np

from hydrolib.core.io.dflowfm.mdu.models import FMModel
from hydrolib.core.io.dflowfm.friction.models import FrictionModel
from hydrolib.core.io.dflowfm.crosssection.models import CrossDefModel, CrossLocModel
from hydrolib.core.io.dflowfm.storagenode.models import StorageNodeModel
from hydrolib.core.io.dflowfm.ext.models import ExtModel, Boundary
from hydrolib.core.io.dflowfm.bc.models import ForcingModel
from hydrolib.core.io.dflowfm.gui.models import BranchModel

from .workflows import helper

__all__ = ["get_process_geodataframe"]


def write_branches_gui(gdf: gpd.GeoDataFrame, fm_model: FMModel) -> None:
    """
    write branches.gui file from branches geodataframe

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        gdf containing the branches
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib
    """
    branches = gdf[["branchId", "branchType", "manhole_up", "manhole_dn"]]
    branches = branches.rename(
        columns={
            "branchId": "name",
            "manhole_up": "sourceCompartmentName",
            "manhole_dn": "targetCompartmentName",
        }
    )
    branches["branchType"] = branches["branchType"].replace(
        {"river": 0, "channel": 0, "pipe": 2, "tunnel": 2, "sewerconnection": 1}
    )
    branchgui_model = BranchModel(branch=branches.to_dict("records"))
    branchgui_model.filepath = fm_model.filepath.with_name(
        branchgui_model._filename() + branchgui_model._ext()
    )
    branchgui_model.save()


def write_friction(gdf: gpd.GeoDataFrame, fm_model: FMModel) -> FMModel:
    """
    write friction files from branches geodataframe

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        gdf containing the branches
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib

    Returns
    -------
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib with the link to the written friction files.
    """
    frictions = gdf[["frictionId", "frictionValue", "frictionType"]]
    frictions = frictions.drop_duplicates(subset="frictionId")

    fm_model.geometry.frictfile = []
    # create a new friction
    for i, row in frictions.iterrows():
        fric_model = FrictionModel(global_=row.to_dict())
        fric_filename = f"{fric_model._filename()}_{i}" + fric_model._ext()
        fric_model.filepath = fm_model.filepath.with_name(fric_filename)
        fric_model.save(fric_model.filepath, recurse=True)

        # save relative path to mdu
        fric_model.filepath = fric_filename
        fm_model.geometry.frictfile.append(fric_model)

    return fm_model


def write_crosssections(gdf: gpd.GeoDataFrame, fm_model: FMModel) -> FMModel:
    """write crosssections into hydrolib-core crsloc and crsdef objects

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        gdf containing the crosssections
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib

    Returns
    -------
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib with the link to the written cross-sections files.
    """
    # crsdef
    # get crsdef from crosssections gpd # FIXME: change this for update case
    gpd_crsdef = gdf[[c for c in gdf.columns if c.startswith("crsdef")]]
    gpd_crsdef = gpd_crsdef.rename(
        columns={c: c.split("_")[1] for c in gpd_crsdef.columns}
    )
    gpd_crsdef = gpd_crsdef.drop_duplicates(subset="id")
    crsdef = CrossDefModel(definition=gpd_crsdef.to_dict("records"))
    fm_model.geometry.crossdeffile = crsdef

    crsdef_filename = crsdef._filename() + ".ini"
    fm_model.geometry.crossdeffile.save(
        fm_model.filepath.with_name(crsdef_filename),
        recurse=False,
    )
    # save relative path to mdu
    fm_model.geometry.crossdeffile.filepath = crsdef_filename

    # crsloc
    # get crsloc from crosssections gpd # FIXME: change this for update case
    gpd_crsloc = gdf[[c for c in gdf.columns if c.startswith("crsloc")]]
    gpd_crsloc = gpd_crsloc.rename(
        columns={c: c.split("_")[1] for c in gpd_crsloc.columns}
    )

    crsloc = CrossLocModel(crosssection=gpd_crsloc.to_dict("records"))
    fm_model.geometry.crosslocfile = crsloc

    crsloc_filename = crsloc._filename() + ".ini"
    fm_model.geometry.crosslocfile.save(
        fm_model.filepath.with_name(crsloc_filename),
        recurse=False,
    )
    # save relative path to mdu
    fm_model.geometry.crosslocfile.filepath = crsloc_filename

    return fm_model


def write_manholes(gdf: gpd.GeoDataFrame, fm_model: FMModel) -> FMModel:
    """
    write manholes into hydrolib-core storage nodes objects

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        gdf containing the manholes
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib

    Returns
    -------
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib with the link to the written storage nodes files.
    """
    storagenodes = StorageNodeModel(storagenode=gdf.to_dict("records"))
    fm_model.geometry.storagenodefile = storagenodes

    storagenodes_filename = storagenodes._filename() + ".ini"
    fm_model.geometry.storagenodefile.save(
        fm_model.filepath.with_name(storagenodes_filename),
        recurse=False,
    )
    # save relative path to mdu
    fm_model.geometry.storagenodefile.filepath = storagenodes_filename

    return fm_model


def read_1dboundary(
    df: pd.DataFrame, quantity: str, nodes: gpd.GeoDataFrame
) -> xr.DataArray:
    """
    Read for a specific quantity the corresponding external and forcing files and parse to xarray

    Parameters
    ----------
    df: pd.DataFrame
        External Model DataFrame filtered for quantity.
    quantity: str
        Name of quantity (eg 'waterlevel').
    nodes: gpd.GeoDataFrame
        Nodes locations of the boundary in df.

    Returns
    -------
    da_out: xr.DataArray
        External and focing values combined into a DataArray for variable quantity.
    """
    # Initialise dataarray attributes
    bc = {"quantity": quantity}
    nodeids = df.nodeid.values
    # Assume one forcing file (hydromt writer) and read
    forcing = df.forcingfile.iloc[0]
    df_forcing = pd.DataFrame([f.__dict__ for f in forcing.forcing])
    # Filter for the current nodes
    df_forcing = df_forcing[np.isin(df_forcing.name, nodeids)]

    # Get data
    # Check if all constant
    if np.all(df_forcing.function == "constant"):
        # Prepare data
        data = np.array([v[0][0] for v in df_forcing.datablock])
        data = data + df_forcing.offset.values * df_forcing.factor.values
        # Prepare dataarray properties
        dims = ["index"]
        coords = dict(index=nodeids)
        bc["function"] = "constant"
        bc["units"] = df_forcing.quantityunitpair.iloc[0][0].unit
        bc["factor"] = 1
        bc["offset"] = 0
    # Check if all timeseries
    elif np.all(df_forcing.function == "timeseries"):
        # Prepare data
        data = np.array()
        for v in df_forcing.datablock:
            data = data.append(np.array([n[1] for n in v]))
        data = data + df_forcing.offset.values * df_forcing.factor.values
        # Assume unique times
        times = np.array([n[0] for n in df_forcing.datablock.iloc[0]])
        # Prepare dataarray properties
        dims = (["index", "time"],)
        coords = dict(index=nodeids, time=times)
        bc["function"] = "timeseries"
        bc["units"] = df_forcing.quantityunitpair.iloc[0][1].unit
        bc["time_unit"] = df_forcing.quantityunitpair.iloc[0][0].unit
        bc["factor"] = 1
        bc["offset"] = 0
    # Else not implemented yet
    else:
        raise NotImplementedError(
            f"ForcingFile with several function for a single variable not implemented yet. Skipping reading forcing for variable {quantity}."
        )

    # Get nodeid coordinates
    node_geoms = nodes.set_index("nodeId").reindex(nodeids)
    xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(node_geoms["geometry"])
    coords["x"] = ("index", xs)
    coords["y"] = ("index", ys)

    # Prep DataArray and add to forcing
    da_out = xr.DataArray(
        data=data,
        dims=dims,
        coords=coords,
        attrs=bc,
    )
    da_out.name = f"{quantity}bnd"

    return da_out


def write_1dboundary(forcing: Dict, fm_model: FMModel) -> FMModel:
    """ "
    write 1dboundary ext and boundary files from forcing dict

    Parameters
    ----------
    forcing: dict of xarray DataArray
        Dict of boundary DataArray for each variable
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib

    Returns
    -------
    fm_model: hydrolib.core FMModel
        DflowFM model object from hydrolib with the link to the written boundary files.
    """
    extdict = list()
    bcdict = list()
    # Loop over forcing dict
    for name, da in forcing.items():
        for i in da.index.values:
            bc = da.attrs.copy()
            # Boundary
            ext = dict()
            ext["quantity"] = bc["quantity"]
            ext["nodeId"] = i
            extdict.append(ext)
            # Forcing
            bc["name"] = i
            if bc["function"] == "constant":
                # one quantityunitpair
                bc["quantityunitpair"] = [{"quantity": da.name, "unit": bc["units"]}]
                # only one value column (no times)
                bc["datablock"] = [[da.sel(index=i).values.item()]]
            else:
                # two quantityunitpair
                bc["quantityunitpair"] = [
                    {
                        "quantity": ["time", da.name],
                        "unit": [bc["time_unit"], bc["units"]]
                        # tuple(("time", bc["time_unit"])),
                        # tuple((da.name, bc["units"])),
                    }
                ]
                bc.pop("time_unit")
                # time/value datablock
                bc["datablock"] = [
                    [t, x] for t, x in zip(da.time.values, da.sel(index=i).values)
                ]
            bc.pop("quantity")
            bc.pop("units")
            bcdict.append(bc)

    forcing_model = ForcingModel(forcing=bcdict)
    forcing_model_filename = forcing_model._filename() + ".bc"

    ext_model = ExtModel()
    ext_model_filename = ext_model._filename() + ".ext"
    for i in range(len(extdict)):
        ext_model.boundary.append(
            Boundary(**{**extdict[i], "forcingFile": forcing_model})
        )
    # assign to model
    fm_model.external_forcing.extforcefilenew = ext_model
    fm_model.external_forcing.extforcefilenew.save(
        fm_model.filepath.with_name(ext_model_filename), recurse=True
    )
    # save relative path to mdu
    fm_model.external_forcing.extforcefilenew.filepath = ext_model_filename

    return fm_model


def get_process_geodataframe(
    self,
    path_or_key: str,
    id_col: str,
    clip_buffer: float = 0,  # TODO: think about whether to keep/remove, maybe good to put into the ini file.
    clip_predicate: str = "contains",
    retype: dict = dict(),
    funcs: dict = dict(),
    variables: list = None,
    required_query: str = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Function to open and process geodataframe.

    This function combines a wrapper around :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataframe`

    Parameters
    ----------
    path_or_key : str
        Data catalog key. If a path to a vector file is provided it will be added
        to the data_catalog with its based on the file basename without extension.
    id_col : str
        The id column name.
    clip_buffer : float, optional
        Buffer around the `bbox` or `geom` area of interest in meters. Defaults to 0.
    clip_predicate : {'contains', 'intersects', 'within', 'overlaps', 'crosses', 'touches'}, optional
        If predicate is provided, the GeoDataFrame is filtered by testing
        the predicate function against each item. Requires bbox or mask.
        Defaults to 'contains'.
    retype : dict, optional
        Dictionary containing retypes. Defaults to an empty dictionary.
    funcs : dict, optional
        A dictionary containing key-value pair describing a column name with a string describing the operation to evaluate. Defaults to an empty dictionary.
    variables : list, optional
        Names of GeoDataFrame columns to return. By default all columns are returned. Defaults to None.
    required_query : str, optional
        The required query. Defaults to None.

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        GeoDataFrame

    Raises
    ------
    ValueError
        If `id_col` does not exist in the opened data frame.
    """

    #    # pop kwargs from catalogue
    #    d = self.data_catalog.to_dict(path_or_key)[path_or_key].pop("kwargs")

    #    # set clipping
    #    clip_buffer = d.get("clip_buffer", clip_buffer)
    #    clip_predicate = d.get("clip_predicate", clip_predicate)  # TODO: in ini file

    # read data + clip data + preprocessing data
    df = self.data_catalog.get_geodataframe(
        path_or_key,
        geom=self.region,
        buffer=clip_buffer,
        clip_predicate=clip_predicate,
        variables=variables,
    )
    self.logger.debug(
        f"GeoDataFrame: {len(df)} feature are read after clipping region with clip_buffer = {clip_buffer}, clip_predicate = {clip_predicate}"
    )

    # retype data
    #    retype = d.get("retype", None)
    df = helper.retype_geodataframe(df, retype)

    # eval funcs on data
    #    funcs = d.get("funcs", None)
    df = helper.eval_funcs(df, funcs)

    # slice data # TODO: test what can be achived by the alias in yml file
    # required_columns = d.get("required_columns", None) can be done with variables arg from get_geodataframe
    # required_query = d.get("required_query", None)
    df = helper.slice_geodataframe(
        df, required_columns=None, required_query=required_query
    )
    self.logger.debug(
        f"GeoDataFrame: {len(df)} feature are sliced after applying required_query = '{required_query}'"
    )

    # index data
    if id_col is None:
        pass
    elif id_col not in df.columns:
        raise ValueError(
            f"GeoDataFrame: cannot index data using id_col = {id_col}. id_col must exist in data columns ({df.columns})"
        )
    else:
        self.logger.debug(f"GeoDataFrame: indexing with id_col: {id_col}")
        df.index = df[id_col]
        df.index.name = id_col

    # remove nan in id
    df_na = df.index.isna()
    if len(df_na) > 0:
        df = df[~df_na]
        self.logger.debug(f"GeoDataFrame: removing index with NaN")

    # remove duplicated
    df_dp = df.duplicated()
    if len(df_dp) > 0:
        df = df.drop_duplicates()
        self.logger.debug(f"GeoDataFrame: removing duplicates")

    # report
    df_num = len(df)
    if df_num == 0:
        self.logger.warning(f"Zero features are read from {path_or_key}")
    else:
        self.logger.info(f"{len(df)} features read from {path_or_key}")

    return df
