# -*- coding: utf-8 -*-

import configparser
import logging
from pathlib import Path

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, Point

from .graphs import gpd_to_digraph

logger = logging.getLogger(__name__)


__all__ = [
    "generate_boundaries_from_branches",
    "select_boundary_type",
    "validate_boundaries",
    "compute_boundary_values",
    "compute_2dboundary_values",
    "compute_meteo_forcings"
]


def generate_boundaries_from_branches(
    branches: gpd.GeoDataFrame, where: str = "both"
) -> gpd.GeoDataFrame:
    """Get the possible boundary locations from the branches with id.

    Parameters
    ----------
    where : {'both', 'upstream', 'downstream'}
        Where at the branches should the boundaries be derived.
        An upstream end node is defined as a node which has 0 incoming branches and 1 outgoing branch.
        A downstream end node is defined as a node which has 1 incoming branch and 0 outgoing branches.

    Returns
    -------
    gpd.GeoDataFrame
        A data frame containing all the upstream and downstream end nodes of the branches
    """

    # convert branches to graph
    G = gpd_to_digraph(branches)

    # get boundary locations at where
    if where == "downstream":
        endnodes = {
            dn: {**d, **{"where": "downstream"}}
            for up, dn, d in G.edges(data=True)
            if G.out_degree[dn] == 0 and G.degree[dn] == 1
        }
    elif where == "upstream":
        endnodes = {
            up: {**d, **{"where": "upstream"}}
            for up, dn, d in G.edges(data=True)
            if G.in_degree[up] == 0 and G.degree[up] == 1
        }
    elif where == "both":
        endnodes = {
            dn: {**d, **{"where": "downstream"}}
            for up, dn, d in G.edges(data=True)
            if G.out_degree[dn] == 0 and G.degree[dn] == 1
        }
        endnodes.update(
            {
                up: {**d, **{"where": "upstream"}}
                for up, dn, d in G.edges(data=True)
                if G.in_degree[up] == 0 and G.degree[up] == 1
            }
        )
    else:
        pass

    if len(endnodes) == 0:
        logger.error(f"cannot generate boundaries for given condition {where}")

    endnodes_pd = (
        pd.DataFrame().from_dict(endnodes, orient="index").drop(columns=["geometry"])
    )
    endnodes_gpd = gpd.GeoDataFrame(
        data=endnodes_pd,
        geometry=[Point(endnode) for endnode in endnodes],
        crs=branches.crs,
    )
    endnodes_gpd.reset_index(inplace=True)
    return endnodes_gpd


def select_boundary_type(
    boundaries: gpd.GeoDataFrame,
    branch_type: str,
    boundary_type: str,
    boundary_locs: str,
    logger=logger,
) -> pd.DataFrame:
    """Select boundary location per branch type and boundary type.

    Parameters
    ----------

    boundaries : gpd.GeoDataFrame
        The boundaries.
    branch_type : {'river', 'pipe'}
        The branch type.
    boundary_type : {'waterlevel', 'discharge'}
        For rivers 'waterlevel' and 'discharge' are supported.
        For pipes 'waterlevel' is supported.
    boundary_locs : {'both', 'upstream', 'downstream'}
        The boundary location to use.
    logger
        The logger to log messages with.

    Returns
    -------
    pd.DataFrame
        A data frame containing the boundary location per branch type and boundary type.
    """

    boundaries_branch_type = boundaries.loc[boundaries["branchType"] == branch_type, :]
    if branch_type == "river":
        if boundary_type == "waterlevel":
            if boundary_locs != "both":
                boundaries_branch_type = boundaries_branch_type.loc[
                    boundaries_branch_type["where"] == boundary_locs, :
                ]
        elif boundary_type == "discharge":
            if boundary_locs != "upstream":
                logger.warning(
                    f"Applying boundary type {boundary_type} selected for {branch_type} boundaries might cause instabilities."
                )
            if boundary_locs != "both":
                boundaries_branch_type = boundaries_branch_type.loc[
                    boundaries_branch_type["where"] == boundary_locs, :
                ]
        else:
            logger.error(
                f"Wrong boundary type {boundary_type} selected for {branch_type} boundaries."
            )
    # TODO: extend
    # for now only downstream boundaries not connected to openwater
    # later add connected to weir (upstream or downstream or intermediate)
    elif branch_type == "pipe":
        if boundary_type == "waterlevel":
            boundaries_branch_type = boundaries_branch_type.loc[
                boundaries_branch_type["where"] == "downstream", :
            ]
        else:
            logger.error(
                f"Wrong boundary type {boundary_type} selected for {branch_type} boundaries."
            )

    return boundaries_branch_type


def validate_boundaries(boundaries: gpd.GeoDataFrame, branch_type: str = "river"):
    """Validate boundaries per branch type.
    Will log a warning if the validation fails.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        The boundaries.
    branch_type : {'river', 'pipe'}
        The branch type.

    """

    if branch_type == "river":  # TODO add other open system branch_type
        for _, bnd in boundaries.iterrows():
            # TODO extended
            if bnd["where"] == "downstream" and bnd["boundary_type"] == "discharge":
                logger.warning(
                    f'Boundary type violates modeller suggestions: using downstream discharge boundary at branch {bnd["branchId"]}'
                )

    if branch_type == "pipe":  # TODO add other close system branch_type
        for _, bnd in boundaries.iterrows():
            # TODO extended
            if bnd["where"] == "upstream":
                logger.warning(
                    f'Boundary type violates modeller suggestions: using upstream boundary at branch {bnd["branchId"]}'
                )


def compute_boundary_values(
    boundaries: gpd.GeoDataFrame,
    da_bnd: xr.DataArray = None,
    boundary_value: float = -2.5,
    boundary_type: str = "waterlevel",
    boundary_unit: str = "m",
    snap_offset: float = 0.1,
    logger=logger,
):
    """
    Compute 1d boundary values

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Point locations of the 1D boundaries to which to add data.

        * Required variables: ['nodeId']
    da_bnd : xr.DataArray, optional
        xr.DataArray containing the boundary timeseries values. If None, uses a constant values for all boundaries.

        * Required variables if netcdf: [``boundary_type``]
    boundary_value : float, optional
        Constant value to use for all boundaries if ``da_bnd`` is None and to
        fill in missing data. By default -2.5 m.
    boundary_type : {'waterlevel', 'discharge'}
        Type of boundary to use. By default "waterlevel".
    boundary_unit : {'m', 'm3/s'}
        Unit corresponding to [boundary_type].
        If ``boundary_type`` = "waterlevel"
            Allowed unit is [m]
        if ''boundary_type`` = "discharge":
            Allowed unit is [m3/s]
        By default m.
    snap_offset : float, optional
        Snapping tolerance to automatically applying boundaries at the correct network nodes.
        By default 0.1, a small snapping is applied to avoid precision errors.
    logger
        Logger to log messages.
    """
    nodata_ids = []
    # Timeseries boundary values
    if da_bnd is not None:
        logger.info(f"Preparing 1D {boundary_type} boundaries from timeseries.")

        # snap user boundary to potential boundary locations to get nodeId
        gdf_bnd = da_bnd.vector.to_gdf()
        gdf_bnd = hydromt.gis_utils.nearest_merge(
            gdf_bnd,
            boundaries,
            max_dist=snap_offset,
            overwrite=True,
        )

        # get boundary data freq in seconds
        _TIMESTR = {"D": "days", "H": "hours", "T": "minutes", "S": "seconds"}
        dt = pd.to_timedelta((da_bnd.time[1].values - da_bnd.time[0].values))
        freq = dt.resolution_string
        multiplier = 1
        if freq == "D":
            logger.warning(
                "time unit days is not supported by the current GUI version: 2022.04"
            ) # converting to hours as temporary solution # FIXME: day is converted to hours temporarily
            multiplier = 24
        if len(
            pd.date_range(da_bnd.time[0].values, da_bnd.time[-1].values, freq=dt)
        ) != len(da_bnd.time):
            logger.error("does not support non-equidistant time-series.")
        freq_name = _TIMESTR[freq]
        freq_step = getattr(dt.components, freq_name)
        bd_times = np.array([(i * freq_step) for i in range(len(da_bnd.time))])
        if multiplier == 24:
            bd_times = np.array([(i * freq_step * multiplier) for i in range(len(da_bnd.time))])
            freq_name = "hours"

        # instantiate xr.DataArray for bnd data
        da_out = xr.DataArray(
            data=da_bnd.data,
            dims=["index", "time"],
            coords=dict(
                index=gdf_bnd["nodeId"],
                time=bd_times,
                x=("index", gdf_bnd.geometry.x.values),
                y=("index", gdf_bnd.geometry.y.values),
            ),
            attrs=dict(
                function="TimeSeries",
                timeInterpolation="Linear",
                quantity=f"{boundary_type}bnd",
                units=f"{boundary_unit}",
                time_unit=f"{freq_name} since {pd.to_datetime(da_bnd.time[0].values)}",  # support only yyyy-mm-dd HH:MM:SS
            ),
        )

        # fill in na using default
        da_out = da_out.fillna(boundary_value)

        # drop na in time
        da_out.dropna(dim='time')

        # add name
        da_out.name = f"{boundary_type}bnd"
    else:
        logger.info(
            f"Using constant value {boundary_value} {boundary_unit} for all {boundary_type} boundaries."
        )
        # instantiate xr.DataArray for bnd data with boundary_value directly
        da_out = xr.DataArray(
            data=np.full((len(boundaries.index)), boundary_value, dtype=np.float32),
            dims=["index"],
            coords=dict(
                index=boundaries["nodeId"],
                x=("index", boundaries.geometry.x.values),
                y=("index", boundaries.geometry.y.values),
            ),
            attrs=dict(
                function="constant",
                offset=0.0,
                factor=1.0,
                quantity=f"{boundary_type}bnd",
                units=f"{boundary_unit}",
            ),
        )
        da_out.name = f"{boundary_type}bnd"

    return da_out.drop_duplicates(dim=...)


def compute_2dboundary_values(
    boundaries: gpd.GeoDataFrame = None,
    df_bnd: pd.DataFrame = None,
    boundary_value: float = 0.0,
    boundary_type: str = "waterlevel",
    boundary_unit: str = "m",
    logger=logger,
):
    """
    Compute 2d boundary timeseries. Line geometry will be converted into supporting points.
    Note that All quantities are specified per support point, except for discharges which are specified per polyline.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame, optional
        line geometry type of locations of the 2D boundaries to which to add data.
        Must be combined with ``df_bnd``.

        * Required variables: ["boundary_id"]
    df_bnd : pd.DataFrame, optional
        pd.DataFrame containing the boundary timeseries values.
        Must be combined with ``boundaries``. Columns must match the "boundary_id" in ``boundaries``.

        * Required variables: ["time"]
    boundary_value : float, optional
        Constant value to fill in missing data. By default 0 m.
    boundary_type : {'waterlevel', 'discharge'}
        Type of boundary to use. By default "waterlevel".
    boundary_unit : {'m', 'm3/s'}
        Unit corresponding to [boundary_type].
        If ``boundary_type`` = "waterlevel"
            Allowed unit is [m]
        if ''boundary_type`` = "discharge":
            Allowed unit is [m3/s]
        By default m.
    logger :
        Logger to log messages.

    Raises
    ------
    ValueError:
        if no boundary to compute.
    """

    # Timeseries boundary values
    if boundaries is None or len(boundaries) == 0:
        raise ValueError("No boundary to compute.")
    else:
        # prepare boundary data
        # get data freq in seconds
        _TIMESTR = {"D": "days", "H": "hours", "T": "minutes", "S": "seconds"}
        dt = df_bnd.time[1] - df_bnd.time[0]
        freq = dt.resolution_string
        multiplier = 1
        if freq == "D":
            logger.warning(
                "time unit days is not supported by the current GUI version: 2022.04"
            )  # converting to hours as temporary solution # FIXME: day is supported in version 2023.02, general question: where to indicate gui version?
            multiplier = 24
        if len(
            pd.date_range(df_bnd.iloc[0, :].time, df_bnd.iloc[-1, :].time, freq=dt)
        ) != len(df_bnd.time):
            logger.error("does not support non-equidistant time-series.")
        freq_name = _TIMESTR[freq]
        freq_step = getattr(dt.components, freq_name)
        bnd_times = np.array([(i * freq_step) for i in range(len(df_bnd.time))])
        if multiplier == 24:
            bnd_times = np.array(
                [(i * freq_step * multiplier) for i in range(len(df_bnd.time))]
            )
            freq_name = "hours"

        # for each boundary apply boundary data
        da_out_dict = {}
        for _index, _bnd in boundaries.iterrows():

            bnd_id = _bnd["boundary_id"]

            # convert line to points
            support_points = pd.DataFrame(
                np.array([[x, y] for x, y in _bnd.geometry.coords[:]]),
                columns=["x", "y"],
            )
            support_points["_id"] = support_points.index + 1
            support_points["id"] = support_points["_id"].astype(str)
            support_points["id"] = support_points["id"].str.zfill(4)
            support_points["name"] = support_points.astype(str).apply(
                lambda x: f"{bnd_id}_{x.id}", axis=1
            )

            # instantiate xr.DataArray for bnd data with boundary_value directly
            da_out = xr.DataArray(
                data=np.full(
                    (len(support_points["name"]), len(bnd_times)),
                    np.tile(df_bnd[bnd_id].values, (len(support_points["name"]), 1)),
                    dtype=np.float32,
                ),
                dims=["index", "time"],
                coords=dict(
                    index=support_points["name"],
                    time=bnd_times,
                    x=("index", support_points.x.values),
                    y=("index", support_points.y.values),
                ),
                attrs=dict(
                    locationfile=bnd_id + ".pli",
                    function="TimeSeries",
                    timeInterpolation="Linear",
                    quantity=f"{boundary_type}bnd",
                    units=f"{boundary_unit}",
                    time_unit=f"{freq_name} since {pd.to_datetime(df_bnd.time[0])}",
                    # support only yyyy-mm-dd HH:MM:SS
                ),
            )
            # fill in na using default
            da_out = da_out.fillna(boundary_value)
            da_out.name = f"{bnd_id}"
            da_out_dict.update({f"{bnd_id}": da_out})

    return da_out_dict


def gpd_to_pli(gdf: gpd.GeoDataFrame, output_dir: Path):
    """function to convert geopandas GeoDataFrame (gdf) into pli files at 'output_dir' directory.
    the geodataframe must has index as stations and geometry of the stations.
    each row of the geodataframe will be converted into a single pli file.
    the file name and the station name will be the index of that row.
    """

    for _, g in gdf.iterrows():
        pli_name = g.index
        pli_coords = g.geometry.coords[:]
        with open(output_dir.joinpath(f"{pli_name}.pli"), "w") as f:
            f.write(f"{pli_name}\n")
            f.write(f"\t{len(pli_coords)} {2}\n")
            for p in pli_coords:
                f.write(f"\t{' '.join(str(pi) for pi in p)}\n")


def df_to_bc(
    df,
    output_dir,
    output_filename="boundary",
    quantity="discharge",
    unit="m3/s",
    freq="H",
):
    """function to convert pandas timeseires 'df' into bc file at 'output_dir'/'output_filename'.bc
    the time series must has time as index, columns names as stations.
    the time series will be first converted into a equidistance timeseries with frequency specified in 'freq'. support [D, H,M,S]
    each columns-wise array will be converted into one bc timeseries.
    The time series has the quantity and unit as specified in 'quantity' nad 'unit'.
    """
    time_unit = {"D": "days", "H": "hours", "M": "minutes", "S": "seconds"}

    df = df.resample(freq).ffill()
    time = df.index
    stations = df.columns

    with open(output_dir.joinpath(f"{output_filename}.bc"), "w") as f:
        f.write(f"[General]\n")
        f.write(f"\tfileVersion = 1.01\n")
        f.write(f"\tfileType = boundConds\n")
        for s in stations:
            d = df[s]
            f.write(f"\n")
            f.write(f"[forcing]\n")
            f.write(f"\tName = {d.name}\n")
            f.write(f"\tfunction = timeSeries\n")
            f.write(f"\ttimeInterpolation = linear\n")
            f.write(f"\tquantity = {quantity}\n")
            f.write(f"\tunit = {unit}\n")
            f.write(f"\tquantity = time\n")
            f.write(f"\tunit = {time_unit[freq]} since {time[0].date()}\n")
            f.write(f"\t0 0\n")
            for i, di in enumerate(d.values):
                f.write(f"\t{i} {di}\n")


def compute_meteo_forcings(
    df_meteo: pd.DataFrame = None,
    fill_value : float = 0.0,
    is_rate : bool = True,
    meteo_location: tuple = None,
    logger = logger,
) -> xr.DataArray:
    """
    Compute meteo forcings

    Parameters
    ----------
    df_meteo : pd.DataFrame, optional
        pd.DataFrame containing the meteo timeseries values. If None, uses ``fill_value``.

        * Required variables: ["precip"]
    meteo_value : float, optional
        Constant value to use for global meteo if ``df_meteo`` is None and to
        fill in missing data in ``df_meteo``. By default 0.0 mm/day.
    is_rate : bool, optional
        Specify if the type of meteo data is direct "rainfall" (False) or "rainfall_rate" (True).
        By default True for "rainfall_rate". Note that Delft3DFM 1D2D Suite 2022.04 supports only "rainfall_rate".
        If rate, unit is expected to be in mm/day and else mm.
    meteo_location : tuple
        Global location for meteo timeseries
    logger
        Logger to log messages.

    Returns
    -------
    da_meteo : xr.DataArray
        xr.DataArray containing the meteo timeseries values. If None, uses ``df_meteo``.

        * Required variables if netcdf: [``precip``]
    """
    # Set units and type
    if is_rate: 
        meteo_type = "rainfall_rate"
        meteo_unit = "mm/day"
    else: 
        meteo_type = "rainfall"
        meteo_unit = "mm"

    # Timeseries boundary values

    logger.info(
        f"Preparing global (spatially uniform) timeseries."
    )
    # get data freq in seconds
    _TIMESTR = {"D": "days", "H": "hours", "T": "minutes", "S": "seconds"}
    dt = (df_meteo.time[1] - df_meteo.time[0])
    freq = dt.resolution_string
    multiplier = 1
    if freq == "D":
        logger.warning(
            "time unit days is not supported by the current GUI version: 2022.04"
        )  # converting to hours as temporary solution # FIXME: day is converted to hours temporarily
        multiplier = 24
    if len(
            pd.date_range(df_meteo.iloc[0,:].time, df_meteo.iloc[-1,:].time, freq=dt)
    ) != len(df_meteo.time):
        logger.error("does not support non-equidistant time-series.")
    freq_name = _TIMESTR[freq]
    freq_step = getattr(dt.components, freq_name)
    meteo_times = np.array([(i * freq_step) for i in range(len(df_meteo.time))])
    if multiplier == 24:
        meteo_times = np.array([(i * freq_step * multiplier) for i in range(len(df_meteo.time))])
        freq_name = "hours"
    # instantiate xr.DataArray for global time series
    da_out = xr.DataArray(
        data=np.full(
            (1, len(df_meteo)), df_meteo["precip"].values, dtype=np.float32
        ),
        dims=["index", "time"],
        coords=dict(
            index=["global"],
            time= meteo_times,
            x=("index", meteo_location[0].values),
            y=("index", meteo_location[1].values),
        ),
        attrs=dict(
            function="TimeSeries",
            timeInterpolation="Linear",
            quantity=f"{meteo_type}",
            units=f"{meteo_unit}",
            time_unit=f"{freq_name} since {pd.to_datetime(df_meteo.time[0])}",
            # support only yyyy-mm-dd HH:MM:SS
        ),
    )
    # fill in na using default
    da_out = da_out.fillna(fill_value)
    da_out.name = f"{meteo_type}"
    da_out.dropna(dim='time')

    return da_out