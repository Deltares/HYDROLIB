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
        if boundary_type == "discharge":
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
        gdf_bnd = da_bnd.vector.to_gdf()
        gdf_bnd["_index"] = gdf_bnd.index
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
            pd.date_range(da_bnd.time[0].values, da_bnd.time[-1].values, freq=freq)
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
            data=np.full(
                (len(boundaries.index), len(bd_times)), boundary_value, dtype=np.float32
            ),
            dims=["index", "time"],
            coords=dict(
                index=boundaries["nodeId"],
                time=bd_times,
                x=("index", boundaries.geometry.x.values),
                y=("index", boundaries.geometry.y.values),
            ),
            attrs=dict(
                function="TimeSeries",
                timeInterpolation="Linear",
                quantity=f"{boundary_type}",
                units=f"{boundary_unit}",
                time_unit=f"{freq_name} since {pd.to_datetime(da_bnd.time[0].values)}",  # support only yyyy-mm-dd HH:MM:SS
            ),
        )
        da_out.name = f"{boundary_type}bnd"
        da_out.dropna(dim = 'time')
        # snap user boundary to potential boundary locations
        boundaries = hydromt.gis_utils.nearest_merge(
            boundaries,
            gdf_bnd,
            max_dist=snap_offset,
            overwrite=True,
        )  # _index will be float
        # remove boundaries without bc values in da_bnd
        boundaries = boundaries[~pd.isnull(boundaries["_index"])]
        nodata_ids = boundaries["nodeId"][
            ~pd.isnull(boundaries["_index"])
        ].values.tolist()
        for i in range(len(boundaries)):
            bc_values = da_bnd.sel(index=int(boundaries["_index"].iloc[i])).values
            # Check if any nodata value, else use default boundary_value
            if np.isnan(bc_values).sum() > 0:
                nodata_ids.append(f'{int(boundaries["_index"].iloc[i])}')
                # send warning about boundary condtitions data set to default values
                logger.warning(
                    f"Nodata found for {boundary_type} boundaries values for nodes {nodata_ids}. Default values of {boundary_value} {boundary_unit} used instead for these nodes."
                )
            else:
                id = boundaries["nodeId"].iloc[i]
                da_out.loc[id, :] = bc_values
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
                quantity=f"{boundary_type}",
                units=f"{boundary_unit}",
            ),
        )
        da_out.name = f"{boundary_type}bnd"

    return da_out


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
