# -*- coding: utf-8 -*-

import logging
import geopandas as gpd
import xarray as xr


logger = logging.getLogger(__name__)


__all__ = ["invert_levels_from_dem"]


def invert_levels_from_dem(gdf: gpd.GeoDataFrame, dem: xr.DataArray, depth: int = -2.0):
    """
    Compute up- and downstream invert levels for pipe lines in gdf.

    Invert levels are computed as DEM - depth - pipe diameter/height.

    Parameters:
    -----------
    gdf: gpd.GeoDataFrame
        Pipes gdf.

        * Required variables: ["shape", "diameter"] or ["shape", "height"] (circle or rectangle shape)
    dem: xr.DataArray
        DEM data array with elevation in m asl.
    depth: int, optional
        Depth of the pipes under the ground in meters. Should be a negative value. By default -2.0 meters
    """
    # Upstream
    upnodes = gpd.GeoDataFrame({}, index=gdf.index, crs=gdf.crs)
    upnodes["geometry"] = [l.coords[0] for l in gdf.geometry]
    gdf["elevtn_up"] = dem.raster.sample(
        upnodes
    ).values  # reproject of dem is done in sample method
    # Downstream
    dnnodes = gpd.GeoDataFrame({}, index=gdf.index, crs=gdf.crs)
    dnnodes["geometry"] = [l.coords[-1] for l in gdf.geometry]
    gdf["elevtn_dn"] = dem.raster.sample(dnnodes).values

    # circle profile
    circle_indexes = gdf.loc[gdf["shape"] == "circle", :].index
    for bi in circle_indexes:
        gdf.loc[bi, "invlev_up"] = (
            gdf.loc[bi, "elevtn_up"] + depth - gdf.loc[bi, "diameter"]
        )
        gdf.loc[bi, "invlev_dn"] = (
            gdf.loc[bi, "elevtn_dn"] + depth - gdf.loc[bi, "diameter"]
        )
    # rectangle profile
    rectangle_indexes = gdf.loc[gdf["shape"] == "rectangle", :].index
    for bi in rectangle_indexes:
        gdf.loc[bi, "invlev_up"] = (
            gdf.loc[bi, "elevtn_up"] + depth - gdf.loc[bi, "height"]
        )
        gdf.loc[bi, "invlev_dn"] = (
            gdf.loc[bi, "elevtn_dn"] + depth - gdf.loc[bi, "height"]
        )

    return gdf
