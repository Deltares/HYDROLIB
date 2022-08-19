# -*- coding: utf-8 -*-

import logging
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point


logger = logging.getLogger(__name__)


__all__ = ["invert_levels_from_dem"]


def invert_levels_from_dem(
    gdf: gpd.GeoDataFrame, dem: xr.DataArray, depth: float = 2.0
):
    """
    Compute up- and downstream invert levels for pipe lines in gdf.

    Invert levels are computed as DEM - depth - pipe diameter/height. # FIXME: by the definition specified here, it should be a postive value

    Parameters:
    -----------
    gdf: gpd.GeoDataFrame
        Pipes gdf.

        * Required variables: ["shape", "diameter"] or ["shape", "height"] (circle or rectangle shape)
    dem: xr.DataArray
        DEM data array with elevation in m asl.
    depth: float, optional
        Depth of the pipes under the ground in meters. Should be a postive value. By default 2.0 meters
    """
    # Upstream
    upnodes = gpd.GeoDataFrame({}, index=gdf.index, crs=gdf.crs)
    upnodes["geometry"] = [Point(l.coords[0]) for l in gdf.geometry]

    # reproject of dem is done in sample method
    gdf["elevtn_up"] = dem.raster.sample(upnodes).values
    # Downstream
    dnnodes = gpd.GeoDataFrame({}, index=gdf.index, crs=gdf.crs)
    dnnodes["geometry"] = [Point(l.coords[-1]) for l in gdf.geometry]
    gdf["elevtn_dn"] = dem.raster.sample(dnnodes).values

    def invert_levels_cross_sections(shape: str, attribute: str):
        indices = gdf.loc[gdf["shape"] == shape, :].index
        for bi in indices:
            gdf.loc[bi, "invlev_up"] = (
                gdf.loc[bi, "elevtn_up"] - depth - gdf.loc[bi, attribute]
            )
            gdf.loc[bi, "invlev_dn"] = (
                gdf.loc[bi, "elevtn_dn"] - depth - gdf.loc[bi, attribute]
            )

    invert_levels_cross_sections("circle", "diameter")
    invert_levels_cross_sections("rectangle", "height")

    return gdf
