import os
import sys
from datetime import datetime

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from read_dhydro import net_nc2gdf, read_nc_data


def inun_dhydro(
    map_path, output_folder, type="level", dtm_path="", sdate="", edate="", filter=False
):
    interpol = 0.5

    ds = nc.Dataset(map_path)
    variables = list(ds.variables)

    # Mesh2d_waterdept, Mesh1d_waterdepth, mesh1d_s1, Mesh1d_s2
    if type == "level":
        # par1 = "mesh1d_s1" if "mesh1d_s1" in variables else "Mesh1d_s1"
        par2 = "mesh2d_s1" if "mesh2d_s1" in variables else "Mesh2d_s1"
        par3 = (
            "mesh2d_waterdepth"
            if "mesh2d_waterdepth" in variables
            else "Mesh2d_waterdepth"
        )
    else:
        # par1 = "mesh1d_waterdepth" if "mesh1d_waterdepth" in variables else "Mesh1d_waterdepth"
        par2 = (
            "mesh2d_waterdepth"
            if "mesh2d_waterdepth" in variables
            else "Mesh2d_waterdepth"
        )

    # create dataframe with data
    # df1 = read_nc_data(ds,par1)
    df2 = read_nc_data(ds, par2)
    if type == "level":
        df3 = read_nc_data(ds, par3)
    else:
        df3 = df2

    # filter data based on time
    # todo check if datas present
    if sdate != "":
        # df1 = df1[df1.index >= datetime.strptime(sdate,"%Y/%m/%d")]
        df2 = df2[df2.index >= datetime.strptime(sdate, "%Y/%m/%d")]
        df3 = df3[df3.index >= datetime.strptime(sdate, "%Y/%m/%d")]
    if edate != "":
        # df1 = df1[df1.index <= datetime.strptime(edate,"%Y/%m/%d")]
        df2 = df2[df2.index <= datetime.strptime(edate, "%Y/%m/%d")]
        df3 = df3[df3.index >= datetime.strptime(sdate, "%Y/%m/%d")]

    # add to geometry
    gdfs = net_nc2gdf(map_path, results=["1d_meshnodes", "2d_faces"])
    # gdf1 = gdfs["1d_meshnodes"]
    gdf2 = gdfs["2d_faces"]
    # gdf1["max"] = df1.max()
    gdf2["max"] = df2.max().where(df3.max() > 0, np.nan)

    if type == "level":
        gdf2.to_file(os.path.join(output_folder, "waterlevel.shp"))
    else:
        gdf2.to_file(os.path.join(output_folder, "waterdepth.shp"))

    cellsize = gdf2.area.max() ** 0.5

    # interpolate waterlevels, only when using waterlevel
    if type == "level":
        gdf2_buf = gpd.GeoDataFrame(
            gdf2.copy(), geometry=gdf2.buffer(gdf2.area ** 0.5 * interpol)
        )
        gdf2 = pd.concat([gdf2_buf, gdf2])
        # gdf2.sort_values("max", ascending=True,inplace=True)

    gdf2.dropna(subset=["max"], inplace=True)

    # inlezen van dtm en goedzetten van meta
    with rasterio.open(dtm_path) as src:
        print("Inlezen van dem en goedzetten nodata.")
        dtm = src.read(1, masked=True)
        dtm[dtm <= -999] = np.nan
        dtm[dtm >= 9999] = np.nan
        meta = src.profile
        meta.update(compress="lzw", count=1, dtype=rasterio.float32)
        resx = src.transform[0]

    # write model results
    filename = "waterlevel.tif" if type == "level" else "waterdepth.tif"
    with rasterio.open(os.path.join(output_folder, filename), "w+", **meta) as out:
        print("Exporteren van modelresultaten.")
        out_arr = out.read(1)
        shapes = ((geom, value) for geom, value in zip(gdf2.geometry, gdf2["max"]))
        values = features.rasterize(
            shapes=shapes, fill=0, out=out_arr, transform=out.transform
        )
        values[values <= -999] = np.nan
        out.write_band(1, values)

    # calcualte inundations
    if type == "level":
        inun = np.where(dtm == np.nan, np.nan, values - dtm)
        inun = np.where(inun <= 0, np.nan, inun)

        # filter inundation area
        if filter == True:
            # create polygons
            dummy = np.where(inun > 0, 1, 0)
            mask = None
            results = (
                {"properties": {"raster_val": v}, "geometry": s}
                for i, (s, v) in enumerate(
                    rasterio.features.shapes(dummy, mask=mask, transform=src.transform)
                )
            )
            geoms = list(results)
            inun_gdf = gpd.GeoDataFrame.from_features(geoms)
            inun_gdf = inun_gdf[inun_gdf.raster_val > 0]

            # apply buffer to determine not connected inundations
            inun_gdf_buf = gpd.GeoDataFrame(
                inun_gdf, geometry=inun_gdf.buffer(resx)
            )  # todo: beter nadenken over buffer
            inun_gdf_buf["dum"] = 1
            inun_gdf_dis = inun_gdf_buf.dissolve(by="dum").explode()
            inun_gdf_sel = inun_gdf_dis[
                inun_gdf_dis.area > cellsize ** 2 * 2
            ].reset_index()  # todo: beter nadenken over oppervlak
            shapes = (
                (geom, value)
                for geom, value in zip(inun_gdf_sel.geometry, inun_gdf_sel["dum"])
            )
            nan_array = np.empty((len(out_arr), len(out_arr[0]))).astype("float32")
            nan_array[:] = np.nan
            filter = rasterio.features.rasterize(
                shapes=shapes, fill=0, out=nan_array, transform=out.transform
            )
            # filter = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)

            # filer inundations
            inun_filter = np.where(filter > 0, inun, np.nan)
        else:
            inun_filter = inun

        # export tif
        print("Exporteren van inundaties.")
        inun_path = os.path.join(output_folder, "waterdepth.tif")
        with rasterio.open(inun_path, "w", **meta) as out:
            out.write(inun_filter, 1)

    print("\nMaken van inundatiegrid afgerond")


if __name__ == "__main__":
    input_path = r"C:\Users\buijerta\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\05 Gedeelde map WRIJ\03_Resultaten\20220214_DR49\modellen\DR49_Bronkhorst_100\dflowfm\output\dr49_map.nc"
    dtm_path = r"C:\Users\buijerta\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\06 Work in Progress\GIS\ahn\dr49\ahn3_2x2_combi_dr49.tif"
    type = "level"  # depth
    output_folder = r"C:/temp/aht"
    inun_dhydro(
        input_path, output_folder, type=type, dtm_path=dtm_path, sdate="", edate=""
    )
