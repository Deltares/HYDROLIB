import os
import sys
from datetime import datetime

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
from geovoronoi import points_to_coords, voronoi_regions_from_coords
from rasterio import features
from shapely.geometry import box

from read_dhydro import net_nc2gdf, read_nc_data

def inun_dhydro(
    nc_path,
    output_folder,
    type="level",
    dtm_path="",
    sdate="",
    edate="",
    domain="",
    filter=False,
    extrapol=0.5,
):

    """
       Calculate inundation depths based on 1D and 2D waterlevels and DEM.

       Parameters:
            nc_path : str
               Path to input nc-file containing the D-hydro model results
            output_folder : str
               Path to the output folder
            type: str
               Analysis type ("type" and "level")
            dtm_path: str
               Path to the dtm in tif format
            sdate: datetime.datetime
               Optional start of period to find maximal level
            edate: datetime.datetime
               Optional end of period to find maximal level
            domain: str
               Optional domain to determine inundations ("1D" or "2D")
            filter: bool
                Optional filtering of the inundations, based on connectivity
            extrapol: float
                Optional extrapolation of 2D results
    ___________________________________________________________________________________________________________
       Warning:
           only 2D results will be used if type=depth
           2D depth is used when both 1D and 2D inundation is plausible
           without filter 1D will only be used outside of 2D.
    """

    ds = nc.Dataset(nc_path)
    EPSG = "EPSG:" + str(ds["projected_coordinate_system"].epsg)
    variables = list(ds.variables)

    # create dataframe with data
    if type == "level":
        par1 = "mesh1d_s1" if "mesh1d_s1" in variables else "Mesh1d_s1"
        par2 = "mesh2d_s1" if "mesh2d_s1" in variables else "Mesh2d_s1"
        par3 = (
            "mesh2d_waterdepth"
            if "mesh2d_waterdepth" in variables
            else "Mesh2d_waterdepth"
        )
        df1 = read_nc_data(ds, par1) if domain.upper() != "2D" else pd.DataFrame()
        df2 = read_nc_data(ds, par2) if domain.upper() != "1D" else pd.DataFrame()
        df3 = read_nc_data(ds, par3) if domain.upper() != "1D" else pd.DataFrame()
    elif type == "depth":
        par2 = (
            "mesh2d_waterdepth"
            if "mesh2d_waterdepth" in variables
            else "Mesh2d_waterdepth"
        )
        df1 = pd.DataFrame()
        df2 = read_nc_data(ds, par2)
        df3 = pd.DataFrame()
    else:
        raise ("Onbekend type: " + str(type))

    # filter data based on time
    sdata = min(df1.index.min() if len(df1>0) else datetime(9999, 1, 1),df2.index.min() if len(df2>0) else datetime(9999, 1, 1),df3.index.min() if len(df3>0) else datetime(9999, 1, 1))
    edata = max(df1.index.max() if len(df1>0) else datetime(1, 1, 1),df2.index.max() if len(df2>0) else datetime(1, 1, 1),df3.index.max() if len(df3>0) else datetime(1, 1, 1))
    if sdate != "":
        if sdate > edata:
            raise("start date later then end date in model results")
        if len(df1) > 0:
            df1 = df1[df1.index >= datetime.strptime(sdate, "%Y/%m/%d")]
        if len(df2) > 0:
            df2 = df2[df2.index >= datetime.strptime(sdate, "%Y/%m/%d")]
        if len(df3) > 0:
            df3 = df3[df3.index >= datetime.strptime(sdate, "%Y/%m/%d")]
    if edate != "":
        if edate < sdata:
            raise("end date earlier then start date in model results")
        if len(df1) > 0:
            df1 = df1[df1.index <= datetime.strptime(edate, "%Y/%m/%d")]
        if len(df2) > 0:
            df2 = df2[df2.index <= datetime.strptime(edate, "%Y/%m/%d")]
        if len(df3) > 0:
            df3 = df3[df3.index >= datetime.strptime(sdate, "%Y/%m/%d")]

    # determine needed geometry
    results = []
    if domain.upper() != "2D":
        results.append("1d_meshnodes")
    if domain.upper() != "2D" and filter == True:
        results.append("1d_branches")
    if domain.upper() != "1D":
        results.append("2d_faces")

    # add to geometry to 1D and 2D
    gdfs = net_nc2gdf(nc_path, results=results)
    if len(df1) > 0:
        gdf1 = gdfs["1d_meshnodes"]
        gdf1["max"] = list(df1.max())  # fails without list
    else:
        gdf1 = gpd.GeoDataFrame()

    if len(df2) > 0:
        gdf2 = gdfs["2d_faces"]
        gdf2["max"] = df2.max().where(
            df3.max() > 0, np.nan
        )  # remove bedlevels when no inundation
    else:
        gdf2 = gpd.GeoDataFrame()

    # read dtm and prepare meta
    with rasterio.open(dtm_path) as src:
        print("Inlezen van dem en goedzetten nodata.")
        dtm = src.read(1, masked=True)
        meta = src.profile
        meta.update(compress="lzw", count=1, dtype=rasterio.float32)
        resx = src.transform[0]
        # correct possible wrong nodata
        dtm[dtm <= -999] = np.nan
        dtm[dtm >= 9999] = np.nan
        #geom_dtm = Polygon([(src.bounds.left,src.bounds.top),(src.bounds.right,src.bounds.top),(src.bounds.right,src.bounds.bottom),(src.bounds.left,src.bounds.bottom)])
        geom_dtm = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

    # create voronois for 1D
    if len(gdf1) > 0:
        # select points outside 2D
        if len(gdf2)>0 and filter == True: # without filtering this will result in wrong results
            gdf1_sel = gdf1[~gdf1.geometry.within(gdf2.geometry.unary_union)]
        else:
            gdf1_sel = gdf1
        coords = points_to_coords(gdf1_sel.geometry)
        #gdf1_sel = gdf1[~gdf1.geometry.within(geom_dtm)]
        geom_areas = pd.concat([gpd.GeoDataFrame(geometry=[geom_dtm.buffer(geom_dtm.area**0.5/10)]), gpd.GeoDataFrame(geometry=[gdf1.unary_union.envelope])]).geometry.unary_union
        vr_area, vr_points, vr_links = voronoi_regions_from_coords(coords, geom_areas) 
        gdf1_area = gpd.GeoDataFrame(geometry=vr_area, crs=EPSG)
        # clip 2D results out of 1D results
        if len(gdf2) > 0:
            gdf1_area = gdf1_area.overlay(
                gdf2, how="difference", keep_geom_type=True
            ).explode(
                ignore_index=True
            ) 
        gdf1_area = gpd.sjoin(gdf1_area, gdf1_sel, how="inner", predicate="intersects")
        gdf1_area.drop(columns=["index_right"], inplace=True)
    else:
        gdf1_area = gpd.GeoDataFrame()

    # write areas if present
    filename = "waterlevel" if type == "level" else "waterdepth"
    if len(gdf1_area) > 0:
        gdf1_area.to_file(os.path.join(output_folder, filename + "1D.shp"))
    if len(gdf2) > 0:
        gdf2.to_file(os.path.join(output_folder, filename + "2D.shp"))

    # extrapolate 2D waterlevels, only when using waterlevel
    if len(gdf2) > 0:
        if type == "level" and extrapol>0:
            gdf2_buf = gpd.GeoDataFrame(
                gdf2.copy(), geometry=gdf2.buffer(gdf2.area ** 0.5 * extrapol), crs=EPSG
            )
            gdf2_extp = pd.concat([gdf2_buf, gdf2])
        else:
            gdf2_extp = gdf2
        gdf2_extp.dropna(subset=["max"], inplace=True)

    # create template array
    nan_array = np.empty((len(dtm), len(dtm[0]))).astype("float32")

    with rasterio.open(
        os.path.join(output_folder, "waterdepth.tif"), "w", **meta
    ) as out:
        # write model results if type is depth
        if type == "depth":
            print("Exporteren van modelresultaten.")
            areas = ((geom, value) for geom, value in zip(gdf2.geometry, gdf2["max"]))
            nan_array[:] = np.nan
            values = features.rasterize(
                shapes=areas, fill=np.nan, out=nan_array, transform=out.transform
            )
            values[values <= -999] = np.nan
            out.write_band(1, values)
        # calcualte inundations from waterlevel
        elif type == "level":
            print("Berekenen van inundaties.")

            # create waterlevel raster and calculate inundations
            if len(gdf1_area) > 0:
                areas1D = (
                    (geom, value)
                    for geom, value in zip(gdf1_area.geometry, gdf1_area["max"])
                )
                nan_array[:] = np.nan
                values1D = features.rasterize(
                    shapes=areas1D, fill=np.nan, out=nan_array, transform=out.transform
                )
                values1D[values1D <= -999] = np.nan
                inun1D = np.where(dtm == np.nan, np.nan, values1D - dtm)
                inun1D = np.where(inun1D <= 0, np.nan, inun1D)
            if len(gdf2_extp) > 0:
                areas2D = (
                    (geom, value) for geom, value in zip(gdf2_extp.geometry, gdf2_extp["max"])
                )
                nan_array[:] = np.nan
                values2D = features.rasterize(
                    shapes=areas2D, fill=np.nan, out=nan_array, transform=out.transform
                )
                values2D[values2D <= -999] = np.nan
                inun2D = np.where(dtm == np.nan, np.nan, values2D - dtm)
                inun2D = np.where(inun2D <= 0, np.nan, inun2D)

            # filter inundation area
            if filter == True:
                print("Filter inundaties.")
                # filter 2D
                if len(gdf2) > 0:
                    # create polygons from 2D inundationraster
                    geoms = list(
                        {"properties": {"raster_val": v}, "geometry": s}
                        for i, (s, v) in enumerate(
                            rasterio.features.shapes(
                                np.where(inun2D > 0, 1, 0),
                                mask=None,
                                transform=src.transform,
                            )
                        )
                    )
                    inun2D_gdf = gpd.GeoDataFrame.from_features(geoms)
                    inun2D_gdf = inun2D_gdf[inun2D_gdf.raster_val > 0]

                    # apply buffer to determine isolated 2D inundations, for instance behind a road or dike.
                    inun2D_gdf["dum"] = 1
                    inun2D_gdf.loc[
                        inun2D_gdf.is_valid == False, "geometry"
                    ] = inun2D_gdf.loc[inun2D_gdf.is_valid == False].geometry.buffer(
                        0
                    )  # repair invalid geometry
                    inun2D_gdf_buf = gpd.GeoDataFrame(
                        geometry=inun2D_gdf.dissolve(by="dum").buffer(resx), crs=EPSG
                    ).explode(
                        index_parts=False
                    )  # todo: deside how big buffer should be

                    # identify small isolated inundations, based on the 2D cellsize
                    cellsize = gdf2.area.max() ** 0.5
                    inun2D_gdf_sel = inun2D_gdf_buf[
                        inun2D_gdf_buf.area > cellsize ** 2 * 2
                    ].reset_index()  # todo: deside how big areas should be
                    shapes = (
                        (geom, value)
                        for geom, value in zip(
                            inun2D_gdf_sel.geometry, inun2D_gdf_sel["dum"]
                        )
                    )
                    nan_array[:] = np.nan
                    filter2D = rasterio.features.rasterize(
                        shapes=shapes, fill=0, out=nan_array, transform=out.transform
                    )

                    # finally filter inundations
                    inun2D = np.where(filter2D > 0, inun2D, np.nan)

                # filter 1D
                if len(gdf1_area) > 0:
                    # create polygons from 1D inundations
                    geoms = list(
                        {"properties": {"raster_val": v}, "geometry": s}
                        for i, (s, v) in enumerate(
                            rasterio.features.shapes(
                                np.where(inun1D > 0, 1, 0),
                                mask=None,
                                transform=src.transform,
                            )
                        )
                    )
                    inun1D_gdf = gpd.GeoDataFrame.from_features(geoms)
                    inun1D_gdf = inun1D_gdf[inun1D_gdf.raster_val > 0]

                    # apply buffer to determine isolated 1D inundations,
                    inun1D_gdf["dum"] = 1
                    inun1D_gdf.loc[
                        inun1D_gdf.is_valid == False, "geometry"
                    ] = inun1D_gdf.loc[inun1D_gdf.is_valid == False].geometry.buffer(
                        0
                    )  # repair invalid geometry
                    inun1D_gdf_buf = gpd.GeoDataFrame(
                        geometry=inun1D_gdf.dissolve(by="dum").buffer(resx), crs=EPSG
                    ).explode(
                        index_parts=False
                    )  # todo: deside how big buffer should be

                    # identify small isolated inundations, based on connection to 1D
                    gdf_branches = gdfs["1d_branches"]
                    gdf_branches["dum"] = 1
                    gdf_branches = gpd.GeoDataFrame(
                        geometry=gdf_branches.dissolve(by="dum").geometry, crs=EPSG
                    ).reset_index(drop=True)
                    inun1D_gdf_sel = gpd.sjoin(
                        inun1D_gdf_buf,
                        gdf_branches,
                        how="inner",
                        predicate="intersects",
                    ).reset_index()
                    shapes = (
                        (geom, value)
                        for geom, value in zip(
                            inun1D_gdf_sel.geometry, inun1D_gdf_sel["dum"]
                        )
                    )
                    nan_array[:] = np.nan
                    filter1D = rasterio.features.rasterize(
                        shapes=shapes, fill=0, out=nan_array, transform=out.transform
                    )

                    # finally filter inundations
                    inun1D = np.where(filter1D > 0, inun1D, np.nan)

            # combine inundations, priorityze 2D
            if len(gdf1_area) > 0 and len(gdf2) > 0:
                inun = np.where(inun2D > 0, inun2D, inun1D)
            elif len(gdf1_area) > 0:
                inun = inun1D
            elif len(gdf2) > 0:
                inun = inun2D
            else:
                raise ("no inundations calculated from 1D and 2D")

            # export tif
            print("Exporteren van inundaties.")
            out.write(inun, 1)

    print("\nMaken van inundatiegrid afgerond")


if __name__ == "__main__":
    input_path = r"C:\Users\buijerta\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\05 Gedeelde map WRIJ\03_Resultaten\20220214_DR49\modellen\DR49_Bronkhorst_100\dflowfm\output\dr49_map.nc"
    dtm_path = r"C:\Users\buijerta\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\06 Work in Progress\GIS\ahn\dr49\ahn3_2x2_combi_dr49.tif"
    type = "level"  # depth
    output_folder = r"C:/temp/aht"
    inun_dhydro(
        input_path,
        output_folder,
        type=type,
        dtm_path=dtm_path,
        sdate="",
        edate="",
        domain="",
        filter=True,
        extrapol = 1.0
    )
