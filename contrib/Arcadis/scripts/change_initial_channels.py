# =============================================================================
#
# License: LGPL
#
# Author: Arjon Buijert Arcadis
#
# =============================================================================


import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from read_dhydro import net_nc2gdf

from hydrolib.core.io.onedfield.models import OneDFieldModel


def initial_dhydro(
    net_nc_path,
    areas_path,
    value_field,
    value_type,
    value_unit,
    global_value,
    output_path,
):
    """
    Create the 1D initial waterlevel D-hydro based on a shape file.
    ___________________________________________________________________________________________________________

    Parameters:
         net_nc_path : str
             Path to input nc-file containing the D-hydro network
         areas_path : str
             Path to shapefile (polygons) with areas containing initial values
         value_field: str
             Column name containing intial values
         value_type: str
             Type of initial value (WaterLevel or WaterDepth)
         value_unit: str
             Unit of initial value ("m")
         global_value: float
             Standard value for waterways that fall outside of the area
         output_path : str
             Path to results-file
    ___________________________________________________________________________________________________________

    Returns:
        initialwaterlevel.ini file
    _______________________________________________________________________________________________________

    Warning:
        Waterways that are in several waterlevel control areas are not (always) processed correctly.

    """
    global_value = float(global_value)

    # read geometries to geopandas
    gdf_areas = gpd.read_file(areas_path)
    gdf_branches = net_nc2gdf(net_nc_path, results=["1d_branches"])["1d_branches"]
    # todo: check projection both files

    initials = determine_initial(gdf_branches, gdf_areas, value_field)

    # prepare result to be writen to file
    df_branch = pd.DataFrame.from_dict(initials).T
    df_branch["branchId"] = [str(f) for f in df_branch.index]
    df_branch["numlocations"] = [len(ch) for ch in df_branch.chainage]
    df_branch = df_branch[df_branch.numlocations > 0]
    df_global = pd.DataFrame(
        data=[[value_type, value_unit, global_value]],
        columns=["quantity", "unit", "value"],
    )

    # write result with hydrolib-core
    writefile = OneDFieldModel(
        branch=df_branch.to_dict("records"), global_=df_global.to_dict("records")[0]
    )
    writefile.save(Path(output_path))
    print("Wegschrijven van initiele situatie gelukt")


## General functions


def determine_initial(gdf_branches, gdf_areas, level_field):
    """Function that determines and changes the initial fields of the D-Hydro project.

    Args:
        gdf_branches : GeoDataFrame
            GDF containing the model 1d branches.
        gdf_areas : GeoDataFrame
            GDF containing the areas with the initial water levels.
        level_field : string
            Name of the field that contains the initial water levels.

    Returns:
        list with new initial water levels.


    """
    nodata_value = -9999

    # prepare data by selecting columns
    gdf1 = gdf_branches[["id", "geometry"]]
    gdf2 = gdf_areas[[level_field, "geometry"]]

    # combine branches with areas
    try:
        gdf_union = gpd.overlay(gdf1, gdf2, how="union", keep_geom_type=True)
    except:
        gdf_union = gpd.overlay(gdf1, gdf2)
        print("Warning: Union failed. Fall back to overlay.")

    # prepare length and remove elements without length (slithers)
    gdf_union["length"] = gdf_union.geometry.length
    gdf_union = gdf_union[gdf_union["length"] > 0]

    # prepare list with branches
    initials = {}
    sortlist = gdf_union.id.unique()
    sortlist.sort()

    # loop over all branches
    for branch_id in sortlist:
        initials[branch_id] = {"chainage": [], "values": []}
        gdf_union_branch = gdf_union[gdf_union["id"] == branch_id]
        chainage = 0
        # simpel branch with only 1 part
        if len(gdf_union_branch) == 1:
            if gdf_union_branch[level_field].iloc[0] > nodata_value:
                initials[branch_id]["chainage"] += [chainage]
                initials[branch_id]["values"] += [gdf_union_branch[level_field].iloc[0]]
        # branches split into multiple parts
        else:
            # find original starting point of branch
            gdf_branches_org = gdf_branches.loc[gdf_branches["id"] == branch_id].iloc[0]
            coords_start = gdf_branches_org.geometry.coords[0]
            gdf_union_branch = gdf_union_branch.explode(index_parts=True, inplace=True)
            gdf_union_branch["coords_start"] = [
                xy.coords[0] for xy in gdf_union_branch["geometry"].tolist()
            ]  # todo: Gives a false positive warining. see https://stackoverflow.com/questions/26666919/add-column-in-dataframe-from-list
            # loop over branch parts, based on start and end coordinates
            for _ in range(len(gdf_union_branch)):
                # find correct first linepart, based on cooridinates
                part = gdf_union_branch[
                    gdf_union_branch["coords_start"] == coords_start
                ]
                part = part.iloc[0]
                # add to list when not nan since D-HYDRO does not support nan
                if part[level_field] > nodata_value:
                    initials[branch_id]["chainage"] += [chainage]
                    initials[branch_id]["values"] += [part[level_field]]
                # prepare next linepart based on chainage
                chainage = chainage + part.length
                coords_start = part.geometry.coords[-1]

    return initials


if __name__ == "__main__":

    dir = os.path.dirname(__file__)
    net_nc_path = os.path.join(dir, r"exampledata\Dellen\Model\dflowfm\dellen_net.nc")
    areas_path = os.path.join(dir, r"exampledata\Dellen\GIS\Peilgebied_Dellen.shp")
    value_field = "GPGWNTPL"
    value_type = "WaterLevel"
    value_unit = "m"
    global_value = 1.0
    output_path = r"C:\temp\Hydrolib\InitialWaterLevel.ini"

    initial_dhydro(
        net_nc_path,
        areas_path,
        value_field,
        value_type,
        value_unit,
        global_value,
        output_path=output_path,
    )
    print("Script finished")
