import os
from pathlib import Path

import geopandas as gpd
import netCDF4 as nc
import pandas as pd
from read_dhydro import net_nc2gdf

from hydrolib.core.io.net.models import Link1d2d, Mesh1d, Network
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

       Parameters:
           net_nc_path : str
               Path to input nc-file containing the D-hydro network
           areas_path : str
               Path to shape file with areas containing initial values
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
    """
    Function that determines and changes the initial fields of the D-Hydro project.

    Parameters
    ----------
    gdf_branches : GeoDataFrame
        GDF containing the model 1d branches.
    gdf_areas : GeoDataFrame
        GDF containing the areas with the initial water levels.
    level_field : string
        Name of the field that contains the initial water levels.

    Returns
    -------
    initial : list
        list containing new initial water levels.

    """
    nodata_value = -9999

    gdf1 = gdf_branches[["id", "geometry"]]
    gdf2 = gdf_areas[[level_field, "geometry"]]

    # TODO: Union nog oplossen dat het niet error geeft, de nan values (oude values) missen nu.
    try:
        gdf_union = gpd.overlay(
            gdf1, gdf2, how="union", keep_geom_type=True
        )  # gdf_union = gpd.sjoin(gdf1, gdf2)
    except:
        gdf_union = gpd.overlay(gdf1, gdf2)
        print("LET OP: Union niet gelukt. Oude waardes worden niet weggeschreven.")

    gdf_union["length"] = gdf_union.geometry.length

    # loop over all branches
    initials = {}
    sortlist = gdf_union.id.unique()
    sortlist.sort()
    for branch_id in sortlist:
        initials[branch_id] = {"chainage": [], "values": []}
        gdf_union_branch = gdf_union[gdf_union["id"] == branch_id]
        chainage = 0
        if len(gdf_union_branch) == 1:  # speed up processing
            if gdf_union_branch[level_field].iloc[0] > nodata_value:
                initials[branch_id]["chainage"] += [chainage]
                initials[branch_id]["values"] += [gdf_union_branch[level_field].iloc[0]]
        else:  # branches split into multiple parts
            # gdf_branches_org = gdf_branches.loc[branch_id]
            gdf_branches_org = gdf_branches.loc[gdf_branches["id"] == branch_id].iloc[0]
            coords_start = gdf_branches_org.geometry.coords[0]
            gdf_union_branch["coords_start"] = [
                xy.coords[0] for xy in gdf_union_branch["geometry"].tolist()
            ]

            for i in range(len(gdf_union_branch)):  # todo: i never used
                # find correct first linepart, based on cooridinates
                part = gdf_union_branch[
                    gdf_union_branch["coords_start"] == coords_start
                ]
                if len(part) == 0:  # needed since sometimes there are weird slithers
                    break
                part = part.iloc[0]
                if (
                    part[level_field] > nodata_value
                ):  # only add when not nan since D-HYDRO does not support nan
                    initials[branch_id]["chainage"] += [chainage]
                    initials[branch_id]["values"] += [part[level_field]]
                # prepare next linepart
                chainage = chainage + part.length
                coords_start = part.geometry.coords[-1]

    return initials


if __name__ == "__main__":

    dir = os.path.dirname(__file__)
    net_nc_path = os.path.join(
        dir, r"exampledata\Zwolle-Minimodel\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    )
    areas_path = os.path.join(dir, r"exampledata\shapes\gebieden.shp")
    value_field = "Level"
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
