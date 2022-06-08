import os

import geopandas as gpd
import netCDF4 as nc
import pandas as pd
from read_dhydro import net_nc2gdf

from hydrolib.core.io.net.models import Link1d2d, Mesh1d, Network


def initial_dhydro(
    net_nc_path, areas_path, value_field, value_type, global_value, output_path
):
    """
       Determine the initial situation for D-hydro based on waterlevel control areas.

       Parameters:
           net_nc_path : str
               Path to input nc-file containing the D-hydro network
           areas_path : str
               Path to shapefile with areas containing initial values
           value_field: str
               Column name of the shape containing intial values
           value_type: str
               Type of initial value (WaterLevel or WaterDepth)
           global_value: float
               Standard value for waterways that fall outside of the area
           output_path : str
               Path to results-file
    ___________________________________________________________________________________________________________
       Warning:
           Waterways that lie in several waterlevel control areas are not (always) processed correctly.

    """
    # TODO: order of split in level is not always right
    global_value = float(global_value)

    ds = nc.Dataset(net_nc_path)
    gdf_areas = gpd.read_file(areas_path)
    initials = determine_initial(ds, gdf_areas, value_field)
    # todo: check projection both files
    write_initial(initials, output_path, value_type, global_value)
    print("Wegschrijven van initiele situatie gelukt")


## General functions


def determine_initial(ds, gdf_areas, level_field):
    """
    Function that determines and changes the initial fields of the D-Hydro project.

    Parameters
    ----------
    ds : netCDF4 dataset
        Dataset of the net_nc file.
    gdf_areas : GeoDataFrame
        GDF containing the areas with the initial water levels.
    level_field : string
        Name of the field that contains the initial water levels.

    Returns
    -------
    initial : list
        list containing new initial water levels.

    """
    gdfs = net_nc2gdf(ds)
    gdf_branch = gdfs["network_branch"]

    gdf1 = gdf_branch.drop(columns=gdf_branch.columns[:-1]).reset_index()
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
        initials[branch_id] = {"chainage": [], "level": []}
        gdf_union_branch = gdf_union[gdf_union["id"] == branch_id]
        chainage = 0
        if len(gdf_union_branch) == 1:  # speed up processing
            if gdf_union_branch[level_field].iloc[0] > -9999:
                initials[branch_id]["chainage"] += [chainage]
                initials[branch_id]["level"] += [gdf_union_branch[level_field].iloc[0]]
        else:  # branches split into multiple parts
            gdf_branch_org = gdf_branch.loc[branch_id]
            coords_start = gdf_branch_org.geometry.coords[0]
            gdf_union_branch["coords_start"] = [
                xy.coords[0] for xy in gdf_union_branch["geometry"].tolist()
            ]
            for i in range(len(gdf_union_branch)):
                # find correct first linepart, based on cooridinates
                part = gdf_union_branch[
                    gdf_union_branch["coords_start"] == coords_start
                ]
                if len(part) == 0:  # needed since sometimes there are weird slithers
                    break
                part = part.iloc[0]
                if (
                    part[level_field] > -9999
                ):  # only add when not nan since D-HYDRO does not support nan
                    initials[branch_id]["chainage"] += [chainage]
                    initials[branch_id]["level"] += [part[level_field]]
                # prepare next linepart
                chainage = chainage + part.length
                coords_start = part.geometry.coords[-1]

        # for index, row in gdf_union[gdf_union["id"] == branch_id].iterrows():
        #    if row[level_field] > -9999:
        #        initials[branch_id]["chainage"] += [row["start"]]
        #        initials[branch_id]["level"] += [row[level_field]]

    return initials


def write_initial(initials, output_location, value_type, global_value=1.0):
    """
    Writer of initial water levels .ini file.

    Parameters
    ----------
    initials : list
        Output of determine_initial function.
    output_location : string
        Folder location used to store the output .ini file.
    value_type : TYPE
        Choice between WaterDepth and WaterLevel.
    global_value : float, optional
        Global initial water level value. The default is 1.0.

    Returns
    -------
    None.

    """
    with open(output_location, "w") as f:
        f.write(
            "[General]\n    fileVersion           = 2.00\n    fileType              = 1dField\n"
        )
        f.write(
            "\n[Global]\n    quantity              = "
            + value_type
            + "\n    unit                  = m\n    value                 = "
            + str(global_value)
            + "\n"
        )

        for initial in initials:
            f.write("\n[Branch]\n    branchId              = " + initial[0] + "\n")
            if len(set(initial[2])) == 1:
                f.write(
                    "    values                = "
                    + "{:8.3f}".format(initial[2][0])
                    + "\n"
                )
            else:
                f.write("    numLocations          = " + str(len(initial[2])) + "\n")
                f.write(
                    "    chainage              = "
                    + " ".join(["{:8.3f}".format(x) for x in initial[1]])
                    + "\n"
                )
                f.write(
                    "    values                = "
                    + " ".join(["{:8.3f}".format(x) for x in initial[2]])
                    + "\n"
                )


if __name__ == "__main__":
    net_nc_path = r"C:\D-Hydro\DR49_Doesburg_Noord_10000\dflowfm\structures.ini"
    areas_path = r"C:\D-Hydro\DR49_Doesburg_Noord_10000\dflowfm\40x40_dr49_ref_net.nc"
    value_field = r"C:\temp\export_dhydro"
    value_type = r"C:\D-Hydro\DR49_Doesburg_Noord_10000\dflowfm\crsloc.ini"
    output_path = r"C:\temp\initial/InitialWaterLevel.ini"

    initial_dhydro(
        net_nc_path,
        areas_path,
        value_field,
        value_type,
        global_value=1.0,
        output_path=output_path,
    )
