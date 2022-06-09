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
    global_value = float(global_value)

    # ds = nc.Dataset(net_nc_path)
    gdf_areas = gpd.read_file(areas_path)
    # hier al reaches uitlezen

    gdf_branches = net_nc2gdf(net_nc_path, results=["1d_branches"])["1d_branches"]

    initials = determine_initial(gdf_branches, gdf_areas, value_field)
    # todo: check projection both files
    write_initial(initials, output_path, value_type, global_value)
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
        initials[branch_id] = {"chainage": [], "level": []}
        gdf_union_branch = gdf_union[gdf_union["id"] == branch_id]
        chainage = 0
        if len(gdf_union_branch) == 1:  # speed up processing
            if gdf_union_branch[level_field].iloc[0] > -9999:
                initials[branch_id]["chainage"] += [chainage]
                initials[branch_id]["level"] += [gdf_union_branch[level_field].iloc[0]]
        else:  # branches split into multiple parts
            # gdf_branches_org = gdf_branches.loc[branch_id]
            gdf_branches_org = gdf_branches.loc[gdf_branches["id"] == branch_id].iloc[0]
            coords_start = gdf_branches_org.geometry.coords[0]
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

    return initials


def write_initial(initials, output_location, value_type="WaterLevel", global_value=0.0):
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

        for branch_id in initials:
            initial = initials[branch_id]
            if len(initial["chainage"]) > 0:
                f.write("\n[Branch]\n    branchId              = " + branch_id + "\n")
                if len(initial["chainage"]) == 1 and initial["chainage"][0] == 0:
                    f.write(
                        "    values                = "
                        + "{:8.3f}".format(initial["level"][0])
                        + "\n"
                    )
                else:
                    f.write(
                        "    numLocations          = "
                        + str(len(initial["chainage"]))
                        + "\n"
                    )
                    f.write(
                        "    chainage              = "
                        + " ".join(["{:8.3f}".format(x) for x in initial["chainage"]])
                        + "\n"
                    )
                    f.write(
                        "    values                = "
                        + " ".join(["{:8.3f}".format(x) for x in initial["level"]])
                        + "\n"
                    )


if __name__ == "__main__":

    dir = os.path.dirname(__file__)
    net_nc_path = os.path.join(
        dir, r"exampledata\Zwolle-Minimodel\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    )
    areas_path = os.path.join(dir, r"exampledata\shapes\gebieden.shp")
    value_field = "Level"
    value_type = "WaterLevel"
    output_path = r"C:\temp\Hydrolib\InitialWaterLevel.ini"

    initial_dhydro(
        net_nc_path,
        areas_path,
        value_field,
        value_type,
        global_value=1.0,
        output_path=output_path,
    )
    print("Script finished")
