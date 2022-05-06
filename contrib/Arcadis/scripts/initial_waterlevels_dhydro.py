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
    gdfs = net_nc2gdf(ds)
    gdf_branch = gdfs["network_branch"]

    gdf1 = gdf_branch.drop(columns=gdf_branch.columns[:-1]).reset_index()
    gdf2 = gdf_areas[[level_field, "geometry"]]

    gdf_union = gpd.overlay(gdf1, gdf2, how="union")
    gdf_union["length"] = gdf_union.geometry.length

    initial = []
    for branch_id in gdf_branch.index.to_list():
        tot_length = 0
        offset = 0.1
        chainage = []
        values = []
        for index, row in gdf_union[gdf_union.id == branch_id].iterrows():
            # order seems not to be consistant
            level = float(row[level_field])
            if level > -9999:
                length = row["length"]
                # only write value if significant
                if length > 4 * offset:
                    values.extend([level, level])
                    chainage.extend([tot_length + offset, tot_length + length - offset])
                tot_length += length
        if len(chainage) > 0:
            initial.append([branch_id, chainage, values])

    return initial


def write_initial(initials, output_location, value_type, global_value=1.0):
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
