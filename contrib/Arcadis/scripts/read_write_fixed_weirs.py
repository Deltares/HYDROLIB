import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.mask
import shapely
from shapely.geometry import (
    LineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import linemerge

from hydrolib.core.io import polyfile
from hydrolib.core.io.polyfile import parser


def shp2pli(input_file, output_file, id, values=[], write_z=True):
    """
    Convert a shapefile to a D-Hydro pli(z) file. This can be a shape with
    points or lines (not mixed). Several columns can be assigned. If there are
    z values, also these z values will be written.
    ___________________________________________________________________________________________________________

    Developer: A Buijert
    ___________________________________________________________________________________________________________

    Parameters
    ----------
    input_file : string
        Path to shapefile
    output_file : string
        Path to output file
    id : string
        Field with id's. Features with the same id will connected and merged in a single line.
    values:
        list of columns which need to be written. empty if only the z value needs to be written.
    ___________________________________________________________________________________________________________

    """
    gdf = gpd.read_file(input_file)
    values = values if isinstance(values, list) else [values]
    write_pli(gdf, output_file, id=id, values=values, write_z=write_z)


def write_pli(gdf, output, id="unique_id", values=[], write_z=True):
    values = values if isinstance(values, list) else [values]
    geom_type = set(gdf.geom_type)  # stil need to check for multi and mixed

    if write_z == True:
        if sum(gdf.geometry.has_z) == len(gdf):
            print("Z-waarde van geometry wordt wegeschreven")
        else:
            sys.exit("z-waarde niet voor alle elementen aanwezig")
    # create dictionary with all values
    print("Wegschrijven .pli voorbereiden")
    pli_dict = {}
    for ID in sorted(list(set(gdf[id]))):
        line = gdf[gdf[id] == ID]
        check_z = line.has_z
        pli_dict[ID] = []
        range_line = range(len(line))
        for i in range_line:
            if geom_type == {"Point"}:
                item = []
                item.append(line.iloc[i].x)
                item.append(line.iloc[i].y)
                if write_z and line.has_z:
                    item.append(line.iloc[i].z)
                for value in values:
                    item.append(line[i][value])
                pli_dict[ID].append(item)
            elif geom_type == {"LineString"}:
                range_coords = range(len(line.iloc[i].geometry.coords))
                for j in range_coords:  # loop over vertex in line segment
                    item = []
                    item.append(line.iloc[i].geometry.coords[j][0])
                    item.append(line.iloc[i].geometry.coords[j][1])
                    # start or end of entire line or middel of linepart
                    if (
                        (i == 0 and j == 0)
                        or (i == range_line[-1] and j == range_coords[-1])
                        or (i in range_line[1:-1] or j in range_coords[1:-1])
                    ):
                        if write_z and check_z.iloc[i]:
                            item.append(line.iloc[i].geometry.coords[j][2])
                        for value in values:
                            item.append(line.iloc[i][value])
                        pli_dict[ID].append(item)
                    # end of linepart
                    elif j == range_coords[-1]:
                        if write_z and check_z.iloc[i]:
                            values_list = [
                                line.iloc[i].geometry.coords[j][2],
                                line.iloc[i + 1].geometry.coords[0][2],
                            ]
                            values_list = [
                                value_list
                                for value_list in values_list
                                if value_list > -9999
                            ]
                            item.append(
                                -9999.0
                                if len(values_list) == 0
                                else sum(values_list) / len(values_list)
                            )
                        for value in values:
                            values_list = [line.iloc[i][value], line.iloc[i + 1][value]]
                            values_list = [
                                value_list
                                for value_list in values_list
                                if value_list > -9999
                            ]
                            item.append(
                                -9999.0
                                if len(values_list) == 0
                                else sum(values_list) / len(values_list)
                            )
                        pli_dict[ID].append(item)
                    else:
                        pass
                        # beginning of linepart, part of a larger line

    # write fixed weir .pli file
    print("Wegschrijven .pli gestart")
    with open(output, "w") as file:
        for ID in pli_dict:
            file.write(ID + "\n")
            line = pli_dict[ID]
            file.write("{:5}".format(len(line)) + "    " + str(len(line[0])) + "\n")
            for part in line:
                file_string = (
                    "{:11.3f}".format(part[0]) + " " + "{:11.3f}".format(part[1])
                )
                for i in part[2:]:
                    file_string += " " + "{:9.3f}".format(i)
                file.write(file_string + "\n")
            file.write("\n")
    print("Wegschrijven .pli afgerond (" + datetime.now().strftime("%H:%M:%S") + ")")


def pli2shp(input_file, output_file):
    gdf = pli2gdf(input_file)
    gdf.to_file(output_file)


def pli2gdf(input_file):
    # read pli file, including z value
    input_path = Path(input_file)
    pli_polyfile = polyfile.parser.read_polyfile(input_path, True)

    list = []
    for pli_object in pli_polyfile["objects"]:
        name = pli_object.metadata.name
        points = pli_object.points
        geometry = LineString(
            [[point.x, point.y, max(point.z, -9999)] for point in points]
        )  # convert nodata to -9999
        list.append({"name": name, "geometry": geometry})

    gdf = gpd.GeoDataFrame(list)

    return gdf


if __name__ == "__main__":
    input_file = r"C:\scripts\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\cleanup_pliz\ZwolleFWnewpli.pliz"
    output_path = r"C:\scripts\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\cleanup_pliz\zwolleFW_pli_fxw.shp"
    # aht_pli2shp(input_file, output_path)
    shp2pli(output_path, input_file, id="name", values=[], write_z=True)
