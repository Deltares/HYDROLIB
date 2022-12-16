# =============================================================================
#
# License: LGPL
#
# Author: Arjon Buijert Arcadis
#
# =============================================================================

import sys
from datetime import datetime
import geopandas as gpd
from read_dhydro import pli2gdf


def shp2pli(input_file, output_file, id, values=[], write_z=True):
    """
    Convert a shapefile to a D-Hydro pli(z) file. This can be a shape with
    points or lines (not mixed). Several columns can be assigned. If there are
    z values, also these z values will be written.
    ___________________________________________________________________________________________________________

    Parameters:
        input_file : string
            Path to shapefile
        output_file : string
            Path to output file
        id : string
            Field with id's. Features with the same id will connected and merged in a single line.
        values:
            list of columns which need to be written. empty if only the z value needs to be written.
    ___________________________________________________________________________________________________________

    Returns:
        Function creates a .pli file at the location that is chosen by the user
        
    """
    gdf = gpd.read_file(input_file)
    values = values if isinstance(values, list) else [values]
    write_pli(gdf, output_file, id=id, values=values, write_z=write_z)


def pli2shp(input_file, output_file):
    gdf = pli2gdf(input_file)
    gdf.to_file(output_file)


def write_pli(gdf, output, id="unique_id", values=[], write_z=True):
    values = values if isinstance(values, list) else [values]
    geom_type = set(gdf.geom_type)  # stil need to check for multi and mixed

    if write_z == True:
        if sum(gdf.geometry.has_z) == len(gdf):
            print("z-value of geometry is written")
        else:
            sys.exit("not all elements have a z-value")
    # create dictionary with all values
    print("Preparing to write .pli file")
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
    print("Starting to write .pli file")
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
    print("Finished writing .pli file (" + datetime.now().strftime("%H:%M:%S") + ")")


if __name__ == "__main__":
    output_path = r"C:\scripts\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\cleanup_pliz\ZwolleFWnewpli.pliz"
    input_file = r"C:\scripts\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\cleanup_pliz\zwolleFW_pli_fxw.shp"
    shp2pli(input_file, output_path, id="name", values=[], write_z=True)
