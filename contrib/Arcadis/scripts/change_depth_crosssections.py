# =========================================================================================
#
# License: LGPL
#
# Author: Stefan de Vries, Waterschap Drents Overijsselse Delta
#
# =========================================================================================
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from clean_dhydro import clean_crsdefs, split_duplicate_crsdef_references
from read_dhydro import read_locations
from shapely.geometry import LineString, Point
from stationpoints import stationpoints

from hydrolib.core.io.crosssection.models import CrossDefModel, CrossLocModel


def change_depth_crosssections(
    mdu_path,
    shape_path,
    column_horizontal,
    column_vertical,
    vertical_distance_type,
    output_path,
):

    """
    Adjust the depth of yz cross sections. Do this over a certain width (deepest part of the profile)

    One could think of two scenario's in which this tool is usefil
    - Deepen the bottom of the crosssection, e.g. due to maintenance/dredging activities
    - Heighten/increase the bottom level of the crosssection, e.g. due to sedimentation processes
    ___________________________________________________________________________________________________________

    Parameters:
         mdu_path : str
            Path to mdu file containing the D-hydro model structure
         shape_path : str
            Path to shape file with polygons describing in which areas to adjust the cross section. The
            user can define multiple polygons within the shapefile and these can have different setting
         column_horizontal : str
             Horizontal width (always in meters) over which the cross section bottom is adjusted. The
             horizontal width/section is always placed in the deepest part of (current) cross section.
             The tool identifies the lowest point of the profile and than looks to the left
             and right to find the lowest section (with the defined width)
         column_vertical : str
             column name in the shapefile that gives the dimensions with which the cross sections should be
             deepened/raised. column should contain float numbers.
             if type_column_vertical = 'distance'/'uniform', the unit is in meters.
             if type_column_vertical = 'referencelevel', the unit is a reference height with a vertical datum.
         vertical_vertical_distance_type : str, only three options
             The cross section can be adjusted vertically using three options. All options can be used to
             raise or lower the profile.
             - 'distance' --> indicates that the bottom of the profile for the user-defined width is flattened
             and moved to a certain level. This level is computed by adding up the bottom level of the profile plus
             the vertical displacement. Positive number is increasing the bottom level, negative number is
             lowering the bottom level. E.g. if the column_vertical contains the number -0.5 and the lowest point
             is currently +5 m NAP, the bottom of the cross section is flattened (width is user defined) and lies
             at +4.5 m NAP
             - 'referencelevel' --> indicates that the bottom of the profile for the user-defined width is flattened
             and moved to a certain reference height that is defined with a vertical datum. In the Netherlands,
             we use meters in Amsterdam Ordnance Datum or Normaal Amsterdams Peil (NAP). E.g. if the column_vertical
             contains the number 4, the bottom of the cross section is flattened (width is user defined) and lies
             at +4 m NAP. In this way, the bottom of the profile can lowered/raised. This can also vary within
             the width-section that is selected. Maybe a part of the section is raised, and another part is lowered.
             - 'uniform' --> indicates that the bottom is uniformly/evenly lowered/raised with a certain distance
             The existing variations/irregularities persist.
    ___________________________________________________________________________________________________________

    Returns:
        Updated crsdef.ini, crsloc.ini

    ___________________________________________________________________________________________________________
       Warning: currently only works for yz and zwRiver cross sections
       zwRiver cross sections are converted to yz and conveyance type
       'segmeted' is used

    """

    ## READ DATA
    # Read locations and definitions of cross sections
    gdfs_results = read_locations(
        mdu_path, ["cross_sections_locations", "cross_sections_definition"]
    )

    # Check which profiles should be adjusted
    crslocs = gdfs_results["cross_sections_locations"]
    print("Amersfoort aangenomen als projectie")
    crslocs.crs = "EPSG:28992"
    gdf_frict = gpd.read_file(shape_path)
    loc_ids = crslocs.loc[
        crslocs.intersects(gdf_frict.dissolve().geometry.iloc[0]), "id"
    ].tolist()  # select only the relevant cross section locations

    ## REMOVE SHARED DEFINITIONS
    crsdefs = gdfs_results["cross_sections_definition"]
    crslocs, crsdefs = split_duplicate_crsdef_references(crslocs, crsdefs, loc_ids)

    # clean crsdefs
    crsdefs = clean_crsdefs(crsdefs)

    ## Adjust selected profile definitions)
    print("Starting to adjust " + str(len(loc_ids)) + " profile definitions")
    for loc_id in loc_ids:  # Loop over the cross sections that are selected as relevant

        def_id = crslocs.loc[crslocs["id"] == loc_id, "definitionid"].iloc[
            0
        ]  # ID cross section definition
        index_loc = crslocs.loc[crslocs["id"] == loc_id, :].index.to_list()[
            0
        ]  # index of the cross section location
        index_def = crsdefs.loc[crsdefs["id"] == def_id, :].index.to_list()[
            0
        ]  # index of the cross section definition

        # Determine the right parameters for this profile
        vertical_distance = gdf_frict.loc[
            gdf_frict.intersects(crslocs.loc[index_loc, "geometry"]),
            column_vertical,
        ].to_list()[0]
        slot_width = gdf_frict.loc[
            gdf_frict.intersects(crslocs.loc[index_loc, "geometry"]),
            column_horizontal,
        ].to_list()[0]
        if (
            isinstance(vertical_distance, (int, float)) == False
            or isinstance(slot_width, (int, float)) == False
        ):
            raise Exception(
                "The input shapefile does not contain int/float numbers for the vertical/horizontal adjustment"
            )
        if (
            (vertical_distance_type != "distance")
            and (vertical_distance_type != "referencelevel")
            and (vertical_distance_type != "uniform")
        ):
            raise Exception(
                "The input does not contain the right key word to describe the type of vertical adjustment"
            )
            print('It is either "distance", "referencelevel" or "uniform"')
        if slot_width < 0:
            raise Exception("Section/slot width should be a positive number")

        if crsdefs.loc[index_def, "type"] == "zwRiver":
            # convert to yz profile
            y = []
            z = []
            if type(crsdefs.loc[index_def, "totalwidths"]) == list:
                if crsdefs.loc[index_def, "totalwidths"] != []:
                    crsdefs.at[index_def, "flowwidths"] = crsdefs.loc[
                        index_def, "totalwidths"
                    ]
            for i in range(int(crsdefs.loc[index_def, "numlevels"])):
                y = (
                    [crsdefs.loc[index_def, "flowwidths"][i] / 2 * -1]
                    + y
                    + [crsdefs.loc[index_def, "flowwidths"][i] / 2]
                )
                z = (
                    [crsdefs.loc[index_def, "levels"][i]]
                    + z
                    + [crsdefs.loc[index_def, "levels"][i]]
                )
            crsdefs.at[index_def, "zcoordinates"] = z
            crsdefs.at[index_def, "ycoordinates"] = y
            crsdefs.loc[index_def, "yzcount"] = len(z)
            crsdefs.loc[index_def, "frictionids"] = ["Channels"]  # will be overwritten
            crsdefs.loc[index_def, "sectioncount"] = 1
            crsdefs.loc[index_def, "type"] = "yz"
            crsdefs.loc[index_def, "thalweg"] = max(y) / 2
            crsdefs.loc[index_def, "singleValuedZ"] = 1
            crsdefs.loc[index_def, "conveyance"] = "segmented"

            crsdefs.loc[index_def, "numlevels"] = None
            crsdefs.loc[index_def, "levels"] = None
            crsdefs.loc[index_def, "flowwidths"] = None
            crsdefs.loc[index_def, "flowwidths"] = None
            crsdefs.loc[index_def, "leveeflowarea"] = None
            crsdefs.loc[index_def, "leveetotalarea"] = None
            crsdefs.loc[index_def, "leveecrestlevel"] = None
            crsdefs.loc[index_def, "leveebaselevel"] = None
            crsdefs.loc[index_def, "mainwidth"] = None
            crsdefs.loc[index_def, "fp1width"] = None
            crsdefs.loc[index_def, "fp2width"] = None
            crsdefs.loc[index_def, "frictiontypes"] = None
            crsdefs.loc[index_def, "frictionvalues"] = None

        if crsdefs.loc[index_def, "type"] == "yz":
            y = crsdefs.loc[index_def, "ycoordinates"].copy()
            z = crsdefs.loc[index_def, "zcoordinates"].copy()
            bot_value = min(z)

            if slot_width < (max(y) - min(y)):
                # if only a certain part of crs should be changed
                ## SELECT LOWEST POINT IN THE PROFILE
                # This will be the starting point when finding the optimal area to deepen/raise the profile
                bot_position = [i for i, zz in enumerate(z) if float(zz) == bot_value]
                if (
                    len(bot_position) > 2
                ):  # If there are two lowest points in de cross section, select the 2 most relevant ones
                    points_next_to_each_other = -1
                    for i in range(
                        len(bot_position) - 2
                    ):  # Check if the lowest points lie next to each other
                        if abs(bot_position[i + 1] - bot_position[i]) == 1:
                            points_next_to_each_other = i
                    if (
                        points_next_to_each_other > -1
                    ):  # If there are two lowest points next to each other, pick these
                        bot_position = bot_position[
                            points_next_to_each_other : points_next_to_each_other + 1
                        ]
                    else:  # Otherwise pick the first two lowest points
                        bot_position = bot_position[0:2]
                if len(bot_position) == 2:  # If there are two lowest points
                    if (y[bot_position[0]] - y[bot_position[1]]) <= slot_width:
                        index_left = bot_position[0]
                        index_right = bot_position[1]
                    else:  # If the distance between the lowest points is too large
                        if abs(bot_position[-1] - bot_position[0]) != 1:
                            # If the lowest points are not next to each other, pick point in the middle based on index (not distance)
                            bot_position = round(
                                abs(bot_position[-1] - bot_position[0]) / 2
                            )  #
                            index_left = bot_position
                            index_right = bot_position
                        else:
                            # If the lowest points are next to each other, add extra point in between them
                            ymiddle = [(y[bot_position[0]] + y[bot_position[1]]) / 2]
                            y = (
                                y[0 : bot_position[0] + 1]
                                + ymiddle
                                + y[bot_position[1] : -1]
                            )
                            z = (
                                z[0 : bot_position[0] + 1]
                                + bot_value
                                + z[bot_position[1] : -1]
                            )
                            index_left = bot_position[0] + 1
                            index_right = bot_position[0] + 1
                elif len(bot_position) == 1:  # If there is only one lowest point
                    # Find index of lowest point
                    index_left = bot_position[0]
                    index_right = bot_position[0]

                y_middle = (y[index_left] + y[index_right]) / 2
                crsdefs.loc[index_def, "thalweg"] = y_middle

                ## ADD EXTRA INTERMEDIATE POINTS IN THE PROFILE
                # This makes the profile line more detailed
                line = []
                for i in range(len(y)):
                    line.append([0, 0])
                    line[i][0] = y[i]
                    line[i][1] = z[i]
                line = LineString(line)
                line2 = gpd.GeoDataFrame(index=[0], data=[0])
                line2["geometry"] = line
                gdf_points = stationpoints(line2, 0.1, midpoint=False)
                # Add the coords from the points
                line = line2.iloc[0]["geometry"].coords[:]
                line += [list(p.coords)[0] for p in gdf_points["geometry"]]
                # Calculate the distance along the line for each point
                dists = [line2.iloc[0, :]["geometry"].project(Point(p)) for p in line]
                # Sort the coords based on the distances
                line = [p for (d, p) in sorted(zip(dists, line))]
                # convert tuple to list
                line = [list(i) for i in line]

                # determine new indices of lowest points based on the refined line
                index_left = [
                    i
                    for i in range(len(line))
                    if line[i] == [y[index_left], z[index_left]]
                ][0]
                index_right = [
                    i
                    for i in range(len(line))
                    if line[i] == [y[index_right], z[index_right]]
                ][0]

                ## FIND LOWEST SECTION OF THE PROFILE OF A CERTAIN WIDTH (USER-DEFINED)
                # first do an estimation, using the steps of the refined profile line
                # starting at lowest point(s), search to the left and right until the lowest section is found
                index_lowest_left = index_left
                index_lowest_right = index_right
                found = 0
                while found == 0:
                    section_width = float(line[index_right][0]) - float(
                        line[index_left][0]
                    )
                    if section_width < slot_width:
                        index_left_new = index_left - 1
                        index_right_new = index_right + 1
                        elevation_left = (
                            float(line[index_left_new][1])
                            if index_left_new > -1
                            else 999
                        )
                        elevation_right = (
                            float(line[index_right_new][1])
                            if index_right_new < len(line)
                            else 999
                        )

                        if (
                            elevation_right - elevation_left
                        ) > 0.01:  # if left point lies at least 1 cm lower
                            index_left = index_left_new
                        elif (
                            elevation_left - elevation_right
                        ) > 0.01:  # if right point lies at least 1 cm lower
                            index_right = index_right_new
                        else:  # if values are equal
                            if abs(
                                line[index_lowest_left][0] - line[index_left][0]
                            ) > abs(
                                line[index_right][0] - line[index_lowest_right][0]
                            ):  # if left point has already moved further away from starting point
                                index_right = index_right_new
                            else:  # if right point has already moved further away from starting point
                                index_left = index_left_new

                    section_width = float(line[index_right][0]) - float(
                        line[index_left][0]
                    )
                    if section_width >= slot_width:
                        found = 1

                # now, make a more refined estimation of the y and z coordinates of the lowest section
                line_line = LineString(line)
                y_left = round(
                    (line[index_right][0] + line[index_left][0]) / 2 - slot_width / 2, 4
                )
                y_left_line = LineString([[y_left, -999], [y_left, 999]])
                z_left = round(line_line.intersection(y_left_line).coords[0][1], 4)
                y_right = round(
                    (line[index_right][0] + line[index_left][0]) / 2 + slot_width / 2, 4
                )
                y_right_line = LineString([[y_right, -999], [y_right, 999]])
                z_right = round(line_line.intersection(y_right_line).coords[0][1], 4)

                ## ADJUST THE PROFILE IN THE SECTION THAT WAS FOUND, BASED ON THE PARAMETERS ENTERED BY THE USER
                pos1 = 0
                pos2 = len(y) - 1
                for i in range(len(y)):
                    if float(y[i]) < y_left:
                        pos1 = i
                    if float(y[i]) > y_right and pos2 == len(y) - 1:
                        pos2 = i

                if vertical_distance_type == "distance":
                    slot_value = bot_value + vertical_distance
                    y_slot = [y[pos1], y_left, y_left + 0.001, y_right - 0.001, y_right]
                    z_slot = [z[pos1], z_left, slot_value, slot_value, z_right]
                elif vertical_distance_type == "referencelevel":
                    slot_value = vertical_distance
                    y_slot = [y[pos1], y_left, y_left + 0.001, y_right - 0.001, y_right]
                    z_slot = [z[pos1], z_left, slot_value, slot_value, z_right]
                elif vertical_distance_type == "uniform":
                    if pos2 - pos1 > 1:  # if there are points in between pos2 and pos1
                        y_slot = y[pos1:pos2]
                        y_slot = (
                            [y_slot[0]]
                            + [y_left, y_left + 0.001]
                            + y_slot[1:-1]
                            + [y_right - 0.001, y_right]
                        )
                        z_slot = z[pos1:pos2]
                        temp = z_slot[1:-1]
                        temp = [i - vertical_distance for i in temp]
                        z_slot = (
                            [z_slot[0]]
                            + [z_left, z_left - vertical_distance]
                            + temp
                            + [z_right - vertical_distance, z_right]
                        )
                    else:
                        y_slot = y[pos1:pos2]
                        y_slot = (
                            [y_slot[0]]
                            + [y_left, y_left + 0.001]
                            + [y_right - 0.001, y_right]
                        )
                        z_slot = z[pos1:pos2]
                        z_slot = (
                            [z_slot[0]]
                            + [z_left, z_left - vertical_distance]
                            + [z_right - vertical_distance, z_right]
                        )
                y_slot = [round(i, 4) for i in y_slot]
                z_slot = [round(i, 4) for i in z_slot]
                y[pos1:pos2] = y_slot
                z[pos1:pos2] = z_slot
            else:
                if vertical_distance_type == "distance":
                    slot_value = bot_value + vertical_distance
                    z = [i * 0 + slot_value for i in z]
                elif vertical_distance_type == "referencelevel":
                    slot_value = vertical_distance
                    z = [i * 0 + slot_value for i in z]
                elif vertical_distance_type == "uniform":
                    crslocs.loc[index_loc, "shift"] = (
                        crslocs.loc[index_loc, "shift"] + vertical_distance
                    )

            ## PUT THE DATA BACK INTO THE DATAFRAME
            temp = crsdefs.loc[:, "ycoordinates"].to_list()
            temp[index_def] = y
            crsdefs["ycoordinates"] = pd.Series(temp)
            temp = crsdefs.loc[:, "zcoordinates"].to_list()
            temp[index_def] = z
            crsdefs["zcoordinates"] = pd.Series(temp)
            crsdefs.loc[index_def, "yzcount"] = len(z)

        else:  # this tool can currently only adjust yz-profiles
            print(
                "Cannot adjust profile definition "
                + str(def_id)
                + " at location "
                + str(loc_id)
                + " because it is not a yz or zwRiver-profile"
            )

    write_cross_section_data(crsdefs, crslocs, output_path)
    print("Finished adjusting " + str(len(loc_ids)) + " profile definitions")


def write_cross_section_data(crsdefs, crslocs, output_path):
    ## WRITE CROSS SECTION DATA
    # crsdefs = crsdefs.drop(columns=["waterlevel", "left_y_waterlevel", "right_y_waterlevel", "method"])
    print("Write cross section definition data")
    tempfile = os.path.join(output_path, "crsdef_temp.ini")
    crsdefs = crsdefs.replace({np.nan: None})
    crossdef_new = CrossDefModel(definition=crsdefs.to_dict("records"))
    crossdef_new.save(Path(tempfile))

    # open crsdef file and clean, remove unnecessary lines
    with open(tempfile, "r") as file:
        def_data = file.readlines()
    filename = os.path.join(output_path, "crsdef.ini")
    proftype = ""
    for num, line in enumerate(def_data, 1):
        if "\n" in line:
            line = line.replace("\n", "")
        if line != "":
            if "=" in line:
                if line.strip()[-1] == "=":
                    # Dont write line, this is a empty line
                    def_data[num - 1] = ""
                elif line.split("= ")[1].strip()[0] == "#":
                    # Dont write line, this is a empty line
                    def_data[num - 1] = ""
                elif line.split("= ")[0].strip() == "conveyance":
                    if line.split("=")[1].strip().split(" ")[0] == "segmented":
                        if (
                            proftype == "yz"
                            or proftype == "zwRiver"
                            or proftype == "zw"
                        ):
                            # Dont write line, default values for conveyance
                            def_data[num - 1] = ""
                else:
                    if line.split("= ")[0].strip() == "type":
                        proftype = line.split("= ")[1].strip()
                        if "#" in line.split("= ")[1].strip():
                            proftype = proftype("#")[0].strip()
    with open(filename, "w") as file:
        file.writelines(def_data)
    os.remove(tempfile)

    print("Write cross section location data")
    tempfile = os.path.join(output_path, "crsloc_temp.ini")
    crslocs = crslocs.replace({np.nan: None})
    crossloc_new = CrossLocModel(crosssection=crslocs.to_dict("records"))
    crossloc_new.save(Path(tempfile))

    # open crsloc file and clean, remove unnecessary lines
    with open(tempfile, "r") as file:
        def_data = file.readlines()
    filename = os.path.join(output_path, "crsloc.ini")
    for num, line in enumerate(def_data, 1):
        if "\n" in line:
            line = line.replace("\n", "")
        if line != "":
            if "=" in line:
                if line[-1] == "=":
                    # Dont write line, this is a empty line
                    def_data[num - 1] = ""
                elif line.split("= ")[1].strip()[0] == "#":
                    # Dont write line, this is a empty line
                    def_data[num - 1] = ""
                elif line.strip() == "locationtype = 1d":
                    # Dont write line, default values locationtype"
                    def_data[num - 1] = ""
    with open(filename, "w") as file:
        file.writelines(def_data)
    os.remove(tempfile)


if __name__ == "__main__":
    # Read shape
    mdu_path = Path(
        r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\flowFM.mdu"
    )
    shape_path = r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\shapes\change_fric.shp"
    column_vertical = "bdm_mNAP"
    vertical_vertical_distance_type = "referencelevel"
    column_horizontal = "breedte"
    output_path = r"C:\Users\devop\Desktop"
    change_depth_crosssections(
        mdu_path,
        shape_path,
        column_horizontal,
        column_vertical,
        vertical_vertical_distance_type,
        output_path,
    )
