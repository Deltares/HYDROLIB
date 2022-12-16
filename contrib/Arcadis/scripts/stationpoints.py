# =========================================================================================
#
# License: LGPL
#
# Author: Arjon Buijert, Arcadis
#
# =========================================================================================

import sys

import geopandas as gpd
import pandas as pd


def create_stationpoints(input_lines, output_points, spacing, midpoint):
    """
    Generate interval points along line
    ___________________________________________________________________________________________________________

    Parameters:
        input_line : str
            Path to shapefile with lines
        output_points : str
            Path to new shapefile with created points
        spacing: float or string
            Spacing of the station points. Can be based on:
            - column name (must be text and present in input_line)
            - percentage (must be text with "%" als last character)
            - given length (must be string or float)
        midpoint: bool
            False = If the resulting lines should be equally spaced (midpoint = False)
            True = If the points define the centres of the equally spaced lines (relevant in case you want to define the locations (stationspoints) of the profiles on channel-lines)

            If input is for example line 0----0 (0 are the start and end nodes, line is 4x '-'long)
            False = 0--x--0 (x is a stationpoint, segments are both 2x '--' long)
            True = 0-x--x-0 (x are the stationpoints of the equally spaced segments
                             Imagine, first make two segments of 2x '--' long, then draw the centroids of these segements.
                             If we split the line at these 2 points, the segments have different lenghts)
    ___________________________________________________________________________________________________________

    Returns:
        Shapefile with stationpoints
    """

    gdf_lines = gpd.read_file(input_lines)
    midpoint = bool(midpoint)

    gdf_points = stationpoints(gdf_lines, spacing, midpoint)
    gdf_points.to_file(output_points)


def stationpoints(gdf_lines, spacing=100, midpoint=False):
    gdf_lines = gdf_lines.copy().reset_index(drop=False)
    if "level_0" in gdf_lines:
        gdf_lines = gdf_lines.drop(columns=["level_0"])
    if "level_1" in gdf_lines:
        gdf_lines = gdf_lines.drop(columns=["level_1"])

    gdf_lines.index.names = ["LineID"]

    # test if spacing is a column, percantage or fixed value
    try:
        if str(spacing) in gdf_lines.columns:
            dynamic_spacing = str(spacing)
        elif str(spacing)[-1] == "%":
            dynamic_spacing = float(str(spacing)[0:-1]) / 100
        else:
            spacing = float(spacing)
            dynamic_spacing = False
    except:
        print("spacing not present as column and not a value")
        sys.exit(1)

    points = []

    for index, row in gdf_lines.iterrows():
        line = gdf_lines.loc[[index]]
        line_length = float(line.length)

        # determine per line the distance between points
        if dynamic_spacing != False:
            if type(dynamic_spacing) == str:
                # op basis van een kolomnaam
                spacing = float(line.iloc[0][dynamic_spacing])
            elif type(dynamic_spacing) == float:
                # op basis van een percentage
                if midpoint == True:
                    spacing = line_length / ((1 / dynamic_spacing) + 1)
                else:
                    spacing = line_length * dynamic_spacing
            else:
                print("Fout in bepalen van afstand")

        # prepare equally spaces points, or equally spaced line parts
        if midpoint == False:
            devisions = int(line_length / spacing)
            part_length = line_length / max(devisions, 1)
            distance = part_length
        else:
            devisions = max(int(line_length / spacing + 0.5), 1)
            part_length = line_length / max(devisions - 1, 1)
            distance = part_length / 2

        # loop over the length of the line
        i = 0
        while distance < line_length:
            point = line.interpolate(distance).tolist()[
                0
            ]  # only keep the geometry of the point
            points.append(line[:-1].values.tolist() + [index, i, distance, point])
            distance = distance + part_length
            i = i + 1

    # create a geodataframe of the points
    df_points = pd.DataFrame(
        points, columns=["LineID", "PointID", "Distance", "geometry"]
    )
    gdf = gpd.GeoDataFrame(df_points, crs=gdf_lines.crs, geometry="geometry").set_index(
        "LineID", drop=True
    )

    # combine the point geometry with the original line data
    gdf_lines.drop(columns=["geometry"], inplace=True)
    gdf_lines = gdf_lines[
        gdf_lines.index.isin(gdf.index.unique())
    ].copy()  # in case lines are to short, they are not included in the stationpoints gdf, therefore also remove these from gdf_lines
    gdf = gpd.GeoDataFrame(pd.concat([gdf_lines, gdf], axis=1)).reset_index(drop=False)

    if "index" in gdf.columns:
        gdf.drop(columns=["index"], inplace=True)
    return gdf
