import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString

from hydrolib.core.dflowfm.crosssection.models import CrossLocModel


def create_branches(network_nc, output_folder=False):
    """
    Create geometry dataset of the branches of the DHydro network
    # TODO: use MDU to find net.nc file

    Args:
        network_nc: path to the _net.nc file of the DHydro network
        output_folder: either False (do not export) or a path to the desired output folder

    Returns:
        branches: A GeoDataFrame of the branches in the DHydro Network
    """
    ds = xr.open_dataset(network_nc)

    # network_key resembles the prefix of the geom_x, geom_y, etc elements in the net_nc. Logic of the exact prexif is currently unknown to script's developers.
    if "network1d_geom_x" in ds.keys():
        network_key = "network1d"
    elif "Network_geom_x" in ds.keys():
        network_key = "Network"
    else:
        network_key = "network"

    if "projected_coordinate_system" in ds.keys():
        crs = ds.projected_coordinate_system.EPSG_code
    else:
        warnings.warn(
            "Caution: no CRS found in network file, assuming default CRS (EPSG:28992 RD New)"
        )
        crs = "EPSG:28992"

    # Create points for all vertexes
    df = pd.concat(
        [
            pd.Series(ds[f"{network_key}_geom_x"].values),
            pd.Series(ds[f"{network_key}_geom_y"].values),
        ],
        axis=1,
    )
    df.columns = [f"{network_key}_geom_x", f"{network_key}_geom_y"]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df[f"{network_key}_geom_x"], df[f"{network_key}_geom_y"]
        ),
    )

    df_branches = pd.DataFrame(
        {
            "node_count": ds[f"{network_key}_geom_node_count"].values.astype(object),
            "branchid": ds[f"{network_key}_branch_id"].values.astype(str),
            "user_length": ds[f"{network_key}_edge_length"].values,
            "edge_start": ds[f"{network_key}_edge_nodes"][:, 0].values.astype(
                float
            ),  # added for route in spatial check
            "edge_end": ds[f"{network_key}_edge_nodes"][:, 1].values.astype(
                float
            ),  # added for route in spatial check
            "line_geometry": None,
            "start_node": None,
            "end_node": None,
        }
    )

    for j in range(len(df_branches)):
        if j == 0:
            start_node = 0
            end_node = 0 + df_branches["node_count"][j]
            df_branches.loc[j, "start_node"] = start_node
            df_branches.loc[j, "end_node"] = end_node
            linestring = LineString(
                gdf.iloc[df_branches["start_node"][j] : df_branches["end_node"][j]][
                    "geometry"
                ].values
            )
            with warnings.catch_warnings():  # This deprication warning is not relevant to this situation
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                df_branches.loc[j, "line_geometry"] = linestring
        else:
            start_node = df_branches["node_count"][:j].sum()
            end_node = start_node + df_branches["node_count"].iloc[j]
            df_branches.loc[j, "start_node"] = start_node
            df_branches.loc[j, "end_node"] = end_node
            linestring = LineString(
                gdf.iloc[df_branches["start_node"][j] : df_branches["end_node"][j]][
                    "geometry"
                ].values
            )
            with warnings.catch_warnings():  # This deprication warning is not relevant to this situation
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                df_branches.loc[j, "line_geometry"] = linestring

    branches = gpd.GeoDataFrame(
        df_branches, geometry=df_branches.line_geometry, crs=crs
    )

    del branches["line_geometry"]
    branches["branchid"] = branches["branchid"].str.strip()

    # add an empty column for 'Strahler'
    branches["Strahler"] = np.nan

    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        branches.to_file(output_folder / "branches.shp")
    return branches


def create_crosssections(branches, crossloc_ini, output_folder=False):
    """
    Create geometry dataset of the crosssection locations, by projecting the points on the branch-network.

    # TODO: use MDU to find crossloc file

    Args:
        branches: geodataframe of the network's branches
        crossloc_ini: pathname to the crosssection location (ini) file
        output_folder: either False (do not export) or a path to the desired output folder

    Returns:
        gdf_cross_loc: GeoDataFrame of the crosssection locations
    """
    crossloc_source = Path(crossloc_ini)
    cross_loc = pd.DataFrame(
        [cs.__dict__ for cs in CrossLocModel(crossloc_source).crosssection]
    )

    df_cross_loc = pd.merge(cross_loc, branches, on="branchid", how="left")
    gdf_cross_loc = gpd.GeoDataFrame(
        df_cross_loc, geometry="geometry", crs=branches.crs
    )
    gdf_cross_loc["length"] = gdf_cross_loc["geometry"].length
    gdf_cross_loc["scaled_offset"] = (
        gdf_cross_loc["chainage"].astype(float)
        / gdf_cross_loc["user_length"]
        * gdf_cross_loc["length"]
    )

    # Use chainage to find location of crosssections
    gdf_cross_loc["cross_loc_geom"] = gdf_cross_loc["geometry"].interpolate(
        gdf_cross_loc["scaled_offset"].astype(float)
    )
    gdf_cross_loc.rename(columns={"geometry": "branch_geometry"}, inplace=True)
    gdf_cross_loc = gdf_cross_loc[
        [
            "id",
            "branchid",
            "chainage",
            "scaled_offset",
            "definitionid",
            "cross_loc_geom",
            "user_length",
            "length",
        ]
    ]
    gdf_cross_loc = gpd.GeoDataFrame(
        gdf_cross_loc, geometry="cross_loc_geom", crs=branches.crs
    )

    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        gdf_cross_loc.to_file(output_folder / "crosssection_locations.shp")
    return gdf_cross_loc


def select_crosssection_locations(
    crosssection_locations, shapefile_path, output_folder=False
):
    """
    Make a selection of the crosssection locations based on a polygon (from shapefile)
    Assumes both crosssection_locations and the shapefile are in the same coordinate system.
    Args:
        crosssection_locations: GeoDataFrame of all crosssection locations
        shapefile_path: path to a shapefile with a polygon of the area to be selected

    Returns:
        selected_cross_loc: GeoDataFrame of the selected crosssection locations
    """
    shapefile_area = gpd.read_file(shapefile_path)
    selected_cross_loc = gpd.clip(crosssection_locations, shapefile_area)
    selected_cross_loc.rename(columns={"cross_loc_geom": "geometry"}, inplace=True)
    selected_cross_loc.rename(columns={"definitionid": "definition"}, inplace=True)
    selected_cross_loc = selected_cross_loc[["branchid", "definition", "geometry"]]
    selected_cross_loc = gpd.GeoDataFrame(
        selected_cross_loc, geometry="geometry", crs="EPSG:28992"
    )
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        selected_cross_loc.to_file(output_folder / "selected_cross_loc.shp")
    return selected_cross_loc
