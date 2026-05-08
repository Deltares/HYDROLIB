from glob import glob
from pathlib import Path

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from shapely.geometry import Point

from hydrolib.core.dflowfm.mdu.models import FMModel


class Results:
    def __init__(self, model_result_folder):
        """General class to read NetCDF results for a DHydro model

        The class automatically parses the model_result_folder for the map.nc result file, and returns results
        on edges (discharge and velocity) and nodes (water leven and depth) on the LAST TIMESTEP of the results.

        Args:
            model_result_folder: folder with DHydro model results

        Variables:
            gdf_lines: A GeoDataFrame with results on edges of the final timestep (discharge and velocity)
            gdf_points: A GeoDataFrame with results on nodes of the final timestep (Water level and depth)

        Functions:
            result_gdf_lines: creates a geodataframe with result of the final timestep on edges (q & v)
            resut_gdf_points: creates a geodataframe with result of the final timestep on nodes (wl & wd)
            result_at_xy: looks up the results closest to this coordinate using a KDTree method.
            export_result_gpkg: exports the results (lines & points) as a geopackage
            export_result_shp: exports the results (lines & points) as a shapefile

        """

        self.model_folder = model_result_folder
        self.gdf_lines = self.result_gdf_lines(self.model_folder)
        self.gdf_points = self.result_gdf_points(self.model_folder)

    @staticmethod
    def result_gdf_lines(model_folder):
        map_nc = glob(f"{model_folder}/**_map.nc")[0]
        ds = nc.Dataset(str(map_nc))

        edges = {"x": "mesh1d_edge_x", "y": "mesh1d_edge_y"}
        var_q = "mesh1d_q1"
        var_v = "mesh1d_u1"

        x = list(ds.variables[edges["x"]][:])
        y = list(ds.variables[edges["y"]][:])
        coords = [Point(xi, yi) for xi, yi in zip(x, y)]

        # Let op! Laatste tijdstap
        q = ds.variables[var_q][-1, :].data
        v = ds.variables[var_v][-1, :].data

        gdf = gpd.GeoDataFrame(geometry=coords, data={"Q": q, "V": v}, crs="EPSG:28992")
        return gdf

    @staticmethod
    def result_gdf_points(model_folder):
        map_nc = glob(f"{model_folder}/**_map.nc")[0]
        ds = nc.Dataset(str(map_nc))

        edges = {"x": "mesh1d_node_x", "y": "mesh1d_node_y"}
        var_wl = "mesh1d_s1"
        # var_bl = "mesh1d_flowelem_bl"
        var_wd = "mesh1d_waterdepth"
        # var_fb = "mesh1d_freeboard"

        x = list(ds.variables[edges["x"]][:])
        y = list(ds.variables[edges["y"]][:])
        coords = [Point(xi, yi) for xi, yi in zip(x, y)]

        # Let op! Laatste tijdstap
        wl = ds.variables[var_wl][-1, :].data
        # bl = ds.variables[var_bl][-1,:].data
        wd = ds.variables[var_wd][-1, :].data
        # fb = ds.variables[var_fb][-1,:].data

        gdf = gpd.GeoDataFrame(
            geometry=coords, data={"WL": wl, "WD": wd}, crs="EPSG:28992"
        )
        return gdf

    def result_at_xy(self, x, y):
        xy = Point(x, y)
        datapoint = gpd.GeoDataFrame(
            geometry=[xy], data={"id": ["get_data"]}, crs="EPSG:28992"
        )
        nA = np.array(list(datapoint.geometry.apply(lambda x: (x.x, x.y))))

        nB = np.array(list(self.gdf_points.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist_b, idx = btree.query(nA, k=1)
        B_nearest = (
            self.gdf_points.iloc[idx].drop(columns="geometry").reset_index(drop=True)
        )

        nC = np.array(list(self.gdf_lines.geometry.apply(lambda x: (x.x, x.y))))
        ctree = cKDTree(nC)
        dist_c, idx = ctree.query(nA, k=1)
        C_nearest = (
            self.gdf_lines.iloc[idx].drop(columns="geometry").reset_index(drop=True)
        )

        gdf = pd.concat(
            [
                datapoint.reset_index(drop=True),
                B_nearest,
                C_nearest,
                pd.Series(dist_b, name="dist_pointdata"),
                pd.Series(dist_c, name="dist_linedata"),
            ],
            axis=1,
        )
        return gdf

    def export_result_gpkg(self, output_dir):
        self.gdf_lines.to_file(
            str(Path(output_dir) / "Results_on_lines.gpkg"), driver="GPKG"
        )
        self.gdf_points.to_file(
            str(Path(output_dir) / "Results_on_points.gpkg"), driver="GPKG"
        )

    def export_result_shp(self, output_dir):
        self.gdf_lines.to_file(str(Path(output_dir) / "Results_on_lines.shp"))
        self.gdf_points.to_file(str(Path(output_dir) / "Results_on_points.shp"))


def plot_profiles(
    bron_model, talud_profiel, geoptimaliseerde_bodembreedte, waterdiepte, profile_id
):
    # HUIDIG model
    base_model = FMModel(bron_model)
    cross_def = pd.DataFrame(
        [cs.__dict__ for cs in base_model.geometry.crossdeffile.definition]
    )
    profielen_yz = cross_def[cross_def["type"] == "yz"]
    profielen_yz = profielen_yz[profielen_yz["id"] == profile_id]

    # normaliseren aan de hand van laagste z punt
    profielen_yz["zcoordinates_normalized"] = ""
    profielen_yz["ycoordinates_normalized"] = ""

    for i in range(len(profielen_yz["id"])):
        # normalizeren over laagste punt z
        profielen_yz["zcoordinates_normalized"].iloc[i] = profielen_yz[
            "zcoordinates"
        ].iloc[i] - (
            np.ones(len(profielen_yz["zcoordinates"].iloc[i]))
            * min(profielen_yz.zcoordinates.iloc[i])
        )

        # normalizeren over y bij dat laagste punt z
        index_laagste_z = profielen_yz.zcoordinates.iloc[i].index(
            min(profielen_yz.zcoordinates.iloc[i])
        )
        profielen_yz["ycoordinates_normalized"].iloc[i] = (
            profielen_yz["ycoordinates"].iloc[i]
            - np.ones(len(profielen_yz["ycoordinates"].iloc[i]))
            * profielen_yz["ycoordinates"].iloc[i][index_laagste_z]
        )

    # GEOPTIMALISEERD model
    geoptimaliseerd_z = [waterdiepte, 0, 0, waterdiepte]
    geoptimaliseerd_y = [
        geoptimaliseerde_bodembreedte * -0.5 - talud_profiel * waterdiepte,
        geoptimaliseerde_bodembreedte * -0.5,
        geoptimaliseerde_bodembreedte * 0.5,
        geoptimaliseerde_bodembreedte * 0.5 + talud_profiel * waterdiepte,
    ]

    # FIGURE PROFIELEN
    fig = go.Figure()
    # huidige model
    fig.add_trace(
        go.Scatter(
            x=profielen_yz.ycoordinates_normalized.iloc[0],
            y=profielen_yz.zcoordinates_normalized.iloc[0],
            mode="lines+markers",
            name="huidig profiel",
        )
    )

    # geoptimaliseerde model
    fig.add_trace(
        go.Scatter(
            x=geoptimaliseerd_y,
            y=geoptimaliseerd_z,
            mode="lines+markers",
            name="geoptimaliseerd profiel",
        )
    )

    # figure layout
    fig.update_layout(title=profielen_yz.id.iloc[0])
    fig.update_xaxes(title_text="y", title_font=dict(size=18, color="crimson"))
    fig.update_yaxes(title_text="z", title_font=dict(size=18, color="crimson"))

    return fig
