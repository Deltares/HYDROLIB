from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from hydrolib.core.io.net.models import Network


def net_nc2gdf(net_ncs):
    """
    This script reads an D-HYDRO *net.nc file and converts it to a dictionary with GeoDataFrames.

    Example:
        gdfs = net_nc2gdf("C:/temp/model/dflowfm/model_net.nc")

    Attributes:
        net_ncs (str): Path to net.nc file

    Result:
        gdfs (dict): Dictionary with all geometries in as GeoDataFrames

    """

    nc_path = Path(net_ncs)
    net = Network()
    nc_model = net.from_file(nc_path)

    # print("Geen projectie in het model, Amersfoort aangenomen")
    EPSG = "EPSG:28992"

    ## 1D
    # 1D nodes mesh
    gdf_mesh1d_nodes = gpd.GeoDataFrame(
        {
            "id": nc_model._mesh1d.mesh1d_node_id,
            "name": nc_model._mesh1d.mesh1d_node_long_name,
            "branch": nc_model._mesh1d.mesh1d_node_branch_id,
            "offset": nc_model._mesh1d.mesh1d_node_branch_offset,
        },
        crs=EPSG,
        geometry=gpd.points_from_xy(
            nc_model._mesh1d.mesh1d_node_x, nc_model._mesh1d.mesh1d_node_y
        ),
    )

    # 1D nodes
    gdf_network1d_nodes = gpd.GeoDataFrame(
        {
            "id": nc_model._mesh1d.network1d_node_id,
            "name": nc_model._mesh1d.network1d_node_long_name,
        },
        crs=EPSG,
        geometry=gpd.points_from_xy(
            nc_model._mesh1d.network1d_node_x, nc_model._mesh1d.network1d_node_y
        ),
    )

    # combine mesh_nodes with edge_nodes from network1d
    gdf_mesh_nodes_start = gpd.GeoDataFrame(
        [
            gdf_network1d_nodes.loc[i]
            for i in np.array(nc_model._mesh1d.network1d_edge_nodes)[:, 0]
        ],
        crs=EPSG,
    ).reset_index(drop=True)
    gdf_mesh_nodes_end = gpd.GeoDataFrame(
        [
            gdf_network1d_nodes.loc[i]
            for i in np.array(nc_model._mesh1d.network1d_edge_nodes)[:, 1]
        ],
        crs=EPSG,
    ).reset_index(drop=True)
    gdf_mesh_nodes_start.index.rename("branch", inplace=True)
    gdf_mesh_nodes_end.index.rename("branch", inplace=True)
    gdf_mesh_nodes_start.reset_index(inplace=True)
    gdf_mesh_nodes_end.reset_index(inplace=True)

    # 1D branches
    gdf_mesh_nodes_total = gpd.GeoDataFrame(
        pd.concat([gdf_mesh_nodes_start, gdf_mesh1d_nodes, gdf_mesh_nodes_end]),
        crs=EPSG,
    )
    gdf_mesh_nodes_total.drop_duplicates(
        subset=["branch", "geometry"], keep="first", inplace=True, ignore_index=True
    )
    geometry = gdf_mesh_nodes_total.groupby(["branch"])["geometry"].apply(
        lambda x: LineString(x.tolist())
    )
    gdf_branches = gpd.GeoDataFrame(
        {
            "id": nc_model._mesh1d.network1d_branch_id,
            "length": nc_model._mesh1d.network1d_branch_length,
            "name": nc_model._mesh1d.network1d_branch_long_name,
            "order": nc_model._mesh1d.network1d_branch_order,
        },
        geometry=geometry,
    )

    ## 2D
    dict_2d_nodes = dict(
        list(
            enumerate(
                zip(
                    nc_model._mesh2d.mesh2d_node_x,
                    nc_model._mesh2d.mesh2d_node_y,
                    nc_model._mesh2d.mesh2d_node_z,
                )
            )
        )
    )

    # create nodes
    gdf_2d_nodes = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            nc_model._mesh2d.mesh2d_node_x,
            nc_model._mesh2d.mesh2d_node_y,
            nc_model._mesh2d.mesh2d_node_z,
        ),
        crs=EPSG,
    )

    # create faces
    gdf_2d_faces = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "X": nc_model._mesh2d.mesh2d_face_x,
                "Y": nc_model._mesh2d.mesh2d_face_y,
                "Z": nc_model._mesh2d.mesh2d_face_z,
            }
        ),
        crs=EPSG,
        geometry=[
            Polygon([dict_2d_nodes[id] for id in np.delete(ids, ids == -999)])
            for ids in nc_model._mesh2d.mesh2d_face_nodes
        ],
    )

    # create edges
    gdf_2d_edges = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "X": nc_model._mesh2d.mesh2d_edge_x,
                "Y": nc_model._mesh2d.mesh2d_edge_y,
                "Z": nc_model._mesh2d.mesh2d_edge_z,
            }
        ),
        crs=EPSG,
        geometry=[
            LineString([dict_2d_nodes[id] for id in np.delete(ids, ids == -999)])
            for ids in nc_model._mesh2d.mesh2d_edge_nodes
        ],
    )

    # convert 1d2d links to dataframe
    l1d2d = nc_model._link1d2d
    if len(l1d2d.link1d2d_id) > 0:
        df_links = pd.DataFrame(
            [
                {
                    "id": l1d2d.link1d2d_id[i],
                    "name": l1d2d.link1d2d_long_name[i],
                    "type": l1d2d.link1d2d_contact_type[i],
                    "1D": l1d2d.link1d2d[i][0],
                    "2D": l1d2d.link1d2d[i][1],
                }
                for i in range(len(l1d2d.link1d2d_id))
            ]
        ).set_index("id")
        geometry = [
            LineString(
                [
                    [
                        gdf_mesh1d_nodes.loc[l1d2d.link1d2d[i][0]].geometry.x,
                        gdf_mesh1d_nodes.loc[l1d2d.link1d2d[i][0]].geometry.y,
                    ],
                    [
                        gdf_2d_faces.loc[l1d2d.link1d2d[i][1]].geometry.centroid.x,
                        gdf_2d_faces.loc[l1d2d.link1d2d[i][1]].geometry.centroid.y,
                    ],
                ]
            )
            for i in range(len(l1d2d.link1d2d_id))
        ]
        gdf_links = gpd.GeoDataFrame(df_links, geometry=geometry, crs=EPSG)
    else:  # no 1D2D links
        gdf_links = gpd.GeoDataFrame(
            columns=["id", "name", "type", "1D", "2D", "geometry"], crs=EPSG
        ).set_index("id")

    return {
        "1d_meshnodes": gdf_mesh1d_nodes,
        "1d_nodes": gdf_network1d_nodes,
        "1d_branches": gdf_branches,
        "2d_nodes": gdf_2d_nodes,
        "2d_edges": gdf_2d_edges,
        "2d_faces": gdf_2d_faces,
        "1d2d_links": gdf_links,
    }
