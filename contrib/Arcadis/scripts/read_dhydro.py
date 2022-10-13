import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString, Point, Polygon

from hydrolib.core.io import polyfile
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.net.models import Network
from hydrolib.core.io.polyfile import parser


def net_nc2gdf(
    net_ncs,
    results=[
        "1d_meshnodes",
        "1d_nodes",
        "1d_branches",
        "1d_edges",
        "2d_nodes",
        "2d_edges",
        "2d_faces",
        "1d2d_links",
    ],
):
    """This script reads an D-HYDRO *net.nc file and converts it to a dictionary with GeoDataFrames.

    Example:
        gdfs = net_nc2gdf("C:/temp/model/dflowfm/model_net.nc")

    Args:
        net_ncs: str
            Path to net.nc file
        results(optional): list
            List containing needed parameters out of the net.nc file.

    Returns:
        gdfs (dict): Dictionary with all geometries in as GeoDataFrames

    """

    if not set(results).issubset(
        [
            "1d_meshnodes",
            "1d_nodes",
            "1d_branches",
            "1d_edges",
            "2d_nodes",
            "2d_edges",
            "2d_faces",
            "1d2d_links",
        ]
    ):
        print("wrong input in results")

    # read nc model
    nc_path = Path(net_ncs)
    net = Network()
    nc_model = net.from_file(nc_path)

    print("Amersfoort aangenomen als projectie")
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
    if "1d_branches" in results:
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
            crs=EPSG,
        )

    # 1D edges
    if "1d_edges" in results:
        gdf_1d_edges = gpd.GeoDataFrame(
            {
                "id": np.arange(0, len(nc_model._mesh1d.mesh1d_edge_branch_id), 1),
                "branch": nc_model._mesh1d.mesh1d_edge_branch_id,
                "offset": nc_model._mesh1d.mesh1d_edge_branch_offset,
            },
            crs=EPSG,
            geometry=gpd.points_from_xy(
                nc_model._mesh1d.mesh1d_edge_x, nc_model._mesh1d.mesh1d_edge_y
            ),
        )

    ## 2D

    # create nodes
    dict_2d_nodes = dict(
        list(
            enumerate(
                zip(
                    nc_model._mesh2d.mesh2d_node_x,
                    nc_model._mesh2d.mesh2d_node_y,
                    nc_model._mesh2d.mesh2d_node_z
                    if len(nc_model._mesh2d.mesh2d_node_z) > 0
                    else [np.nan] * len(nc_model._mesh2d.mesh2d_node_x),
                )
            )
        )
    )
    if (
        "2d_nodes" in results
        or "2d_faces" in results
        or "2d_edges" in results
        or "1d2d_links" in results
    ):
        gdf_2d_nodes = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                nc_model._mesh2d.mesh2d_node_x,
                nc_model._mesh2d.mesh2d_node_y,
                nc_model._mesh2d.mesh2d_node_z
                if len(nc_model._mesh2d.mesh2d_node_z) > 0
                else [np.nan] * len(nc_model._mesh2d.mesh2d_node_x),
            ),
            crs=EPSG,
        )

    # create faces
    if "2d_faces" in results or "1d2d_links" in results:
        gdf_2d_faces = gpd.GeoDataFrame(
            pd.DataFrame(
                {
                    "X": nc_model._mesh2d.mesh2d_face_x,
                    "Y": nc_model._mesh2d.mesh2d_face_y,
                    "Z": nc_model._mesh2d.mesh2d_face_z
                    if len(nc_model._mesh2d.mesh2d_face_z) > 0
                    else [np.nan] * len(nc_model._mesh2d.mesh2d_face_x),
                }
            ),
            crs=EPSG,
            geometry=[
                Polygon([dict_2d_nodes[id] for id in np.delete(ids, ids < 0)])
                for ids in nc_model._mesh2d.mesh2d_face_nodes
            ],
        )

    # create edges
    if "2d_edges" in results:
        gdf_2d_edges = gpd.GeoDataFrame(
            pd.DataFrame(
                {
                    "X": nc_model._mesh2d.mesh2d_edge_x,
                    "Y": nc_model._mesh2d.mesh2d_edge_y,
                    "Z": nc_model._mesh2d.mesh2d_edge_z
                    if len(nc_model._mesh2d.mesh2d_edge_z) > 0
                    else [np.nan] * len(nc_model._mesh2d.mesh2d_edge_x),
                }
            ),
            crs=EPSG,
            geometry=[
                LineString([dict_2d_nodes[id] for id in np.delete(ids, ids == -999)])
                for ids in nc_model._mesh2d.mesh2d_edge_nodes
            ],
        )

    # create 1d2d links
    if "1d2d_links" in results:
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
            print("no 1d2d links in model")

    gdfs_results = {}
    if "1d_meshnodes" in results:
        gdfs_results["1d_meshnodes"] = gdf_mesh1d_nodes
    if "1d_nodes" in results:
        gdfs_results["1d_nodes"] = gdf_network1d_nodes
    if "1d_branches" in results:
        gdfs_results["1d_branches"] = gdf_branches
    if "1d_edges" in results:
        gdfs_results["1d_edges"] = gdf_1d_edges
    if "2d_nodes" in results:
        gdfs_results["2d_nodes"] = gdf_2d_nodes
    if "2d_edges" in results:
        gdfs_results["2d_edges"] = gdf_2d_edges
    if "2d_faces" in results:
        gdfs_results["2d_faces"] = gdf_2d_faces
    if "1d2d_links" in results:
        gdfs_results["1d2d_links"] = gdf_links

    # join branch information when present
    branches_path = os.path.join(os.path.dirname(net_ncs), "branches.gui")
    if "1d_branches" in gdfs_results.keys() and os.path.isfile(branches_path):
        branches_df = branch_gui2df(branches_path)
        gdfs_results["1d_branches"] = gdfs_results["1d_branches"].join(
            branches_df.set_index("name"), on="id"
        )

    return gdfs_results


def map_nc2gdf(input_path, param):
    """This script reads an D-HYDRO *map.nc file and converts it to a GeoDataFrame for a chosen parameter. Currently, not all parameters work.

    Example:
        gdf = map_nc2gdf("C:/temp/model/dflowfm/output/FlowFM_map.nc")

    Args:
        input_path: str
            Path to map.nc file
        param: str
            Chosen parameter by the user

    Returns:
        gdf: GeoDataFrame
            GeoDataFrame with chosen parameter output. With the point names as index and the timesteps as columns.

    """

    # Open mapfile
    ds = xr.open_dataset(input_path)
    EPSG = ds["projected_coordinate_system"].EPSG_code
    if EPSG == "EPSG:0":
        print("Geen projectie in het model, Amersfoort aangenomen")
        EPSG = "EPSG:28992"

    # User can give input to which parameter is needed.
    choice_params = [x for x in list(ds.variables) if x.startswith("mesh1d_")]
    choice_params.append([x for x in list(ds.variables) if x.startswith("Mesh2d_")])
    print("The possible parameters are:\n", choice_params)
    par = param

    # Check if user wants 1D points and store all data in geodataframe
    if ds[par].mesh[-2:] == "1d":
        prefix = (
            ds[par].mesh + "_" + ds[par].location
        )  # Make standard prefix to get all data
        xcor = ds[prefix + "_x"].data  # Get xdata based on prefix
        ycor = ds[prefix + "_y"].data  # Get ydata based on prefix
        if prefix.split("_")[1] == "node":
            id = [
                x.decode("utf-8").strip() for x in ds[prefix + "_id"].data
            ]  # Get ID's of the nodes
        else:
            id = ds["mesh1d_nEdges"].values

        time = ds["time"].data  # Get timedata
        data = ds[par].data  # Get defined data
        df = pd.DataFrame(data=data, index=time, columns=id).T
        geom = gpd.points_from_xy(list(xcor), list(ycor))
        gdf = gpd.GeoDataFrame(df, geometry=geom)

    # Check if user wants 2D points, this script takes the middle point of the grid cells as x-y values. Store all data in geodataframe.
    if ds[par].mesh[-2:] == "2d":
        prefix = (
            ds[par].mesh + "_"
        )  # + ds[par].location)    #Make standard prefix to get all data
        xcor = ds[prefix + "face_x"].data  # Get xdata based on prefix
        ycor = ds[prefix + "face_y"].data  # Get ydata based on prefix
        id = ds[prefix + "nFaces"].data  # Get ID's of the nodes
        time = ds["time"].data  # Get timedata
        data = ds[par].data  # Get defined data

        df = pd.DataFrame(data=data, index=time, columns=id).T
        geom = gpd.points_from_xy(list(xcor), list(ycor))
        gdf = gpd.GeoDataFrame(df, geometry=geom)  # Make geodataframe with geometry

    return gdf


def hisnc_2gdf(input_path):
    """This script reads an D-HYDRO *his.nc file and converts it to a dictionary containing several geodataframes for all the output.

    Example:
        gdf = hisnc_2gdf("C:/temp/model/dflowfm/output/FlowFM_his.nc")

    Args:
        input_path (str): Path to his.nc file

    Returns:
        gdfs: dict
            Dictionary containing separate GeoDataFrames of output his. With the point names as index and the timesteps as columns.

    """

    # Open hisfile
    ds = xr.open_dataset(input_path)
    EPSG = ds["projected_coordinate_system"].EPSG_code
    if EPSG == "EPSG:0":
        print("Geen projectie in het model, Amersfoort aangenomen")
        EPSG = "EPSG:28992"

    gdfs = {}

    # The HIS file consists of different datapoints. Where the obesrvation points i.e. stations contain data and the structures.
    # read general data
    time = ds["time"].data

    # Read all waterbalance data
    wbs = [x for x in list(ds.variables) if x.startswith("water_balance")]
    wb = gpd.GeoDataFrame(index=time)
    for par in wbs:
        data = ds[par].data
        wb[str(par)] = data
    gdfs["water_balance"] = wb

    # read observation points with data
    obspoint_id = [x.decode("utf-8").strip() for x in ds["station_id"].data]
    obspoint_xcor = ds["station_geom_node_coordx"].data
    obspoint_ycor = ds["station_geom_node_coordy"].data
    geom = gpd.points_from_xy(list(obspoint_xcor), list(obspoint_ycor))
    obsdatanames = [
        "waterlevel",
        "bedlevel",
        "waterdepth",
        "taus",
        "x_velocity",
        "y_velocity",
        "velocity_magnitude",
        "discharge_magnitude",
    ]
    for obsname in obsdatanames:
        data = ds[obsname].data
        if len(data) == len(time):
            df = pd.DataFrame(data=data, index=time, columns=obspoint_id).T
            gdf = gpd.GeoDataFrame(df, geometry=geom)
            gdfs[str(obsname)] = gdf
        else:
            print(obsname + " is empty, skipping")
            continue

    # Read structure data
    strucs = [
        "general_structure",
        "pump",
        "weirgen",
        "orifice",
        "bridge",
        "culvert",
        "uniweir",
    ]

    for struc in strucs:
        strucgdfs = {}
        # voor elke structure de data inladen en in een gdf, opslaan in strucgdfs
        struc_data = [x for x in list(ds.variables) if x.startswith(struc)]
        struc_id = [x.decode("utf-8").strip() for x in ds[struc + "_id"].data]
        struc_xcorall = ds[struc + "_geom_node_coordx"].data
        struc_ycorall = ds[struc + "_geom_node_coordy"].data

        # Get average middle points because it are now lines.
        # TODO: Point data omzetten naar polylines, wordt nu nog gemiddelde van genomen
        points = [Point(xy) for xy in zip(struc_xcorall, struc_ycorall)]
        struc_xcorgem = [
            sum([pair[0].x, pair[1].x]) / 2 for pair in zip(points[1::2], points[0::2])
        ]
        struc_ycorgem = [
            sum([pair[0].y, pair[1].y]) / 2 for pair in zip(points[1::2], points[0::2])
        ]
        avgpoints = [Point(xy) for xy in zip(struc_xcorgem, struc_ycorgem)]

        # TODO: Hij herkent nu nog niet de linestring format in geodataframe, daarom gemiddelde punt gepakt.
        # if ds[struc +"_geom"].geometry_type == "line":
        #     itx = iter(struc_xcorall)
        #     for x in itx:
        #         struc_xcorgem.append([x,next(itx)])
        #     ity = iter(struc_ycorall)
        #     for y in ity:
        #         struc_ycorgem.append([y,next(ity)])

        # Check if amount of xy coordinates matches with the geom_node_count:
        if sum(list(ds[struc + "_geom_node_count"].data)) == 2 * len(struc_ycorgem):
            print("Length is correct.")
        else:
            print("Length is INCORRECT: Please check")

        # Schrijf alle data weg naar een losse dataframe
        if struc == "pump":
            for struc_dat in struc_data[7:]:
                data = ds[struc_dat].data
                df = pd.DataFrame(data=data, index=time, columns=struc_id).T
                gdf = gpd.GeoDataFrame(df, geometry=avgpoints)
                strucgdfs[str(struc_dat)] = gdf
        else:
            for struc_dat in struc_data[5:]:
                data = ds[struc_dat].data
                df = pd.DataFrame(data=data, index=time, columns=struc_id).T
                gdf = gpd.GeoDataFrame(df, geometry=avgpoints)
                strucgdfs[str(struc_dat)] = gdf
        gdfs[str(struc)] = strucgdfs

    # import compound structures
    cmpnd_data = [x for x in list(ds.variables) if x.startswith("cmpstru")]
    cmpnd_id = [x.decode("utf-8").strip() for x in ds["cmpstru_id"].data]
    cmpndgdfs = {}
    for cmpnd_dat in cmpnd_data[1:]:
        data = ds[cmpnd_dat].data
        df = pd.DataFrame(data=data, index=time, columns=cmpnd_id).T
        cmpndgdfs[str(cmpnd_dat)] = df
    gdfs[str("cmpstru")] = cmpndgdfs

    # import laterals
    lat_data = [x for x in list(ds.variables) if x.startswith("lateral")]
    lats_id = [x.decode("utf-8").strip() for x in ds["lateral_id"].data]
    lats_xcor = ds["lateral_geom_node_coordx"].data
    lats_ycor = ds["lateral_geom_node_coordy"].data
    geom = gpd.points_from_xy(list(lats_xcor), list(lats_ycor))
    latsgdfs = {}
    for lat_dat in lat_data[5:]:
        data = ds[lat_dat].data
        df = pd.DataFrame(data=data, index=time, columns=lats_id).T
        gdf = gpd.GeoDataFrame(df, geometry=geom)
        latsgdfs[str(lat_dat)] = gdf
    gdfs[str("lateral")] = latsgdfs

    # Done

    return gdfs


def chainage2gdf(df, gdf_branches, chainage="chainage", x="x", y="y", branch_id="id"):
    """Gets dataframe as input, converts chainage to x,y datapoints.

    Args
        df : Pandas DataFrame
            containing data to be changed.
        gdf_branches : GeoDataFrame
            gdf containing branches with chainage.
        chainage : String, optional
            Name of the column where "chainage" is defined. The default is "chainage".
        x : str, optional
            Name of the column where "x" is defined. The default is "x".
        y : String, optional
            Name of the column where "y" is defined. The default is "y".
        branch_id : String, optional
            Name of the column where the branch_id is defined. The default is "id".

    Returns:
        gdf : GeoDataFrame
            gdf containing the data with xy.

    """
    # TODO andere objecten
    for index, row in df.iterrows():
        branchid = row.branchid
        chainage = float(row.chainage)
        if branchid != None:
            branch = gdf_branches[gdf_branches[branch_id] == branchid].iloc[0]
            geom = branch.geometry.interpolate(chainage)
            df.loc[index, [x, y]] = [geom.coords[0][0], geom.coords[0][1]]
        elif row.xcoordinates != None:
            df.loc[index, [x, y]] = [
                sum(row["xcoordinates"]) / len(row["xcoordinates"]),
                sum(row["ycoordinates"]) / len(row["ycoordinates"]),
            ]

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))
    return gdf


def branch_gui2df(branch_file):
    """Reads the branch_gui into a gdf


    Args:
        branch_file : str
            path to the branch.gui file.

    Returns:
        df : DataFrame
            DataFrame containing contents of the branch.gui file.

    """
    # branch file
    with open(branch_file, mode="r") as f:
        text = [x.strip().splitlines() for x in f.read().split("[Branch]") if x != ""]
        branches = []
        for branch in text:
            td = {}
            for item in branch:
                item = item.split("#")[0].split("=")
                td[item[0].strip()] = item[1].strip()
            branches.append(td)
        df = pd.DataFrame(branches)
    return df


def read_nc_data(ds, par):
    """Reads .nc data


    Args:
        ds : xarray.DataArray
            Data array of the nc file.
        par : str
            String of the chosen parameter to read.

    Returns:
        df : DataFrame
            DataFrame containing the time as index, columns with data.

    """

    data_params = [
        x
        for x in list(ds.variables)
        if x.startswith(ds[par].mesh + "_" + ds[par].location)
    ]
    ds_params_coords = list(ds[par].coords)[0:2]

    id = ds[par].mesh + "_" + ds[par].location + "_id"

    data = ds[par].data.tolist()
    index = ds["time"].data
    if id in ds.variables:
        columns = [
            id.tostring().decode("utf-8").strip() for id in ds.variables[id].data
        ]
    else:
        columns = list(range(len(data[0])))

    df = pd.DataFrame(data=data, index=index, columns=columns)
    return df


def pli2gdf(input_file):
    """reads .pli file into gdf.

    Args:
    input_file : str
        input pli file.

    Returns:
    gdf : GeoDataFrame
        .pli file containing contents.

    """
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


def read_locations(
    input_mdu,
    results=[
        "cross_sections_locations",
        "cross_sections_definition",
        "structures",
        "boundaries",
        "laterals",
    ],
):
    """Use an input_mdu to read all locations into a GeoDataFrame. The user can choose which locations to read. If none are defined, all locations are read.
    The user can choose cross section locations, cross section definitions, structures, boundaries, laterals.

    Args:
        input_mdu : Path()
            Path to input_mdu in dflowfm folder. The dflowfm needs to be cleaned to be read with hydrolib.
        results : gdfs (dict)
            Dictionary with all locations in as GeoDataFrames

    Returns:
        gdfs_results : gdfs (dict)
            Dictionary with all locations in as GeoDataFrames.

    """

    # initial results dictionary
    gdfs_results = {}

    # read fm model and net_nc
    fm = FMModel(input_mdu)
    netnc_path = os.path.join(input_mdu.parent, str(fm.geometry.netfile.filepath))
    gdfs = net_nc2gdf(netnc_path)

    # read cross sections
    if "cross_sections_locations" in results:
        crsloc = pd.DataFrame(
            [f.__dict__ for f in fm.geometry.crosslocfile.crosssection]
        )
        gdf_crslocs = chainage2gdf(crsloc, gdfs["1d_branches"])
        gdf_crslocs.dropna(axis=1, how="all", inplace=True)
        gdf_crslocs.drop(columns=["comments", "x", "y"], inplace=True)

        gdfs_results["cross_sections_locations"] = gdf_crslocs

    if "cross_sections_definition" in results:
        crsdef = pd.DataFrame([f.__dict__ for f in fm.geometry.crossdeffile.definition])
        gdfs_results["cross_sections_definition"] = crsdef
        # TODO: read the data of the cross section definitions

    # read structures
    if "structures" in results:

        structures = pd.DataFrame(
            [f.__dict__ for f in fm.geometry.structurefile[0].structure]
        )

        # TODO: na deze functie is "type" veranderd van soort structure naar een "Point"
        gdf_strucs = chainage2gdf(structures, gdfs["1d_branches"])
        gdf_strucs.dropna(axis=1, how="all", inplace=True)
        gdf_strucs.drop(columns=["comments", "structureids", "x", "y"], inplace=True)

        gdfs_results["structures"] = gdf_strucs

    # read boundaries
    if "boundaries" in results:
        bclocs = pd.DataFrame(
            [f.__dict__ for f in fm.external_forcing.extforcefilenew.boundary]
        )
        # get the geometry from the 1d nodes of the net_file
        geom = gdfs["1d_nodes"].geometry.loc[
            gdfs["1d_nodes"]["id"].isin(list(bclocs.nodeid))
        ]

        gdf_bclocs = gpd.GeoDataFrame(bclocs, geometry=geom.values)
        gdf_bclocs.dropna(axis=1, how="all", inplace=True)
        gdf_bclocs.drop(columns=["comments", "forcingfile"], inplace=True)

        gdfs_results["laterals"] = gdf_bclocs

    # read laterals
    if "laterals" in results:
        latlocs = pd.DataFrame(
            [f.__dict__ for f in fm.external_forcing.extforcefilenew.lateral]
        )
        gdf_lats = chainage2gdf(latlocs, gdfs["1d_branches"])

        gdf_lats.dropna(axis=1, how="all", inplace=True)
        gdf_lats.drop(columns=["comments", "name", "x", "y", "discharge"], inplace=True)

        gdfs_results["laterals"] = gdf_lats

    return gdfs_results


if __name__ == "__main__":
    inputf = Path(
        r"C:\Users\delanger3781\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\05 Gedeelde map WRIJ\03_Resultaten\20220214_DR49\modellen\DR49_Bronkhorst_1000\dflowfm\output\dr49_map.nc"
    )
    ds = xr.open_dataset(inputf)
    print(list(ds.variables))
    hisnc_2gdf(
        Path(
            r"C:\Users\delanger3781\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\05 Gedeelde map WRIJ\03_Resultaten\20220214_DR49\modellen\DR49_Bronkhorst_1000\dflowfm\output\dr49_his.nc"
        )
    )
