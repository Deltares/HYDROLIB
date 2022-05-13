from shapely.geometry import Polygon, MultiPolygon
from typing import Union
from hydrolib.core.io.net.models import Network
from hydrolib.dhydamo.geometry.models import GeometryList
from hydrolib.dhydamo.geometry import common
import numpy as np


def add_2dmesh_rectilinear(
    network: Network,
    polygon: Union[Polygon, MultiPolygon],
    dx: float,
    dy: float,
    deletemeshoption: int = 1,
) -> None:

    # Loop over polygons if a MultiPolygon is given
    plist = common.as_polygon_list(polygon)
    if len(plist) > 1:
        for part in plist:
            add_2dmesh_rectilinear(network, part, dx, dy, deletemeshoption)
        return None

    # Store present 2d mesh (to be able to add)
    existing_mesh2d = network._mesh2d.get_mesh2d()

    # Create new network
    network.mesh2d_create_rectilinear_within_extent(
        extent=polygon.bounds,
        dx=dx,
        dy=dy,
    )

    # Clip and clean
    mesh2d_clip_and_clean(
        network=network,
        geometrylist=GeometryList.from_geometry(polygon),
        deletemeshoption=deletemeshoption,
        inside=False,
    )

    # Merge with existing network
    if existing_mesh2d.node_x.size > 0:
        new_mesh2d = network._mesh2d.get_mesh2d()
        # Modify count for indexing variables
        new_mesh2d.edge_nodes += existing_mesh2d.edge_nodes.max() + 1
        new_mesh2d.face_nodes += existing_mesh2d.edge_nodes.max() + 1
        # Add all variables to existing mesh
        variables = [
            "node_x",
            "node_y",
            "edge_nodes",
            "face_nodes",
            "nodes_per_face",
            "edge_x",
            "edge_y",
            "face_x",
            "face_y",
        ]
        for var in variables:
            setattr(
                existing_mesh2d,
                var,
                np.concatenate(
                    [getattr(existing_mesh2d, var), getattr(new_mesh2d, var)]
                ),
            )
        # Process merged mesh
        network._mesh2d._process(existing_mesh2d)


def add_2dmesh_triangular(
    network: Network, polygon: Union[Polygon, MultiPolygon], edge_length: float = None
) -> None:

    meshkernel = network._mesh2d.meshkernel
    for polygon in common.as_polygon_list(polygon):

        # Interpolate coordinates on polygon with edge_length distance
        if edge_length is not None:
            polygon = common.interp_polygon(polygon, dist=edge_length)

        # Add triangular mesh within polygon
        meshkernel.mesh2d_make_mesh_from_polygon(GeometryList.from_geometry(polygon))

    network._mesh2d._process(network._mesh2d.get_mesh2d())


def mesh2d_clip_and_clean(
    network: Network,
    geometrylist: GeometryList,
    deletemeshoption: int = 1,
    inside=True,
) -> None:
    network.mesh2d_clip_mesh(geometrylist, deletemeshoption, inside)

    # Remove hanging edges
    network._mesh2d.meshkernel.mesh2d_delete_hanging_edges()
    network._mesh2d._process(network._mesh2d.meshkernel.mesh2d_get())
