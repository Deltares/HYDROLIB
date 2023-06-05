import logging
from typing import List, Union
from enum import Enum
from pathlib import Path
import numpy as np
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.prepared import prep

from hydrolib.core.dflowfm.net.models import Branch, Network
from hydrolib.core.dflowfm.net.reader import UgridReader
from dhydamo.geometry import common, rasterstats, spatial
from dhydamo.geometry.models import GeometryList

from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator

import geopandas as gpd

logger = logging.getLogger(__name__)


class FillOption(Enum):
    INTERPOLATE = "interpolate"
    FILL_VALUE = "fill_value"
    NEAREST = "nearest"


class RasterStatPosition(Enum):
    NODE = "node"
    FACE = "face"


def mesh2d_add_rectilinear(
    network: Network,
    polygon: Union[Polygon, MultiPolygon],
    dx: float,
    dy: float,
    deletemeshoption: int = 1,
) -> None:
    """Add 2d rectilinear mesh to network. A new network is created, clipped, and merged
    with the existing network.

    Args:
        network (Network): Network object to which the mesh is added
        polygon (Union[Polygon, MultiPolygon]): Geometry within which the mesh is generated
        dx (float): Horizontal mesh spacing
        dy (float): Vertical mesh spacing
        deletemeshoption (int, optional): Option for clipping mesh. Defaults to 1.

    Returns:
        _type_: _description_
    """

    # Loop over polygons if a MultiPolygon is given
    plist = common.as_polygon_list(polygon)
    if len(plist) > 1:
        for part in plist:
            mesh2d_add_rectilinear(network, part, dx, dy, deletemeshoption)
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
    mesh2d_clip(
        network=network,
        polygon=GeometryList.from_geometry(polygon),
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


def mesh2d_from_netcdf(network: Network, path: Union[Path, str]) -> None:
    reader = UgridReader(path)
    reader.read_mesh2d(network._mesh2d)


def mesh2d_add_triangular(
    network: Network, polygon: Union[Polygon, MultiPolygon], edge_length: float = None
) -> None:
    """Add triangular mesh to existing network. An orthogonal mesh is generated by the
    meshkernel, which likely means that the given geometry is not completely filled. The
    triangle discretization is determined based on the coordinates on the boundary of the
    provided geometry. Giving an edge_length will discretize the polygon for you, but
    you can also do this yourself.

    Args:
        network (Network): Network object to which the mesh is added
        polygon (Union[Polygon, MultiPolygon]): Geometry within which the mesh is generated
        edge_length (float, optional): Distance for which the polygon boundary is discretized (by approximation). Defaults to None.
    """

    meshkernel = network._mesh2d.meshkernel
    for polygon in common.as_polygon_list(polygon):
        # Interpolate coordinates on polygon with edge_length distance
        if edge_length is not None:
            polygon = common.interp_polygon(polygon, dist=edge_length)

        # Add triangular mesh within polygon
        meshkernel.mesh2d_make_mesh_from_polygon(GeometryList.from_geometry(polygon))

    network._mesh2d._process(network._mesh2d.get_mesh2d())


def mesh2d_clip(
    network: Network,
    polygon: Union[GeometryList, Union[Polygon, MultiPolygon]],
    deletemeshoption: int = 1,
    inside=True,
) -> None:
    """Clip the mesh (currently implemented for 2d) and clean remaining hanging edges.

    Args:
        network (Network): Network for which the mesh is clipped
        polygon (Union[GeometryList, Union[Polygon, MultiPolygon]]): Polygon within which the mesh is clipped
        deletemeshoption (int, optional): Options for deleting nodes inside/outside polygon. Defaults to 1.
        inside (bool, optional): Whether to clip inside or outside the polygon. Defaults to True.
    """

    if isinstance(polygon, GeometryList):
        geo = polygon.to_geometry()
        if not isinstance(geo, (Polygon, MultiPolygon)):
            raise TypeError(
                f"Expected to provided geometrylist to be interpreted as Polygon or MultiPolygon, not a {type(geo)}."
            )
    elif isinstance(polygon, (Polygon, MultiPolygon)):
        polygon = GeometryList.from_geometry(polygon)

    network.mesh2d_clip_mesh(polygon, deletemeshoption, inside)

    # Remove hanging edges
    network._mesh2d.meshkernel.mesh2d_delete_hanging_edges()
    network._mesh2d._process(network._mesh2d.meshkernel.mesh2d_get())


def mesh2d_refine(
    network: Network, polygon: Union[Polygon, MultiPolygon], steps: int
) -> None:
    """Refine mesh 2d within (list of) polygon or multipolygon, with a certain
    number of refinement steps.

    Args:
        network (Network): Network for which the mesh is clipped
        polygon (Union[GeometryList, Union[Polygon, MultiPolygon]]): Polygon within which the mesh is clipped
        steps (int): Number of steps in the refinement
    """
    # Check if any polygon contains holes (does not work)
    for polygon in common.as_polygon_list(polygon):
        if len(polygon.interiors) > 0:
            raise NotImplementedError(
                "Refining within polygons with holes does not work. Remove holes before using this function (e.g., polygon = Polygon(polygon.exterior))."
            )

    for polygon in common.as_polygon_list(polygon):
        network.mesh2d_refine_mesh(GeometryList.from_geometry(polygon), level=steps)


def mesh1d_add_branch_from_linestring(
    network: Network,
    linestring: LineString,
    node_distance: Union[float, int],
    name: Union[str, None] = None,
    structure_chainage: Union[List[float], None] = None,
    max_dist_to_struc: Union[float, None] = None,
) -> str:
    """Add branch to 1d mesh, from a LineString geometry.
    The branch is discretized with the given node distance.
    The position of a structure can be provided, just like the max distance
    of a mesh node to a structure

    Args:
        network (Network): Network to which the branch is added
        linestring (LineString): The geometry of the new branch
        node_distance (Union[float, int]): Preferred node distance between branch nodes
        name (Union[str, None], optional): Name of the branch. If not given, a name is generated. Defaults to None.
        structure_chainage (Union[List[float], None], optional): Positions of structures along the branch. Defaults to Union[float, None]=None.
        max_dist_to_struc (Union[float, None], optional): Max distance of a mesh point to a structure. Defaults to Union[float, None]=None.

    Returns:
        str: name of added branch
    """

    branch = Branch(geometry=np.array(linestring.coords[:]))
    branch.generate_nodes(
        mesh1d_edge_length=node_distance,
        structure_chainage=structure_chainage,
        max_dist_to_struc=max_dist_to_struc,
    )
    branchid = network.mesh1d_add_branch(branch, name=name)

    return branchid


def mesh1d_add_branches_from_gdf(
    network: Network,
    branches: gpd.GeoDataFrame,
    branch_name_col: str,
    node_distance: float,
    max_dist_to_struc: float = None,
    structures=None,
) -> None:
    """Function to generate branches from geodataframe

    Args:
        network (Network): The network to which the branches are added
        branches (gpd.GeoDataFrame): GeoDataFrame with branches
        branch_name_col (str): Name of the column in the GeoDataFrame with the branchnames
        node_distance (float): Preferred 1d mesh distance
        max_dist_to_struc (float, optional): Maximum distance to structure. Defaults to None.
        structures (gpd.GeoDataFrame, optional): GeoDataFrame with structures. Must contain a column branchid and chainage. Defaults to None.
    """

    # Create empty dictionary for structure chainage
    structure_chainage = {}

    # If structures are given, collect offsets per branch
    if structures is not None:
        # Get structure data from dfs
        ids_offsets = structures[["branchid", "chainage", "id"]].copy()
        idx = structures["branchid"] != ""
        if idx.any():
            logger.warning("Some structures are not linked to a branch.")
        ids_offsets = ids_offsets.loc[idx, :]

        # For each branch
        for branchid, group in ids_offsets.groupby("branchid")["chainage", "id"]:
            # Check if structures are located at the same offset
            u, c = np.unique(group.chainage, return_counts=True)
            if any(c > 1):
                logger.warning(
                    "Structures {} have the same location.".format(
                        ", ".join(group.loc[np.isin(group, u[c > 1]), "id"].tolist())
                    )
                )
            # Add to dictionary
            structure_chainage[branchid] = u

    # Loop over all branches, and add structures
    for branchname, geometry in zip(
        branches[branch_name_col].tolist(), branches["geometry"].tolist()
    ):
        # Create branch
        branch = Branch(geometry=np.array(geometry.coords[:]))
        # Generate nodes on branch
        branch.generate_nodes(
            mesh1d_edge_length=node_distance,
            structure_chainage=structure_chainage[branchname]
            if branchname in structure_chainage
            else None,
            max_dist_to_struc=max_dist_to_struc,
        )
        network.mesh1d_add_branch(branch, name=branchname)


def mesh1d_set_branch_order(network: Network, branchids: list, idx: int = None) -> None:
    """
    Group branch ids so that the cross sections are
    interpolated along the branch.

    Parameters
    ----------
    branchids : list
        List of branches to group
    idx : int
        Order number with which to update a branch
    """
    # Get the ids (integers) of the branch names given by the user
    branchidx = np.isin(network._mesh1d.network1d_branch_id, branchids)
    # Get current order
    branchorder = network._mesh1d.network1d_branch_order
    # Update
    if idx is None:
        branchorder[branchidx] = branchorder.max() + 1
    else:
        if not isinstance(idx, int):
            raise TypeError("Expected integer.")
        branchorder[branchidx] = idx
    # Save
    network._mesh1d.network1d_branch_order = branchorder


def links1d2d_add_links_1d_to_2d(
    network: Network,
    branchids: List[str] = None,
    within: Union[Polygon, MultiPolygon] = None,
    max_length: float = np.inf,
) -> None:
    """Function to add 1d2d links to network, by generating them from 1d to 2d.
    Branchids can be specified for 1d branches that need to be linked.
    A (Multi)Polygon can be provided were links should be made.

    Args:
        network (Network): Network in which the connections are made
        branchids (List[str], optional): List of branchid's to connect. If None, all branches are connected. Defaults to None.
        within (Union[Polygon, MultiPolygon], optional): Area within which connections are made. Defaults to None.
        max_length (float, optional): Max edge length. Defaults to None.
    """
    # Load 1d and 2d in meshkernel
    network._mesh1d._set_mesh1d()
    network._mesh2d._set_mesh2d()

    if within is None:
        # If not provided, create a box from the maximum bounds
        xmin = min(
            network._mesh1d.mesh1d_node_x.min(), network._mesh2d.mesh2d_node_x.min()
        )
        xmax = max(
            network._mesh1d.mesh1d_node_x.max(), network._mesh2d.mesh2d_node_x.max()
        )
        ymin = min(
            network._mesh1d.mesh1d_node_y.min(), network._mesh2d.mesh2d_node_y.min()
        )
        ymax = max(
            network._mesh1d.mesh1d_node_y.max(), network._mesh2d.mesh2d_node_y.max()
        )

        within = box(xmin, ymin, xmax, ymax)

    # If a 'within' polygon was provided, convert it to a geometrylist
    geometrylist = GeometryList.from_geometry(within)

    # Get the nodes for the specific branch ids
    node_mask = network._mesh1d.get_node_mask(branchids)

    # Get the already present links. These are not filtered on length
    npresent = len(network._link1d2d.link1d2d)

    # Generate links
    network._link1d2d._link_from_1d_to_2d(node_mask, polygon=geometrylist)

    # Filter the links that are longer than the max distance
    id1d = network._link1d2d.link1d2d[npresent:, 0]
    id2d = network._link1d2d.link1d2d[npresent:, 1]
    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x[id1d], network._mesh1d.mesh1d_node_y[id1d]],
        axis=1,
    )
    faces2d = np.stack(
        [network._mesh2d.mesh2d_face_x[id2d], network._mesh2d.mesh2d_face_y[id2d]],
        axis=1,
    )
    lengths = np.hypot(nodes1d[:, 0] - faces2d[:, 0], nodes1d[:, 1] - faces2d[:, 1])
    keep = np.concatenate(
        [np.arange(npresent), np.where(lengths < max_length)[0] + npresent]
    )
    _filter_links_on_idx(network, keep)

    # subtract 1 from both columns of the indices as this is added my meshkernel in the writing process
    # network._link1d2d.link1d2d[:, 1] = network._link1d2d.link1d2d[:, 1] - 1


def _filter_links_on_idx(network: Network, keep: np.ndarray) -> None:
    # Select the remaining links
    network._link1d2d.link1d2d = network._link1d2d.link1d2d[keep]
    network._link1d2d.link1d2d_contact_type = network._link1d2d.link1d2d_contact_type[
        keep
    ]
    network._link1d2d.link1d2d_id = network._link1d2d.link1d2d_id[keep]
    network._link1d2d.link1d2d_long_name = network._link1d2d.link1d2d_long_name[keep]


def links1d2d_add_links_2d_to_1d_embedded(
    network: Network,
    branchids: List[str] = None,
    within: Union[Polygon, MultiPolygon] = None,
) -> None:
    """Generates links from 2d to 1d, where the 2d mesh intersects the 1d mesh: the 'embedded' links.

    To find the intersecting cells in an efficient way, we follow we the next steps. 1) Get the
    maximum length of a face edge. 2) Buffer the branches with this length. 3) Find all face nodes
    within this buffered geometry. 4) Check for each of the corresponding faces if it crossed the
    branches.

    Args:
        network (Network): Network in which the links are made. Should contain a 1d and 2d mesh
        branchids (List[str], optional): List is branch id's for which the connections are made. Defaults to None.
        within (Union[Polygon, MultiPolygon], optional): Clipping polygon for 2d mesh that is. Defaults to None.

    """
    # Load 1d and 2d in meshkernel
    network._mesh1d._set_mesh1d()
    network._mesh2d._set_mesh2d()

    # Get the max edge distance
    nodes2d = np.stack(
        [network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1
    )
    edge_node_crds = nodes2d[network._mesh2d.mesh2d_edge_nodes]

    diff = edge_node_crds[:, 0, :] - edge_node_crds[:, 1, :]
    maxdiff = np.hypot(diff[:, 0], diff[:, 1]).max()

    # Create multilinestring from branches
    # branchnrs = np.unique(network._mesh1d.mesh1d_node_branch_id)
    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1
    )

    # Create a prepared multilinestring of the 1d network, to check for intersections
    mls = MultiLineString(nodes1d[network._mesh1d.mesh1d_edge_nodes].tolist())
    mls_prep = prep(mls)

    # Buffer the branches with the max cell distances
    area = mls.buffer(maxdiff)

    # If a within polygon is provided, clip the buffered area with this polygon.
    if within is not None:
        area = area.intersection(within)

    # Create an array with 2d facecenters and check which intersect the (clipped) area
    faces2d = np.stack(
        [network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y], axis=1
    )
    mpgl = GeometryList(*faces2d.T.copy())
    idx = np.zeros(len(faces2d), dtype=bool)
    for subarea in common.as_polygon_list(area):
        subarea = GeometryList.from_geometry(subarea)
        idx |= (
            network.meshkernel.polygon_get_included_points(subarea, mpgl).values == 1.0
        )

    # Check for each of the remaining faces, if it actually crosses the branches
    nodes2d = np.stack(
        [network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1
    )
    where = np.where(idx)[0]
    for i, face_crds in enumerate(nodes2d[network._mesh2d.mesh2d_face_nodes[idx]]):
        if not mls_prep.intersects(LineString(face_crds)):
            idx[where[i]] = False

    # Use the remaining points to create the links
    multipoint = GeometryList(
        x_coordinates=faces2d[idx, 0], y_coordinates=faces2d[idx, 1]
    )

    # Get the nodes for the specific branch ids
    # TODO: The node mask does not seem to work
    node_mask = network._mesh1d.get_node_mask(branchids)

    # Generate links
    network._link1d2d._link_from_2d_to_1d_embedded(node_mask, points=multipoint)


def links1d2d_add_links_2d_to_1d_lateral(
    network: Network,
    dist_factor: Union[float, None] = 2.0,
    branchids: List[str] = None,
    within: Union[Polygon, MultiPolygon] = None,
    max_length: float = np.inf,
) -> None:
    """Generate 1d2d links from the 2d mesh to the 1d mesh, with a lateral connection.
    If a link is kept, is determined based on the distance between the face center and
    the intersection with the 2d mesh exterior. By default, links with an intersection
    distance larger than 2 times the center to edge distance of the cell, are removed.
    Note that for a square cell with a direct link out of the cell (without passing any
    other cells) this max distance is sqrt(2) = 1.414. The default value of 2 provides
    some flexibility. Note that a link with more than 1 intersection with the 2d mesh
    boundary is removed anyway.

    Furthermore:
    - Branch ids can be specified to connect only specific branches.
    - A 'within' polygon can be given to only connect 2d cells within this polygon.
    - A max link length can be given to limit the link length.

    Args:
        network (Network): Network in which the links are made. Should contain a 1d and 2d mesh
        dist_factor (Union[float, None], optional): Factor to determine which links are kept (see description above). Defaults to 2.0.
        branchids (List[str], optional): List is branch id's for which the conncetions are made. Defaults to None.
        within (Union[Polygon, MultiPolygon], optional): Clipping polygon for 2d mesh that is. Defaults to None.
        max_length (float, optional): Max edge length. Defaults to None.
    """

    # Load 1d and 2d in meshkernel
    network._mesh1d._set_mesh1d()
    network._mesh2d._set_mesh2d()

    geometrylist = network.meshkernel.mesh2d_get_mesh_boundaries_as_polygons()
    mpboundaries = GeometryList(**geometrylist.__dict__).to_geometry()
    if within is not None:
        # If a 'within' polygon was provided, get the intersection with the meshboundaries
        # and convert it to a geometrylist
        # Note that the provided meshboundaries is a (list of) polygon(s). Holes are provided
        # as polygons as well, which dont make it a valid MultiPolygon
        if isinstance(mpboundaries, Polygon):
            geometrylist = GeometryList.from_geometry(
            MultiPolygon(
                [mpboundaries]
            ))
        else:
            geometrylist = GeometryList.from_geometry(
                MultiPolygon(
                    common.as_polygon_list(
                        [geom.intersection(within) for geom in mpboundaries.geoms]
                    )
                )
            )

    # Get the nodes for the specific branch ids
    node_mask = network._mesh1d.get_node_mask(branchids)

    # Get the already present links. These are not filtered subsequently
    npresent = len(network._link1d2d.link1d2d)

    # Generate links
    network._link1d2d._link_from_2d_to_1d_lateral(
        node_mask, polygon=geometrylist, search_radius=max_length
    )

    # If the provided distance factor was None, no further selection is needed, all links are kept.
    if dist_factor is None:
        return

    # Create multilinestring
    if isinstance(mpboundaries, Polygon):
        multilinestring = MultiLineString([mpboundaries.exterior])
    else: 
        multilinestring = MultiLineString([poly.exterior for poly in mpboundaries.geoms])

    # Find the links that intersect the boundary close to the origin
    id1d = network._link1d2d.link1d2d[npresent:, 0]
    id2d = network._link1d2d.link1d2d[npresent:, 1]

    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x[id1d], network._mesh1d.mesh1d_node_y[id1d]],
        axis=1,
    )
    faces2d = np.stack(
        [network._mesh2d.mesh2d_face_x[id2d], network._mesh2d.mesh2d_face_y[id2d]],
        axis=1,
    )
    nodes2d = np.stack(
        [network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1
    )
    face_node_crds = nodes2d[network._mesh2d.mesh2d_face_nodes[id2d]]

    # Calculate distance between face edge and face center
    x1 = np.take(face_node_crds, 0, axis=2)
    y1 = np.take(face_node_crds, 1, axis=2)
    face_node_crds[:] = np.roll(face_node_crds, 1, axis=1)
    x2 = np.take(face_node_crds, 0, axis=2)
    y2 = np.take(face_node_crds, 1, axis=2)
    x0, y0 = faces2d[:, 0], faces2d[:, 1]
    distance = (
        np.absolute((x2 - x1) * (y1 - y0[:, None]) - (x1 - x0[:, None]) * (y2 - y1))
        / np.hypot(x2 - x1, y2 - y1)
    ).mean(axis=1)

    # Check which links to keep
    keep = list(range(npresent))
    for i, (node1d, face2d, comp_dist) in enumerate(
        zip(nodes1d, faces2d, distance * dist_factor)
    ):
        isect = multilinestring.intersection(LineString([face2d, node1d]))

        # If the intersection is for some reason not a Point of Multipoint, skip it.
        if not isinstance(isect, (Point, MultiPoint)):
            continue

        # Skip the link if it has more than one intersection with the boundary
        isect_list = common.as_point_list(isect)
        if len(isect_list) != 1:
            continue

        # If the distance to the mesh 2d exterior intersection is smaller than
        # the compared distance, keep it.
        dist = np.hypot(*(face2d - isect_list[0]))
        if dist < comp_dist:
            keep.append(i)

    # Select the remaining links
    _filter_links_on_idx(network, keep)


def links1d2d_remove_within(
    network: Network, within: Union[Polygon, MultiPolygon]
) -> None:
    """Remove 1d2d links within a given polygon or multipolygon

    Args:
        network (Network): The network from which the links are removed
        within (Union[Polygon, MultiPolygon]): The polygon that indicates which to remove
    """

    # Create an array with 2d facecenters and 1d nodes, that form the links
    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1
    )[network._link1d2d.link1d2d[:, 0]]
    faces2d = np.stack(
        [network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y], axis=1
    )[network._link1d2d.link1d2d[:, 1]]

    # Create GeometryList MultiPoint object
    mpgl_faces2d = GeometryList(*faces2d.T.copy())
    mpgl_nodes1d = GeometryList(*nodes1d.T.copy())
    idx = np.zeros(len(network._link1d2d.link1d2d), dtype=bool)

    # Check which links intersect the provided area
    for polygon in common.as_polygon_list(within):
        subarea = GeometryList.from_geometry(polygon)
        idx |= (
            network.meshkernel.polygon_get_included_points(subarea, mpgl_faces2d).values
            == 1.0
        )
        idx |= (
            network.meshkernel.polygon_get_included_points(subarea, mpgl_nodes1d).values
            == 1.0
        )

    # Remove these links
    keep = ~idx
    _filter_links_on_idx(network, keep)


def links1d2d_remove_1d_endpoints(network: Network) -> None:
    """Method to remove 1d2d links from end points of the 1d mesh. The GUI
    will interpret every endpoint as a boundary conditions, which does not
    allow a 1d 2d link at the same node. To avoid problems with this, use
    this method.
    """
    # Can only be done after links have been generated
    if not list(network._mesh1d.network1d_node_id) or not list(
        network._mesh2d.mesh2d_face_nodes
    ):
        return None

    # Create an array with 2d facecenters and 1d nodes, that form the links
    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1
    )[network._link1d2d.link1d2d[:, 0]]

    faces2d = np.stack(
        [network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y], axis=1
    )[network._link1d2d.link1d2d[:, 1]]

    # Select 1d nodes that are only present in a single edge
    edge_nodes = network._mesh1d.network1d_edge_nodes
    edgeid, counts = np.unique(edge_nodes, return_counts=True)
    to_remove = edgeid[counts == 1]

    keep = np.ones(len(network._link1d2d.link1d2d), dtype=bool)
    for i in to_remove:
        network_node = (
            network._mesh1d.network1d_node_x[i],
            network._mesh1d.network1d_node_y[i],
        )
        if np.isin(network_node, nodes1d)[0]:
            index = np.argwhere(np.isin(nodes1d, network_node)).ravel()[0]
            keep[index] = False

    _filter_links_on_idx(network, keep)
    # # Create GeometryList MultiPoint object
    # mpgl_faces2d = GeometryList(*faces2d.T.copy())
    # mpgl_nodes1d = GeometryList(*nodes1d.T.copy())
    # idx = np.zeros(len(network._link1d2d.link1d2d), dtype=bool)

    # # Remove these links
    # keep = ~to_remove


def mesh2d_altitude_from_raster(
    network,
    rasterpath,
    where: RasterStatPosition = "face",
    stat="mean",
    fill_option: FillOption = "fill_value",
    fill_value=None,
):
    """
    Method to determine level of nodes

    This function works faster for large amounts of cells, since it does not
    draw polygons but checks for nearest neighbours (voronoi) based
    on interpolation.

    Note that the raster is not clipped. Any values outside the bounds are
    also taken into account.
    """

    if isinstance(fill_option, str):
        fill_option = FillOption(fill_option)

    if isinstance(where, str):
        where = RasterStatPosition(where)

    # Select points on faces or nodes
    logger.info("Creating GeoDataFrame of cell faces.")

    # Get coordinates where z-values are derived or centered
    xy = np.stack(
        [
            getattr(network._mesh2d, f"mesh2d_{where.value}_x"),
            getattr(network._mesh2d, f"mesh2d_{where.value}_y"),
        ],
        axis=1,
    )

    # Create cells as polygons
    xy_facenodes = np.stack(
        [network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1
    )
    cells = network._mesh2d.mesh2d_face_nodes
    nodatavalue = np.iinfo(cells.dtype).min
    indices = cells != nodatavalue
    indices = cells != -2147483648
    cells = [xy_facenodes[cell[index]] for cell, index in zip(cells, indices)]
    facedata = gpd.GeoDataFrame(geometry=[Polygon(cell) for cell in cells])

    if where == RasterStatPosition.FACE:
        # Get raster statistics
        facedata.index = np.arange(len(xy), dtype=np.uint32) + 1
        facedata["crds"] = [cell for cell in cells]

        df = rasterstats.raster_stats_fine_cells(rasterpath, facedata, stats=[stat])
        # Get z values
        zvalues = df[stat].values

    elif where == RasterStatPosition.NODE:
        logger.info(
            "Generating voronoi polygons around cell centers for determining raster statistics."
        )

        facedata = spatial.get_voronoi_around_nodes(xy, facedata)
        # Get raster statistics
        df = rasterstats.raster_stats_fine_cells(rasterpath, facedata, stats=[stat])
        # Get z values
        zvalues = df[stat].values

    isnan = np.isnan(zvalues)
    if isnan.any():
        # If interpolation or fill_option, but all are NaN, raise error.
        if isnan.all() and fill_option in [FillOption.NEAREST, FillOption.INTERPOLATE]:
            raise ValueError(
                "Only NaN values found, interpolation or nearest not possible."
            )

        # Fill fill_option values
        if fill_option == FillOption.FILL_VALUE:
            if fill_value is None:
                raise ValueError("Provide a fill_value (keyword argument).")
            # With default value
            zvalues[isnan] = fill_value

        elif fill_option == FillOption.NEAREST:
            # By looking for the nearest value in the grid
            # Create a KDTree of the known points
            tree = KDTree(data=xy[~isnan])
            idx = tree.query(x=xy[isnan])[1]
            zvalues[isnan] = zvalues[~isnan][idx]

        elif fill_option == FillOption.INTERPOLATE:
            if fill_value is None:
                raise ValueError(
                    "Provide a fill_value (keyword argument) to fill values that cannot be interpolated."
                )
            # By interpolating
            isnan = np.isnan(zvalues)
            interp = LinearNDInterpolator(
                xy[~isnan], zvalues[~isnan], fill_value=fill_value
            )
            zvalues[isnan] = interp(xy[isnan])

    # Set values to mesh geometry
    setattr(network._mesh2d, f"mesh2d_{where.value}_z", zvalues)
