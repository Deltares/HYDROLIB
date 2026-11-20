"""Create and edit the 1D mesh, the 2D mesh and the 1D2D links of a D-Flow FM network."""

import logging
from collections.abc import Hashable
from enum import Enum
from pathlib import Path

import geopandas as gpd
import meshkernel as mk
import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from hydrolib.core.dflowfm.net.models import Branch, Mesh2d, Network
from hydrolib.core.dflowfm.net.reader import UgridReader
from hydrolib.dhydamo.geometry import common, spatial, zonal
from hydrolib.dhydamo.geometry.models import GeometryList

logger = logging.getLogger(__name__)


class FillOption(Enum):
    """How to fill nodes or faces without a value from the raster."""

    INTERPOLATE = "interpolate"
    FILL_VALUE = "fill_value"
    NEAREST = "nearest"


class RasterStatPosition(Enum):
    """Where a 2D mesh gets the raster statistic: at the nodes or at the faces."""

    NODE = "node"
    FACE = "face"


def mesh2d_add_rectilinear(
    network: Network,
    polygon: Polygon | MultiPolygon,
    dx: float,
    dy: float,
    deletemeshoption: mk.DeleteMeshOption = mk.DeleteMeshOption.INSIDE_NOT_INTERSECTED,
) -> None:
    """Add a 2D rectilinear mesh to the network.

    A new mesh is created over the bounds of each polygon, clipped to the polygon, and
    merged with the existing 2D mesh.

    Parameters
    ----------
    network : Network
        Network to which the mesh is added.
    polygon : Polygon or MultiPolygon
        Geometry within which the mesh is generated.
    dx : float
        Horizontal mesh spacing.
    dy : float
        Vertical mesh spacing.
    deletemeshoption : mk.DeleteMeshOption, optional
        Option for clipping the mesh. Defaults to
        ``mk.DeleteMeshOption.INSIDE_NOT_INTERSECTED``.
    """
    # Loop over polygons and create a grid for each polygon bounds
    mesh2d_hc = Mesh2d(meshkernel=mk.MeshKernel())
    polygon_list = GeometryList.from_geometry(polygon)
    for part in polygon_list.geoms:
        xmin, ymin, xmax, ymax = part.bounds
        rows = int((ymax - ymin) / dy)
        columns = int((xmax - xmin) / dx)
        if rows > 0 and columns > 0:
            mesh2d_hc.create_rectilinear(extent=part.bounds, dx=dx, dy=dy)

    # Store present 2d mesh (to be able to add)
    existing_mesh2d = network._mesh2d.get_mesh2d()

    # Create new network
    mesh2d_output = mesh2d_hc.get_mesh2d()
    network._mesh2d._set_mesh2d(
        mesh2d_output.node_x,
        mesh2d_output.node_y,
        mesh2d_output.edge_nodes,
    )

    # Clip and clean
    mesh2d_clip(
        network=network,
        polygon=polygon_list,
        deletemeshoption=deletemeshoption,
        inside=False,
    )

    # Merge with existing network
    if existing_mesh2d.node_x.size > 0:
        # Add all variables to existing mesh
        new_mesh2d = network._mesh2d.get_mesh2d()
        network._mesh2d._set_mesh2d(
            np.concatenate([existing_mesh2d.node_x, new_mesh2d.node_x]),
            np.concatenate([existing_mesh2d.node_y, new_mesh2d.node_y]),
            np.concatenate(
                [
                    existing_mesh2d.edge_nodes,
                    new_mesh2d.edge_nodes + existing_mesh2d.edge_nodes.max() + 1,
                ]
            ),
        )


def mesh2d_from_netcdf(network: Network, path: Path | str) -> None:
    """Read a 2d mesh from a UGRID netCDF file into the network.

    Parameters
    ----------
    network : Network
        Network to which the mesh is added.
    path : Path or str
        Path to the UGRID netCDF file.
    """
    reader = UgridReader(Path(path))
    reader.read_mesh2d(network._mesh2d)


def mesh2d_add_triangular(network: Network, polygon: Polygon | MultiPolygon, edge_length: float | None = None) -> None:
    """Add a triangular mesh to the network.

    meshkernel generates an orthogonal mesh, which likely means that the given geometry
    is not completely filled. The triangle size follows only from the spacing of the
    coordinates on the boundary of the geometry: triangles are small where boundary
    points are close together and larger where they are far apart. Giving an
    ``edge_length`` discretizes the polygon for you, but you can also do this yourself.

    Parameters
    ----------
    network : Network
        Network to which the mesh is added.
    polygon : Polygon or MultiPolygon
        Geometry within which the mesh is generated.
    edge_length : float, optional
        Distance at which the polygon boundary is discretized (by approximation). This
        is also the approximate edge length of the triangles. Defaults to None: the
        existing boundary coordinates determine the triangle size.
    """
    meshkernel = network._mesh2d.meshkernel
    for poly in common.as_polygon_list(polygon):
        # Interpolate coordinates on polygon with edge_length distance
        if edge_length is not None:
            poly = common.interp_polygon(poly, dist=edge_length)

        # Add triangular mesh within polygon
        # No scale_factor: meshkernel>=8.4 ignores it (Deltares/MeshKernelPy#291)
        meshkernel.mesh2d_make_triangular_mesh_from_polygon(_geomlist_from_polygon(poly))


def mesh2d_clip(
    network: Network,
    polygon: GeometryList | Polygon | MultiPolygon,
    deletemeshoption: mk.DeleteMeshOption = mk.DeleteMeshOption.INSIDE_AND_INTERSECTED,
    inside: bool = True,
) -> None:
    """Clip the mesh (currently implemented for 2d) and clean remaining hanging edges.

    Parameters
    ----------
    network : Network
        Network for which the mesh is clipped.
    polygon : GeometryList, Polygon or MultiPolygon
        Polygon within which the mesh is clipped.
    deletemeshoption : mk.DeleteMeshOption, optional
        Option for deleting nodes inside or outside the polygon. Defaults to
        ``mk.DeleteMeshOption.INSIDE_AND_INTERSECTED``.
    inside : bool, optional
        Delete the mesh inside the polygon (True) or outside it (False). Defaults to True.

    Raises
    ------
    TypeError
        If a GeometryList is not a Polygon or MultiPolygon.
    """
    if isinstance(polygon, GeometryList):
        geo = polygon.to_geometry()
        if not isinstance(geo, (Polygon, MultiPolygon)):
            raise TypeError(
                f"Expected to provided geometrylist to be interpreted as Polygon or MultiPolygon, not a {type(geo)}."
            )
    else:
        geo = polygon

    if inside:
        # Clip each single polygon
        for part in common.as_polygon_list(geo):
            network.mesh2d_clip_mesh(_geomlist_from_polygon(part), deletemeshoption, inside)
    else:
        # Keep exterior
        clip_pol_ext = _geomlist_from_multipolygon(geo, "exterior")
        if clip_pol_ext.x_coordinates.size > 0:
            network._mesh2d.meshkernel.mesh2d_delete(
                geometry_list=clip_pol_ext,
                delete_option=deletemeshoption,
                invert_deletion=True,
            )
        # # Clip interior
        clip_pol_int = _geomlist_from_multipolygon(geo, "interior")
        if clip_pol_int.x_coordinates.size > 0:
            network._mesh2d.meshkernel.mesh2d_delete(
                geometry_list=clip_pol_int,
                delete_option=deletemeshoption,
                invert_deletion=False,
            )

    # Remove hanging edges
    network._mesh2d.meshkernel.mesh2d_delete_hanging_edges()
    network._mesh2d.meshkernel.mesh2d_get()


def _geomlist_from_polygon(polygon: Polygon | MultiPolygon) -> mk.GeometryList:
    """Convert a shapely Polygon or MultiPolygon to a meshkernel GeometryList."""
    gl = GeometryList()
    gl = gl.from_geometry(polygon)
    return mk.GeometryList(
        gl.x_coordinates, gl.y_coordinates, gl.values, gl.geometry_separator, gl.inner_outer_separator
    )


def _geomlist_from_multipolygon(polygons: Polygon | MultiPolygon, mode: str) -> mk.GeometryList:
    """Convert the exterior or interior rings of polygons to a meshkernel GeometryList.

    Parameters
    ----------
    polygons : Polygon or MultiPolygon
        Polygon(s) to convert.
    mode : str
        ``"exterior"`` or ``"interior"``: which rings to convert.

    Returns
    -------
    mk.GeometryList
        The rings, separated by -999.

    Raises
    ------
    ValueError
        If the mode is not ``"exterior"`` or ``"interior"``.
    """
    poly_separator = -999
    x_crds: list[float] = []
    y_crds: list[float] = []
    for polygon in common.as_polygon_list(polygons):
        x_geom, y_geom = [], []
        if mode == "exterior":
            x_ext, y_ext = np.array(polygon.exterior.coords[:]).T
            x_geom.append(x_ext)
            y_geom.append(y_ext)
        elif mode == "interior":
            for int_geom in polygon.interiors:
                x_int, y_int = np.array(int_geom.coords[:]).T
                x_geom.append(x_int)
                y_geom.append(y_int)
        else:
            raise ValueError(f"Expected mode to be either 'interior' or 'exterior' (got: '{mode}')")

        for xc, yc in zip(x_geom, y_geom):
            if len(x_crds) > 0:
                x_crds.append(poly_separator)
                y_crds.append(poly_separator)
            x_crds.extend(xc)
            y_crds.extend(yc)

    gl = mk.GeometryList(x_coordinates=x_crds, y_coordinates=y_crds, inner_outer_separator=-998)

    return gl


def mesh2d_refine(
    network: Network,
    polygon: Polygon | MultiPolygon,
    steps: int,
    refine_parameters: dict | None = None,
) -> None:
    """Refine the 2D mesh within a polygon, with a number of refinement steps.

    Parameters
    ----------
    network : Network
        Network for which the mesh is refined.
    polygon : Polygon or MultiPolygon
        Polygon within which the mesh is refined.
    steps : int
        Number of steps in the refinement.
    refine_parameters : dict, optional
        Overrides for the default ``mk.MeshRefinementParameters`` attributes. Defaults
        to None.
    """
    parameters = mk.MeshRefinementParameters(
        refine_intersected=True,
        use_mass_center_when_refining=True,
        min_edge_size=1.0,
        refinement_type=2,
        connect_hanging_nodes=True,
        account_for_samples_outside_face=True,
        max_refinement_iterations=1,
        smoothing_iterations=5,
        max_courant_time=120.0,
        directional_refinement=False,
    )
    if refine_parameters is not None:
        for key in refine_parameters:
            setattr(parameters, key, refine_parameters[key])

    for poly in common.as_polygon_list(polygon):
        network.mesh2d_refine_mesh(_geomlist_from_polygon(poly), level=steps, parameters=parameters)


def mesh1d_add_branch_from_linestring(
    network: Network,
    linestring: LineString,
    node_distance: float,
    name: str | None = None,
    structure_chainage: list[float] | None = None,
    max_dist_to_struc: float | None = None,
) -> str:
    """Add a branch to the 1D mesh, from a LineString geometry.

    The branch is discretized with the given node distance. The position of structures
    can be provided, as well as the maximum distance of a mesh node to a structure.

    Parameters
    ----------
    network : Network
        Network to which the branch is added.
    linestring : LineString
        Geometry of the new branch.
    node_distance : float
        Preferred distance between the branch nodes.
    name : str, optional
        Name of the branch. Defaults to None: a name is generated.
    structure_chainage : list[float], optional
        Positions of structures along the branch. Defaults to None.
    max_dist_to_struc : float, optional
        Maximum distance of a mesh node to a structure. Defaults to None.

    Returns
    -------
    str
        Name of the added branch.
    """
    branch = Branch(geometry=np.array(linestring.coords[:]))
    branch.generate_nodes(
        mesh1d_edge_length=node_distance,
        structure_chainage=structure_chainage,
        max_dist_to_struc=max_dist_to_struc,
    )
    # hydrolib-core annotates mesh1d_add_branch with -> None, but it returns the branch name:
    # https://github.com/Deltares/HYDROLIB-core/issues/1313
    return network.mesh1d_add_branch(branch, name=name)  # type: ignore[func-returns-value, return-value]


def mesh1d_add_branches_from_gdf(
    network: Network,
    branches: gpd.GeoDataFrame,
    branch_name_col: str,
    node_distance: float,
    max_dist_to_struc: float | None = None,
    structures: pd.DataFrame | None = None,
) -> None:
    """Add the branches of a GeoDataFrame to the 1D mesh.

    With ``structures``, each branch gets a mesh node between its structures. Structures
    without a branch, or at the same location, are logged.

    Parameters
    ----------
    network : Network
        Network to which the branches are added.
    branches : gpd.GeoDataFrame
        Branches, with LineString geometries.
    branch_name_col : str
        Column with the branch names.
    node_distance : float
        Preferred distance between the mesh nodes.
    max_dist_to_struc : float, optional
        Maximum distance of a mesh node to a structure. Defaults to None.
    structures : pd.DataFrame, optional
        Structures, with the columns ``id``, ``branchid`` and ``chainage``. Defaults to
        None.
    """
    # Create empty dictionary for structure chainage
    structure_chainage: dict[Hashable, list[float]] = {}

    # If structures are given, collect offsets per branch
    if structures is not None:
        # Get structure data from dfs
        ids_offsets = structures[["branchid", "chainage", "id"]].copy()
        idx = structures["branchid"] != ""
        missing = structures.branchid.isna()
        if missing.any():
            n_missing = int(missing.sum())
            if n_missing > 5:
                logger.warning(
                    "%d structures are not linked to a branch (too many to show).",
                    n_missing,
                )
            else:
                logger.warning(
                    "%d structures (%s) are not linked to a branch.",
                    n_missing,
                    structures.loc[missing, "id"].to_list(),
                )
        ids_offsets = ids_offsets.loc[idx, :]

        # For each branch
        for branchid, group in ids_offsets.groupby("branchid")[["chainage", "id"]]:
            # Check if structures are located at the same offset
            u, c = np.unique(group.chainage, return_counts=True)
            if any(c > 1):
                logger.warning(
                    "Structures %s have the same location.",
                    ", ".join(str(v) for v in group.loc[group.chainage.isin(u[c > 1]), "id"].tolist()),
                )
            # Add to dictionary
            structure_chainage[branchid] = u.tolist()

    # Loop over all branches, and add structures
    for branchname, geometry in zip(branches[branch_name_col].tolist(), branches["geometry"].tolist()):
        # Create branch
        branch = Branch(geometry=np.array(geometry.coords[:]))
        # Generate nodes on branch
        branch.generate_nodes(
            mesh1d_edge_length=node_distance,
            structure_chainage=structure_chainage.get(branchname, None),
            max_dist_to_struc=max_dist_to_struc,
        )
        network.mesh1d_add_branch(branch, name=branchname)


def mesh1d_order_numbers_from_attribute(
    branches: gpd.GeoDataFrame,
    missing: list[str],
    order_attribute: str,
    network: Network,
    exceptions: list[str] | None = None,
) -> list[str]:
    """Give branches with the same attribute value the same order number.

    Cross sections are interpolated along branches with the same order number. Each
    attribute value that has at least one branch with a cross section gets a number.
    The column ``order`` is added to ``branches``.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
        HyDAMO branches, with a ``code`` column.
    missing : list[str]
        Codes of the branches that have no cross section so far.
    order_attribute : str
        Column of ``branches`` by which the order numbers are given.
    network : Network
        Network to which the order numbers are written.
    exceptions : list[str], optional
        Codes of the branches that get no order number. Defaults to None.

    Returns
    -------
    list[str]
        Sorted codes of the branches that still have no cross section.
    """
    if exceptions is None:
        exceptions = []

    # order numbers start at 1: only numbers > 0 are written to the network below
    j = 1
    branches["order"] = np.nan
    for value in branches[order_attribute].unique():
        if value is not None and not all(
            x in missing for x in branches.loc[branches.loc[:, order_attribute] == value, "code"]
        ):
            branches.loc[branches.loc[:, order_attribute] == value, "order"] = int(j)
            j = j + 1
    for exception in exceptions:
        branches.loc[branches.code == exception, "order"] = -1

    interpolation: list[str] = []
    for ordernr in branches.order.unique():
        if ordernr > 0:
            mesh1d_set_branch_order(
                network,
                branches.code[branches.order == ordernr].to_list(),
                idx=int(ordernr),
            )
            interpolation = interpolation + branches.code[branches.order == ordernr].to_list()
    missing_after_interpolation = np.setdiff1d(missing, interpolation)
    return missing_after_interpolation.tolist()


def mesh1d_set_branch_order(network: Network, branchids: list[str], idx: int | None = None) -> None:
    """Give branches the same order number, so cross sections are interpolated along them.

    Parameters
    ----------
    network : Network
        Network in which the branch order is set.
    branchids : list[str]
        Branches to group.
    idx : int, optional
        Order number to give the branches. Defaults to None: one more than the current
        highest order number.

    Raises
    ------
    TypeError
        If ``idx`` is not an integer.
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
    branchids: list[str] | None = None,
    within: Polygon | MultiPolygon | None = None,
    max_length: float = np.inf,
) -> None:
    """Add 1D2D links to the network, generated from the 1D nodes to the 2D faces.

    Parameters
    ----------
    network : Network
        Network in which the links are made.
    branchids : list[str], optional
        Branches to link. Defaults to None: all branches.
    within : Polygon or MultiPolygon, optional
        Area within which links are made. Defaults to None: the extent of both meshes.
    max_length : float, optional
        Maximum length of a link. Defaults to ``np.inf``.
    """
    # Fetch the 2d mesh once and reuse it below - each network._mesh2d.mesh2d_*
    # property otherwise re-fetches and re-validates the whole mesh from meshkernel.
    mesh2d_output = network._mesh2d.get_mesh2d()

    if within is None:
        # If not provided, create a box from the maximum bounds
        xmin = min(network._mesh1d.mesh1d_node_x.min(), mesh2d_output.node_x.min())
        xmax = max(network._mesh1d.mesh1d_node_x.max(), mesh2d_output.node_x.max())
        ymin = min(network._mesh1d.mesh1d_node_y.min(), mesh2d_output.node_y.min())
        ymax = max(network._mesh1d.mesh1d_node_y.max(), mesh2d_output.node_y.max())

        within = box(xmin, ymin, xmax, ymax)

    # If a 'within' polygon was provided, convert it to a geometrylist
    geometrylist = _geomlist_from_polygon(within)

    # Get the nodes for the specific branch ids
    node_mask = network._mesh1d.get_node_mask(branchids)

    # Snapshot the already present links. They are restored after the call below
    # regardless of whether that call preserves them itself (see _filter_links_on_idx).
    present_links = network._link1d2d.link1d2d

    # Generate links
    network._link1d2d._link_from_1d_to_2d(node_mask, polygon=geometrylist)

    # Filter the newly generated links that are longer than the max distance.
    # Links that were already present are never subject to this filter.
    current_links = network._link1d2d.link1d2d
    # TODO: fix in hydrolib-core, link1d2d is annotated np.ndarray[np.int32] (int32 read as shape):
    # https://github.com/Deltares/HYDROLIB-core/issues/1314
    is_new = _new_link_mask(current_links, present_links)  # type: ignore[arg-type]
    id1d = current_links[is_new, 0]
    id2d = current_links[is_new, 1]
    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x[id1d], network._mesh1d.mesh1d_node_y[id1d]],
        axis=1,
    )
    faces2d = np.stack(
        [mesh2d_output.face_x[id2d], mesh2d_output.face_y[id2d]],
        axis=1,
    )
    lengths = np.hypot(nodes1d[:, 0] - faces2d[:, 0], nodes1d[:, 1] - faces2d[:, 1])
    keep_mask = ~is_new
    keep_mask[is_new] = lengths < max_length
    keep = np.nonzero(keep_mask)[0]
    # TODO: fix in hydrolib-core, link1d2d is annotated np.ndarray[np.int32] (int32 read as shape):
    # https://github.com/Deltares/HYDROLIB-core/issues/1314
    _filter_links_on_idx(network, keep, present_links)  # type: ignore[arg-type]


def _new_link_mask(current: np.ndarray, present_links: np.ndarray) -> np.ndarray:
    """Boolean mask over `current`, True for links not already present in `present_links`.

    Identifies which of the current 1d2d contacts were just added by a meshkernel
    compute call, without assuming that call appends contacts (rather than replacing
    the whole set) or preserves their original order.

    Duplicate rows in either array don't need special handling here - each row is
    tested independently and any duplicates that end up in `keep` are removed later
    by _filter_links_on_idx's final np.unique.

    Parameters
    ----------
    current : np.ndarray
        Current (mesh1d_idx, mesh2d_idx) contacts, shape (N, 2).
    present_links : np.ndarray
        Snapshot of contacts taken before the meshkernel compute call, shape (M, 2).
    """
    if not len(current) or not len(present_links):
        return np.ones(len(current), dtype=bool)
    # Combine each (mesh1d_idx, mesh2d_idx) pair into a single index, the same
    # way np.ravel_multi_index turns e.g. a (row, col) pair into one flat array
    # index. That lets np.isin compare whole pairs at once via its fast integer
    # hash-table path, instead of a Python-level loop over tuples.
    dims = (
        int(max(current[:, 0].max(), present_links[:, 0].max())) + 1,
        int(max(current[:, 1].max(), present_links[:, 1].max())) + 1,
    )
    current_keys = np.ravel_multi_index((current[:, 0], current[:, 1]), dims)
    present_keys = np.ravel_multi_index((present_links[:, 0], present_links[:, 1]), dims)
    return ~np.isin(current_keys, present_keys)


def _filter_links_on_idx(
    network: Network,
    keep: np.ndarray | list[int],
    present_links: np.ndarray | None = None,
) -> None:
    """Set the network's 1d2d contacts to `keep`, merged with `present_links`.

    Every function that *adds* 1d2d links must snapshot network._link1d2d.link1d2d
    before its meshkernel contacts computation and pass that snapshot as
    `present_links` here, even though `keep` is computed from the state *after*
    that computation. `present_links` is always merged back in - regardless of
    whether it also survives `keep` - because MeshKernel's contacts_compute_*
    calls are not all guaranteed to preserve previously existing contacts. This
    is the safety net that stops links created by an earlier call from being
    silently discarded by a later one.

    Functions that only *remove* links (filtering the current contacts without
    an intervening compute call) have nothing to restore and can omit it.

    Parameters
    ----------
    network : Network
        Network whose 1d2d contacts are set.
    keep : np.ndarray or list[int]
        Boolean mask or integer indices into the current contacts, selecting which of
        them to keep.
    present_links : np.ndarray, optional
        Snapshot of contacts to merge back in regardless of `keep`. Defaults to None.
    """
    # Select the remaining links
    contacts = network._link1d2d.meshkernel.contacts_get()
    mesh1d_indices = contacts.mesh1d_indices[keep]
    mesh2d_indices = contacts.mesh2d_indices[keep]
    if present_links is not None and len(present_links):
        mesh1d_indices = np.concatenate([mesh1d_indices, present_links[:, 0]])
        mesh2d_indices = np.concatenate([mesh2d_indices, present_links[:, 1]])
    if mesh1d_indices.size:
        # De-duplicate while keeping first occurrence
        pairs = np.column_stack([mesh1d_indices, mesh2d_indices])
        _, unique_idx = np.unique(pairs, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        mesh1d_indices = mesh1d_indices[unique_idx]
        mesh2d_indices = mesh2d_indices[unique_idx]
    contacts.mesh1d_indices = np.array(mesh1d_indices).astype(np.int32)
    contacts.mesh2d_indices = np.array(mesh2d_indices).astype(np.int32)
    network._link1d2d.meshkernel.contacts_set(contacts)


def links1d2d_add_links_2d_to_1d_embedded(
    network: Network,
    branchids: list[str] | None = None,
    within: Polygon | MultiPolygon | None = None,
) -> None:
    """Add 1D2D links where the 2D mesh overlaps the 1D mesh: the 'embedded' links.

    Each 2D face that crosses a branch is linked to the 1D mesh. To find these faces
    efficiently:

    1. get the maximum length of a face edge;
    2. buffer the branches with this length;
    3. find the face centers within this buffer;
    4. check for each of these faces if it crosses the branches.

    Parameters
    ----------
    network : Network
        Network in which the links are made, with a 1D and a 2D mesh.
    branchids : list[str], optional
        Branches to link. Defaults to None: all branches.
    within : Polygon or MultiPolygon, optional
        Area within which links are made. Defaults to None: everywhere.

    Raises
    ------
    TypeError
        If clipping the buffered branches with ``within`` does not give a polygon.
    """
    # Fetch the 2d mesh once and reuse it below - each network._mesh2d.mesh2d_*
    # property otherwise re-fetches and re-validates the whole mesh from meshkernel.
    mesh2d_output = network._mesh2d.get_mesh2d()

    # Get the max edge distance
    nodes2d = np.stack([mesh2d_output.node_x, mesh2d_output.node_y], axis=1)
    edge_node_crds = nodes2d[network._mesh2d.mesh2d_edge_nodes]

    diff = edge_node_crds[:, 0, :] - edge_node_crds[:, 1, :]
    maxdiff = np.hypot(diff[:, 0], diff[:, 1]).max()

    # Create multilinestring from branches
    nodes1d = np.stack([network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1)

    # Buffer the branches with the max cell distances
    mls = MultiLineString(nodes1d[network._mesh1d.mesh1d_edge_nodes].tolist())
    area: Polygon | MultiPolygon = mls.buffer(maxdiff)

    # If a within polygon is provided, clip the buffered area with this polygon.
    if within is not None:
        clipped = area.intersection(within)
        if not isinstance(clipped, (Polygon, MultiPolygon)):
            raise TypeError(f"Expected the clipped area to be a Polygon or MultiPolygon, not a {clipped.geom_type}.")
        area = clipped

    # Create an array with 2d facecenters and check which intersect the (clipped) area
    faces2d = np.stack([mesh2d_output.face_x, mesh2d_output.face_y], axis=1)
    mpgl = mk.GeometryList(*faces2d.T.copy())
    idx = np.zeros(len(faces2d), dtype=bool)

    for subarea in common.as_polygon_list(area):
        subarea = GeometryList.from_polygon(subarea)
        sub_idx = network.meshkernel.polygon_get_included_points(subarea, mpgl)
        # values of a meshkernel GeometryList, not pandas
        idx |= sub_idx.values == 1  # noqa: PD011

    # Check for each of the remaining faces, if it actually crosses the branches
    where = np.nonzero(idx)[0]

    # Return early if there are no candidates
    if not where.size:
        return

    # Check which face crosses a branch
    candidate_face_nodes = network._mesh2d.mesh2d_face_nodes[idx]
    face_polygons = []
    for face_idxs in candidate_face_nodes:
        # nodes2d always has 4 columns if the grid is a mix of triangular and
        # rectangular, but a triangular cell only has 3 nodes. A triangular
        # cell will then have a column of invalid indices. This needs to be
        # checked per cell.
        # Use a polygon instead of linestring to enforce a closed ring and to
        # test both boundary and/or face intersection.
        valid_face_idxs = face_idxs[face_idxs >= 0]
        face_polygons.append(Polygon(nodes2d[valid_face_idxs]))
    idx[where] = shapely.intersects(face_polygons, mls)

    # Use the remaining points to create the links
    multipoint = mk.GeometryList(x_coordinates=faces2d[idx, 0], y_coordinates=faces2d[idx, 1])

    # Get the nodes for the specific branch ids
    node_mask = network._mesh1d.get_node_mask(branchids)

    # Snapshot the already present links. They are restored after the call below
    # regardless of whether that call preserves them itself (see _filter_links_on_idx).
    present_links = network._link1d2d.link1d2d

    # Generate links
    network._link1d2d._link_from_2d_to_1d_embedded(node_mask, polygons=multipoint)

    # Nothing computed here is filtered on its own merits, so keep it all
    keep = np.arange(len(network._link1d2d.link1d2d))
    # TODO: fix in hydrolib-core, link1d2d is annotated np.ndarray[np.int32] (int32 read as shape):
    # https://github.com/Deltares/HYDROLIB-core/issues/1314
    _filter_links_on_idx(network, keep, present_links)  # type: ignore[arg-type]


def links1d2d_add_links_2d_to_1d_lateral(
    network: Network,
    dist_factor: float | None = 2.0,
    branchids: list[str] | None = None,
    within: Polygon | MultiPolygon | None = None,
    max_length: float = np.inf,
) -> None:
    """Add lateral 1D2D links, from the 2D mesh boundary to the 1D mesh beside it.

    Whether a link is kept depends on the distance from the face center to where the
    link crosses the boundary of the 2D mesh. By default, links that cross further than
    2 times the mean distance from the face center to its nodes are removed. For a
    square cell with a direct link out of the cell (without passing other cells) this
    ratio is at most sqrt(2) = 1.414; the default of 2 gives some margin. A link that
    crosses the 2D mesh boundary more than once is always removed.

    Parameters
    ----------
    network : Network
        Network in which the links are made, with a 1D and a 2D mesh.
    dist_factor : float, optional
        Factor that determines which links are kept, see above. Defaults to 2.0. With
        None, all links are kept.
    branchids : list[str], optional
        Branches to link. Defaults to None: all branches.
    within : Polygon or MultiPolygon, optional
        Area within which 2D faces are linked. Defaults to None: everywhere.
    max_length : float, optional
        Maximum length of a link. Defaults to ``np.inf``.

    Raises
    ------
    TypeError
        If the mesh boundaries, or their intersection with ``within``, are not polygons.
    """
    geometrylist = network.meshkernel.mesh2d_get_mesh_boundaries_as_polygons()
    mpboundaries = GeometryList(**geometrylist.__dict__).to_geometry()
    if not isinstance(mpboundaries, (Polygon, MultiPolygon)):
        raise TypeError(f"Expected the mesh boundaries to be polygons, not a {mpboundaries.geom_type}.")
    if within is not None:
        # If a 'within' polygon was provided, get the intersection with the meshboundaries
        # and convert it to a geometrylist
        # Note that the provided meshboundaries is a (list of) polygon(s). Holes are provided
        # as polygons as well, which dont make it a valid MultiPolygon
        clipped: list[Polygon | MultiPolygon] = []
        for boundary in common.as_polygon_list(mpboundaries):
            part = boundary.intersection(within)
            if not isinstance(part, (Polygon, MultiPolygon)):
                raise TypeError(
                    f"Expected the clipped mesh boundary to be a Polygon or MultiPolygon, not a {part.geom_type}."
                )
            clipped.append(part)
        geometrylist = GeometryList.from_geometry(MultiPolygon(common.as_polygon_list(clipped)))

    # Get the nodes for the specific branch ids
    node_mask = network._mesh1d.get_node_mask(branchids)

    # Snapshot the already present links. They are restored after the call below
    # regardless of whether that call preserves them itself (see _filter_links_on_idx).
    present_links = network._link1d2d.link1d2d

    # Generate links
    network._link1d2d._link_from_2d_to_1d_lateral(node_mask, polygon=geometrylist, search_radius=max_length)

    # If the provided distance factor was None, no further selection is needed, all links are kept.
    if dist_factor is None:
        return

    # Create multilinestring
    multilinestring = MultiLineString([poly.exterior for poly in common.as_polygon_list(mpboundaries)])

    # Find the links that intersect the boundary close to the origin
    id1d = network._link1d2d.link1d2d[:, 0]
    id2d = network._link1d2d.link1d2d[:, 1]

    nodes1d = np.stack(
        [network._mesh1d.mesh1d_node_x[id1d], network._mesh1d.mesh1d_node_y[id1d]],
        axis=1,
    )
    # Fetch the 2d mesh once and reuse it below - each network._mesh2d.mesh2d_*
    # property otherwise re-fetches and re-validates the whole mesh from meshkernel.
    mesh2d_output = network._mesh2d.get_mesh2d()
    faces2d = np.stack(
        [mesh2d_output.face_x[id2d], mesh2d_output.face_y[id2d]],
        axis=1,
    )
    nodes2d = np.stack([mesh2d_output.node_x, mesh2d_output.node_y], axis=1)
    nodes2d_idx = network._mesh2d.mesh2d_face_nodes[id2d]
    keep = []
    for i, (node1d, face2d, node2d) in enumerate(zip(nodes1d, faces2d, nodes2d_idx)):
        # nodes2d always has 4 columns if the grid is a mix of triangular and
        # rectangular, but a triangular cell only has 3 nodes. A triangular
        # cell will then have a column of invalid indices. This needs to be
        # checked per cell.
        face_node_crds = nodes2d[node2d[node2d >= 0]]

        # Calculate distance between face edge and face center
        x1 = face_node_crds[:, 0]
        y1 = face_node_crds[:, 1]
        x0, y0 = face2d[0], face2d[1]
        distance = np.hypot(x1 - x0, y1 - y0).mean()

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
        dist = np.hypot(*(face2d - isect_list[0].coords[:][0]))
        if dist < dist_factor * distance:
            keep.append(i)

    # Select the remaining links
    # TODO: fix in hydrolib-core, link1d2d is annotated np.ndarray[np.int32] (int32 read as shape):
    # https://github.com/Deltares/HYDROLIB-core/issues/1314
    _filter_links_on_idx(network, keep, present_links=present_links)  # type: ignore[arg-type]


def links1d2d_remove_within(network: Network, within: Polygon | MultiPolygon) -> None:
    """Remove the 1D2D links with the 1D node or the 2D face center inside a polygon.

    Parameters
    ----------
    network : Network
        Network from which the links are removed.
    within : Polygon or MultiPolygon
        Area in which links are removed.
    """
    # Create an array with 2d facecenters and 1d nodes, that form the links
    links = network._link1d2d.link1d2d
    nodes1d = np.stack(
        [
            network._mesh1d.mesh1d_node_x[links[:, 0]],
            network._mesh1d.mesh1d_node_y[links[:, 0]],
        ],
        axis=1,
    )
    # Fetch the 2d mesh once and reuse it below - each network._mesh2d.mesh2d_*
    # property otherwise re-fetches and re-validates the whole mesh from meshkernel.
    mesh2d_output = network._mesh2d.get_mesh2d()
    faces2d = np.stack(
        [
            mesh2d_output.face_x[links[:, 1]],
            mesh2d_output.face_y[links[:, 1]],
        ],
        axis=1,
    )

    # Create GeometryList MultiPoint object (dedupe to avoid MK duplicate-point issues)
    uniq_faces2d, inv_faces2d = np.unique(faces2d, axis=0, return_inverse=True)
    uniq_nodes1d, inv_nodes1d = np.unique(nodes1d, axis=0, return_inverse=True)
    mpgl_faces2d = GeometryList(*uniq_faces2d.T.copy())
    mpgl_nodes1d = GeometryList(*uniq_nodes1d.T.copy())
    idx_faces = np.zeros(len(uniq_faces2d), dtype=bool)
    idx_nodes = np.zeros(len(uniq_nodes1d), dtype=bool)

    # Check which links intersect the provided area
    for polygon in common.as_polygon_list(within):
        subarea = GeometryList.from_geometry(polygon)
        # values of a meshkernel GeometryList, not pandas
        idx_faces |= (
            network.meshkernel.polygon_get_included_points(subarea, mpgl_faces2d).values  # noqa: PD011
            == 1
        )
        idx_nodes |= (
            network.meshkernel.polygon_get_included_points(subarea, mpgl_nodes1d).values  # noqa: PD011
            == 1
        )

    # Remove these links
    idx = idx_faces[inv_faces2d] | idx_nodes[inv_nodes1d]
    keep = ~idx
    _filter_links_on_idx(network, keep)


def links1d2d_remove_1d_endpoints(network: Network) -> None:
    """Remove the 1D2D links at the end points of the 1D mesh.

    The GUI interprets every end point as a boundary condition, which does not allow a
    1D2D link at the same node. Logs a warning if there are no 1D nodes, 2D faces or
    links.

    Parameters
    ----------
    network : Network
        Network from which the links are removed.
    """
    # Can only be done after links have been generated
    if not list(network._mesh1d.network1d_node_id) or not list(network._mesh2d.mesh2d_face_nodes):
        logger.warning("1d nodes or 2d faces are not present.")
        return

    # Fetch the current 1d2d links once - like network._mesh2d.mesh2d_*, this
    # property re-fetches and re-validates the contacts from meshkernel on every access.
    link1d2d = network._link1d2d.link1d2d
    # TODO: fix in hydrolib-core, link1d2d is annotated np.ndarray[np.int32] (int32 read as shape):
    # https://github.com/Deltares/HYDROLIB-core/issues/1314
    if link1d2d.shape[0] == 0:  # type: ignore[index]
        logger.warning("No 1d2d-links present.")
        return

    # Select mesh1d nodes that are only present in a single edge.
    # link1d2d[:, 0] stores mesh1d node indices, so endpoint detection must
    # use the same mesh1d index space.
    edge_nodes = network._mesh1d.mesh1d_edge_nodes
    edgeid, counts = np.unique(edge_nodes, return_counts=True)
    to_remove = edgeid[counts == 1]

    # Remove links connected to 1d endpoint nodes.
    link_nodes = link1d2d[:, 0]
    keep = ~np.isin(link_nodes, to_remove)

    _filter_links_on_idx(network, keep)


def mesh2d_altitude_from_raster(
    network: Network,
    rasterpath: Path | str,
    where: RasterStatPosition | str = "face",
    stat: str = "mean",
    fill_option: FillOption | str = "fill_value",
    fill_value: float | None = None,
    *,
    min_coverage_frac: float = 0.0,
    coverage_weight: zonal.CoverageWeight = "fraction",
) -> None:
    """Set the level of the 2D faces or nodes from a raster.

    For faces, the statistic is taken over the face polygons; for nodes, over the
    Voronoi cells around the nodes, clipped to the mesh.

    Note that the raster is not clipped. Any values outside the bounds are
    also taken into account.

    Parameters
    ----------
    network : Network
        Network for which the level is determined, with a 2D mesh.
    rasterpath : Path or str
        Raster file from which the levels are read.
    where : RasterStatPosition or str, optional
        Set the level of the faces (``"face"``) or of the nodes (``"node"``). Defaults
        to ``"face"``.
    stat : str, optional
        exactextract statistic of the raster cells per face or node, such as ``"mean"``,
        ``"median"`` or ``"quantile(q=0.9)"``. Defaults to ``"mean"``.
    fill_option : FillOption or str, optional
        How to fill faces or nodes without a raster value: ``"fill_value"``,
        ``"nearest"`` or ``"interpolate"``. Defaults to ``"fill_value"``.
    fill_value : float, optional
        Value for ``"fill_value"``, and for values that cannot be interpolated with
        ``"interpolate"``. Defaults to None.
    min_coverage_frac : float, optional
        Smallest fraction of a raster cell a face or node polygon must cover for the
        cell to count. Defaults to 0: every touched cell.
    coverage_weight : str, optional
        ``"fraction"`` counts raster cells by the fraction of them the polygon covers,
        ``"none"`` counts them fully. Defaults to ``"fraction"``.

    Raises
    ------
    ValueError
        If all values are missing with ``"nearest"`` or ``"interpolate"``, or if
        ``fill_value`` is needed but not given.
    """
    if isinstance(fill_option, str):
        fill_option = FillOption(fill_option)

    if isinstance(where, str):
        where = RasterStatPosition(where)

    # Select points on faces or nodes
    logger.info("Creating GeoDataFrame of cell faces.")

    # Fetch the 2d mesh once and reuse it below - each network._mesh2d.mesh2d_*
    # property otherwise re-fetches and re-validates the whole mesh from meshkernel.
    mesh2d_output = network._mesh2d.get_mesh2d()

    # Get coordinates where z-values are derived or centered
    xy = np.stack(
        [
            getattr(mesh2d_output, f"{where.value}_x"),
            getattr(mesh2d_output, f"{where.value}_y"),
        ],
        axis=1,
    )

    # Create cells as polygons
    xy_facenodes = np.stack([mesh2d_output.node_x, mesh2d_output.node_y], axis=1)
    # Use raw MeshKernel connectivity instead of mesh2d_face_nodes, whose
    # fixed-width rows contain padding for triangles and other shorter faces.
    # nodes_per_face marks each face boundary in the flattened coordinates.
    face_node_counts = mesh2d_output.nodes_per_face
    face_coords = xy_facenodes[mesh2d_output.face_nodes]
    # Shapely's ``indices`` groups coordinates into rings. Repeat each face ID
    # by its node count so every coordinate is assigned to the correct face.
    face_ids = np.arange(len(face_node_counts))
    face_idx = np.repeat(face_ids, face_node_counts)
    facedata = gpd.GeoDataFrame(geometry=shapely.polygons(shapely.linearrings(face_coords, indices=face_idx)))

    if where == RasterStatPosition.NODE:
        logger.info("Generating voronoi polygons around cell centers for determining raster statistics.")
        facedata = spatial.get_voronoi_around_nodes(xy, facedata)

    df = zonal.zonal_stats(
        facedata,
        rasterpath,
        stat,
        strategy="raster-sequential",
        min_coverage_frac=min_coverage_frac,
        coverage_weight=coverage_weight,
    )
    # By position: exactextract names e.g. "quantile(q=0.9)" "quantile_90".
    zvalues = df.iloc[:, 0].to_numpy()

    isnan = np.isnan(zvalues)
    if isnan.any():
        # If interpolation or fill_option, but all are NaN, raise error.
        if isnan.all() and fill_option in [FillOption.NEAREST, FillOption.INTERPOLATE]:
            raise ValueError("Only NaN values found, interpolation or nearest not possible.")

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
                raise ValueError("Provide a fill_value (keyword argument) to fill values that cannot be interpolated.")
            # By interpolating
            isnan = np.isnan(zvalues)
            interp = LinearNDInterpolator(xy[~isnan], zvalues[~isnan], fill_value=fill_value)
            zvalues[isnan] = interp(xy[isnan])

    # Set values to mesh geometry
    setattr(network._mesh2d, f"mesh2d_{where.value}_z", zvalues)
