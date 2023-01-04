import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from meshkernel.py_structures import DeleteMeshOption
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box

from hydrolib.core.io.dflowfm.mdu.models import FMModel
from hydrolib.core.io.dflowfm.net.models import Branch
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import common, mesh, viz
from hydrolib.dhydamo.geometry.models import GeometryList
from tests.dhydamo.io import test_from_hydamo

hydamo_data_path = (
    Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "tests" / "data"
)


@pytest.mark.plots
def test_create_2d_rectilinear():

    # Define polygon
    bbox = (1.0, -2.0, 3.0, 4.0)

    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    polygon = box(0, 0, 10, 10)

    mesh.mesh2d_add_rectilinear(
        network,
        polygon,
        dx=5,
        dy=5,
        deletemeshoption=DeleteMeshOption.ALL_FACE_CIRCUMCENTERS,
    )

    np.testing.assert_array_equal(
        network._mesh2d.mesh2d_node_x,
        np.array([0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0]),
    )
    np.testing.assert_array_equal(
        network._mesh2d.mesh2d_node_y,
        np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0]),
    )

    # viz.plot_network(network)


def _get_circle_polygon(
    radius, n_points: int = 361, xoff: float = 0, yoff: float = 0
) -> Polygon:
    theta = np.linspace(0, 2 * np.pi, n_points)
    coords = np.c_[np.sin(theta), np.cos(theta)] * radius
    coords[:, 0] += xoff
    coords[:, 1] += yoff
    circle = Polygon(coords)
    return circle


@pytest.mark.plots
def test_create_2d_rectilinear_within_circle():

    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    # Create circular polygon
    circle = _get_circle_polygon(radius=10)

    # Add mesh and clip part outside circle
    mesh.mesh2d_add_rectilinear(
        network,
        circle,
        dx=2,
        dy=2,
        deletemeshoption=DeleteMeshOption.ALL_FACE_CIRCUMCENTERS,
    )

    # Plot to verify
    fig, ax = plt.subplots()
    viz.plot_network(network, ax=ax)
    ax.plot(*circle.exterior.coords.xy, color="red", ls="--")
    plt.show()

    # Test if 80 (of the 100) cells are left
    assert len(network._mesh2d.mesh2d_face_x) == 80


@pytest.mark.xfail
@pytest.mark.plots
def test_create_2d_triangular_within_circle():

    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    # Create circular polygon
    circle = _get_circle_polygon(radius=10)

    # Add mesh and clip part outside circle
    mesh.mesh2d_add_triangular(network, circle, edge_length=2)

    np.testing.assert_array_equal(
        np.c_[network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y].round(3),
        np.array(
            [
                [0.0, 10.0],
                [1.886, 9.219],
                [3.771, 8.438],
                [5.657, 7.657],
                [7.266, 6.6],
                [8.047, 4.714],
                [8.828, 2.828],
                [9.609, 0.943],
                [9.609, -0.943],
                [8.828, -2.828],
                [8.047, -4.714],
                [7.266, -6.6],
                [5.657, -7.657],
                [3.771, -8.438],
                [1.886, -9.219],
                [0.0, -10.0],
                [-1.886, -9.219],
                [-3.771, -8.438],
                [-5.657, -7.657],
                [-7.266, -6.6],
                [-8.047, -4.714],
                [-8.828, -2.828],
                [-9.609, -0.943],
                [-9.609, 0.943],
                [-8.828, 2.828],
                [-8.047, 4.714],
                [-7.266, 6.6],
                [-5.657, 7.657],
                [-3.771, 8.438],
                [-1.886, 9.219],
                [-6.267, -0.0],
                [-4.687, -4.427],
                [-4.687, 4.427],
                [4.687, -4.427],
                [4.687, 4.427],
                [-8.071, -0.0],
                [-5.653, -5.897],
                [-5.653, 5.897],
                [5.653, -5.897],
                [5.653, 5.897],
                [6.267, 0.0],
                [-8.226, -1.474],
                [-6.599, -3.009],
                [-7.169, 1.757],
                [-3.492, -6.264],
                [-6.385, 4.364],
                [-3.492, 6.264],
                [6.861, -5.327],
                [6.489, -3.152],
                [3.492, -6.264],
                [3.492, 6.264],
                [6.385, 4.364],
                [8.071, 0.0],
                [-6.208, -4.48],
                [-8.358, 1.258],
                [-4.675, 1.927],
                [-4.492, -5.607],
                [-4.816, -6.779],
                [-3.243, -4.795],
                [-0.763, -5.972],
                [-3.313, -2.051],
                [-2.169, -5.733],
                [-3.665, -3.433],
                [-2.231, -7.492],
                [-1.052, -3.37],
                [-5.007, -2.355],
                [-2.367, -3.027],
                [-1.108, -0.863],
                [-1.945, -4.393],
                [-7.385, 3.244],
                [-5.98, 2.689],
                [-4.691, 6.779],
                [4.691, -6.779],
                [4.492, 5.607],
                [4.816, 6.779],
                [3.243, 4.795],
                [0.763, 5.972],
                [3.313, 2.051],
                [2.169, 5.733],
                [3.665, 3.433],
                [2.231, 7.492],
                [1.044, 3.365],
                [5.606, 2.202],
                [2.364, 3.028],
                [1.109, 0.861],
                [1.942, 4.389],
                [8.226, 1.474],
                [7.17, 2.755],
                [7.169, -1.52],
                [-5.886, 1.306],
                [-4.558, 0.21],
                [-0.276, -7.999],
                [-6.28, -1.522],
                [-5.047, -1.026],
                [0.276, 7.999],
                [5.502, 3.461],
                [5.204, -1.658],
                [7.613, -2.663],
                [-2.351, 1.223],
                [5.077, -3.067],
                [2.643, -2.137],
                [3.991, -2.259],
                [3.678, -0.1],
                [3.18, -3.715],
                [1.063, -3.555],
                [3.51, -4.968],
                [2.042, -4.684],
                [0.503, -5.03],
                [0.748, -6.681],
                [1.13, -8.034],
                [2.201, -7.002],
                [-3.647, 1.135],
                [-3.006, -0.408],
                [-3.116, 2.888],
                [4.984, -0.348],
                [4.582, 1.101],
                [-4.37, 3.178],
                [-1.004, 2.849],
                [-2.288, 7.523],
                [-1.409, 5.477],
                [-1.075, 8.131],
                [-0.53, 6.733],
                [-2.283, 6.313],
                [-2.83, 4.866],
                [-1.656, 4.094],
                [-0.219, 4.054],
                [0.779, 4.753],
                [5.989, -4.357],
                [6.781, 1.355],
                [-1.092, -7.123],
                [-0.724, -9.08],
                [1.092, 7.123],
                [2.556, 0.816],
                [1.785, -0.685],
                [0.316, -0.407],
                [-0.619, 1.06],
                [0.07, -2.091],
                [0.375, 2.095],
                [1.354, -2.289],
                [2.561, -8.184],
            ]
        ),
    )

    # Plot to verify
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    ax.plot(*circle.exterior.coords.xy, color="red", ls="--")
    plt.show()


@pytest.mark.plots
def test_create_2d_rectangular_from_multipolygon():

    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    # Create 2 part mesh
    polygon1 = box(0, 0, 10, 10)
    polygon2 = box(12, 2, 19, 9)
    multipolygon = MultiPolygon([polygon1, polygon2])

    mesh.mesh2d_add_rectilinear(
        network,
        multipolygon,
        dx=1,
        dy=1,
        deletemeshoption=DeleteMeshOption.ALL_FACE_CIRCUMCENTERS,
    )
    assert len(network._mesh2d.mesh2d_face_x) == 149

    x = np.linspace(0, 20, 101)
    river = LineString(np.c_[x, np.sin(x / 3) + 5]).buffer(0.5)

    # Refine along river
    refinement = river.buffer(1)
    mesh.mesh2d_refine(network, refinement, steps=1)
    assert len(network._mesh2d.mesh2d_face_x) == 411

    # Clip river
    mesh.mesh2d_clip(
        network=network,
        polygon=GeometryList.from_geometry(river),
        deletemeshoption=DeleteMeshOption.ALL_NODES,
    )
    assert len(network._mesh2d.mesh2d_face_x) == 303

    # Plot to verify
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    ax.plot(*polygon1.exterior.coords.xy, color="k", ls="--")
    ax.plot(*polygon2.exterior.coords.xy, color="k", ls="--")
    ax.plot(*river.exterior.coords.xy, color="r", ls="--")
    ax.plot(*refinement.exterior.coords.xy, color="g", ls="--")
    plt.show()


@pytest.mark.xfail
@pytest.mark.plots
def test_create_2d_triangular_from_multipolygon():

    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    circle1 = _get_circle_polygon(radius=10)
    circle2 = _get_circle_polygon(radius=7, xoff=20, yoff=2)
    refinement_box = box(0, 0, 16, 6)

    multipolygon = MultiPolygon([circle1, circle2])

    mesh.mesh2d_add_triangular(network, multipolygon, edge_length=2)

    # Check bounds and number of faces
    assert len(network._mesh2d.mesh2d_face_x) == 352
    assert len(network._mesh2d.mesh2d_edge_x) == 553

    # Refine mesh
    mesh.mesh2d_refine(network, refinement_box, steps=1)

    # Check bounds and number of faces
    assert len(network._mesh2d.mesh2d_face_x) == 636
    assert len(network._mesh2d.mesh2d_edge_x) == 984

    # Plot to verify
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    ax.plot(*circle1.exterior.coords.xy, color="k", ls="--")
    ax.plot(*circle2.exterior.coords.xy, color="k", ls="--")
    ax.plot(*refinement_box.exterior.coords.xy, color="g", ls="--")
    plt.show()


@pytest.mark.plots
def test_2d_clip_outside_polygon():

    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    rectangle = box(-10, -10, 10, 10)

    dmo = DeleteMeshOption.ALL_FACE_CIRCUMCENTERS
    mesh.mesh2d_add_rectilinear(network, rectangle, dx=1, dy=1, deletemeshoption=dmo)

    clipgeo = box(-8, -8, 8, 8).difference(
        MultiPolygon([box(-6, -1, -4, 2), box(4, 5, 7, 7)])
    )

    mesh.mesh2d_clip(network, clipgeo, deletemeshoption=1, inside=False)

    assert len(network._mesh2d.mesh2d_node_x) == 285

    # Plot to verify
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    ax.plot(*clipgeo.exterior.coords.xy, color="k", ls="--")
    for hole in clipgeo.interiors:
        ax.plot(*hole.coords.xy, color="r", ls="--")
    plt.show()


@pytest.mark.plots
def test_2d_clip_inside_multipolygon():

    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    rectangle = box(-10, -10, 10, 10)

    dmo = DeleteMeshOption.ALL_FACE_CIRCUMCENTERS
    mesh.mesh2d_add_rectilinear(network, rectangle, dx=1, dy=1, deletemeshoption=dmo)

    clipgeo = MultiPolygon([box(-6, -1, -4, 2), box(4, 5, 7.2, 7.2)])

    mesh.mesh2d_clip(network, clipgeo, deletemeshoption=1, inside=True)
    assert len(network._mesh2d.mesh2d_node_x) == 437

    # Plot to verify
    fig, ax = plt.subplots()

    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    for polygon in clipgeo.geoms:
        ax.plot(*polygon.exterior.coords.xy, color="r", ls="--")
    plt.show()


@pytest.mark.plots
def test_1d_add_branch_from_linestring():

    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    x = np.linspace(0, 20, 101)
    branch = LineString(np.c_[x, np.sin(x / 3) + 5])

    # a multilinestring does not work...
    mesh.mesh1d_add_branch_from_linestring(network, branch, node_distance=3)

    # Plot to verify
    fig, ax = plt.subplots()

    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    ax.autoscale_view()
    for ls in common.as_linestring_list(branch):
        ax.plot(*ls.coords.xy, color="k", ls="-", lw=3, alpha=0.2)
    plt.show()


def _prepare_1d2d_mesh():
    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    # Generate 1d
    branch = LineString([[-9, -3], [0, 4]])

    branchids = mesh.mesh1d_add_branch_from_linestring(network, branch, node_distance=1)

    # Generate 2d
    areas = MultiPolygon([box(-8, -10, 8, -2), box(-8, 2, 8, 10)])
    hole = box(-6, -6, 6, -4)
    mesh.mesh2d_add_rectilinear(network, areas, dx=0.5, dy=0.5)
    mesh.mesh2d_clip(network, hole)

    within = box(-10, -10, 12, 10).difference(
        LineString([[-2, -10], [2, 10]]).buffer(2)
    )

    return network, within, branchids


@pytest.mark.plots
def test_links1d2d_add_links_1d_to_2d():

    network, within, branchids = _prepare_1d2d_mesh()

    # Generate all links
    mesh.links1d2d_add_links_1d_to_2d(network)
    assert len(network._link1d2d.link1d2d) == 14
    network._link1d2d.clear()

    # Generate links within polygon, with smaller distance factor, with max length, and for the first branch
    mesh.links1d2d_add_links_1d_to_2d(network, within=within, branchids=[branchids])
    assert len(network._link1d2d.link1d2d) == 13
    network._link1d2d.clear()

    # Generate links within polygon, with smaller distance factor, with max length, and for the first branch
    mesh.links1d2d_add_links_1d_to_2d(
        network, within=within, max_length=2, branchids=[branchids]
    )
    assert len(network._link1d2d.link1d2d) == 7
    network._link1d2d.clear()

    # Generate links within polygon
    mesh.links1d2d_add_links_1d_to_2d(network, within=within)
    assert len(network._link1d2d.link1d2d) == 13

    # Plot to verify
    fig, ax = plt.subplots(figsize=(5, 5))

    viz.plot_network(network, ax=ax)

    for polygon in common.as_polygon_list(within):
        ax.fill(*polygon.exterior.coords.xy, color="g", ls="-", lw=0, alpha=0.05)
        ax.plot(*polygon.exterior.coords.xy, color="g", ls="-", lw=0.5)
    ax.set_aspect(1.0)
    ax.autoscale_view()

    plt.show()


@pytest.mark.plots
def test_links1d2d_add_links_2d_to_1d_embedded():

    network, within, branchids = _prepare_1d2d_mesh()

    # Generate all links
    mesh.links1d2d_add_links_2d_to_1d_embedded(network)
    assert len(network._link1d2d.link1d2d) == 13
    network._link1d2d.clear()

    # TODO: The node mask does not seem to work. Fix in meshkernel
    # Generate links within polygon, with smaller distance factor, with max length, and for the first branch
    # mesh.links1d2d_add_links_2d_to_1d_embedded(
    #     network, within=within, branchids=[branchids[0]]
    # )
    # assert len(network._link1d2d.link1d2d) == 22
    # network._link1d2d.clear()

    # Generate links within polygon
    mesh.links1d2d_add_links_2d_to_1d_embedded(network, within=within)
    assert len(network._link1d2d.link1d2d) == 5

    # Plot to verify
    fig, ax = plt.subplots(figsize=(5, 5))

    viz.plot_network(network, ax=ax)

    for polygon in common.as_polygon_list(within):
        ax.fill(*polygon.exterior.coords.xy, color="g", ls="-", lw=0, alpha=0.05)
        ax.plot(*polygon.exterior.coords.xy, color="g", ls="-", lw=0.5)
    ax.set_aspect(1.0)
    ax.autoscale_view()

    plt.show()


@pytest.mark.plots
def test_links1d2d_add_links_2d_to_1d_lateral():

    network, within, branchids = _prepare_1d2d_mesh()

    # Generate all links
    mesh.links1d2d_add_links_2d_to_1d_lateral(network)
    assert len(network._link1d2d.link1d2d) == 30
    network._link1d2d.clear()

    # Generate links within polygon and with smaller distance factor
    mesh.links1d2d_add_links_2d_to_1d_lateral(network, within=within, dist_factor=1.5)
    assert len(network._link1d2d.link1d2d) == 20
    network._link1d2d.clear()

    # Generate links within polygon, with smaller distance factor, and with max length
    mesh.links1d2d_add_links_2d_to_1d_lateral(
        network, within=within, dist_factor=1.5, max_length=2
    )
    assert len(network._link1d2d.link1d2d) == 11
    network._link1d2d.clear()

    # Generate links within polygon, with smaller distance factor, with max length, and for the first branch
    mesh.links1d2d_add_links_2d_to_1d_lateral(
        network, within=within, dist_factor=1.5, max_length=2, branchids=[branchids]
    )
    assert len(network._link1d2d.link1d2d) == 11
    network._link1d2d.clear()

    # Generate links within polygon
    mesh.links1d2d_add_links_2d_to_1d_lateral(network, within=within)
    assert len(network._link1d2d.link1d2d) == 24

    # Plot the final result verify
    fig, ax = plt.subplots(figsize=(5, 5))

    viz.plot_network(network, ax=ax)

    for polygon in common.as_polygon_list(within):
        ax.fill(*polygon.exterior.coords.xy, color="g", ls="-", lw=0, alpha=0.05)
        ax.plot(*polygon.exterior.coords.xy, color="g", ls="-", lw=0.5)
    ax.set_aspect(1.0)
    ax.autoscale_view()

    plt.show()


@pytest.mark.plots
def test_linkd1d2d_remove_links_within_polygon():

    network, within, _ = _prepare_1d2d_mesh()
    within = within.buffer(-2)

    # Generate all links
    mesh.links1d2d_add_links_1d_to_2d(network)
    mesh.links1d2d_remove_within(network, within=within)

    # Plot to verify
    fig, ax = plt.subplots(figsize=(5, 5))

    viz.plot_network(network, ax=ax)

    for polygon in common.as_polygon_list(within):
        ax.fill(*polygon.exterior.coords.xy, color="g", ls="-", lw=0, alpha=0.05)
        ax.plot(*polygon.exterior.coords.xy, color="g", ls="-", lw=0.5)
    ax.set_aspect(1.0)
    ax.autoscale_view()

    plt.show()


def _prepare_hydamo(culverts: bool = False):

    # initialize a hydamo object
    extent_file = hydamo_data_path / "OLO_stroomgebied_incl.maas.shp"
    assert extent_file.exists()
    hydamo = HyDAMO(extent_file=extent_file)

    # all data is contained in one geopackage called 'Example model'
    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    # read branchs
    hydamo.branches.read_gpkg_layer(
        str(gpkg_file), layer_name="HydroObject", index_col="code"
    )

    # Read management device
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")

    # read culverts
    if culverts:
        hydamo.culverts.read_gpkg_layer(
            gpkg_file, layer_name="DuikerSifonHevel", index_col="code"
        )
        hydamo.culverts.snap_to_branch(hydamo.branches, snap_method="ends", maxdist=5)
        hydamo.culverts.dropna(axis=0, inplace=True, subset=["branch_offset"])

        # Connect to management_device
        idx = hydamo.management_device.loc[
            hydamo.management_device["duikersifonhevelid"].notnull()
        ].index
        for i in idx:
            globid = hydamo.culverts.loc[
                hydamo.culverts["code"].eq(
                    hydamo.management_device.at[i, "duikersifonhevelid"]
                ),
                "globalid",
            ].values[0]
            hydamo.management_device.at[i, "duikersifonhevelid"] = globid

        # Convert culverts
        hydamo.structures.convert.culverts(
            hydamo.culverts, management_device=hydamo.management_device
        )

    return hydamo


@pytest.mark.parametrize(
    "where,fill_option,fill_value,outcome",
    [
        ("face", "interpolate", 10.0, 8629.457),
        ("face", "fill_value", 10.0, 8629.457),
        ("face", "nearest", None, 9050.679),
        ("node", "interpolate", 10.0, 6541.38),
        ("node", "fill_value", 10.0, 6526.393),
        ("node", "nearest", None, 6978.605),
    ],
)
def test_mesh2d_altitude_from_raster(where, fill_option, fill_value, outcome):

    rasterpath = hydamo_data_path / "rasters" / "AHN_2m_clipped_filled.tif"
    assert rasterpath.exists()

    # Create HyDAMO object for extent
    hydamo = _prepare_hydamo()
    extent2d = hydamo.branches.unary_union.buffer(200)
    # Shift extent 2 km to right, such that some cells will have no-data values
    extent2d = translate(extent2d, xoff=2000)

    # Create FMModel
    fm = FMModel()
    cellsize = 200

    # Add 2D Mesh, partly triangular, partly rectangular
    xcenter = extent2d.centroid.coords[0][0]
    centerline = LineString([(xcenter, -1e6), (xcenter, 1e6)]).buffer(cellsize / 2)
    parts = extent2d.difference(centerline).geoms

    network = fm.geometry.netfile.network
    mesh.mesh2d_add_triangular(network=network, polygon=parts[0], edge_length=cellsize)
    mesh.mesh2d_add_rectilinear(
        network=network, polygon=parts[1], dx=cellsize, dy=cellsize * 1.5
    )

    # Derive z-values from ahn
    mesh.mesh2d_altitude_from_raster(
        network=network,
        rasterpath=rasterpath,
        where=where,
        stat="mean",
        fill_option=fill_option,
        fill_value=fill_value,
    )

    assert np.float32(
        getattr(network._mesh2d, f"mesh2d_{where}_z").sum()
    ) == np.float32(outcome)


def test_mesh1d_add_branches_from_gdf():

    # Create full HyDAMO object (use from other test)
    hydamo = test_from_hydamo.test_hydamo_object_from_gpkg()

    fm = FMModel()

    network = fm.geometry.netfile.network

    structures = structures = hydamo.structures.as_dataframe(
        rweirs=True,
        bridges=False,
        uweirs=False,
        culverts=True,
        orifices=False,
        pumps=False,
    )

    mesh.mesh1d_add_branches_from_gdf(
        network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=structures,
    )
    # Plot to verify
    fig, ax = plt.subplots()

    ax.set_aspect(1.0)
    viz.plot_network(network, ax=ax)
    ax.autoscale_view()
    plt.show()
