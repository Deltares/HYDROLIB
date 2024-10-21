import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from meshkernel.py_structures import DeleteMeshOption
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Polygon, box

sys.path.append(".")
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import common, mesh, viz
from hydrolib.dhydamo.geometry.models import GeometryList
from tests.dhydamo.io import test_from_hydamo

hydamo_data_path = (
    Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "tests" / "data"
)


def test_create_2d_rectilinear(do_plot=False):
    # Define polygon
    # bbox = (1.0, -2.0, 3.0, 4.0)

    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    polygon = box(0, 0, 10, 10)

    mesh.mesh2d_add_rectilinear(
        network,
        polygon,
        dx=5,
        dy=5,
        deletemeshoption=DeleteMeshOption.INSIDE_AND_INTERSECTED
    )

    np.testing.assert_array_equal(
        network._mesh2d.mesh2d_node_x,
        np.array([0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0]),
    )
    np.testing.assert_array_equal(
        network._mesh2d.mesh2d_node_y,
        np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0]),
    )

    if do_plot:
        viz.plot_network(network)
        plt.show()


def _get_circle_polygon(
    radius, n_points: int = 361, xoff: float = 0, yoff: float = 0
) -> Polygon:
    theta = np.linspace(0, 2 * np.pi, n_points)
    coords = np.c_[np.sin(theta), np.cos(theta)] * radius
    coords[:, 0] += xoff
    coords[:, 1] += yoff
    circle = Polygon(coords)
    return circle


def test_create_2d_rectilinear_within_circle(do_plot=False):
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
        deletemeshoption=DeleteMeshOption.INSIDE_AND_INTERSECTED,
    )

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()
        viz.plot_network(network, ax=ax)
        ax.plot(*circle.exterior.coords.xy, color="red", ls="--")
        plt.show()

    # Test if 80 (of the 100) cells are left
    assert len(network._mesh2d.mesh2d_face_x) == 88


@pytest.mark.xfail
def test_create_2d_triangular_within_circle(do_plot=False):
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    # Create circular polygon
    circle = _get_circle_polygon(radius=10)

    # Add mesh and clip part outside circle
    mesh.mesh2d_add_triangular(network, circle, edge_length=2)

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()
        ax.set_aspect(1.0)
        viz.plot_network(network, ax=ax)
        ax.plot(*circle.exterior.coords.xy, color="red", ls="--")
        plt.show()


def test_create_2d_rectangular_from_multipolygon(do_plot=False):
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
        deletemeshoption=DeleteMeshOption.INSIDE_AND_INTERSECTED,
    )
    assert len(network._mesh2d.mesh2d_face_x) == 149

    x = np.linspace(0, 20, 101)
    river = LineString(np.c_[x, np.sin(x / 3) + 5]).buffer(0.5)

    # Refine along river
    refinement = river.buffer(1.0)
    mesh.mesh2d_refine(network, refinement, steps=1)

    # Clip river
    mesh.mesh2d_clip(
        network=network,
        polygon=GeometryList.from_geometry(river),
        deletemeshoption=DeleteMeshOption.INSIDE_AND_INTERSECTED, #ALL_NODES,
    )

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()
        ax.set_aspect(1.0)
        viz.plot_network(network, ax=ax)
        ax.plot(*polygon1.exterior.coords.xy, color="k", ls="--")
        ax.plot(*polygon2.exterior.coords.xy, color="k", ls="--")
        ax.plot(*river.exterior.coords.xy, color="r", ls="--")
        ax.plot(*refinement.exterior.coords.xy, color="g", ls="--")
        plt.show()


@pytest.mark.xfail
def test_create_2d_triangular_from_multipolygon(do_plot=False):
    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    circle1 = _get_circle_polygon(radius=10)
    circle2 = _get_circle_polygon(radius=7, xoff=20, yoff=2)
    refinement_box = box(0, 0, 16, 6)

    multipolygon = MultiPolygon([circle1, circle2])

    mesh.mesh2d_add_triangular(network, multipolygon, edge_length=2)

    # Check bounds and number of faces
    assert len(network._mesh2d.mesh2d_face_x) == 376
    assert len(network._mesh2d.mesh2d_edge_x) == 589

    # Refine mesh
    mesh.mesh2d_refine(network, refinement_box, steps=1)

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()
        ax.set_aspect(1.0)
        viz.plot_network(network, ax=ax)
        ax.plot(*circle1.exterior.coords.xy, color="k", ls="--")
        ax.plot(*circle2.exterior.coords.xy, color="k", ls="--")
        ax.plot(*refinement_box.exterior.coords.xy, color="g", ls="--")
        plt.show()


def test_2d_clip_outside_polygon(do_plot=False):
    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    rectangle = box(-10, -10, 10, 10)

    dmo = DeleteMeshOption.INSIDE_NOT_INTERSECTED
    mesh.mesh2d_add_rectilinear(network, rectangle, dx=1, dy=1, deletemeshoption=dmo)

    clipgeo = box(-8, -8, 8, 8).difference(
        MultiPolygon([box(-6, -1, -4, 2), box(4, 5, 7, 7)])
    )

    mesh.mesh2d_clip(network, clipgeo, deletemeshoption= DeleteMeshOption.INSIDE_AND_INTERSECTED, inside=False)
    assert len(network._mesh2d.mesh2d_face_x) == 284

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()
        ax.set_aspect(1.0)
        viz.plot_network(network, ax=ax)
        ax.plot(*clipgeo.exterior.coords.xy, color="k", ls="--")
        for hole in clipgeo.interiors:
            ax.plot(*hole.coords.xy, color="r", ls="--")
        plt.show()


def test_2d_clip_inside_multipolygon(do_plot=False):
    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    rectangle = box(-10, -10, 10, 10)
    mesh.mesh2d_add_rectilinear(
        network,
        rectangle,
        dx=1,
        dy=1,
        deletemeshoption=DeleteMeshOption.INSIDE_NOT_INTERSECTED
    )

    clipgeo = MultiPolygon([box(-6, -1, -4, 2), box(4, 5, 7.2, 7.2)])
    mesh.mesh2d_clip(
        network,
        clipgeo,
        deletemeshoption=DeleteMeshOption.INSIDE_AND_INTERSECTED,
        inside=True,
    )
    assert len(network._mesh2d.mesh2d_node_x) == 337

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()
        ax.set_aspect(1.0)
        viz.plot_network(network, ax=ax)
        for polygon in clipgeo.geoms:
            ax.plot(*polygon.exterior.coords.xy, color="r", ls="--")
        plt.show()


def test_1d_add_branch_from_linestring(do_plot=False):
    # Define polygon
    fmmodel = FMModel()
    network = fmmodel.geometry.netfile.network

    x = np.linspace(0, 20, 101)
    branch = LineString(np.c_[x, np.sin(x / 3) + 5])

    # a multilinestring does not work...
    mesh.mesh1d_add_branch_from_linestring(network, branch, node_distance=3)

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots()

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


def test_links1d2d_add_links_1d_to_2d(do_plot=False):
    network, within, _ = _prepare_1d2d_mesh()

    # Generate all links
    mesh.links1d2d_add_links_1d_to_2d(network)

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots(figsize=(5, 5))

        viz.plot_network(network, ax=ax)

        for polygon in common.as_polygon_list(within):
            ax.fill(*polygon.exterior.coords.xy, color="g", ls="-", lw=0, alpha=0.05)
            ax.plot(*polygon.exterior.coords.xy, color="g", ls="-", lw=0.5)
        ax.set_aspect(1.0)
        ax.autoscale_view()

        plt.show()


def test_links1d2d_add_links_2d_to_1d_lateral(do_plot=False):
    network, within, branchids = _prepare_1d2d_mesh()

    # Generate all links
    mesh.links1d2d_add_links_2d_to_1d_lateral(network)

    # Plot the final result verify
    if do_plot:
        _, ax = plt.subplots(figsize=(5, 5))

        viz.plot_network(network, ax=ax)

        for polygon in common.as_polygon_list(within):
            ax.fill(*polygon.exterior.coords.xy, color="g", ls="-", lw=0, alpha=0.05)
            ax.plot(*polygon.exterior.coords.xy, color="g", ls="-", lw=0.5)
        ax.set_aspect(1.0)
        ax.autoscale_view()

        plt.show()


def test_linkd1d2d_remove_links_within_polygon(do_plot=False):
    network, within, _ = _prepare_1d2d_mesh()
    within = within.buffer(-2)

    # Generate all links
    mesh.links1d2d_add_links_1d_to_2d(network)

    # Plot to verify
    if do_plot:
        _, ax = plt.subplots(figsize=(5, 5))

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

def test_mesh1d_add_branches_from_gdf(do_plot=False):
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
    if do_plot:
        _, ax = plt.subplots()

        ax.set_aspect(1.0)
        viz.plot_network(network, ax=ax)
        ax.autoscale_view()
        plt.show()

