import pytest
import sys

sys.path.append(".")
from tests.dhydamo.geometry import test_mesh


@pytest.mark.plot
def test_plot_create_2d_rectilinear():
    test_mesh.test_create_2d_rectilinear(do_plot=True)


@pytest.mark.plot
def test_plot_create_2d_rectilinear_within_circle():
    test_mesh.test_create_2d_rectilinear_within_circle(do_plot=True)


@pytest.mark.plot
def test_plot_create_2d_triangular_within_circle():
    test_mesh.test_create_2d_triangular_within_circle(do_plot=True)


@pytest.mark.plot
def test_plot_create_2d_rectangular_from_multipolygon():
    test_mesh.test_create_2d_rectangular_from_multipolygon(do_plot=True)


@pytest.mark.plot
def test_plot_create_2d_triangular_from_multipolygon():
    test_mesh.test_create_2d_triangular_from_multipolygon(do_plot=True)


@pytest.mark.plot
def test_plot_2d_clip_outside_polygon():
    test_mesh.test_2d_clip_outside_polygon(do_plot=True)


@pytest.mark.plot
def test_plot_2d_clip_inside_multipolygon():
    test_mesh.test_2d_clip_inside_multipolygon(do_plot=True)


@pytest.mark.plot
def test_plot_1d_add_branch_from_linestring():
    test_mesh.test_1d_add_branch_from_linestring(do_plot=True)


@pytest.mark.plot
def test_plot_links1d2d_add_links_1d_to_2d(): 
    test_mesh.test_links1d2d_add_links_1d_to_2d(do_plot=True)

@pytest.mark.plot
def test_plot_links1d2d_add_links_2d_to_1d_lateral():
    test_mesh.test_links1d2d_add_links_2d_to_1d_lateral(do_plot=True)


@pytest.mark.plot
def test_plot_linkd1d2d_remove_links_within_polygon():
    test_mesh.test_linkd1d2d_remove_links_within_polygon(do_plot=True)


@pytest.mark.plot
def test_plot_mesh1d_add_branches_from_gdf():
    test_mesh.test_mesh1d_add_branches_from_gdf(do_plot=True)
