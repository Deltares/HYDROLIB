import pytest
import sys

import numpy as np

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
def test_2d_refine_ring_geometry():
    test_mesh.test_2d_refine_ring_geometry(do_plot=True)


@pytest.mark.plot
def test_plot_2d_clip_outside_polygon():
    test_mesh.test_2d_clip_outside_polygon(do_plot=True)


@pytest.mark.plot
def test_plot_2d_clip_inside_multipolygon():
    test_mesh.test_2d_clip_inside_multipolygon(do_plot=True)


@pytest.mark.plot
def test_plot_2d_clip_inside_polygon_with_holes():
    test_mesh.test_plot_2d_clip_inside_polygon_with_holes(do_plot=True)


@pytest.mark.plot
def test_plot_1d_add_branch_from_linestring():
    test_mesh.test_1d_add_branch_from_linestring(do_plot=True)


@pytest.mark.plot
@pytest.mark.parametrize(
    "b_within,b_branchids,b_refine,max_length,b_plot,outcome",
    [
        (False, False, False, np.inf, True, 15),
        (False, False, False, 1, True, 11),
        (True, False, False, np.inf, True, 11),
        (False, True, False, np.inf, True, 8),
        (False, False, True, np.inf, True, 15),
    ],
)
def test_plot_links1d2d_add_links_1d_to_2d(b_within, b_branchids, b_refine, max_length, b_plot, outcome):
    test_mesh.test_links1d2d_add_links_1d_to_2d(b_within, b_branchids, b_refine, max_length, b_plot, outcome)


@pytest.mark.plot
@pytest.mark.parametrize(
    "b_within,b_branchids,b_refine, max_length, b_plot,outcome",
    [
        (False, False, False, np.inf, True, 99),
        (False, False, False, 10, True, 99),
        (True, False, False, np.inf, True, 99),
        (False, True, False,  np.inf, True, 99),
        (False, False, True,  np.inf, True, 99),
    ],
)
def test_plot_links1d2d_add_links_2d_to_1d_lateral(b_within, b_branchids, b_refine, max_length, b_plot, outcome):
    test_mesh.test_links1d2d_add_links_2d_to_1d_lateral(b_within, b_branchids, b_refine, max_length, b_plot, outcome)


@pytest.mark.plot
@pytest.mark.parametrize(
    "b_within,b_branchids,b_refine,b_plot,outcome",
    [
        (False, False, False, False, 24),
        (True, False, False, False, 11),
        (False, True, False, False, 9),
        (False, False, True, False, 48),
    ],
)
def test_plot_links1d2d_add_links_2d_to_1d_embedded(b_within, b_branchids, b_refine, b_plot, outcome):
    test_mesh.test_links1d2d_add_links_2d_to_1d_embedded(b_within, b_branchids, b_refine, b_plot, outcome)


@pytest.mark.plot
def test_plot_linkd1d2d_remove_links_within_polygon():
    test_mesh.test_linkd1d2d_remove_links_within_polygon(do_plot=True)


@pytest.mark.plot
def test_plot_mesh1d_add_branches_from_gdf():
    test_mesh.test_mesh1d_add_branches_from_gdf(do_plot=True)
