import geopandas as gpd
import numpy as np
import shapely
from hydrolib.dhydamo.geometry import spatial
from shapely.geometry import MultiPolygon, box


def test_voronoi_around_nodes_cells_cover_the_faces():
    # 3x3 nodes of a 2x2 mesh: each cell contains its own node, and together the cells cover the faces exactly
    faces = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)])
    x, y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    nodes = np.stack([x.ravel(), y.ravel()], axis=1)

    cells = spatial.get_voronoi_around_nodes(nodes, faces)

    assert cells.index.tolist() == list(range(1, 10))
    assert shapely.intersects_xy(cells.geometry.to_numpy(), nodes[:, 0], nodes[:, 1]).all()
    # corner, edge and centre node
    assert cells.geometry[1].equals(box(0, 0, 0.5, 0.5))
    assert cells.geometry[2].equals(box(0.5, 0, 1.5, 0.5))
    assert cells.geometry[5].equals(box(0.5, 0.5, 1.5, 1.5))
    np.testing.assert_allclose(cells.area.sum(), 4.0)
    assert cells.union_all().equals(box(0, 0, 2, 2))


def test_voronoi_around_nodes_clipped_into_two_parts():
    # The cell of the second node reaches into the first area, so the faces split it into two parts; both
    # are kept. This used to fail with "'MultiPolygon' object is not subscriptable".
    faces = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(0, 12, 10, 13)])
    nodes = np.array([[5.0, 5.0], [5.0, 12.5]])

    cells = spatial.get_voronoi_around_nodes(nodes, faces)

    # the bisector of the two nodes is at y = 8.75
    assert cells.geometry[1].equals(box(0, 0, 10, 8.75))
    assert cells.geometry[2].equals(MultiPolygon([box(0, 8.75, 10, 10), box(0, 12, 10, 13)]))


def test_voronoi_around_nodes_duplicate_nodes():
    # Meshes added next to each other share border nodes; each copy gets the same cell
    faces = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]])

    cells = spatial.get_voronoi_around_nodes(nodes, faces)

    assert len(cells) == len(nodes)
    assert cells.geometry[2].equals(box(0.5, 0, 1.5, 0.5))
    assert cells.geometry[3].equals(cells.geometry[2])
    assert cells.geometry[7].equals(cells.geometry[6])


def test_voronoi_around_nodes_node_outside_faces():
    # A node outside the faces gets an empty cell, so the cells stay aligned with the nodes
    faces = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])

    cells = spatial.get_voronoi_around_nodes(nodes, faces)

    assert len(cells) == len(nodes)
    assert cells.geometry[5].is_empty
    assert cells.geometry[1].equals(box(0, 0, 0.5, 0.5))
