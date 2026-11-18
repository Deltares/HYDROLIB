import folium
import geopandas as gpd
import numpy as np
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.geometry import mesh, viz
from meshkernel.py_structures import DeleteMeshOption
from shapely.geometry import box
from tests.dhydamo.io.test_spatial_fields import _branches


def test_plot_network_returns_map():
    # smoke test: a 1D network gives a folium map, without background tiles and without opening it
    fm = FMModel()
    network = fm.geometry.netfile.network
    mesh.mesh1d_add_branches_from_gdf(network, branches=_branches(), branch_name_col="code", node_distance=5)
    m = viz.plot_network(network, background=False, crs=28992)
    assert isinstance(m, folium.Map)
    layers = [child.layer_name for child in m._children.values() if isinstance(child, folium.GeoJson)]
    assert layers == ["Mesh 1D"]


def test_plot_network_face_levels():
    # the face levels are drawn as an image overlay over the extent of the face centres
    network = FMModel().geometry.netfile.network
    mesh.mesh2d_add_rectilinear(
        network, box(0, 0, 100, 100), dx=10, dy=10, deletemeshoption=DeleteMeshOption.INSIDE_AND_INTERSECTED
    )
    face_x, face_y = network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y
    network._mesh2d.mesh2d_face_z = np.arange(len(face_x), dtype=float)

    m = viz.plot_network(network, background=False, crs=28992, face_z_kwargs={})

    overlays = [child for child in m._children.values() if isinstance(child, folium.raster_layers.ImageOverlay)]
    assert len(overlays) == 1
    centres = gpd.GeoSeries(gpd.points_from_xy(face_x, face_y), crs=28992).to_crs(4326)
    lon0, lat0, lon1, lat1 = centres.total_bounds
    np.testing.assert_allclose(overlays[0].bounds, [[lat0, lon0], [lat1, lon1]])
