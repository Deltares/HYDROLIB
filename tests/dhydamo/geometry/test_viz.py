import folium
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.geometry import mesh, viz
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
