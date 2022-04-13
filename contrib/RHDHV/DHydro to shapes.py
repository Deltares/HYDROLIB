import netCDF4 as nc
from pathlib import Path
from shapely.geometry import Point
import geopandas as gpd
from scipy import spatial

def to_shape(mapfn, export_folder):
    """
    Function that converts map nc DHydro result to a shapefile (once as esri shapefile, once as geopackage).
    Two results are made: one with waterlevel, waterdepth and bottomlevel (locations are mesh1d nodes),
    the other shapefile contains velocity and discharge (locations are mesh1d edges).
    Default projection: EPSG 22892 (RD New)
    Exports only the final timestep.
    Args:
        mapfn: filename of the map.nc file

    Returns:
        gdf_wl_wd_bl: Geodataframe with results for: Waterlevel, Bottomlevel and Waterdepth.
        gdf_q_v: Geodataframe with results for: Discharge and Velocity
    """
    export_folder = Path(export_folder)
    export_folder.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(mapfn)

    coords = {'x': "mesh1d_node_x", "y": "mesh1d_node_y"}
    var_wl = "mesh1d_s1"
    var_bl = "mesh1d_flowelem_bl"
    var_wd = 'mesh1d_waterdepth'

    x = list(ds.variables[coords['x']][:])
    y = list(ds.variables[coords['y']][:])
    coords = [Point(xi, yi) for xi, yi in zip(x, y)]

    wl = ds.variables[var_wl][-1,:].data
    bl = ds.variables[var_bl][:].data
    wd = ds.variables[var_wd][-1,:].data

    gdf_wl_wd_bl = gpd.GeoDataFrame(geometry=coords, data={'Waterlevel': wl,
                                                           'Bottom': bl,
                                                           'Waterdepth': wd},
                                    crs = 'EPSG:28992')
    gdf_wl_wd_bl.to_file(export_folder/f'Waterstand_en_diepte.gpkg', driver='GPKG')
    gdf_wl_wd_bl.to_file(export_folder/f'Waterstand_en_diepte.shp')

    # This is a lazy way to do it. Creates a point on (presumably) the center of the line of the mesh1d edge.
    # Preferred would be to have the geometry be the linesegment between two mesh1d nodes.
    edges = {'x': 'mesh1d_edge_x', 'y': 'mesh1d_edge_y'}
    var_q = 'mesh1d_q1'
    var_v = 'mesh1d_u1'

    x = list(ds.variables[edges['x']][:])
    y = list(ds.variables[edges['y']][:])
    coords = [Point(xi, yi) for xi, yi in zip(x, y)]

    q = ds.variables[var_q][-1,:].data
    v = ds.variables[var_v][-1,:].data

    gdf_q_v = gpd.GeoDataFrame(geometry=coords, data={'Q': q, 'V': v}, crs = 'EPSG:28992')
    gdf_q_v.to_file(export_folder/f'Debiet_en_snelheid.gpkg', driver='GPKG')
    gdf_q_v.to_file(export_folder/f'Debiet_en_snelheid.shp')
    return gdf_wl_wd_bl, gdf_q_v


def read_result(map_fn, point: tuple, variable):
    # Will create a spatial mesh based on nodes x&y, looks up which datapoint is closest to the input point
    # Returns the last timestep of that datapoint for the given variable (velocity, waterdepth or discharge)
    vars = {'velocity': 'mesh1d_u1',
            'waterdepth': 'mesh1d_waterdepth',
            'discharge': 'mesh1d_q1'}
    layer = vars[variable]
    ds = nc.Dataset(map_fn)
    mesh = [(x, y) for x, y in zip(ds.variables['mesh1d_node_x'][:], ds.variables['mesh1d_node_y'][:])]
    # TODO: Check if the kd tree really works as expected.
    mytree = spatial.cKDTree(data=mesh)
    _, closest = mytree.query(point)
    last = ds.variables[layer][-1, closest].data
    return last

if __name__ == '__main__':
    mapfn = Path(r'c:\Dev\Hydrolib_optimizer\src\moergestels_broek\DFM_OUTPUT_moergestels_broek\moergestels_broek_map.nc')
    point = (140311, 393413)
    variable = 'velocity'

    print(read_result(mapfn, point, variable))