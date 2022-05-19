#Inladen packages
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import LineString
from shapely.errors import ShapelyDeprecationWarning
from pathlib import Path
import numpy as np
import warnings
from hydrolib.core.io.crosssection.models import CrossLocModel
from sympy import solve
from sympy import Symbol
import math


#Functie voor het inladen van het te optimaliseren gebied in de vorm van een shapefile
def select_crosssections(shapefile_path, model_network_nc, crossdef_filename, output_folder):
    output_folder = Path(output_folder)
    shapefile_area = gpd.read_file(shapefile_path)
    ds = xr.open_dataset(model_network_nc)

    #maak een geodataframe van alle nodes
    df = pd.concat([pd.Series(ds['network1d_geom_x'].values), pd.Series(ds['network1d_geom_y'].values)], axis=1)
    df.columns = ['network1d_geom_x', 'network1d_geom_y']
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.network1d_geom_x, df.network1d_geom_y))
    gdf.to_file(driver = 'ESRI Shapefile',filename=str(output_folder/'network1d_geomxandy.shp'))

    #aantal nodes per branch id
    df_branches = pd.concat([pd.Series(ds['network1d_geom_node_count'].values), pd.Series(ds['network1d_branch_id'].values.astype(str))], axis=1)
    df_branches.columns = ['network1d_geom_node_count', 'branchid']

    #maak aparte linestring aan voor elke branch id met alle nodes die erbij horen
    df_branches['line_geometry'] = ''
    df_branches['start_node'] = ''
    df_branches['end_node'] = ''

    for j in range(len(df_branches)):
        if j == 0:
            start_node = 0
            end_node = 0 + df_branches['network1d_geom_node_count'][j]
            df_branches.loc[j, 'start_node'] = start_node
            df_branches.loc[j, 'end_node'] = end_node
            linestring = LineString(gdf.iloc[df_branches['start_node'][j] : df_branches['end_node'][j]]['geometry'].values)
            with warnings.catch_warnings(): # This deprication warning is not relevant to this situation
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                df_branches.loc[j, 'line_geometry'] = linestring
        else:
            start_node = df_branches['network1d_geom_node_count'][:j].sum()
            end_node = start_node + df_branches['network1d_geom_node_count'].iloc[j]
            df_branches.loc[j, 'start_node'] = start_node
            df_branches.loc[j, 'end_node'] = end_node
            linestring = LineString(gdf.iloc[df_branches['start_node'][j] : df_branches['end_node'][j]]['geometry'].values)
            with warnings.catch_warnings(): # This deprication warning is not relevant to this situation 
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                df_branches.loc[j, 'line_geometry'] = linestring

    gdf_branches = gpd.GeoDataFrame(df_branches, geometry=df_branches.line_geometry)
    del gdf_branches['line_geometry']
    gdf_branches['branchid'] = gdf_branches['branchid'].str.strip()

    #Lees de cross_section_locations file uit:
    crossloc_source = Path(crossdef_filename)
    cross_loc = pd.DataFrame([cs.__dict__ for cs in CrossLocModel(crossloc_source).crosssection])

    #Koppel deze cross section locations aan de geometrie van de branches 
    df_cross_loc = pd.merge(cross_loc, gdf_branches, on='branchid', how='left')
    gdf_cross_loc = gpd.GeoDataFrame(df_cross_loc, geometry='geometry')

    #lengte van geometrie branch hoeft niet hetzelfde te zijn aan de gedefenieerde user length, dus bereken de geschaalde offset (chainage)
    df_lengths= pd.concat([pd.Series(ds['network1d_edge_length'].values), pd.Series(ds['network1d_branch_id'].values.astype(str))], axis=1)
    df_lengths.columns = ['user_length', 'branchid']
    df_lengths['branchid'] = df_lengths['branchid'].str.strip()
    gdf_cross_loc = pd.merge(gdf_cross_loc, df_lengths, on='branchid', how='left')
    gdf_cross_loc['length'] = gdf_cross_loc['geometry'].length
    gdf_cross_loc['geschaalde_offset'] = gdf_cross_loc['chainage'].astype(float) / gdf_cross_loc['user_length'] * gdf_cross_loc['length']


    #Gebruik de chainage om de punt geometrie van de cross_loc profielen te definiëren:
    gdf_cross_loc['cross_loc_geom'] = gdf_cross_loc['geometry'].interpolate(gdf_cross_loc['geschaalde_offset'].astype(float))
    gdf_cross_loc.rename(columns={'geometry': 'branch_geometry'}, inplace=True)
    gdf_cross_loc = gdf_cross_loc[['id', 'branchid', 'chainage', 'geschaalde_offset', 'definitionid', 'cross_loc_geom', 'user_length', 'length']]
    gdf_cross_loc = gpd.GeoDataFrame(gdf_cross_loc, geometry='cross_loc_geom')
    gdf_cross_loc.to_file(str(output_folder/'cross_loc_geometry.shp'),driver = 'ESRI Shapefile')

    #Clip deze profiles geometrieën op de opgegeven optimization area
    points_clip = gpd.clip(gdf_cross_loc, shapefile_area)
    points_clip.to_file(str(output_folder/'cross_loc_geometry_within_optimization_area.shp'), driver = 'ESRI Shapefile')
    points_clip.rename(columns={'cross_loc_geom': 'geometry'}, inplace=True)
    points_clip = points_clip[['branchid', 'definitionid', 'geometry']]
    return points_clip
    

def search_window(b_start_value, bandwidth_perc, iterations):
    max_bound_b = b_start_value * (1 + (bandwidth_perc / 100))
    min_bound_b = b_start_value * (1 - (bandwidth_perc / 100))
    b_waardes_binnen_zoekruimte = np.linspace(min_bound_b, max_bound_b, iterations)
    return b_waardes_binnen_zoekruimte

# Ideas:
# Checks with material (sand/clay/peat/etc) if talud & velocity are okay (vademecum)

def determine_v_with_manning(d, talud, b, slope, kmanning):
    A = ((talud * d) * d)+ (b * d)
    R = A / (b + math.sqrt(d ** 2 + (d * talud) ** 2) * 2)
    V = R ** (2 / 3) * kmanning * slope ** (1 / 2)
    return V

def check_QVA(Q_target, d, talud, b, slope, kmanning, allowed_variation=0.05):
    # Belangrijke keuze: 0.05% aanpassen (stop dit in notitie)
    if b < 0:
        b = 1
        print("First guess for bottom width was negative. Trying to find a positive bottom width, starting at 1 m...")

    V = determine_v_with_manning(d, talud, b, slope, kmanning)
    A = ((talud * d) * d)+ (b * d)
    Q = V * A
    print(f"Initial: width: {b:.2f}, V: {V:.4f}, Q: {Q:.4f}")
    deviation_from_target = (Q - Q_target) / Q_target
    counter = 0
    while abs(deviation_from_target) > allowed_variation:
        counter += 1
        stepsize = 0.05  # %
        if deviation_from_target < - allowed_variation:
            b *= (1+stepsize)
            V = determine_v_with_manning(d, talud, b, slope, kmanning)
            A = ((talud * d) * d) + (b * d)
            Q = V * A
            deviation_from_target = (Q - Q_target) / Q_target
            print(f"Adjustment {counter}: new width: {b:.2f}, V: {V:.4f}, Q: {Q:.4f}")
        elif deviation_from_target > allowed_variation:
            b *= (1-stepsize)
            V = determine_v_with_manning(d, talud, b, slope, kmanning)
            A = ((talud * d) * d) + (b * d)
            Q = V * A
            deviation_from_target = (Q - Q_target) / Q_target
            print(f"Adjustment {counter}: new width: {b:.2f}, V: {V:.4f}, Q: {Q:.4f}")
        if counter == 30:
            print("Failed to find suitable initial bottom width in 30 tries, please check if your inputs are correct "
                  "and use the returned bottom width with caution.")
            return b
    else:
        return b


def bottom_width(kmanning, slope, talud, diepte, V_target):
    # Let op: dit is symmetrisch!
    R_23 = V_target / (kmanning * slope ** (1/2))
    b = Symbol('b')
    A = (b * diepte) + (diepte * diepte * talud)
    P = b + (2 * math.sqrt(diepte ** 2 + (diepte * talud)**2))
    eq = (A/P) ** (2/3) - R_23
    return solve(eq, b)

    
    
    

if __name__ == '__main__':
    u_gewenst = 0.22
    afvoer = 1.5
    waterdiepte = 0.80
    talud_profiel = 1
    verhang = 0.5 / 1400
    strickler_ks = 10

    import plotly.express as px
    import numpy as np
    x_vals = np.linspace(-1, 5, 1000)
    fig = px.line(x=x_vals, y=[determine_v_with_manning(waterdiepte, talud_profiel, b, verhang, strickler_ks) for b in x_vals])
    # fig.write_html("C:/local/check_v_with_manning.html")

    width = bottom_width(strickler_ks, verhang, talud_profiel, waterdiepte, u_gewenst)
    print(width)
