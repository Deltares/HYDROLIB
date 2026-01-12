from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from hydrolib.dhydamo.core import hydamo
from hydrolib.dhydamo.geometry import mesh
from shapely.geometry import Point
import geopandas as gpd
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from hydrolib.core.dflowfm.mdu.models import FMModel
from tests.dhydamo.io.test_from_hydamo import _hydamo_object_from_gpkg

hydamo_data_path = (
    Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "tests" / "data"
)

@pytest.mark.parametrize(
    "his_file, location_type, location_id, variable, starttime, endtime, outcome",
     [
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'observation_point', 'ObsP_113GIS' , 'waterlevel'           ,                  None,                  None, (12.36, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'weir'             , 'S_96550'     , 'discharge'            ,                  None,                  None, (2.34, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'weir'             , 'S_96550'     , 'waterlevel_upstream'  ,                  None,                  None, (17.11, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'weir'             , 'S_96550'     , 'waterlevel_downstream',                  None,                  None, (17.00, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'weir'             , 'S_96550'     , 'waterlevel_downstream', '2016-06-01 12:00:00', '2016-06-02 12:00:00', (16.96, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'orifice'          , 'orifice_test', 'waterlevel_upstream'  ,                  None,                  None, (18.41, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'pump'             , '114GIS_1'    , 'discharge'            ,                  None,                  None, (0.17, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'uweir'            , 'uweir_test'  , 'discharge'            ,                  None,                  None, (-0.01, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'bridge'           , 'KBR_test'    , 'discharge'            ,                  None,                  None,  (2.47, Point)),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'compound'         , 'cmp_114GIS'  , 'discharge'            ,                  None,                  None,  (0.17, type(None))),
        (hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 'culvert'          , 'D_25561'     , 'discharge'            ,                  None,                  None,  (5.59, Point)),         
    ],
)
def test_get_timeseries_from_other_model(his_file, location_type, location_id, variable, starttime, endtime, outcome):
    hydamo = HyDAMO()  

    geom, series = hydamo.external_forcings.convert.timeseries_from_other_model(his_file=his_file, location_type=location_type, location_id=location_id, variable=variable, starttime=starttime, endtime=endtime)

    assert np.round(np.mean(series),2) == outcome[0]
    assert isinstance(geom, outcome[1])

    return geom, series

def test_add_boundary_from_other_model():
    # Get full hydamo object
    hydamo, _ = _hydamo_object_from_gpkg()  
    fm = FMModel()

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=None,
    )

    geom, series = hydamo.external_forcings.convert.timeseries_from_other_model(his_file=hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 
                                                        location_type='culvert',
                                                        location_id= 'D_25561', 
                                                        variable='discharge'
                                                        )

    hydamo.external_forcings.add_boundary_condition(
        'S_96544',
        geom,
        "dischargebnd",
        series,
        fm.geometry.netfile.network,
    )
    
    assert len(hydamo.external_forcings.boundary_nodes) == 1


def test_add_lateral_from_other_model():
    # Get full hydamo object
    hydamo, _ = _hydamo_object_from_gpkg()  
        
    geom, series = hydamo.external_forcings.convert.timeseries_from_other_model(his_file=hydamo_data_path / Path('other_model/fm/output/test_his.nc'), 
                                                        location_type='culvert',
                                                        location_id= 'D_25561', 
                                                        variable='discharge'
                                                        )
    
    point = ExtendedGeoDataFrame(geotype=Point)
    point.set_data(gpd.GeoDataFrame(geometry=[geom]))
    point.snap_to_branch(
        hydamo.branches,
        snap_method='overal',
        maxdist=5,
    )

    hydamo.external_forcings.add_lateral(
        'S_96544',
        point.branch_id.squeeze(),
        str(point.branch_offset.squeeze()),
        series,
        )
    
    assert len(hydamo.external_forcings.lateral_nodes) == 1

