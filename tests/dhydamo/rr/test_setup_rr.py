import os
import sys
from pathlib import Path

from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from tests.dhydamo.io import test_from_hydamo


def test_setup_rr_model(hydamo=None):

    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/tests/model").resolve()
    assert output_path.exists()

    drrmodel = DRRModel()

    if hydamo is None:
        hydamo = test_from_hydamo.test_hydamo_object_from_gpkg()

    # all data and settings to create the RR-model
    lu_file = data_path / "rasters" / "sobek_landuse.tif"
    ahn_file = data_path / "rasters" / "AHN_2m_clipped_filled.tif"
    soil_file = data_path / "rasters" / "sobek_soil.tif"
    surface_storage = 10.0
    infiltration_capacity = 100.0
    initial_gwd = 1.2  # water level depth below surface

    runoff_resistance = 1.0
    infil_resistance = 300.0
    layer_depths = [0.0, 1.0, 2.0]
    layer_resistances = [30, 200, 10000]
    street_storage = 10.0
    sewer_storage = 10.0
    pumpcapacity = 10.0
    roof_storage = 10.0
    meteo_areas = hydamo.catchments

    drrmodel.unpaved.io.unpaved_from_input(
        hydamo.catchments,
        lu_file,
        ahn_file,
        soil_file,
        surface_storage,
        infiltration_capacity,
        initial_gwd,
        meteo_areas,
    )
    drrmodel.unpaved.io.ernst_from_input(
        hydamo.catchments,
        depths=layer_depths,
        resistance=layer_resistances,
        infiltration_resistance=infil_resistance,
        runoff_resistance=runoff_resistance,
    )
    assert len([i[1]['ga'] for i in drrmodel.unpaved.unp_nodes.items() if float(i[1]['ga']) > 0.0]) == 129
    
    drrmodel.paved.io.paved_from_input(
        catchments=hydamo.catchments,
        landuse=lu_file,
        surface_level=ahn_file,
        street_storage=street_storage,
        sewer_storage=sewer_storage,
        pump_capacity=pumpcapacity,
        meteo_areas=meteo_areas,
        zonalstats_alltouched=True,
    )
    # assert len([i[1]['ar'] for i in drrmodel.paved.pav_nodes.items() if float(i[1]['ar']) > 0.0]) == 107

    drrmodel.greenhouse.io.greenhouse_from_input(
        hydamo.catchments,
        lu_file,
        ahn_file,
        roof_storage,
        meteo_areas,
        zonalstats_alltouched=True,
    )
    assert len([i[1]['ar'] for i in drrmodel.greenhouse.gh_nodes.items() if float(i[1]['ar']) > 0.0]) == 1
    
    drrmodel.openwater.io.openwater_from_input(
        hydamo.catchments, lu_file, meteo_areas, zonalstats_alltouched=True
    )

    # assert len([i[1]['ar'] for i in drrmodel.openwater.ow_nodes.items() if float(i[1]['ar']) > 0.0]) == 116
    
    drrmodel.external_forcings.io.boundary_from_input(
        hydamo.laterals, hydamo.catchments, drrmodel
    )
    
    assert len(drrmodel.external_forcings.boundary_nodes) == 121

    seepage_folder = data_path / "rasters" / "seepage"
    precip_folder = data_path / "rasters" / "precipitation"
    evap_folder = data_path / "rasters" / "evaporation"
    drrmodel.external_forcings.io.seepage_from_input(hydamo.catchments, seepage_folder)
    drrmodel.external_forcings.io.precip_from_input(
        meteo_areas, precip_folder=precip_folder, precip_file=None
    )
    drrmodel.external_forcings.io.evap_from_input(
        meteo_areas, evap_folder=evap_folder, evap_file=None
    )

    assert len(drrmodel.external_forcings.precip) == 129
    assert len(drrmodel.external_forcings.evap) == 129
    assert len(drrmodel.external_forcings.seepage) == 129

    drrmodel.d3b_parameters["Timestepsize"] = 300
    drrmodel.d3b_parameters[
        "StartTime"
    ] = "'2016/06/01;00:00:00'"  # should be equal to refdate for D-HYDRO
    drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;00:00:00'"
    drrmodel.d3b_parameters["RestartIn"] = 0
    drrmodel.d3b_parameters["RestartOut"] = 0
    drrmodel.d3b_parameters["RestartFileNamePrefix"] = "Test"
    drrmodel.d3b_parameters["UnsaturatedZone"] = 1
    drrmodel.d3b_parameters["UnpavedPercolationLikeSobek213"] = -1
    drrmodel.d3b_parameters["VolumeCheckFactorToCF"] = 100000

    hydamo.external_forcings.convert.laterals(
        hydamo.laterals,
        lateral_discharges=None,
        rr_boundaries=drrmodel.external_forcings.boundary_nodes,
    )

    rr_writer = DRRWriter(
        drrmodel, output_dir=output_path, name="test", wwtp=(199000.0, 396000.0)
    )
    rr_writer.write_all()

    return drrmodel
