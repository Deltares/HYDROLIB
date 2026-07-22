from pathlib import Path

from shapely.geometry import Point, Polygon, box

from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from tests.dhydamo.io import test_from_hydamo


def _setup_rr_model(hydamo=None):
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()
    
    drrmodel = DRRModel()

    if hydamo is None:
        hydamo, _ = test_from_hydamo._hydamo_object_from_gpkg()

    # all data and settings to create the RR-model
    lu_file = data_path / "rasters" / "sobek_landuse2.tif"
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
    pumpcapacity = data_path / "rasters/pumpcap.tif"
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

    drrmodel.greenhouse.io.greenhouse_from_input(
        hydamo.catchments,
        lu_file,
        ahn_file,
        roof_storage,
        meteo_areas,
        zonalstats_alltouched=True,
    )

    drrmodel.openwater.io.openwater_from_input(
        hydamo.catchments, lu_file, meteo_areas, zonalstats_alltouched=True
    )

    drrmodel.external_forcings.io.boundary_from_input(
        hydamo.laterals, hydamo.catchments, drrmodel
    )

    seepage_folder = data_path / "rasters" / "seepage"
    precip_folder = data_path /  "rasters" / "precipitation"
    evap_folder = data_path / "rasters" / "evaporation"
    drrmodel.external_forcings.io.seepage_from_input(hydamo.catchments, seepage_folder)
    drrmodel.external_forcings.io.precip_from_input(
        meteo_areas, precip_folder=precip_folder, precip_file=None
    )
    drrmodel.external_forcings.io.evap_from_input(
        meteo_areas, evap_folder=evap_folder, evap_file=None
    )

    return drrmodel


def test_setup_rr_model(hydamo=None):
    drrmodel = _setup_rr_model(hydamo=hydamo)
    assert len([i[1]['ga'] for i in drrmodel.unpaved.unp_nodes.items() if float(i[1]['ga']) > 0.0]) == 121
    assert len([i[1]['ar'] for i in drrmodel.paved.pav_nodes.items() if float(i[1]['ar']) > 0.0]) == 101
    assert len([i[1]['ar'] for i in drrmodel.greenhouse.gh_nodes.items() if float(i[1]['ar']) > 0.0]) == 1
    assert len([i[1]['ar'] for i in drrmodel.openwater.ow_nodes.items() if float(i[1]['ar']) > 0.0]) == 113
    assert len(drrmodel.external_forcings.boundary_nodes) == 121
    assert len(drrmodel.external_forcings.precip) == 121
    assert len(drrmodel.external_forcings.evap) == 121
    assert len(drrmodel.external_forcings.seepage) == 121


def test_boundary_from_input_drops_zero_area_catchments_without_crash():
    # boundary_from_input first bulk-drops every catchment whose boundary_node
    # has no RR-node with area > 0 ("not_occurring"), then used to loop over
    # the same catchments a second time and re-drop them individually. Since
    # the first pass already removed them, the second pass crashed with
    # "IndexError: single positional indexer is out-of-bounds".
    catchments = ExtendedGeoDataFrame(
        geotype=Polygon,
        required_columns=["code", "geometry", "globalid", "lateraleknoopid", "boundary_node"],
        data={
            "code": ["cat_1", "cat_2"],
            "globalid": ["g1", "g2"],
            "lateraleknoopid": ["lat_g1", "lat_g2"],
            "boundary_node": ["bnd_1", "bnd_2"],
        },
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:28992",
    )
    laterals = ExtendedGeoDataFrame(
        geotype=Point,
        required_columns=["code", "geometry", "globalid"],
        data={
            "code": ["lat_1", "lat_2"],
            "globalid": ["lat_g1", "lat_g2"],
        },
        geometry=[Point(0.5, 0.5), Point(2.5, 2.5)],
        crs="EPSG:28992",
    )

    # A fresh DRRModel has no unpaved/paved/greenhouse/openwater nodes at all,
    # so both catchments count as "not_occurring" and must be dropped.
    drrmodel = DRRModel()

    drrmodel.external_forcings.io.boundary_from_input(laterals, catchments, drrmodel)

    assert len(catchments) == 0
    assert len(drrmodel.external_forcings.boundary_nodes) == 0
