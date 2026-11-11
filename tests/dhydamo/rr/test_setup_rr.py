import re
from pathlib import Path

import pytest
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from hydrolib.dhydamo.io import drrwriter
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from shapely.geometry import Point, Polygon, box
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
    )

    drrmodel.greenhouse.io.greenhouse_from_input(
        hydamo.catchments,
        lu_file,
        ahn_file,
        roof_storage,
        meteo_areas,
    )

    drrmodel.openwater.io.openwater_from_input(
        hydamo.catchments, lu_file, meteo_areas
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


def test_setup_rr_model():
    drrmodel = _setup_rr_model()
    assert len([i[1]['ga'] for i in drrmodel.unpaved.unp_nodes.items() if float(i[1]['ga']) > 0.0]) == 121
    assert len([i[1]['ar'] for i in drrmodel.paved.pav_nodes.items() if float(i[1]['ar']) > 0.0]) == 101
    assert len([i[1]['ar'] for i in drrmodel.greenhouse.gh_nodes.items() if float(i[1]['ar']) > 0.0]) == 1
    assert len([i[1]['ar'] for i in drrmodel.openwater.ow_nodes.items() if float(i[1]['ar']) > 0.0]) == 113
    assert len(drrmodel.external_forcings.boundary_nodes) == 121
    assert len(drrmodel.external_forcings.precip) == 121
    assert len(drrmodel.external_forcings.evap) == 121
    assert len(drrmodel.external_forcings.seepage) == 121

    assert drrmodel.unpaved.unp_nodes["15.0"]["ga"] == "1375"
    assert drrmodel.unpaved.unp_nodes["15.0"]["lv"] == "16.93"
    assert drrmodel.paved.pav_nodes["15.0"]["lv"] == "16.93"
    assert drrmodel.greenhouse.gh_nodes["15.0"]["sl"] == "16.93"
    assert drrmodel.openwater.ow_nodes["15.0"]["ar"] == "900.0"

    assert drrmodel.external_forcings.precip["ms_15.0"]["precip"].iloc[2] == pytest.approx(
        0.08, abs=1e-7
    )
    assert drrmodel.external_forcings.evap["ms_15.0"]["evap"].iloc[1] == pytest.approx(
        2.9, abs=1e-7
    )
    # Catchment 15.0 (55 x 45 m) touches a few 250 m seepage cells, weighted by
    # its coverage of each; counting them fully gives -0.079156. IDF seepage is
    # float32-sourced, hence the looser tolerance.
    assert drrmodel.external_forcings.seepage["sep_15.0"]["seepage"].iloc[1] == pytest.approx(
        -0.033760, abs=1e-6
    )


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


def _fnm_input_files(fnm_path):
    """File names of the input files (flag I) in a Sobek_3b.fnm file."""
    names = []
    for line in Path(fnm_path).read_text(encoding="latin-1").splitlines():
        match = re.match(r"\s*'([^']+)'\s*\*.*\s(I|I/O)\s*$", line)
        if match:
            names.append(match.group(1))
    return names


def test_rr_file_names_match_fnm():
    # File systems on Linux are case-sensitive: the names drrwriter uses must match Sobek_3b.fnm and the
    # files copied from resources exactly. Checked on the source, so that it also fails on Windows.
    resources = Path(drrwriter.__file__).parent.parent / "resources" / "RR"
    fnm_text = (resources / "Sobek_3b.fnm").read_text(encoding="latin-1")
    fnm_names = {name.lower(): name for name in re.findall(r"^\s*'([^']+)'", fnm_text, re.MULTILINE)}
    resource_names = {path.name.lower(): path.name for path in resources.iterdir()}
    source = Path(drrwriter.__file__).read_text(encoding="utf-8")
    used = re.findall(r'os\.path\.join\(\s*self\.output_dir\s*,\s*["\']([^"\']+)["\']', source)
    assert used

    for name in used:
        assert fnm_names.get(name.lower(), name) == name, f"drrwriter uses {name}, Sobek_3b.fnm {fnm_names[name.lower()]}"
        assert resource_names.get(name.lower(), name) == name, f"drrwriter uses {name}, resources {resource_names[name.lower()]}"


def test_rr_written_files_match_fnm(tmp_path):
    drrmodel = _setup_rr_model()
    drrmodel.d3b_parameters["Timestepsize"] = "60"
    drrmodel.d3b_parameters["StartTime"] = "'2016/06/01;00:00:00'"
    drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;00:00:00'"
    DRRWriter(drrmodel, output_dir=tmp_path, wwtp=(199000.0, 396000.0)).write_all()

    rr_dir = tmp_path / "rr"
    written = [path.name for path in rr_dir.iterdir()]
    by_lower = {}
    for name in written:
        by_lower.setdefault(name.lower(), []).append(name)
    # no files that only differ in case (on Linux these would be two different files)
    assert all(len(names) == 1 for names in by_lower.values()), by_lower

    # every input file in Sobek_3b.fnm that is present has exactly that name
    for name in _fnm_input_files(rr_dir / "Sobek_3b.fnm"):
        if name.lower() in by_lower:
            assert name in written, f"Sobek_3b.fnm refers to {name}, written as {by_lower[name.lower()][0]}"
