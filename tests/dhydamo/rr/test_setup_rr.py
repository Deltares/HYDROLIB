import copy
import re
from pathlib import Path

import pandas as pd
import pytest
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io import drrwriter
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from shapely.geometry import Point, Polygon, box

from tests.dhydamo.io import test_from_hydamo


def _setup_rr_model(hydamo: HyDAMO | None = None) -> DRRModel:
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

    drrmodel.openwater.io.openwater_from_input(hydamo.catchments, lu_file, meteo_areas)

    drrmodel.external_forcings.io.boundary_from_input(
        hydamo.laterals, hydamo.catchments, drrmodel
    )

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

    return drrmodel


@pytest.fixture(scope="module")
def rr_model() -> DRRModel:
    """Return the RR model of the sample data, built once for this module (takes a few seconds)."""
    return _setup_rr_model()


def test_setup_rr_model(rr_model: DRRModel) -> None:
    drrmodel = rr_model
    assert (
        len(
            [
                i[1]["ga"]
                for i in drrmodel.unpaved.unp_nodes.items()
                if float(i[1]["ga"]) > 0.0
            ]
        )
        == 121
    )
    assert (
        len(
            [
                i[1]["ar"]
                for i in drrmodel.paved.pav_nodes.items()
                if float(i[1]["ar"]) > 0.0
            ]
        )
        == 101
    )
    assert (
        len(
            [
                i[1]["ar"]
                for i in drrmodel.greenhouse.gh_nodes.items()
                if float(i[1]["ar"]) > 0.0
            ]
        )
        == 1
    )
    assert (
        len(
            [
                i[1]["ar"]
                for i in drrmodel.openwater.ow_nodes.items()
                if float(i[1]["ar"]) > 0.0
            ]
        )
        == 113
    )
    assert len(drrmodel.external_forcings.boundary_nodes) == 121
    assert len(drrmodel.external_forcings.precip) == 121
    assert len(drrmodel.external_forcings.evap) == 121
    assert len(drrmodel.external_forcings.seepage) == 121

    assert drrmodel.unpaved.unp_nodes["15.0"]["ga"] == "1375"
    assert drrmodel.unpaved.unp_nodes["15.0"]["lv"] == "16.93"
    assert drrmodel.paved.pav_nodes["15.0"]["lv"] == "16.93"
    assert drrmodel.greenhouse.gh_nodes["15.0"]["sl"] == "16.93"
    assert drrmodel.openwater.ow_nodes["15.0"]["ar"] == "900.0"

    # series per meteo area, not the path of a meteo file
    assert isinstance(drrmodel.external_forcings.precip, dict)
    assert isinstance(drrmodel.external_forcings.evap, dict)
    assert drrmodel.external_forcings.precip["ms_15.0"]["precip"].iloc[
        2
    ] == pytest.approx(0.08, abs=1e-7)
    assert drrmodel.external_forcings.evap["ms_15.0"]["evap"].iloc[1] == pytest.approx(
        2.9, abs=1e-7
    )
    # Catchment 15.0 (55 x 45 m) touches a few 250 m seepage cells, weighted by
    # its coverage of each; counting them fully gives -0.079156. IDF seepage is
    # float32-sourced, hence the looser tolerance.
    assert drrmodel.external_forcings.seepage["sep_15.0"]["seepage"].iloc[
        1
    ] == pytest.approx(-0.033760, abs=1e-6)


def test_setup_rr_model_with_sewer_areas_and_greenhouses() -> None:
    """RR model as in the example notebook: sewer areas, overflows, greenhouses and their laterals.

    Expected values were checked against the code before the mypy refactor of
    drrreader.py.
    """
    data_path = Path("hydrolib/sample_data/data").resolve()
    hydamo, _ = test_from_hydamo._hydamo_object_from_gpkg()
    lu_file = data_path / "rasters" / "sobek_landuse2.tif"
    ahn_file = data_path / "rasters" / "AHN_2m_clipped_filled.tif"
    soil_file = data_path / "rasters" / "sobek_soil.tif"
    meteo_areas = hydamo.catchments

    hydamo.greenhouse_areas._read_gpkg_layer(
        data_path / "greenhouses.gpkg", layer_name="greenhouses", index_col="code"
    )
    hydamo.greenhouse_laterals._read_gpkg_layer(
        data_path / "greenhouse_laterals.gpkg", layer_name="laterals"
    )
    hydamo.greenhouse_laterals.snap_to_branch(
        hydamo.branches, snap_method="overal", maxdist=1100
    )
    hydamo.greenhouse_areas["roof_storage_mm"] = 5.0
    hydamo.greenhouse_areas["basin_storage_class"] = range(
        1, len(hydamo.greenhouse_areas) + 1
    )
    hydamo.sewer_areas.read_shp(
        str(data_path / "rioleringsgebieden.shp"),
        index_col="code",
        column_mapping={
            "Code": "code",
            "Berging_mm": "riool_berging_mm",
            "POC_m3s": "riool_poc_m3s",
        },
    )
    hydamo.overflows.read_shp(
        str(data_path / "overstorten.shp"),
        column_mapping={"codegerela": "codegerelateerdobject"},
    )
    hydamo.overflows.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)

    drrmodel = DRRModel()
    drrmodel.unpaved.io.unpaved_from_input(
        hydamo.catchments,
        lu_file,
        ahn_file,
        soil_file,
        10.0,
        100.0,
        1.2,
        meteo_areas,
        greenhouse_areas=hydamo.greenhouse_areas,
    )
    drrmodel.paved.io.paved_from_input(
        catchments=hydamo.catchments,
        landuse=lu_file,
        surface_level=ahn_file,
        sewer_areas=hydamo.sewer_areas,
        overflows=hydamo.overflows,
        street_storage=10.0,
        sewer_storage=10.0,
        pump_capacity=data_path / "rasters/pumpcap.tif",
        meteo_areas=meteo_areas,
    )
    drrmodel.greenhouse.io.greenhouse_from_input(
        catchments=hydamo.catchments,
        landuse=lu_file,
        surface_level=ahn_file,
        greenhouse_areas=hydamo.greenhouse_areas,
        greenhouse_laterals=hydamo.greenhouse_laterals,
        roof_storage=5.0,
        basin_storage_class=3,
        meteo_areas=meteo_areas,
    )
    drrmodel.openwater.io.openwater_from_input(hydamo.catchments, lu_file, meteo_areas)
    drrmodel.external_forcings.io.boundary_from_input(
        hydamo.laterals,
        hydamo.catchments,
        drrmodel,
        overflows=hydamo.overflows,
        greenhouse_laterals=hydamo.greenhouse_laterals,
    )
    hydamo.external_forcings.convert.laterals(
        hydamo.laterals,
        overflows=hydamo.overflows,
        greenhouse_laterals=hydamo.greenhouse_laterals,
        lateral_discharges=None,
        rr_boundaries=drrmodel.external_forcings.boundary_nodes,
    )

    # 121 catchments, plus 12 sewer areas (paved) and 5 greenhouse areas
    assert len(drrmodel.paved.pav_nodes) == 133
    assert len(drrmodel.greenhouse.gh_nodes) == 126
    assert len(drrmodel.external_forcings.boundary_nodes) == 138
    assert len(hydamo.external_forcings.lateral_nodes) == 138

    # a sewer area, discharging through its overflow
    sewer = drrmodel.paved.pav_nodes["BBB_de_Kuyper"]
    assert float(sewer["ar"]) == pytest.approx(50786.4169, abs=1e-3)
    assert sewer["lv"] == "15.95"
    assert sewer["qc"] == "0.12113900"
    assert sewer["ms"] == "ms_434.0"
    assert sewer["boundary_node"] == "BBB_de_Kuyper"

    # a greenhouse area with roof storage and basin storage class from its own columns
    greenhouse = drrmodel.greenhouse.gh_nodes["kas_2"]
    assert greenhouse["ar"] == "4788.29"
    assert greenhouse["sd"] == "5.00"
    assert greenhouse["basin_storage_class"] == "1"
    assert greenhouse["ms"] == "ms_15.0"
    assert greenhouse["boundary_node"] == "lat_kas_2"

    # overflows and greenhouse laterals are RR boundaries, coupled to the FM laterals
    assert "BBB_de_Kuyper" in drrmodel.external_forcings.boundary_nodes
    assert "lat_kas_2" in drrmodel.external_forcings.boundary_nodes
    assert (
        hydamo.external_forcings.lateral_nodes["BBB_de_Kuyper"]["discharge"]
        == "realtime"
    )
    assert (
        hydamo.external_forcings.lateral_nodes["lat_kas_2"]["discharge"] == "realtime"
    )


def test_boundary_from_input_drops_zero_area_catchments_without_crash() -> None:
    # boundary_from_input first bulk-drops every catchment whose boundary_node
    # has no RR-node with area > 0 ("not_occurring"), then used to loop over
    # the same catchments a second time and re-drop them individually. Since
    # the first pass already removed them, the second pass crashed with
    # "IndexError: single positional indexer is out-of-bounds".
    catchments = ExtendedGeoDataFrame(
        geotype=Polygon,
        required_columns=[
            "code",
            "geometry",
            "globalid",
            "lateraleknoopid",
            "boundary_node",
        ],
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


def _fnm_input_files(fnm_path: Path) -> list[str]:
    """File names of the input files (flag I) in a Sobek_3b.fnm file."""
    names = []
    for line in Path(fnm_path).read_text(encoding="latin-1").splitlines():
        match = re.match(r"\s*'([^']+)'\s*\*.*\s(I|I/O)\s*$", line)
        if match:
            names.append(match.group(1))
    return names


def test_rr_file_names_match_fnm() -> None:
    # File systems on Linux are case-sensitive: the names drrwriter uses must match Sobek_3b.fnm and the
    # files copied from resources exactly. Checked on the source, so that it also fails on Windows.
    resources = Path(drrwriter.__file__).parent.parent / "resources" / "RR"
    fnm_text = (resources / "Sobek_3b.fnm").read_text(encoding="latin-1")
    fnm_names = {
        name.lower(): name
        for name in re.findall(r"^\s*'([^']+)'", fnm_text, re.MULTILINE)
    }
    resource_names = {path.name.lower(): path.name for path in resources.iterdir()}
    source = Path(drrwriter.__file__).read_text(encoding="utf-8")
    used = re.findall(
        r'os\.path\.join\(\s*self\.output_dir\s*,\s*["\']([^"\']+)["\']', source
    )
    assert used

    for name in used:
        assert fnm_names.get(name.lower(), name) == name, (
            f"drrwriter uses {name}, Sobek_3b.fnm {fnm_names[name.lower()]}"
        )
        assert resource_names.get(name.lower(), name) == name, (
            f"drrwriter uses {name}, resources {resource_names[name.lower()]}"
        )


def test_rr_written_files_match_fnm(rr_model: DRRModel, tmp_path: Path) -> None:
    drrmodel = copy.deepcopy(
        rr_model
    )  # the time settings below should not leak into other tests
    drrmodel.d3b_parameters["Timestepsize"] = "60"
    drrmodel.d3b_parameters["StartTime"] = "'2016/06/01;00:00:00'"
    drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;00:00:00'"
    DRRWriter(drrmodel, output_dir=tmp_path, wwtp=(199000.0, 396000.0)).write_all()

    rr_dir = tmp_path / "rr"
    written = [path.name for path in rr_dir.iterdir()]
    by_lower: dict[str, list[str]] = {}
    for name in written:
        by_lower.setdefault(name.lower(), []).append(name)
    # no files that only differ in case (on Linux these would be two different files)
    assert all(len(names) == 1 for names in by_lower.values()), by_lower

    # every input file in Sobek_3b.fnm that is present has exactly that name
    for name in _fnm_input_files(rr_dir / "Sobek_3b.fnm"):
        if name.lower() in by_lower:
            assert name in written, (
                f"Sobek_3b.fnm refers to {name}, written as {by_lower[name.lower()][0]}"
            )


def test_add_precip_after_a_meteo_file_raises() -> None:
    # precipitation from a file and series per meteo area cannot be combined
    forcings = DRRModel().external_forcings
    forcings.precip = "DEFAULT.BUI"
    with pytest.raises(TypeError, match="already read from DEFAULT.BUI"):
        forcings.add_precip("ms_1", pd.Series([1.0]))
