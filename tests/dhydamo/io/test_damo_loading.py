"""Loading DAMO GeoPackages of every version into HYDROLIB's data model."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Literal
from unittest import mock

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.dhydamo.core.data_model import (
    OWN_TABLES,
    TABLES,
    DamoRelations,
)
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame
from hydrolib.dhydamo.io.damo_versions import SUPPORTED_HYDAMO_VERSIONS, loader, steps
from shapely.geometry import LineString, Point, Polygon

from tests.dhydamo.io.damo_gpkg import (
    DATA_PATH,
    _damo_names,
    _install_fake_validator,
    _make_minimal_25_gpkg,
    _sturing,
    _syntax_validator,
    _write_branch,
    _write_layer,
    _write_stuw,
)


def _make_roughness_gpkg(
    tmp_path: Path, roughness_type: str, parameter: str, culvert_shape: str
) -> Path:
    """Write the layers whose column names changed in DAMO 2.4 and 2.5.

    The column names are those of the DAMO version under test: *roughness_type*
    is ``typeruwheid`` in 2.2 and ``typemethoderuwheid`` from 2.4, *parameter*
    is ``soortparameter`` up to 2.4 and ``typeparameter`` from 2.5, and
    *culvert_shape* is ``vormkoker`` up to 2.4 and ``vormduikersifonhevel`` from 2.5.
    """
    gpkg_path = tmp_path / "roughness.gpkg"
    _write_layer(
        gpkg_path,
        "NormgeparamProfielWaarde",
        gpd.GeoDataFrame(
            {
                "normgeparamprofielid": ["np-guid"],
                parameter: ["bodembreedte"],
                "waarde": [2.0],
                "ruwheidlaag": [30.0],
                "ruwheidhoog": [20.0],
                roughness_type: ["Manning"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Brug",
        gpd.GeoDataFrame(
            {
                "code": ["BR1"],
                "globalid": ["brug-guid"],
                "lengte": [5.0],
                "intreeverlies": [0.5],
                "uittreeverlies": [1.0],
                "ruwheid": [30.0],
                roughness_type: ["Manning"],
                "geometry": [Point(2, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "DuikerSifonHevel",
        gpd.GeoDataFrame(
            {
                "code": ["DU1"],
                "globalid": ["duiker-guid"],
                "lengte": [10.0],
                "hoogteopening": [1.0],
                "breedteopening": [1.0],
                "hoogtebinnenonderkantbene": [0.0],
                "hoogtebinnenonderkantbov": [0.1],
                culvert_shape: ["Rond"],
                "intreeverlies": [0.5],
                "uittreeverlies": [1.0],
                "ruwheid": [75.0],
                roughness_type: ["StricklerKs"],
                "geometry": [LineString([(3, 0), (4, 0)])],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "RuwheidProfiel",
        gpd.GeoDataFrame(
            {
                "profielpuntid": ["pp-guid"],
                "ruwheidlaag": [30.0],
                "ruwheidhoog": [20.0],
                roughness_type: ["Manning"],
                "geometry": [Point(0, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    return gpkg_path


def _assert_canonical_roughness_columns(hydamo: HyDAMO) -> None:
    """Assert that the renamed DAMO columns arrive under their 2.2 names."""
    assert hydamo.param_profile_values["typeruwheid"].tolist() == ["Manning"]
    assert hydamo.param_profile_values["soortparameter"].tolist() == ["bodembreedte"]
    assert hydamo.bridges.loc["BR1", "typeruwheid"] == "Manning"
    assert hydamo.culverts.loc["DU1", "typeruwheid"] == "StricklerKs"
    assert hydamo.culverts.loc["DU1", "vormkoker"] == "Rond"
    assert hydamo.profile_roughness["typeruwheid"].tolist() == ["Manning"]


def test_load_from_gpkg_maps_damo_24_roughness_type(tmp_path: Path) -> None:
    gpkg_file = _make_roughness_gpkg(
        tmp_path,
        roughness_type="typemethoderuwheid",
        parameter="soortparameter",
        culvert_shape="vormkoker",
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.4")

    _assert_canonical_roughness_columns(hydamo)


def test_load_from_gpkg_maps_damo_25_parameter_and_culvert_shape(
    tmp_path: Path,
) -> None:
    gpkg_file = _make_roughness_gpkg(
        tmp_path,
        roughness_type="typemethoderuwheid",
        parameter="typeparameter",
        culvert_shape="vormduikersifonhevel",
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    _assert_canonical_roughness_columns(hydamo)


def _write_control_layers(gpkg_path: Path, names: dict[str, str]) -> None:
    """Write a branch, an opening, its management device and a Sturing, with *names*."""
    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": ["H1"],
                "globalid": ["hydro-guid"],
                "geometry": [LineString([(0, 0), (10, 0)])],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": ["opening-guid"],
                "stuwid": ["stuw-guid"],
                names["regelmiddelid"]: ["device-guid"],
                "laagstedoorstroombreedte": [1.2],
                "laagstedoorstroomhoogte": [0.4],
                "afvoercoefficient": [1.0],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        names["device_layer"],
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["device-guid"],
                "kunstwerkopeningid": ["opening-guid"],
                "overlaatonderlaat": ["onderlaat"],
                names["soortregelmiddel"]: ["schuif"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "code": ["CTRL1"],
                "globalid": ["mgmt-guid"],
                names["regelmiddelid"]: ["device-guid"],
                "pompid": [None],
                names["stuurvariabele"]: ["hoogte opening"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_delivers_damo_22_column_names(
    tmp_path: Path, version: str
) -> None:
    names = _damo_names(version)
    gpkg_file = _make_roughness_gpkg(
        tmp_path,
        roughness_type=names["typeruwheid"],
        parameter=names["soortparameter"],
        culvert_shape=names["vormkoker"],
    )
    _write_control_layers(gpkg_file, names)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    _assert_canonical_roughness_columns(hydamo)
    assert hydamo.opening["regelmiddelid"].tolist() == ["device-guid"]
    assert hydamo.management.loc["CTRL1", "regelmiddelid"] == "device-guid"
    assert hydamo.management.loc["CTRL1", "stuurvariabele"] == "hoogte opening"
    device = hydamo.management_device
    assert device["globalid"].tolist() == ["device-guid"]
    assert device["soortregelmiddel"].tolist() == ["schuif"]
    assert "soortafsluitmiddel" not in device.columns


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_passes_undeclared_columns_through(
    tmp_path: Path, version: str
) -> None:
    gpkg_file = tmp_path / "branch.gpkg"
    _write_branch(gpkg_file, opmerking=["own column"])

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert hydamo.branches.loc["H1", "opmerking"] == "own column"


def test_dropping_undeclared_columns_makes_a_builder_fail_on_reading_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The test-only switch the example-model builds use (_hydamo_object_from_gpkg):
    # a builder reading a column the data model doesn't declare fails in CI.
    monkeypatch.setattr(loader, "_KEEP_UNDECLARED_COLUMNS", False)
    gpkg_file = tmp_path / "branch.gpkg"
    _write_branch(gpkg_file, naam=["Beek"], opmerking=["own column"])

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    assert hydamo.branches.loc["H1", "naam"] == "Beek"
    assert "opmerking" not in hydamo.branches.columns
    with pytest.raises(KeyError, match="opmerking"):
        mesh.mesh1d_order_numbers_from_attribute(
            hydamo.branches, [], order_attribute="opmerking", network=mock.Mock()
        )


def test_dropping_undeclared_columns_drops_a_derived_column_the_file_supplies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # boundary_node is derived from the laterals; without laterals the derivation
    # leaves it alone, so the guard must not let the file's own value in.
    monkeypatch.setattr(loader, "_KEEP_UNDECLARED_COLUMNS", False)
    gpkg_file = tmp_path / "catchments.gpkg"
    _write_layer(
        gpkg_file,
        "afvoergebiedaanvoergebied",
        gpd.GeoDataFrame(
            {
                "code": ["AG1"],
                "globalid": ["ag-guid"],
                "lateraleknoopid": ["lat-guid"],
                "boundary_node": ["from the file"],
                "geometry": [Polygon([(0, 0), (1, 0), (1, 1)])],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    assert hydamo.catchments.index.tolist() == ["AG1"]
    assert "boundary_node" not in hydamo.catchments.columns


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_names_layer_and_missing_required_column(
    tmp_path: Path, version: str
) -> None:
    gpkg_file = tmp_path / "weir.gpkg"
    _write_stuw(gpkg_file, afvoercoefficient=None)

    with pytest.raises(KeyError, match=r'layer "Stuw".*"afvoercoefficient"'):
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)


@pytest.mark.parametrize("missing", ["profiellijnid", "codevolgnummer"])
def test_load_from_gpkg_names_layer_and_missing_point_grouping_column(
    tmp_path: Path, missing: str
) -> None:
    gpkg_file = tmp_path / "profile.gpkg"
    points = {
        "code": ["PP1", "PP2"],
        "globalid": ["line-guid", "line-guid"],
        "profiellijnid": ["line-guid", "line-guid"],
        "codevolgnummer": [1, 2],
    }
    del points[missing]
    _write_layer(
        gpkg_file,
        "ProfielPunt",
        gpd.GeoDataFrame(
            {**points, "geometry": [Point(0, 0, 1.0), Point(1, 0, 0.0)]},
            crs="EPSG:28992",
        ),
    )

    with pytest.raises(ValueError, match=rf'layer "ProfielPunt".*"{missing}"'):
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")


def test_load_from_gpkg_replaces_the_data_of_an_earlier_load(tmp_path: Path) -> None:
    first, second = tmp_path / "first.gpkg", tmp_path / "second.gpkg"
    _write_layer(
        first,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": ["A1", "A2"],
                "globalid": ["a1-guid", "a2-guid"],
                "afvoercoefficient": [1.0, 1.0],
                "naam": ["first", "second"],
                "geometry": [Point(0, 0), Point(1, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(first, "Sturing", _sturing([{"code": "CTRL1"}]))
    _write_stuw(second)
    hydamo = HyDAMO().load_from_gpkg(first, hydamo_version="2.2")
    weirs = hydamo.weirs

    hydamo.load_from_gpkg(second, hydamo_version="2.2")

    # a reference taken before the second load sees its data
    assert weirs is hydamo.weirs
    assert weirs.index.tolist() == ["ST1"]
    assert "naam" not in weirs.columns
    # the second file has no Sturing layer: nothing of the first file's is left
    assert hydamo.management.empty
    assert list(hydamo.management.columns) == list(HyDAMO().management.columns)


@pytest.mark.parametrize("table_name", ["sewer_areas", "management"])
def test_set_data_drops_the_columns_the_new_data_lacks(table_name: str) -> None:
    table = getattr(HyDAMO(), table_name)
    first = {"code": ["A"], "globalid": ["a-guid"], "naam": ["first"]}
    second = {"code": ["B"], "globalid": ["b-guid"]}
    if table_name == "sewer_areas":
        area = Point(0, 0).buffer(1)
        table.set_data(gpd.GeoDataFrame({**first, "geometry": [area]}), "code")
        table.set_data(gpd.GeoDataFrame({**second, "geometry": [area]}), "code")
    else:
        table.set_data(pd.DataFrame(first), "code")
        table.set_data(pd.DataFrame(second), "code")

    assert table.index.tolist() == ["B"]
    assert "naam" not in table.columns


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_loads_without_optional_columns(
    tmp_path: Path, version: str
) -> None:
    gpkg_file = tmp_path / "weir.gpkg"
    _write_stuw(gpkg_file)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert hydamo.weirs.index.tolist() == ["ST1"]
    assert "naam" not in hydamo.weirs.columns
    assert "laagstedoorstroomhoogte" not in hydamo.weirs.columns


def test_load_from_gpkg_leaves_columns_the_version_lacks_absent(
    tmp_path: Path,
) -> None:
    # This 2.5 Afsluitmiddel has no maximalehoogteonderkant, from which hoogteopening
    # is derived.
    gpkg_file = tmp_path / "device_25.gpkg"
    _write_layer(
        gpkg_file,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["device-guid"],
                "kunstwerkopeningid": ["opening-guid"],
                "overlaatonderlaat": ["onderlaat"],
                "typeafsluitmiddel": ["schuif"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert hydamo.management_device["globalid"].tolist() == ["device-guid"]
    assert "hoogteopening" not in hydamo.management_device.columns


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_derives_profile_code_and_catchment_boundary_node(
    tmp_path: Path, version: str
) -> None:
    gpkg_file = tmp_path / "derived.gpkg"
    _write_layer(
        gpkg_file,
        "ProfielPunt",
        gpd.GeoDataFrame(
            {
                "code": ["PP2", "PP1"],
                "globalid": ["line-guid", "line-guid"],
                "profiellijnid": ["line-guid", "line-guid"],
                "codevolgnummer": [2, 1],
                "geometry": [Point(1, 0, 0.0), Point(0, 0, 1.0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_file,
        "lateraleknoop",
        gpd.GeoDataFrame(
            {
                "code": ["LAT1"],
                "globalid": ["lateral-guid"],
                "geometry": [Point(5, 5)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_file,
        "afvoergebiedaanvoergebied",
        gpd.GeoDataFrame(
            {
                "code": ["C1"],
                "globalid": ["catchment-guid"],
                "lateraleknoopid": ["lateral-guid"],
                "geometry": [Point(5, 5).buffer(2)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert hydamo.profile["code"].tolist() == ["line-guid"]
    assert hydamo.profile.index.tolist() == ["line-guid"]
    assert hydamo.catchments.loc["C1", "boundary_node"] == "LAT1"
    assert hydamo.laterals.index.tolist() == ["LAT1"]


def test_supported_damo_converter_versions() -> None:
    assert SUPPORTED_HYDAMO_VERSIONS == ("2.2", "2.3", "2.4", "2.5", "2.6")


def test_load_from_gpkg_reads_a_damo_26_file_like_its_25_equivalent(
    tmp_path: Path,
) -> None:
    # DAMO 2.6 renamed none of the columns HYDROLIB reads, so a 2.5 file has 2.6's
    # real column names for them.
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    hydamo_25 = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")
    hydamo_26 = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.6")

    assert hydamo_26.source_damo_version == "2.6"
    for name in TABLES:
        pd.testing.assert_frame_equal(
            getattr(hydamo_26, name), getattr(hydamo_25, name), obj=name
        )
    assert hydamo_26.management_device.loc[0, "hoogteopening"] == pytest.approx(0.3)


def test_load_from_gpkg_uses_versioned_converter_for_damo_22() -> None:
    gpkg_file = DATA_PATH / "Example_model.gpkg"
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    assert hydamo.source_damo_version == "2.2"
    assert hydamo.validation_result is None
    assert len(hydamo.branches) == 61
    assert len(hydamo.management_device) == 32
    assert len(hydamo.pumps) == 3


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_load_from_gpkg_passthrough_versions_behave_like_22(version: str) -> None:
    gpkg_file = DATA_PATH / "Example_model.gpkg"
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert hydamo.source_damo_version == version
    assert len(hydamo.branches) == 61
    assert len(hydamo.management_device) == 32
    assert len(hydamo.pumps) == 3


def test_load_from_gpkg_maps_damo_25_to_canonical_internal_structure(
    tmp_path: Path,
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert hydamo.source_damo_version == "2.5"
    # Undeclared columns pass through under their 2.5 names, though DAMO renamed them.
    assert hydamo.branches.loc["H1", "categorieoppervlaktewater"] == "primair"
    assert "categorieoppwaterlichaam" not in hydamo.branches.columns

    assert hydamo.management_device.loc[0, "soortregelmiddel"] == "schuif"
    # 0.3 m above the opening's crest of 0.4 m
    assert hydamo.management_device.loc[0, "hoogteopening"] == pytest.approx(0.3)
    assert (
        hydamo.management_device.loc[0, "typeregelbaarheid"]
        == "Regelbaar - lokaal automatisch - elektronisch"
    )
    assert "soortregelbaarheid" not in hydamo.management_device.columns

    assert "regelmiddelid" in hydamo.management.columns
    assert "stuurvariabele" in hydamo.management.columns
    assert hydamo.management.loc["mgmt-guid", "regelmiddelid"] == "afsluit-guid"
    assert hydamo.management.loc["mgmt-guid", "stuurvariabele"] == "hoogte opening"

    assert "regelmiddelid" in hydamo.opening.columns
    assert hydamo.opening.loc[0, "regelmiddelid"] == "afsluit-guid"


def test_damo_25_relations_resolve_expected_rows(tmp_path: Path) -> None:
    hydamo = HyDAMO().load_from_gpkg(
        _make_minimal_25_gpkg(tmp_path), hydamo_version="2.5"
    )

    opening = hydamo.opening.loc[[0]]
    management = hydamo.management.loc[["mgmt-guid"]]
    weir = DamoRelations.OPENING_WEIR.parents_of(opening, hydamo.weirs)
    device_from_opening = DamoRelations.OPENING_MANAGEMENT_DEVICE.parents_of(
        opening, hydamo.management_device
    )
    device_from_management = DamoRelations.MANAGEMENT_DEVICE.parents_of(
        management, hydamo.management_device
    )
    opening_from_device = DamoRelations.MANAGEMENT_DEVICE_OPENING.parents_of(
        device_from_management, hydamo.opening
    )

    assert weir["globalid"].tolist() == ["stuw-guid"]
    assert device_from_opening["globalid"].tolist() == ["afsluit-guid"]
    assert device_from_management["globalid"].tolist() == ["afsluit-guid"]
    assert opening_from_device["globalid"].tolist() == ["opening-guid"]


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_indexes_weirs_by_unique_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    gpkg_path = tmp_path / "weirs.gpkg"
    _write_layer(
        gpkg_path,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": ["ST2", "ST1", "ST2"],
                "globalid": ["stuw-guid-1", "stuw-guid-2", "stuw-guid-3"],
                "afvoercoefficient": [1.0] * 3,
                "geometry": [Point(1, 0), Point(2, 0), Point(3, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version=version)

    assert hydamo.weirs.index.tolist() == ["ST2", "ST1", "ST2_1"]
    assert hydamo.weirs["code"].tolist() == ["ST2", "ST1", "ST2_1"]
    assert "Index column 'code' contains duplicates (['ST2'])" in caplog.text


def test_load_from_gpkg_skips_existing_codes_when_suffixing_duplicates(
    tmp_path: Path,
) -> None:
    gpkg_path = tmp_path / "weirs.gpkg"
    _write_layer(
        gpkg_path,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": ["ST2", "ST2", "ST2_1"],
                "globalid": ["stuw-guid-1", "stuw-guid-2", "stuw-guid-3"],
                "afvoercoefficient": [1.0] * 3,
                "geometry": [Point(1, 0), Point(2, 0), Point(3, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.5")

    assert hydamo.weirs.index.tolist() == ["ST2", "ST2_2", "ST2_1"]


def test_load_from_gpkg_makes_non_spatial_index_unique(tmp_path: Path) -> None:
    # Sturing has no geometry in HYDROLIB; duplicate codes must get a suffix, as
    # for spatial tables, so that dropping one row by label drops only that row.
    gpkg_path = tmp_path / "duplicate_sturing.gpkg"
    _write_layer(
        gpkg_path,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "code": ["CTRL1", "CTRL1"],
                "globalid": ["mgmt-guid-1", "mgmt-guid-2"],
                "geometry": [Point(5, 0), Point(6, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.2")

    assert hydamo.management.index.tolist() == ["CTRL1", "CTRL1_1"]


def test_load_from_gpkg_fills_missing_management_code_from_globalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # DAMO defines no code for Sturing, so HYDROLIB fills it from globalid.
    gpkg_path = tmp_path / "sturing_22.gpkg"
    _write_layer(
        gpkg_path,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "globalid": ["mgmt-guid"],
                "regelmiddelid": ["regel-guid"],
                "stuurvariabele": ["hoogte opening"],
                "typecontroller": ["PID"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.2")

    assert hydamo.management.index.tolist() == ["mgmt-guid"]
    assert hydamo.management.loc["mgmt-guid", "code"] == "mgmt-guid"
    assert hydamo.management.loc["mgmt-guid", "regelmiddelid"] == "regel-guid"
    assert (
        'HyDAMO layer "Sturing": filled "code" from "globalid" in 1 row(s): mgmt-guid'
        in caplog.text
    )


def test_load_from_gpkg_fills_empty_and_blank_management_codes_from_globalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Blank counts as empty, as it does for the identity check after the fill.
    gpkg_path = tmp_path / "sturing_22.gpkg"
    _write_layer(
        gpkg_path,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "code": ["CTRL1", None, "", "  "],
                "globalid": [
                    "mgmt-guid-1",
                    "mgmt-guid-2",
                    "mgmt-guid-3",
                    "mgmt-guid-4",
                ],
                "geometry": [Point(5, 0), Point(6, 0), Point(7, 0), Point(8, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.2")

    assert hydamo.management.index.tolist() == [
        "CTRL1",
        "mgmt-guid-2",
        "mgmt-guid-3",
        "mgmt-guid-4",
    ]
    assert (
        'HyDAMO layer "Sturing": filled "code" from "globalid" in 3 row(s): '
        "mgmt-guid-2, mgmt-guid-3, mgmt-guid-4" in caplog.text
    )


def test_load_from_gpkg_rejects_empty_profiellijnid_naming_the_layer_and_rows(
    tmp_path: Path,
) -> None:
    # profiellijnid becomes the profile's code; without the check, a null one drops
    # its points silently and a blank one becomes a cross-section with an empty id.
    gpkg_file = tmp_path / "profile.gpkg"
    _write_layer(
        gpkg_file,
        "ProfielPunt",
        gpd.GeoDataFrame(
            {
                "code": ["PP1", "PP2", "PP3", "PP4"],
                "globalid": ["p1", "p2", "p3", "p4"],
                "profiellijnid": ["line-guid", "line-guid", None, " "],
                "codevolgnummer": [1, 2, 1, 2],
                "geometry": [
                    Point(0, 0, 1.0),
                    Point(1, 0, 0.0),
                    Point(2, 0, 1.0),
                    Point(3, 0, 0.0),
                ],
            },
            crs="EPSG:28992",
        ),
    )

    with pytest.raises(
        ValueError,
        match=r'layer "ProfielPunt".*"profiellijnid" is empty in 2 row\(s\): p3, p4',
    ):
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")


@pytest.mark.parametrize("validation_mode", ["off", "warn", "strict"])
@pytest.mark.parametrize("layer", ["Stuw", "lateraleknoop"])
def test_load_from_gpkg_rejects_empty_codes_naming_the_layer_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_mode: Literal["off", "warn", "strict"],
    layer: str,
) -> None:
    # code is the identity of these tables: DAMO allows it empty, HYDROLIB doesn't.
    if validation_mode != "off":
        _install_fake_validator(monkeypatch, _syntax_validator({}))
    gpkg_file = tmp_path / "empty_codes.gpkg"
    _write_layer(
        gpkg_file,
        layer,
        gpd.GeoDataFrame(
            {
                "code": ["A1", None, " "],
                "globalid": ["g1", "g2", "g3"],
                "afvoercoefficient": [1.0, 1.0, 1.0],
                "geometry": [Point(1, 0), Point(2, 0), Point(3, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    with pytest.raises(
        ValueError, match=rf'layer "{layer}".*"code" is empty in 2 row\(s\): g2, g3'
    ):
        HyDAMO().load_from_gpkg(
            gpkg_file, hydamo_version="2.5", validation_mode=validation_mode
        )


def test_load_from_gpkg_names_at_most_ten_rows_with_an_empty_code(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "empty_codes.gpkg"
    _write_layer(
        gpkg_file,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": [None] * 12,
                "globalid": [f"g{i}" for i in range(12)],
                "afvoercoefficient": [1.0] * 12,
                "geometry": [Point(i, 0) for i in range(12)],
            },
            crs="EPSG:28992",
        ),
    )

    with pytest.raises(
        ValueError,
        match=r'"code" is empty in 12 row\(s\): g0, g1, g2, g3, g4, g5, g6, g7, g8, '
        r"g9 and 2 more\.",
    ):
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")


def test_load_from_gpkg_reports_the_errors_of_all_layers_at_once(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "empty_codes.gpkg"
    for layer in ("Stuw", "lateraleknoop"):
        _write_layer(
            gpkg_file,
            layer,
            gpd.GeoDataFrame(
                {
                    "code": ["A1", None],
                    "globalid": [f"{layer}-1", f"{layer}-2"],
                    "afvoercoefficient": [1.0, 1.0],
                    "geometry": [Point(1, 0), Point(2, 0)],
                },
                crs="EPSG:28992",
            ),
        )

    with pytest.raises(ValueError, match='"code" is empty') as excinfo:
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    lines = str(excinfo.value).splitlines()
    assert any('layer "Stuw"' in line and "Stuw-2" in line for line in lines)
    assert any(
        'layer "lateraleknoop"' in line and "lateraleknoop-2" in line for line in lines
    )


def test_load_from_gpkg_raises_a_missing_column_before_other_layer_errors(
    tmp_path: Path,
) -> None:
    # Stuw loads before lateraleknoop: its empty code doesn't hide the missing column.
    gpkg_file = tmp_path / "errors.gpkg"
    _write_layer(
        gpkg_file,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": [None],
                "globalid": ["g1"],
                "afvoercoefficient": [1.0],
                "geometry": [Point(1, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_file,
        "lateraleknoop",
        gpd.GeoDataFrame(
            {"globalid": ["g2"], "geometry": [Point(2, 0)]},
            crs="EPSG:28992",
        ),
    )

    with pytest.raises(KeyError, match='layer "lateraleknoop".*"code" not found'):
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")


def test_load_from_gpkg_rejects_a_branch_layer_without_code(tmp_path: Path) -> None:
    gpkg_file = tmp_path / "branch.gpkg"
    _write_layer(
        gpkg_file,
        "HydroObject",
        gpd.GeoDataFrame(
            {"globalid": ["hydro-guid"], "geometry": [LineString([(0, 0), (10, 0)])]},
            crs="EPSG:28992",
        ),
    )

    with pytest.raises(KeyError, match='layer "HydroObject".*"code" not found'):
        HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")


def test_load_from_gpkg_suffixes_duplicate_lateral_codes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gpkg_file = tmp_path / "laterals.gpkg"
    _write_layer(
        gpkg_file,
        "lateraleknoop",
        gpd.GeoDataFrame(
            {
                "code": ["L1", "L1"],
                "globalid": ["g1", "g2"],
                "geometry": [Point(1, 0), Point(2, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert hydamo.laterals.index.tolist() == ["L1", "L1_1"]
    assert "Index column 'code' contains duplicates (['L1'])" in caplog.text


def test_load_from_gpkg_loads_a_management_device_without_code(tmp_path: Path) -> None:
    # Devices are joined on globalid, and code is optional on the 2.5 Afsluitmiddel.
    gpkg_file = tmp_path / "device.gpkg"
    _write_layer(
        gpkg_file,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "globalid": ["device-guid"],
                "kunstwerkopeningid": ["opening-guid"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert hydamo.management_device["globalid"].tolist() == ["device-guid"]


def test_hydamo_has_the_data_model_tables_with_their_declared_types() -> None:
    # HyDAMO creates its tables from the data model, and declares their types in the
    # class body for mypy: both lists must name the same tables, with the same types.
    geotypes = {name: table.geotype for name, table in TABLES.items()}
    geotypes.update({name: table.geotype for name, table in OWN_TABLES.items()})
    declared = {
        name: hint
        for name, hint in typing.get_type_hints(HyDAMO).items()
        if hint in (ExtendedDataFrame, ExtendedGeoDataFrame)
    }
    hydamo = HyDAMO()

    assert set(declared) == set(geotypes)
    for name, geotype in geotypes.items():
        expected = ExtendedDataFrame if geotype is None else ExtendedGeoDataFrame
        assert declared[name] is expected, name
        assert type(getattr(hydamo, name)) is expected, name


def test_source_damo_version_is_none_until_loaded() -> None:
    assert HyDAMO().source_damo_version is None


def test_load_from_gpkg_requires_hydamo_version() -> None:
    # Without a version, a 2.5 or 2.6 file would be read as another version and
    # columns would go missing without an error. The call fails before the file
    # is opened, so the path is a placeholder.
    with pytest.raises(TypeError, match="hydamo_version"):
        HyDAMO().load_from_gpkg("irrelevant.gpkg")  # type: ignore[call-arg]


def test_load_from_gpkg_raises_for_unsupported_version() -> None:
    hydamo = HyDAMO()
    with pytest.raises(
        ValueError,
        match=r'Unsupported HyDAMO DAMO version "2\.7"\. '
        r"Supported versions: 2\.2, 2\.3, 2\.4, 2\.5, 2\.6\.",
    ):
        hydamo.load_from_gpkg("irrelevant.gpkg", hydamo_version="2.7")


def test_load_from_gpkg_renames_the_order_column_before_grouping_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A step that renames the column ordering the points must apply to the grouping
    # too; before, grouping looked for the DAMO 2.2 name in the file.
    renamed = dataclasses.replace(
        steps.STEPS[-1], columns={"profile": {"codevolgnummer": "volgnummer"}}
    )
    monkeypatch.setattr(steps, "STEPS", (*steps.STEPS[:-1], renamed))
    gpkg_file = tmp_path / "profile.gpkg"
    _write_layer(
        gpkg_file,
        "ProfielPunt",
        gpd.GeoDataFrame(
            {
                "code": ["PP2", "PP1"],
                "globalid": ["p2", "p1"],
                "profiellijnid": ["line-guid", "line-guid"],
                "volgnummer": [2, 1],
                "geometry": [Point(1, 0, 0.0), Point(0, 0, 1.0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.6")

    line = hydamo.profile.geometry.iloc[0]
    assert list(line.coords) == [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
