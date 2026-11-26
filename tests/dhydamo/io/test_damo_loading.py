from __future__ import annotations

import json
import sys
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from unittest import mock

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo import validation
from hydrolib.dhydamo.converters.hydamo2df import RoughnessVariant
from hydrolib.dhydamo.core.data_model import (
    OWN_TABLES,
    TABLES,
    DamoRelations,
)
from hydrolib.dhydamo.core.drtc import CREST_LEVEL, DRTCModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame
from hydrolib.dhydamo.io.damo_versions import SUPPORTED_HYDAMO_VERSIONS, loader
from hydrolib.dhydamo.validation import HyDAMOValidationError
from pydantic import ValidationError
from shapely.geometry import LineString, Point, Polygon

DATA_PATH = Path("hydrolib/sample_data/data").resolve()

# A fake hydamo_validation.validator(**settings) returns this function, called as
# validate(directory=..., raise_error=...). It returns (datamodel, layers_summary,
# result_summary); the layers summary may be None when a test doesn't need it.
Validate = Callable[
    ..., tuple[None, types.SimpleNamespace | None, types.SimpleNamespace]
]


def _install_fake_validator(
    monkeypatch: pytest.MonkeyPatch, validator_factory: Callable[..., Validate]
) -> None:
    """Inject a fake hydamo_validation module; skip the schema scan, which needs the real package."""
    monkeypatch.setitem(
        sys.modules,
        "hydamo_validation",
        types.SimpleNamespace(validator=validator_factory),
    )
    monkeypatch.setattr(
        validation, "_schema_version_for_damo", lambda _module, _version: ("1.4", [])
    )


def _syntax_validator(layers: dict[str, pd.DataFrame]) -> Callable[..., Validate]:
    """Return a fake validator factory whose run reports *layers* as syntax results.

    Each layer frame holds ``syntax_<column>`` strings in the validatiemodule's
    format, e.g. ``"{guid} -> NULL (fouten: 3)"``, or ``""`` for a valid value.
    """
    layers_summary = types.SimpleNamespace(
        geo_types={layer: "Point" for layer in layers}, **layers
    )

    def _factory(**kwargs: object) -> Validate:
        def _run(
            directory: Path, raise_error: bool = False
        ) -> tuple[None, types.SimpleNamespace, types.SimpleNamespace]:
            return (
                None,
                layers_summary,
                types.SimpleNamespace(success=True, status="finished", error=[]),
            )

        return _run

    return _factory


def _write_layer(gpkg_path: Path, layer_name: str, gdf: gpd.GeoDataFrame) -> None:
    mode: Literal["w", "a"] = "w" if not gpkg_path.exists() else "a"
    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode=mode, engine="pyogrio")


def _write_stuw(gpkg_path: Path, afvoercoefficient: float | None = 1.0) -> None:
    """Write weir ST1 (globalid ``stuw-guid``) at (5, 0).

    With *afvoercoefficient* ``None``, the layer has no ``afvoercoefficient`` column.
    """
    columns: dict[str, list[object]] = {"code": ["ST1"], "globalid": ["stuw-guid"]}
    if afvoercoefficient is not None:
        columns["afvoercoefficient"] = [afvoercoefficient]
    _write_layer(
        gpkg_path,
        "Stuw",
        gpd.GeoDataFrame({**columns, "geometry": [Point(5, 0)]}, crs="EPSG:28992"),
    )


def _convert_weirs(hydamo: HyDAMO) -> None:
    """Build the D-Flow FM weirs, with every weir placed on branch H1."""
    hydamo.weirs["branch_id"] = "H1"
    hydamo.weirs["branch_offset"] = 5.0
    hydamo.structures.convert.weirs(
        weirs=hydamo.weirs,
        opening=hydamo.opening,
        management_device=hydamo.management_device,
    )


def _make_minimal_25_gpkg(tmp_path: Path) -> Path:
    gpkg_path = tmp_path / "minimal_25.gpkg"

    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": ["H1"],
                "globalid": ["hydro-guid"],
                "categorieoppervlaktewater": ["primair"],
                "geometry": [LineString([(0, 0), (10, 0)])],
            },
            crs="EPSG:28992",
        ),
    )
    _write_stuw(gpkg_path)
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "code": ["KO1"],
                "globalid": ["opening-guid"],
                "stuwid": ["stuw-guid"],
                "afsluitmiddelid": ["afsluit-guid"],
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
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["afsluit-guid"],
                "kunstwerkopeningid": ["opening-guid"],
                "overlaatonderlaat": ["onderlaat"],
                "typeafsluitmiddel": ["schuif"],
                "typeregelbaarheid": ["Regelbaar - lokaal automatisch - elektronisch"],
                "maximalehoogteonderkant": [0.3],
                "afvoercoefficient": [0.8],
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
                "globalid": ["mgmt-guid"],
                "afsluitmiddelid": ["afsluit-guid"],
                "pompid": [None],
                "typecontroller": ["PID"],
                "typestuurvariabele": ["hoogte opening"],
                "doelvariabele": ["waterstand"],
                "bovengrens": [1.0],
                "ondergrens": [0.0],
                "streefwaarde": [0.5],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    return gpkg_path


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


def _damo_names(version: str) -> dict[str, str]:
    """Return ``{2.2 name: name in *version*}`` of the columns DAMO renamed.

    The management device is Regelmiddel up to DAMO 2.4 and Afsluitmiddel from 2.5;
    its type column is ``soortregelmiddel`` on Regelmiddel and ``typeafsluitmiddel``
    on the 2.5 Afsluitmiddel, and its opening height ``hoogteopening`` and
    ``maximalehoogteonderkant``.

    Two keys are not columns: ``device_layer`` is the management device's layer, and
    ``onderlaat`` is the schema's value of ``overlaatonderlaat``, capitalised from 2.5.
    """
    names = {
        "typeruwheid": "typeruwheid",
        "soortparameter": "soortparameter",
        "vormkoker": "vormkoker",
        "regelmiddelid": "regelmiddelid",
        "stuurvariabele": "stuurvariabele",
        "device_layer": "Regelmiddel",
        "soortregelmiddel": "soortregelmiddel",
        "hoogteopening": "hoogteopening",
        "onderlaat": "onderlaat",
    }
    if version >= "2.4":
        names["typeruwheid"] = "typemethoderuwheid"
    if version >= "2.5":
        names["soortparameter"] = "typeparameter"
        names["vormkoker"] = "vormduikersifonhevel"
        names["regelmiddelid"] = "afsluitmiddelid"
        names["stuurvariabele"] = "typestuurvariabele"
        names["device_layer"] = "Afsluitmiddel"
        names["soortregelmiddel"] = "typeafsluitmiddel"
        names["hoogteopening"] = "maximalehoogteonderkant"
        names["onderlaat"] = "Onderlaat"
    return names


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


def _write_branch(gpkg_path: Path, **columns: list[object]) -> None:
    """Write branch H1 with the extra *columns*."""
    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": ["H1"],
                "globalid": ["hydro-guid"],
                **columns,
                "geometry": [LineString([(0, 0), (10, 0)])],
            },
            crs="EPSG:28992",
        ),
    )


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
    # This 2.5 Afsluitmiddel has no maximalehoogteonderkant, the source of hoogteopening.
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


def _write_weir_with_opening(gpkg_path: Path) -> None:
    """Write branch H1 and weir ST1 with opening ``opening-guid``, crest at 0.4 m."""
    _write_branch(gpkg_path)
    _write_stuw(gpkg_path)
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": ["opening-guid"],
                "stuwid": ["stuw-guid"],
                "laagstedoorstroombreedte": [1.2],
                "laagstedoorstroomhoogte": [0.4],
                "afvoercoefficient": [1.0],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )


def _write_culvert(gpkg_path: Path, names: dict[str, str]) -> None:
    """Write culvert DU1 (globalid ``duiker-guid``, 2 m high) on branch H1."""
    _write_layer(
        gpkg_path,
        "DuikerSifonHevel",
        gpd.GeoDataFrame(
            {
                "code": ["DU1"],
                "globalid": ["duiker-guid"],
                "lengte": [1.0],
                "hoogteopening": [2.0],
                "breedteopening": [1.0],
                "hoogtebinnenonderkantbene": [0.0],
                "hoogtebinnenonderkantbov": [0.1],
                names["vormkoker"]: ["rond"],
                "intreeverlies": [0.5],
                "uittreeverlies": [1.0],
                names["typeruwheid"]: ["StricklerKs"],
                "ruwheid": [75.0],
                "geometry": [LineString([(3, 0), (4, 0)])],
            },
            crs="EPSG:28992",
        ),
    )


def _write_devices(
    gpkg_path: Path, names: dict[str, str], rows: list[dict[str, object]]
) -> None:
    """Write the management devices *rows*, keyed by their DAMO 2.2 names, under *names*."""
    renamed = [
        {names.get(column, column): value for column, value in row.items()}
        for row in rows
    ]
    for row in renamed:
        row["geometry"] = Point(5, 0)
    _write_layer(
        gpkg_path,
        names["device_layer"],
        gpd.GeoDataFrame(renamed, crs="EPSG:28992"),
    )


def _convert_culverts(hydamo: HyDAMO) -> pd.DataFrame:
    """Build the D-Flow FM culverts on branch H1 and return them, by id."""
    hydamo.culverts["branch_id"] = "H1"
    hydamo.culverts["branch_offset"] = 3.5
    hydamo.structures.convert.culverts(
        hydamo.culverts, management_device=hydamo.management_device
    )
    return hydamo.structures.culverts_df.set_index("id")


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_gives_an_orifice_the_gate_height_of_its_device(
    tmp_path: Path, version: str
) -> None:
    names = _damo_names(version)
    gpkg_file = tmp_path / "orifice.gpkg"
    _write_weir_with_opening(gpkg_file)
    _write_devices(
        gpkg_file,
        names,
        [
            {
                "code": "AF1",
                "globalid": "device-guid",
                "kunstwerkopeningid": "opening-guid",
                "overlaatonderlaat": names["onderlaat"],
                "soortregelmiddel": "schuif",
                "hoogteopening": 0.3,
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    _convert_weirs(hydamo)

    orifice = hydamo.structures.orifices_df.set_index("id").loc["ST1"]
    # The gate's lower edge is the crest plus the device's opening height.
    assert orifice["gateloweredgelevel"] == pytest.approx(0.4 + 0.3)


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_gives_a_culvert_the_valve_of_its_device(
    tmp_path: Path, version: str
) -> None:
    names = _damo_names(version)
    gpkg_file = tmp_path / "valve.gpkg"
    _write_branch(gpkg_file)
    _write_culvert(gpkg_file, names)
    _write_devices(
        gpkg_file,
        names,
        [
            {
                "code": "AF1",
                "globalid": "device-guid",
                "duikersifonhevelid": "duiker-guid",
                "overlaatonderlaat": None,
                "soortregelmiddel": "schuif",
                "hoogteopening": 0.3,
                "afvoercoefficient": 0.8,
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    culvert = _convert_culverts(hydamo).loc["DU1"]

    assert culvert["valveonoff"] == 1
    assert culvert["valveopeningheight"] == pytest.approx(0.3)
    # Relative to the culvert's opening height of 2 m.
    assert culvert["relopening"] == pytest.approx([0.15])
    assert culvert["allowedflowdir"] == "both"


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_load_from_gpkg_gives_a_culvert_the_non_return_valve_of_its_device(
    tmp_path: Path, version: str
) -> None:
    names = _damo_names(version)
    gpkg_file = tmp_path / "non_return.gpkg"
    _write_branch(gpkg_file)
    _write_culvert(gpkg_file, names)
    _write_devices(
        gpkg_file,
        names,
        [
            {
                "code": "AF1",
                "globalid": "device-guid",
                "duikersifonhevelid": "duiker-guid",
                "overlaatonderlaat": None,
                "soortregelmiddel": "terugslagklep",
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    culvert = _convert_culverts(hydamo).loc["DU1"]

    assert culvert["allowedflowdir"] == "positive"
    assert culvert["valveonoff"] == 0


def test_load_from_gpkg_leaves_closing_devices_of_locks_coupures_and_tunnels_out_of_the_model(
    tmp_path: Path,
) -> None:
    # From DAMO 2.5, Afsluitmiddel also holds the closing devices of locks, coupures
    # and tunnels. They load as management devices, but nothing joins on them.
    names = _damo_names("2.5")
    gpkg_file = tmp_path / "closing_devices.gpkg"
    _write_weir_with_opening(gpkg_file)
    _write_culvert(gpkg_file, names)
    structure_device = {"overlaatonderlaat": "Nvt", "hoogteopening": 2.0}
    _write_devices(
        gpkg_file,
        names,
        [
            {
                "code": "AF-ST1",
                "globalid": "weir-device-guid",
                "kunstwerkopeningid": "opening-guid",
                "overlaatonderlaat": "Onderlaat",
                "soortregelmiddel": "schuif",
                "hoogteopening": 0.3,
            },
            {
                "code": "AF-DU1",
                "globalid": "culvert-device-guid",
                "duikersifonhevelid": "duiker-guid",
                "overlaatonderlaat": "Nvt",
                "soortregelmiddel": "schuif",
                "hoogteopening": 0.3,
                "afvoercoefficient": 0.8,
            },
            {
                **structure_device,
                "code": "AF-SL",
                "globalid": "lock-device-guid",
                "sluisid": "sluis-guid",
                "soortregelmiddel": "deur",
            },
            {
                **structure_device,
                "code": "AF-CP",
                "globalid": "coupure-device-guid",
                "coupureid": "coupure-guid",
                "soortregelmiddel": "schotbalk sponning",
            },
            {
                **structure_device,
                "code": "AF-TN",
                "globalid": "tunnel-device-guid",
                "tunnelid": "tunnel-guid",
                "soortregelmiddel": "zandzakken",
            },
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")
    _convert_weirs(hydamo)
    culverts = _convert_culverts(hydamo)

    assert len(hydamo.management_device) == 5
    orifices = hydamo.structures.orifices_df.set_index("id")
    assert orifices.index.tolist() == ["ST1"]
    assert orifices.loc["ST1", "gateloweredgelevel"] == pytest.approx(0.4 + 0.3)
    assert hydamo.structures.rweirs_df.empty
    assert culverts.index.tolist() == ["DU1"]
    assert culverts.loc["DU1", "relopening"] == pytest.approx([0.15])


@pytest.mark.parametrize("old_first", [False, True], ids=["new_first", "old_first"])
def test_load_from_gpkg_prefers_the_versioned_column_over_a_stale_old_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, old_first: bool
) -> None:
    # A 2.5 file that still carries the 2.4 Regelmiddel columns next to the 2.5 ones,
    # e.g. after a migration. The 2.5 columns count, whatever the column order.
    gpkg_file = tmp_path / "stale_columns.gpkg"
    _write_weir_with_opening(gpkg_file)
    _write_culvert(gpkg_file, _damo_names("2.5"))
    stale = {"hoogteopening": 0.1, "soortregelmiddel": "terugslagklep"}
    current = {"maximalehoogteonderkant": 0.3, "typeafsluitmiddel": "schuif"}
    device_columns = {**stale, **current} if old_first else {**current, **stale}
    _write_layer(
        gpkg_file,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            [
                {
                    "code": "AF-ST1",
                    "globalid": "weir-device-guid",
                    "kunstwerkopeningid": "opening-guid",
                    "overlaatonderlaat": "Onderlaat",
                    **device_columns,
                    "geometry": Point(5, 0),
                },
                {
                    "code": "AF-DU1",
                    "globalid": "culvert-device-guid",
                    "duikersifonhevelid": "duiker-guid",
                    "overlaatonderlaat": "Nvt",
                    **device_columns,
                    "afvoercoefficient": 0.8,
                    "geometry": Point(5, 0),
                },
            ],
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")
    _convert_weirs(hydamo)
    culverts = _convert_culverts(hydamo)

    assert not hydamo.management_device.columns.duplicated().any()
    orifices = hydamo.structures.orifices_df.set_index("id")
    assert orifices.loc["ST1", "gateloweredgelevel"] == pytest.approx(0.4 + 0.3)
    assert culverts.loc["DU1", "valveonoff"] == 1
    assert culverts.loc["DU1", "relopening"] == pytest.approx([0.15])
    assert culverts.loc["DU1", "allowedflowdir"] == "both"
    assert (
        'HyDAMO layer "Afsluitmiddel" has both "maximalehoogteonderkant" and '
        '"hoogteopening"' in caplog.text
    )


def test_load_from_gpkg_loads_closing_devices_without_overlaatonderlaat(
    tmp_path: Path,
) -> None:
    # Optional on the 2.5 Afsluitmiddel, which also holds the devices of locks.
    gpkg_file = tmp_path / "lock_devices.gpkg"
    _write_layer(
        gpkg_file,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF-SL"],
                "globalid": ["lock-device-guid"],
                "sluisid": ["sluis-guid"],
                "typeafsluitmiddel": ["deur"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert hydamo.management_device["globalid"].tolist() == ["lock-device-guid"]
    assert "overlaatonderlaat" not in hydamo.management_device.columns


@pytest.mark.parametrize(
    ("kind_column", "reason"),
    [
        ({}, 'has no "overlaatonderlaat"'),
        ({"overlaatonderlaat": ["Nvt"]}, 'has "overlaatonderlaat" "Nvt"'),
    ],
    ids=["column_absent", "neither_kind"],
)
def test_weir_conversion_skips_a_weir_whose_device_is_neither_overflow_nor_underflow(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    kind_column: dict[str, list[object]],
    reason: str,
) -> None:
    gpkg_file = tmp_path / "weir_without_kind.gpkg"
    _write_weir_with_opening(gpkg_file)
    _write_layer(
        gpkg_file,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["device-guid"],
                "kunstwerkopeningid": ["opening-guid"],
                "typeafsluitmiddel": ["schuif"],
                **kind_column,
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    _convert_weirs(hydamo)

    assert hydamo.structures.orifices_df.empty
    assert hydamo.structures.rweirs_df.empty
    assert f"Skipping weir ST1: its management device AF1 {reason}" in caplog.text


def test_hydamo_has_no_closing_device_table() -> None:
    # DAMO 2.2 Afsluitmiddel was loaded as closing_device, but nothing read it.
    assert not hasattr(HyDAMO(), "closing_device")


def _write_universal_weir(
    gpkg_path: Path, weir_crest: float | None, opening_crest: float | None
) -> None:
    """Write universal weir ST1 on branch H1: a weir with a profile and one opening.

    The profile's lowest point is at 0.2 m. With *weir_crest* ``None``, the weir has
    no ``laagstedoorstroomhoogte`` column, as from DAMO 2.4; with *opening_crest*
    ``None``, the opening's crest is empty.
    """
    _write_branch(gpkg_path)
    weir: dict[str, list[object]] = {
        "code": ["ST1"],
        "globalid": ["stuw-guid"],
        "afvoercoefficient": [1.0],
    }
    if weir_crest is not None:
        weir["laagstedoorstroomhoogte"] = [weir_crest]
    _write_layer(
        gpkg_path,
        "Stuw",
        gpd.GeoDataFrame({**weir, "geometry": [Point(5, 0)]}, crs="EPSG:28992"),
    )
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": ["opening-guid"],
                "stuwid": ["stuw-guid"],
                "laagstedoorstroombreedte": [1.2],
                "laagstedoorstroomhoogte": [
                    float("nan") if opening_crest is None else opening_crest
                ],
                "afvoercoefficient": [1.0],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "profielgroep",
        gpd.GeoDataFrame(
            {"globalid": ["group-guid"], "stuwid": ["stuw-guid"], "geometry": [None]},
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "profiellijn",
        gpd.GeoDataFrame(
            {
                "globalid": ["line-guid"],
                "profielgroepid": ["group-guid"],
                "geometry": [LineString([(5, -1), (5, 1)])],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "ProfielPunt",
        gpd.GeoDataFrame(
            {
                "code": ["P1", "P2", "P3"],
                "globalid": ["p1-guid", "p2-guid", "p3-guid"],
                "profiellijnid": ["line-guid"] * 3,
                "codevolgnummer": [1, 2, 3],
                "geometry": [Point(5, -1, 1.0), Point(5, 0, 0.2), Point(5, 1, 1.0)],
            },
            crs="EPSG:28992",
        ),
    )


def _universal_weir_crest(hydamo: HyDAMO) -> float:
    """Build the D-Flow FM weirs and return the crest level of universal weir ST1."""
    hydamo.weirs["branch_id"] = "H1"
    hydamo.weirs["branch_offset"] = 5.0
    hydamo.structures.convert.weirs(
        weirs=hydamo.weirs,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        profiles=hydamo.profile,
        opening=hydamo.opening,
        management_device=hydamo.management_device,
    )
    uweirs = hydamo.structures.uweirs_df.set_index("id")
    return float(uweirs["crestlevel"].astype(float)["ST1"])


@pytest.mark.parametrize("version", ["2.4", "2.5"])
def test_universal_weir_takes_the_crest_of_its_opening(
    tmp_path: Path, version: str
) -> None:
    # From DAMO 2.4, the crest is on the opening (Kunstwerkopening), not on the weir.
    gpkg_file = tmp_path / "uweir.gpkg"
    _write_universal_weir(gpkg_file, weir_crest=None, opening_crest=0.7)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert hydamo.weirs.loc["ST1", "laagstedoorstroomhoogte"] == pytest.approx(0.7)
    assert _universal_weir_crest(hydamo) == pytest.approx(0.7)


def test_universal_weir_without_a_crest_takes_the_lowest_profile_point(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "uweir.gpkg"
    _write_universal_weir(gpkg_file, weir_crest=None, opening_crest=None)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert _universal_weir_crest(hydamo) == pytest.approx(0.2)


@pytest.mark.parametrize("version", ["2.2", "2.3"])
def test_universal_weir_keeps_its_own_crest(tmp_path: Path, version: str) -> None:
    # Up to DAMO 2.3 the weir has its own crest, which wins over its opening's.
    gpkg_file = tmp_path / "uweir.gpkg"
    _write_universal_weir(gpkg_file, weir_crest=0.5, opening_crest=0.7)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert _universal_weir_crest(hydamo) == pytest.approx(0.5)


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


def _make_weir_without_device_gpkg(tmp_path: Path, n_openings: int) -> Path:
    """Write a weir with *n_openings* openings, none of which has a management device.

    The only management device belongs to another opening. The openings have no
    ``code``: DAMO does not define one for Kunstwerkopening.
    """
    gpkg_path = tmp_path / "weir_without_device.gpkg"
    _write_stuw(gpkg_path)
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": [f"opening-guid-{i}" for i in range(n_openings)],
                "stuwid": ["stuw-guid"] * n_openings,
                "laagstedoorstroombreedte": [1.2] * n_openings,
                "laagstedoorstroomhoogte": [0.4] * n_openings,
                "afvoercoefficient": [1.0] * n_openings,
                "geometry": [Point(5, 0)] * n_openings,
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["afsluit-guid"],
                "kunstwerkopeningid": ["other-opening-guid"],
                "overlaatonderlaat": ["overlaat"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    return gpkg_path


@pytest.mark.parametrize("n_openings", [1, 2])
def test_weir_opening_without_device_logs_opening_globalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, n_openings: int
) -> None:
    gpkg_file = _make_weir_without_device_gpkg(tmp_path, n_openings)
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    _convert_weirs(hydamo)

    assert "Skipping opening opening-guid-0 (weir ST1)" in caplog.text


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
    assert hydamo_26.management_device.loc[0, "hoogteopening"] == 0.3


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
    assert hydamo.management_device.loc[0, "hoogteopening"] == 0.3
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


def test_drtc_resolves_canonical_damo_25_management_links(tmp_path: Path) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")
    # No manual stuwid injection: the weir link is resolved from the static
    # management_device.kunstwerkopeningid -> opening.globalid -> opening.stuwid
    # -> weirs.globalid chain already present in the gpkg fixture (see
    # DRTCModel._resolve_weir_via_opening). Only the built D-FlowFM structure
    # needs to be faked here, since structure conversion isn't run in this test.
    hydamo.structures.rweirs_df = pd.DataFrame({"id": ["ST1"]})

    fm = FMModel()
    fm.time.refdate = 20160601
    fm.time.tstop = 3600

    drtc = DRTCModel(hydamo, fm, output_path=tmp_path)
    assert drtc._resolve_management_structure(hydamo.management.iloc[0]) == (
        "weir",
        "ST1",
    )


PID_GLOBAL = {"ki": -0.05, "kp": -0.03, "kd": 0.0, "maxspeed": 0.00033}
PID_OWN = {"ki": -0.03, "kp": -0.0, "kd": 0.0, "maxspeed": 0.00033}
INTERVAL_GLOBAL = {"deadband": 0.1, "maxspeed": 0.00033}
INTERVAL_OWN = {"deadband": 0.2, "maxspeed": 0.001}


def _sturing(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    """Return Sturing rows, in the given order, that steer a weir crest.

    Each row gives at least ``code``, ``afsluitmiddelid``, ``typecontroller`` and
    ``streefwaarde``. Its ``globalid`` is derived from the code, and the other columns
    get shared defaults, which a row can override.
    """
    defaults = {
        "pompid": None,
        "typestuurvariabele": "bovenkant afsluitmiddel",
        "doelvariabele": "waterstand",
        "meetlocatieid": "Obs1",
        "bovengrens": 1.0,
        "ondergrens": 0.0,
        "geometry": Point(5, 0),
    }
    records = []
    for row in rows:
        records.append({**defaults, "globalid": f"mgmt-guid-{row['code']}", **row})
    return gpd.GeoDataFrame(records, crs="EPSG:28992")


def _controlled_compound_weir_drtc(
    tmp_path: Path, sturing: gpd.GeoDataFrame | None = None
) -> DRTCModel:
    """Load a compound weir ST1 with two openings, controlled by *sturing*.

    The openings become the structures ``ST1_1`` (closing device ``afsluit-guid-1``)
    and ``ST1_2`` (``afsluit-guid-2``). By default, ``CTRL_PID`` gives ``ST1_1`` a PID
    controller and ``CTRL_INTERVAL`` gives ``ST1_2`` an interval controller.
    """
    if sturing is None:
        sturing = _sturing(
            [
                {
                    "code": "CTRL_PID",
                    "afsluitmiddelid": "afsluit-guid-1",
                    "typecontroller": "PID",
                    "streefwaarde": 0.5,
                },
                {
                    "code": "CTRL_INTERVAL",
                    "afsluitmiddelid": "afsluit-guid-2",
                    "typecontroller": "interval",
                    "streefwaarde": 0.5,
                },
            ]
        )
    gpkg_path = tmp_path / "controlled_compound_weir.gpkg"
    points = [Point(5, 0)] * 2
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
    _write_stuw(gpkg_path)
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": ["opening-guid-1", "opening-guid-2"],
                "stuwid": ["stuw-guid"] * 2,
                "laagstedoorstroombreedte": [1.2] * 2,
                "laagstedoorstroomhoogte": [0.4] * 2,
                "afvoercoefficient": [1.0] * 2,
                "geometry": points,
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1", "AF2"],
                "globalid": ["afsluit-guid-1", "afsluit-guid-2"],
                "kunstwerkopeningid": ["opening-guid-1", "opening-guid-2"],
                "overlaatonderlaat": ["overlaat"] * 2,
                "geometry": points,
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Sturing",
        sturing,
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.5")
    _convert_weirs(hydamo)

    fm = FMModel()
    fm.time.refdate = 20160601
    fm.time.tstop = 3600
    return DRTCModel(hydamo, fm, output_path=tmp_path)


def _drtc_warnings(caplog: pytest.LogCaptureFixture, text: str) -> list[str]:
    """Return the D-RTC warnings that contain *text*."""
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "hydrolib.dhydamo.core.drtc" and record.levelname == "WARNING"
    ]
    return [message for message in messages if text in message]


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
@pytest.mark.parametrize(
    ("in_file", "loaded"),
    [
        # every DAMO schema allows only the value with the trailing comma
        ("bovenkant afsluitmiddel,", "bovenkant afsluitmiddel"),
        ("bovenkant afsluitmiddel", "bovenkant afsluitmiddel"),
        ("hoogte opening", "hoogte opening"),
        # only the exact typo is rewritten: no general stripping
        ("bovenkant afsluitmiddel, ", "bovenkant afsluitmiddel, "),
        (" bovenkant afsluitmiddel,", " bovenkant afsluitmiddel,"),
        (None, None),
    ],
    ids=["typo", "correct", "other", "trailing_space", "leading_space", "empty"],
)
def test_load_from_gpkg_rewrites_the_steering_variable_typo(
    tmp_path: Path, version: str, in_file: str | None, loaded: str | None
) -> None:
    gpkg_file = tmp_path / "sturing.gpkg"
    _write_layer(
        gpkg_file,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "code": ["CTRL1"],
                "globalid": ["mgmt-guid"],
                _damo_names(version)["stuurvariabele"]: [in_file],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    value = hydamo.management.loc["CTRL1", "stuurvariabele"]
    if loaded is None:
        assert pd.isna(value)
    else:
        assert value == loaded


def _crest_controlled_weir_sturing(steering_variable: str) -> gpd.GeoDataFrame:
    """Return Sturing ``CTRL_PID``: a PID controller of ``ST1_1`` on *steering_variable*."""
    return _sturing(
        [
            {
                "code": "CTRL_PID",
                "afsluitmiddelid": "afsluit-guid-1",
                "typecontroller": "PID",
                "streefwaarde": 0.5,
                "typestuurvariabele": steering_variable,
            }
        ]
    )


def test_drtc_builds_a_crest_controller_from_the_schema_steering_variable(
    tmp_path: Path,
) -> None:
    sturing = _crest_controlled_weir_sturing("bovenkant afsluitmiddel,")
    drtc = _controlled_compound_weir_drtc(tmp_path, sturing)

    drtc.from_hydamo(
        pid_settings={"global": PID_GLOBAL},
        interval_settings={"global": INTERVAL_GLOBAL},
    )

    assert drtc.pid_controllers["ST1_1"]["steering_variable"] == CREST_LEVEL


def test_drtc_still_rejects_other_steering_variables(tmp_path: Path) -> None:
    sturing = _crest_controlled_weir_sturing("bovenkant afsluitmiddel, ")
    drtc = _controlled_compound_weir_drtc(tmp_path, sturing)

    with pytest.raises(ValueError, match="Invalid value for steering variable"):
        drtc.from_hydamo(
            pid_settings={"global": PID_GLOBAL},
            interval_settings={"global": INTERVAL_GLOBAL},
        )


def test_drtc_applies_settings_keyed_by_compound_weir_opening_id(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    drtc = _controlled_compound_weir_drtc(tmp_path)

    drtc.from_hydamo(
        pid_settings={"global": PID_GLOBAL, "ST1_1": PID_OWN},
        interval_settings={"global": INTERVAL_GLOBAL, "ST1_2": INTERVAL_OWN},
    )

    assert drtc.pid_controllers["ST1_1"]["kp"] == PID_OWN["kp"]
    assert drtc.pid_controllers["ST1_1"]["ki"] == PID_OWN["ki"]
    assert drtc.interval_controllers["ST1_2"]["deadband"] == INTERVAL_OWN["deadband"]
    assert drtc.interval_controllers["ST1_2"]["max_speed"] == INTERVAL_OWN["maxspeed"]
    assert _drtc_warnings(caplog, "_settings") == []


def test_drtc_applies_global_settings_to_controllers_without_own_key(
    tmp_path: Path,
) -> None:
    drtc = _controlled_compound_weir_drtc(tmp_path)

    drtc.from_hydamo(
        pid_settings={"global": PID_GLOBAL},
        interval_settings={"global": INTERVAL_GLOBAL},
    )

    assert drtc.pid_controllers["ST1_1"]["kp"] == PID_GLOBAL["kp"]
    assert drtc.interval_controllers["ST1_2"]["deadband"] == INTERVAL_GLOBAL["deadband"]


def test_drtc_warns_once_per_settings_dict_about_keys_matching_no_structure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # CTRL_PID is the Sturing code, not a structure id. ST1_2 is a structure, but
    # it has an interval controller, so it matches no PID-controlled structure.
    drtc = _controlled_compound_weir_drtc(tmp_path)

    drtc.from_hydamo(
        pid_settings={"global": PID_GLOBAL, "CTRL_PID": PID_OWN, "ST1_2": PID_OWN},
        interval_settings={"global": INTERVAL_GLOBAL, "S_unknown": INTERVAL_OWN},
    )

    (pid_warning,) = _drtc_warnings(caplog, "pid_settings")
    assert "'CTRL_PID'" in pid_warning
    assert "'ST1_2'" in pid_warning
    (interval_warning,) = _drtc_warnings(caplog, "interval_settings")
    assert "'S_unknown'" in interval_warning
    assert drtc.pid_controllers["ST1_1"]["kp"] == PID_GLOBAL["kp"]


def test_drtc_warns_about_unknown_keys_of_mixed_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    drtc = _controlled_compound_weir_drtc(tmp_path)

    drtc.from_hydamo(
        pid_settings={"global": PID_GLOBAL, 1: PID_OWN, "S_typo": PID_OWN},
        interval_settings={"global": INTERVAL_GLOBAL},
    )

    (pid_warning,) = _drtc_warnings(caplog, "pid_settings")
    assert "1" in pid_warning
    assert "'S_typo'" in pid_warning


def test_drtc_names_structure_without_own_settings_or_global(tmp_path: Path) -> None:
    # Without "global", every controlled structure needs its own key, so a
    # misspelt key must show up as the structure id that has no settings.
    drtc = _controlled_compound_weir_drtc(tmp_path)

    with pytest.raises(
        ValueError, match="pid_settings has no key 'ST1_1' and no 'global' fallback"
    ):
        drtc.from_hydamo(
            pid_settings={"ST1_l": PID_OWN},
            interval_settings={"ST1_2": INTERVAL_OWN},
        )


@pytest.mark.parametrize("second_type", ["PID", "interval"])
def test_drtc_uses_first_sturing_row_for_a_structure_controlled_twice(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, second_type: str
) -> None:
    # Z_FIRST sorts after A_SECOND, so a sort by code or globalid would fail here.
    sturing = _sturing(
        [
            {
                "code": "Z_FIRST",
                "afsluitmiddelid": "afsluit-guid-1",
                "typecontroller": "PID",
                "streefwaarde": 0.5,
            },
            {
                "code": "A_SECOND",
                "afsluitmiddelid": "afsluit-guid-1",
                "typecontroller": second_type,
                "streefwaarde": 0.7,
            },
        ]
    )
    drtc = _controlled_compound_weir_drtc(tmp_path, sturing)

    drtc.from_hydamo(
        pid_settings={"global": PID_GLOBAL},
        interval_settings={"global": INTERVAL_GLOBAL},
    )

    assert drtc.pid_controllers["ST1_1"]["setpoint"] == 0.5
    assert "ST1_1" not in drtc.interval_controllers
    (warning,) = _drtc_warnings(caplog, "controlled by")
    assert "ST1_1" in warning
    assert "Z_FIRST" in warning
    assert "A_SECOND" in warning


@pytest.mark.parametrize(
    "types", [("nvt", "PID"), ("PID", "nvt")], ids=["nvt_first", "nvt_last"]
)
def test_drtc_ignores_sturing_rows_without_supported_controller_for_conflicts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, types: tuple[str, str]
) -> None:
    # An "nvt" row creates no controller, so it doesn't compete with the PID row.
    sturing = _sturing(
        [
            {
                "code": "Z_FIRST",
                "afsluitmiddelid": "afsluit-guid-1",
                "typecontroller": types[0],
                "streefwaarde": 0.5,
            },
            {
                "code": "A_SECOND",
                "afsluitmiddelid": "afsluit-guid-1",
                "typecontroller": types[1],
                "streefwaarde": 0.5,
            },
        ]
    )
    drtc = _controlled_compound_weir_drtc(tmp_path, sturing)

    drtc.from_hydamo(pid_settings={"global": PID_GLOBAL})

    assert list(drtc.pid_controllers) == ["ST1_1"]
    assert _drtc_warnings(caplog, "controlled by") == []
    assert len(_drtc_warnings(caplog, "not a valid controller type")) == 1


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


def _write_weirs_with_openings(gpkg_path: Path, weirs: list[tuple[str, int]]) -> None:
    """Write weirs, given as ``(code, number of openings)``, on branch H1.

    Each opening gets an overflow closing device, so each becomes a structure.
    """
    weir_rows = []
    opening_rows = []
    device_rows = []
    for weir_number, (code, n_openings) in enumerate(weirs):
        weir_guid = f"stuw-guid-{weir_number}"
        weir_rows.append(
            {
                "code": code,
                "globalid": weir_guid,
                "afvoercoefficient": 1.0,
                "geometry": Point(5, 0),
            }
        )
        for opening_number in range(n_openings):
            opening_guid = f"opening-guid-{weir_number}-{opening_number}"
            opening_rows.append(
                {
                    "globalid": opening_guid,
                    "stuwid": weir_guid,
                    "laagstedoorstroombreedte": 1.2,
                    "laagstedoorstroomhoogte": 0.4,
                    "afvoercoefficient": 1.0,
                    "geometry": Point(5, 0),
                }
            )
            device_rows.append(
                {
                    "code": f"AF-{weir_number}-{opening_number}",
                    "globalid": f"afsluit-guid-{weir_number}-{opening_number}",
                    "kunstwerkopeningid": opening_guid,
                    "overlaatonderlaat": "overlaat",
                    "geometry": Point(5, 0),
                }
            )
    branch = {
        "code": "H1",
        "globalid": "hydro-guid",
        "geometry": LineString([(0, 0), (10, 0)]),
    }
    _write_layer(gpkg_path, "HydroObject", gpd.GeoDataFrame([branch], crs="EPSG:28992"))
    _write_layer(gpkg_path, "Stuw", gpd.GeoDataFrame(weir_rows, crs="EPSG:28992"))
    _write_layer(
        gpkg_path, "Kunstwerkopening", gpd.GeoDataFrame(opening_rows, crs="EPSG:28992")
    )
    _write_layer(
        gpkg_path, "Afsluitmiddel", gpd.GeoDataFrame(device_rows, crs="EPSG:28992")
    )


@pytest.mark.parametrize(
    ("weirs", "message"),
    [
        # the second ST2 becomes ST2_1, the id of the first one's first opening
        (
            [("ST2", 2), ("ST2", 1)],
            "Structure id ST2_1 of weir ST2_1 is also the id of opening 1 of weir ST2",
        ),
        (
            [("ST2", 2), ("ST2_1", 1)],
            "Structure id ST2_1 of weir ST2_1 is also the id of opening 1 of weir ST2",
        ),
        (
            [("ST2_1", 1), ("ST2", 2)],
            "Structure id ST2_1 of opening 1 of weir ST2 is also the id of weir ST2_1",
        ),
    ],
    ids=["duplicate_code", "compound_first", "single_first"],
)
def test_weir_conversion_raises_on_structure_id_collision(
    tmp_path: Path, weirs: list[tuple[str, int]], message: str
) -> None:
    gpkg_path = tmp_path / "colliding_weirs.gpkg"
    _write_weirs_with_openings(gpkg_path, weirs)
    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.5")

    with pytest.raises(ValueError, match=message):
        _convert_weirs(hydamo)


def _load_weir_far_from_branches(tmp_path: Path) -> HyDAMO:
    """Load weir ST1 at (5, 0) with its opening and device, and a branch 100 m away."""
    gpkg_path = tmp_path / "weir_far_from_branches.gpkg"
    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": ["H1"],
                "globalid": ["hydro-guid"],
                "geometry": [LineString([(0, 100), (10, 100)])],
            },
            crs="EPSG:28992",
        ),
    )
    _write_stuw(gpkg_path)
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": ["opening-guid"],
                "stuwid": ["stuw-guid"],
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
        "Afsluitmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["afsluit-guid"],
                "kunstwerkopeningid": ["opening-guid"],
                "overlaatonderlaat": ["overlaat"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
    return HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.5")


def test_snap_to_branch_and_drop_drops_a_weir_with_its_opening_and_device(
    tmp_path: Path,
) -> None:
    hydamo = _load_weir_far_from_branches(tmp_path)

    hydamo.snap_to_branch_and_drop(
        hydamo.weirs, hydamo.branches, snap_method="overal", source_table="weirs"
    )

    assert hydamo.weirs.empty
    assert hydamo.opening.empty
    assert hydamo.management_device.empty


def test_snap_to_branch_and_drop_rejects_a_misspelt_source_table_before_dropping(
    tmp_path: Path,
) -> None:
    hydamo = _load_weir_far_from_branches(tmp_path)

    with pytest.raises(ValueError, match="'WEIRS'"):
        hydamo.snap_to_branch_and_drop(
            hydamo.weirs, hydamo.branches, snap_method="overal", source_table="WEIRS"
        )

    assert hydamo.weirs.index.tolist() == ["ST1"]
    assert hydamo.opening["globalid"].tolist() == ["opening-guid"]
    assert hydamo.management_device["globalid"].tolist() == ["afsluit-guid"]


def test_weir_conversion_gives_each_opening_a_unique_structure_id(
    tmp_path: Path,
) -> None:
    gpkg_path = tmp_path / "weirs.gpkg"
    _write_weirs_with_openings(gpkg_path, [("ST2", 2), ("ST3", 1)])
    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.5")

    _convert_weirs(hydamo)

    assert hydamo.structures.rweirs_df["id"].tolist() == ["ST2_1", "ST2_2", "ST3"]


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


def test_drtc_error_names_management_code_filled_from_globalid(tmp_path: Path) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")
    hydamo.structures.rweirs_df = pd.DataFrame({"id": ["ST1"]})

    fm = FMModel()
    fm.time.refdate = 20160601
    fm.time.tstop = 3600
    drtc = DRTCModel(hydamo, fm, output_path=tmp_path)

    with pytest.raises(ValueError, match="mgmt-guid contains a PID controller"):
        drtc.from_hydamo(pid_settings=None)


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


def test_load_from_gpkg_fills_empty_management_code_from_globalid(
    tmp_path: Path,
) -> None:
    gpkg_path = tmp_path / "sturing_22.gpkg"
    _write_layer(
        gpkg_path,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "code": ["CTRL1", None],
                "globalid": ["mgmt-guid-1", "mgmt-guid-2"],
                "geometry": [Point(5, 0), Point(6, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.2")

    assert hydamo.management.index.tolist() == ["CTRL1", "mgmt-guid-2"]


def test_load_from_gpkg_fills_blank_management_code_from_globalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Blank counts as empty, as it does for the identity check after the fill.
    gpkg_path = tmp_path / "sturing_22.gpkg"
    _write_layer(
        gpkg_path,
        "Sturing",
        gpd.GeoDataFrame(
            {
                "code": ["CTRL1", "", "  "],
                "globalid": ["mgmt-guid-1", "mgmt-guid-2", "mgmt-guid-3"],
                "geometry": [Point(5, 0), Point(6, 0), Point(7, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.2")

    assert hydamo.management.index.tolist() == ["CTRL1", "mgmt-guid-2", "mgmt-guid-3"]
    assert (
        'HyDAMO layer "Sturing": filled "code" from "globalid" in 2 row(s): '
        "mgmt-guid-2, mgmt-guid-3" in caplog.text
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


def test_load_from_gpkg_with_validation_warn_generates_rules_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    captured: dict[str, object] = {}

    def _validator_factory(**kwargs: object) -> Validate:
        captured["factory_kwargs"] = kwargs

        def _run(
            directory: Path, raise_error: bool = False
        ) -> tuple[None, None, types.SimpleNamespace]:
            rules = json.loads((Path(directory) / "validationrules.json").read_text())
            captured["rules"] = rules
            captured["dataset_files"] = sorted(
                p.name for p in (Path(directory) / "datasets").iterdir()
            )
            return (
                None,
                None,
                types.SimpleNamespace(success=True, status="finished", error=[]),
            )

        return _run

    _install_fake_validator(monkeypatch, _validator_factory)

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file,
        hydamo_version="2.5",
        validation_mode="warn",
    )

    assert hydamo.validation_result is not None
    assert hydamo.validation_result.success is True
    assert captured["factory_kwargs"] == {
        "output_types": [],
        "coverages": {},
        "log_level": "INFO",
    }
    assert captured["dataset_files"] == [gpkg_file.name]
    rules = captured["rules"]
    assert isinstance(rules, dict)
    assert rules["hydamo_version"] == "2.5"
    assert rules["schema"]  # version determined at runtime from installed package
    assert any(obj["object"] == "hydroobject" for obj in rules["objects"])


def test_validation_reports_syntax_findings_per_layer_and_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    stuw = pd.DataFrame(
        {
            "code": ["ST1", "ST2", "ST3", "ST4"],
            "syntax_globalid": [
                "{dup} -> NULL (fouten: 3)",
                "{dup} -> NULL (fouten: 3)",
                "None -> NULL (fouten: 4)",
                "",
            ],
            "syntax_oordeel": [False, False, False, True],
        }
    )
    _install_fake_validator(monkeypatch, _syntax_validator({"stuw": stuw}))

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file, hydamo_version="2.5", validation_mode="warn"
    )

    assert hydamo.validation_result is not None
    findings = {
        (f.layer, f.column, f.code): f.row_count
        for f in hydamo.validation_result.syntax_findings
    }
    assert findings == {("stuw", "globalid", 3): 2, ("stuw", "globalid", 4): 1}


def test_validation_findings_read_the_validatiemodule_result_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The strings as hydamo_validation 1.7.0 writes them: the value shown is the
    # one after emptying, so an unconvertible "bogus" and an empty value both
    # read "nan -> NULL (fouten: 1)".
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    stuw = pd.DataFrame(
        {
            "code": ["ST1", "ST2", "ST3"],
            # converted (2) is no rejection; unconvertible (1) always is
            "syntax_kruinbreedte": [
                "(fouten: 2)",
                "nan -> NULL (fouten: 1)",
                "NaT -> NULL (fouten: 1)",
            ],
            "syntax_geometry": ["", "Point -> NULL (7)", ""],
            # a rejected value that itself looks like an error marker
            "syntax_naam": ["foo (fouten: 3) -> NULL (fouten: 6)", "", ""],
            # a whole missing required column: no value, only the codes
            "syntax_afvoercoefficient": ["(fouten: 4)", "(fouten: 4)", "(fouten: 4)"],
        }
    )
    _install_fake_validator(monkeypatch, _syntax_validator({"stuw": stuw}))

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file, hydamo_version="2.5", validation_mode="warn"
    )

    assert hydamo.validation_result is not None
    findings = {
        (f.column, f.code): (f.row_count, f.examples)
        for f in hydamo.validation_result.syntax_findings
    }
    assert findings == {
        ("kruinbreedte", 1): (2, ["ST2", "ST3"]),
        ("geometry", 7): (1, ["Point"]),
        ("naam", 6): (1, ["foo (fouten: 3)"]),
        ("afvoercoefficient", 4): (3, ["ST1", "ST2", "ST3"]),
    }


@pytest.mark.parametrize(
    ("layer", "column"),
    [
        ("stuw", "globalid"),
        # DAMO 2.5 name of the culvert shape HYDROLIB reads as vormkoker
        ("duikersifonhevel", "vormduikersifonhevel"),
        # a spatial table whose required columns omit its geometry
        ("profiellijn", "geometry"),
    ],
)
def test_validation_strict_raises_on_findings_in_columns_hydrolib_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str, column: str
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    results = pd.DataFrame({f"syntax_{column}": ["None -> NULL (fouten: 4)"]})
    _install_fake_validator(monkeypatch, _syntax_validator({layer: results}))

    with pytest.raises(HyDAMOValidationError, match=f"{layer}.{column}"):
        HyDAMO().load_from_gpkg(
            gpkg_file, hydamo_version="2.5", validation_mode="strict"
        )


def test_validation_strict_loads_despite_findings_in_columns_hydrolib_ignores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    stuw = pd.DataFrame({"syntax_objectid": ["nan -> NULL (fouten: 3,4)"]})
    _install_fake_validator(monkeypatch, _syntax_validator({"stuw": stuw}))

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file, hydamo_version="2.5", validation_mode="strict"
    )

    assert hydamo.validation_result is not None
    assert [
        (f.column, f.code, f.read_by_hydrolib)
        for f in hydamo.validation_result.syntax_findings
    ] == [("objectid", 3, False), ("objectid", 4, False)]
    assert not hydamo.weirs.empty


def test_validation_warn_logs_findings_in_columns_hydrolib_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    stuw = pd.DataFrame(
        {
            "code": ["ST1", "ST2", "ST3"],
            "syntax_globalid": [
                "{dup} -> NULL (fouten: 3)",
                "{dup} -> NULL (fouten: 3)",
                "None -> NULL (fouten: 4)",
            ],
            "syntax_objectid": ["nan -> NULL (fouten: 3,4)", "", ""],
        }
    )
    pomp = pd.DataFrame(
        {"code": ["P1"], "syntax_pomprichting": ["x -> NULL (fouten: 6)"]}
    )
    _install_fake_validator(
        monkeypatch, _syntax_validator({"stuw": stuw, "pomp": pomp})
    )

    with caplog.at_level("INFO", logger="hydrolib.dhydamo.validation"):
        hydamo = HyDAMO().load_from_gpkg(
            gpkg_file, hydamo_version="2.5", validation_mode="warn"
        )

    assert hydamo.validation_result is not None
    assert [
        (f.column, f.code, f.meaning, f.examples)
        for f in hydamo.validation_result.syntax_findings
        if f.layer == "stuw" and f.column == "globalid"
    ] == [
        ("globalid", 3, "not unique", ["{dup}"]),
        # a missing value has no value to show: name the row by its code
        ("globalid", 4, "missing", ["ST3"]),
    ]
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and r.name == "hydrolib.dhydamo.validation"
    ]
    assert len(warnings) == 1
    assert "stuw" in warnings[0]
    assert "globalid: 2 rows not unique (e.g. {dup})" in warnings[0]
    assert "globalid: 1 rows missing (e.g. ST3)" in warnings[0]
    assert "objectid" not in warnings[0]
    infos = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    # stuw.objectid (codes 3 and 4) and pomp.pomprichting
    assert any("3 other findings" in message for message in infos)


def test_validation_strict_loads_a_clean_geopackage_without_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    stuw = pd.DataFrame(
        {"code": ["ST1"], "syntax_globalid": [""], "syntax_kruinbreedte": [""]}
    )
    _install_fake_validator(monkeypatch, _syntax_validator({"stuw": stuw}))

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file, hydamo_version="2.5", validation_mode="strict"
    )

    assert hydamo.validation_result is not None
    assert hydamo.validation_result.syntax_findings == []


def test_load_from_gpkg_with_validation_strict_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    def _validator_factory(**kwargs: object) -> Validate:
        def _run(
            directory: Path, raise_error: bool = False
        ) -> tuple[None, None, types.SimpleNamespace]:
            return (
                None,
                None,
                types.SimpleNamespace(
                    success=False, status="failed", error=["bad data"]
                ),
            )

        return _run

    _install_fake_validator(monkeypatch, _validator_factory)

    hydamo = HyDAMO()
    with pytest.raises(HyDAMOValidationError):
        hydamo.load_from_gpkg(
            gpkg_file,
            hydamo_version="2.5",
            validation_mode="strict",
        )


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


def test_validate_gpkg_requires_hydamo_version() -> None:
    with pytest.raises(ValidationError, match="hydamo_version"):
        HyDAMO().validate_gpkg("irrelevant.gpkg")  # type: ignore[call-arg]


def test_load_from_gpkg_raises_for_unsupported_version() -> None:
    hydamo = HyDAMO()
    with pytest.raises(
        ValueError,
        match=r'Unsupported HyDAMO DAMO version "2\.7"\. '
        r"Supported versions: 2\.2, 2\.3, 2\.4, 2\.5, 2\.6\.",
    ):
        hydamo.load_from_gpkg("irrelevant.gpkg", hydamo_version="2.7")


def test_load_from_gpkg_validation_warn_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    def _validator_factory(**kwargs: object) -> Validate:
        def _run(
            directory: Path, raise_error: bool = False
        ) -> tuple[None, None, types.SimpleNamespace]:
            return (
                None,
                None,
                types.SimpleNamespace(
                    success=False, status="failed", error=["bad data"]
                ),
            )

        return _run

    _install_fake_validator(monkeypatch, _validator_factory)

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file, hydamo_version="2.5", validation_mode="warn"
    )

    assert hydamo.validation_result is not None
    assert hydamo.validation_result.success is False
    assert hydamo.validation_result.errors == ["bad data"]
    assert not hydamo.branches.empty


def test_validate_gpkg_returns_result_and_stores_on_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    def _validator_factory(**kwargs: object) -> Validate:
        def _run(
            directory: Path, raise_error: bool = False
        ) -> tuple[None, None, types.SimpleNamespace]:
            return (
                None,
                None,
                types.SimpleNamespace(success=True, status="finished", error=[]),
            )

        return _run

    _install_fake_validator(monkeypatch, _validator_factory)

    hydamo = HyDAMO()
    result = hydamo.validate_gpkg(gpkg_file, hydamo_version="2.5")

    assert result is not None
    assert result.success is True
    assert hydamo.validation_result is result


@pytest.mark.requires_hydamo_validation
@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="hydamo-validation is only available for Python 3.12",
)
def test_real_validation_reports_example_model_globalid_findings(
    tmp_path: Path,
) -> None:
    """Run the real validatiemodule on the example model and read its findings.

    Skipped unless hydamo-validation is installed (Python 3.12, pip install
    hydrolib[validation]).
    """
    # The validatiemodule crashes on a layer stored without the geometry its
    # schema requires (Regelmiddel in the example) and on NULL geometries. Give
    # those rows a placeholder point, so every layer is validated.
    original = DATA_PATH / "Example_model.gpkg"
    clean_gpkg = tmp_path / "clean_for_validation.gpkg"
    placeholder = Point(0, 0)
    mode: Literal["w", "a"] = "w"
    for layer_name in gpd.list_layers(original).name.tolist():
        layer = gpd.read_file(original, layer=layer_name, engine="pyogrio")
        if isinstance(layer, gpd.GeoDataFrame):
            geometries = layer.geometry.to_numpy()
            geometries[pd.isna(geometries)] = placeholder
            layer.geometry = geometries
        else:
            layer = gpd.GeoDataFrame(
                layer, geometry=[placeholder] * len(layer), crs="EPSG:28992"
            )
        layer.to_file(
            clean_gpkg, layer=layer_name, driver="GPKG", mode=mode, engine="pyogrio"
        )
        mode = "a"

    hydamo = HyDAMO()
    result = hydamo.validate_gpkg(clean_gpkg, hydamo_version="2.2")

    assert result.success is True
    findings = {
        (f.layer, f.column, f.code): f
        for f in result.syntax_findings
        if f.column == "globalid"
    }
    # split branches share their globalid on purpose
    assert findings[("hydroobject", "globalid", 3)].read_by_hydrolib
    # the example's control data (ticket 07)
    assert findings[("sturing", "globalid", 3)].row_count > 0
    assert findings[("sturing", "globalid", 4)].row_count > 0
    assert findings[("regelmiddel", "globalid", 3)].examples == [
        "{bb24467d-3de6-4a5f-a3a4-b89e334c6824}"
    ]


def _write_param_profiles(
    gpkg_path: Path,
    branches: dict[str, str],
    links: list[tuple[str, str]],
    widths: dict[str, float],
) -> None:
    """Write branches, their links to parameterised profiles, and the profiles' values.

    *branches* maps a code to a ``globalid``; each branch is a 100 m line. *links* are
    ``(hydroobjectid, normgeparamprofielid)`` rows of ``hydroobject_normgp``. Each
    profile in *widths* gets a rectangular profile of that bottom width.
    """
    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": list(branches),
                "globalid": list(branches.values()),
                "geometry": [
                    LineString([(200 * i, 0), (200 * i + 100, 0)])
                    for i in range(len(branches))
                ],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "hydroobject_normgp",
        gpd.GeoDataFrame(
            {
                "hydroobjectid": [branch for branch, _ in links],
                "normgeparamprofielid": [profile for _, profile in links],
                # undeclared, so dropped by the tests
                "nen3610id": [f"NORMGP.{i}" for i in range(len(links))],
                "geometry": [Point(0, 0)] * len(links),
            },
            crs="EPSG:28992",
        ),
    )
    rows = []
    for profile, width in widths.items():
        parameters = {
            "bodembreedte": width,
            "bodemhoogte bovenstrooms": 1.0,
            "bodemhoogte benedenstrooms": 0.5,
            # no slope: a rectangle
            "taludhelling linkerzijde": float("nan"),
        }
        for parameter, value in parameters.items():
            rows.append(
                {
                    "normgeparamprofielid": profile,
                    "soortparameter": parameter,
                    "waarde": value,
                    "ruwheidlaag": 30.0,
                    "ruwheidhoog": 20.0,
                    "typeruwheid": "Manning",
                    "geometry": Point(0, 0),
                }
            )
    _write_layer(
        gpkg_path,
        "NormgeparamProfielWaarde",
        gpd.GeoDataFrame(rows, crs="EPSG:28992"),
    )


def _param_profile_widths(gpkg_path: Path) -> dict[str, float]:
    """Load *gpkg_path* without undeclared columns and build its cross sections.

    Returns the bottom width of the cross sections of each branch.
    """
    hydamo = HyDAMO()
    # Without undeclared columns, so building fails on a column read undeclared.
    with mock.patch.object(loader, "_KEEP_UNDECLARED_COLUMNS", False):
        hydamo.load_from_gpkg(gpkg_path, hydamo_version="2.2")
    hydamo.crosssections.convert.profiles(
        branches=hydamo.branches,
        roughness_variant=RoughnessVariant.HIGH,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
    )
    widths = {}
    for location in hydamo.crosssections.crosssection_loc.values():
        definition = hydamo.crosssections.crosssection_def[location["definitionId"]]
        widths[location["branchid"]] = definition["width"]
    return widths


def test_load_from_gpkg_reads_param_profiles_from_hydroobject_normgp(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file, {"W1": "w1-guid"}, [("w1-guid", "np-guid")], {"np-guid": 4.0}
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    assert hydamo.param_profile[
        ["hydroobjectid", "normgeparamprofielid"]
    ].to_numpy().tolist() == [["w1-guid", "np-guid"]]
    assert _param_profile_widths(gpkg_file) == {"W1": 4.0}


def test_a_param_profile_linked_to_two_branches_applies_to_both(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid", "W2": "w2-guid"},
        [("w1-guid", "np-guid"), ("w2-guid", "np-guid")],
        {"np-guid": 4.0},
    )

    assert _param_profile_widths(gpkg_file) == {"W1": 4.0, "W2": 4.0}


def test_a_branch_linked_to_two_param_profiles_gets_the_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid"},
        [("w1-guid", "np-1"), ("w1-guid", "np-2"), ("w1-guid", "np-3")],
        {"np-1": 4.0, "np-2": 6.0, "np-3": 8.0},
    )

    assert _param_profile_widths(gpkg_file) == {"W1": 4.0}
    assert (
        "Branch W1 is linked to 3 parameterised profiles; using np-1 and ignoring "
        "np-2, np-3." in caplog.text
    )


def test_a_param_profile_link_matching_two_branches_goes_to_the_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Split parts sharing a globalid: invalid DAMO, as in the example model.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1_0": "w1-guid", "W1_1": "w1-guid"},
        [("w1-guid", "np-guid")],
        {"np-guid": 4.0},
    )

    assert _param_profile_widths(gpkg_file) == {"W1_0": 4.0}
    assert (
        "Parameterised profile np-guid: hydroobjectid w1-guid matches branches "
        "W1_0, W1_1; only W1_0 gets the profile." in caplog.text
    )


def test_a_param_profile_link_to_a_branch_with_a_cross_section_is_skipped(
    tmp_path: Path,
) -> None:
    # Measured profiles come first; before, this link raised a KeyError.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid", "W2": "w2-guid"},
        [("w1-guid", "np-1"), ("w2-guid", "np-2")],
        {"np-1": 4.0, "np-2": 6.0},
    )
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")
    measured = hydamo.crosssections.add_rectangle_definition(
        height=2.0,
        width=1.0,
        closed=False,
        roughnesstype="Manning",
        roughnessvalue=0.03,
        name="measured",
    )
    hydamo.crosssections.add_crosssection_location(
        branchid="W1", chainage=50.0, definition=measured
    )

    hydamo.crosssections.convert.profiles(
        branches=hydamo.branches,
        roughness_variant=RoughnessVariant.HIGH,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
    )

    widths: dict[str, set[float]] = {}
    for location in hydamo.crosssections.crosssection_loc.values():
        definition = hydamo.crosssections.crosssection_def[location["definitionId"]]
        widths.setdefault(location["branchid"], set()).add(definition["width"])
    assert widths == {"W1": {1.0}, "W2": {6.0}}
