from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo import validation
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.core.relations import RelationsHyDAMO
from hydrolib.dhydamo.io.damo_converters import (
    SUPPORTED_HYDAMO_VERSIONS,
    get_damo_converter,
)
from hydrolib.dhydamo.validation import HyDAMOValidationError
from shapely.geometry import LineString, Point

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
    _write_layer(
        gpkg_path,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": ["ST1"],
                "globalid": ["stuw-guid"],
                "afvoercoefficient": [1.0],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
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
                "hoogteopening": [0.3],
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


def _make_weir_without_device_gpkg(tmp_path: Path, n_openings: int) -> Path:
    """Write a weir with *n_openings* openings, none of which has a management device.

    The only management device belongs to another opening. The openings have no
    ``code``: DAMO does not define one for Kunstwerkopening.
    """
    gpkg_path = tmp_path / "weir_without_device.gpkg"
    _write_layer(
        gpkg_path,
        "Stuw",
        gpd.GeoDataFrame(
            {
                "code": ["ST1"],
                "globalid": ["stuw-guid"],
                "afvoercoefficient": [1.0],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )
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
    hydamo.weirs["branch_id"] = "H1"
    hydamo.weirs["branch_offset"] = 5.0

    hydamo.structures.convert.weirs(
        weirs=hydamo.weirs,
        opening=hydamo.opening,
        management_device=hydamo.management_device,
    )

    assert "Skipping opening opening-guid-0 (weir ST1)" in caplog.text


def test_supported_damo_converter_versions() -> None:
    assert SUPPORTED_HYDAMO_VERSIONS == ("2.2", "2.3", "2.4", "2.5")
    for version in SUPPORTED_HYDAMO_VERSIONS:
        assert get_damo_converter(version).hydamo_version == version


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
    assert "categorieoppwaterlichaam" in hydamo.branches.columns
    assert hydamo.branches.loc["H1", "categorieoppwaterlichaam"] == "primair"

    assert "soortafsluitmiddel" in hydamo.management_device.columns
    assert hydamo.management_device.loc[0, "soortafsluitmiddel"] == "schuif"
    assert "soortregelbaarheid" in hydamo.management_device.columns
    assert (
        hydamo.management_device.loc[0, "soortregelbaarheid"]
        == "Regelbaar - lokaal automatisch - elektronisch"
    )

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
    weir = RelationsHyDAMO.OPENING_WEIR.parents_of(opening, hydamo.weirs)
    device_from_opening = RelationsHyDAMO.OPENING_MANAGEMENT_DEVICE.parents_of(
        opening, hydamo.management_device
    )
    device_from_management = RelationsHyDAMO.MANAGEMENT_DEVICE.parents_of(
        management, hydamo.management_device
    )
    opening_from_device = RelationsHyDAMO.MANAGEMENT_DEVICE_OPENING.parents_of(
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
    tmp_path: Path,
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
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
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


def test_load_from_gpkg_raises_for_unsupported_version() -> None:
    hydamo = HyDAMO()
    with pytest.raises(ValueError, match="Unsupported"):
        hydamo.load_from_gpkg("irrelevant.gpkg", hydamo_version="9.9")


def test_load_from_gpkg_v22_postprocess_copies_soortregelmiddel(tmp_path: Path) -> None:
    gpkg_path = tmp_path / "v22_postprocess.gpkg"
    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": ["H1"],
                "globalid": ["h-guid"],
                "geometry": [LineString([(0, 0), (10, 0)])],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "Regelmiddel",
        gpd.GeoDataFrame(
            {
                "code": ["AF1"],
                "globalid": ["af-guid"],
                "soortregelmiddel": ["schuif"],
                "overlaatonderlaat": ["onderlaat"],
                "geometry": [Point(5, 0)],
            },
            crs="EPSG:28992",
        ),
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_path, hydamo_version="2.2")

    assert "soortafsluitmiddel" in hydamo.management_device.columns
    assert hydamo.management_device.loc[0, "soortafsluitmiddel"] == "schuif"


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
