"""Validating DAMO GeoPackages with the validatiemodule."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.validation import HyDAMOValidationError
from pydantic import ValidationError
from shapely.geometry import Point

from tests.dhydamo.io.damo_gpkg import (
    DATA_PATH,
    Validate,
    _install_fake_validator,
    _make_minimal_25_gpkg,
    _syntax_validator,
)


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


def test_validate_gpkg_requires_hydamo_version() -> None:
    with pytest.raises(ValidationError, match="hydamo_version"):
        HyDAMO().validate_gpkg("irrelevant.gpkg")  # type: ignore[call-arg]


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
def test_real_validation_of_example_model(
    tmp_path: Path,
) -> None:
    """Run the real validatiemodule on the example model: only the two schema findings remain.

    Skipped unless hydamo-validation is installed (Python 3.12, pip install
    hydrolib[validation]).
    """
    # The validatiemodule crashes on a layer stored without the geometry its
    # schema requires and on NULL geometries (the example's tables without
    # geometry, e.g. Sturing). Give those rows a placeholder point, so every layer
    # is validated.
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
    findings = {(f.layer, f.column, f.code) for f in result.syntax_findings}
    # The example follows DAMO 2.2, apart from two findings that come from the schema:
    # it requires pompid on every Sturing row, also on rows that control a weir, and
    # it only allows the steering variable with a trailing comma ("bovenkant
    # afsluitmiddel,"), which loading rewrites (VALUE_REWRITES).
    assert findings == {("sturing", "pompid", 4), ("sturing", "stuurvariabele", 6)}
