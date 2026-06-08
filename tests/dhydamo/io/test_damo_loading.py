from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.core.dflowfm.mdu.models import FMModel
from shapely.geometry import LineString, Point

from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.io.damo_converters import (
    SUPPORTED_HYDAMO_VERSIONS,
    get_damo_converter,
)
from hydrolib.dhydamo.validation import HyDAMOValidationError


DATA_PATH = Path("hydrolib/tests/data").resolve()


def _write_layer(gpkg_path: Path, layer_name: str, gdf: gpd.GeoDataFrame) -> None:
    mode = "w" if not gpkg_path.exists() else "a"
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
                "code": ["CTRL1"],
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


def test_supported_damo_converter_versions():
    assert SUPPORTED_HYDAMO_VERSIONS == ("2.2", "2.3", "2.4", "2.5")
    for version in SUPPORTED_HYDAMO_VERSIONS:
        assert get_damo_converter(version).hydamo_version == version


def test_load_from_gpkg_uses_versioned_converter_for_damo_22():
    gpkg_file = DATA_PATH / "Example_model.gpkg"
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    assert hydamo.source_damo_version == "2.2"
    assert len(hydamo.branches) == 61
    assert len(hydamo.management_device) == 32
    assert len(hydamo.pumps) == 3


@pytest.mark.parametrize("version", ["2.3", "2.4"])
def test_load_from_gpkg_passthrough_versions_behave_like_22(version):
    gpkg_file = DATA_PATH / "Example_model.gpkg"
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)

    assert hydamo.source_damo_version == version
    assert len(hydamo.branches) == 61
    assert len(hydamo.management_device) == 32
    assert len(hydamo.pumps) == 3


def test_load_from_gpkg_maps_damo_25_to_canonical_internal_structure(tmp_path: Path):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")

    assert hydamo.source_damo_version == "2.5"
    assert "categorieoppwaterlichaam" in hydamo.branches.columns
    assert hydamo.branches.at["H1", "categorieoppwaterlichaam"] == "primair"

    assert "soortafsluitmiddel" in hydamo.management_device.columns
    assert hydamo.management_device.at[0, "soortafsluitmiddel"] == "schuif"
    assert "soortregelbaarheid" in hydamo.management_device.columns
    assert hydamo.management_device.at[0, "soortregelbaarheid"] == "Regelbaar - lokaal automatisch - elektronisch"

    assert "regelmiddelid" in hydamo.management.columns
    assert "stuurvariabele" in hydamo.management.columns
    assert hydamo.management.at["CTRL1", "regelmiddelid"] == "afsluit-guid"
    assert hydamo.management.at["CTRL1", "stuurvariabele"] == "hoogte opening"

    assert "regelmiddelid" in hydamo.opening.columns
    assert hydamo.opening.at[0, "regelmiddelid"] == "afsluit-guid"


def test_drtc_resolves_canonical_damo_25_management_links(tmp_path: Path):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5")
    hydamo.management["stuwid"] = "ST1"
    hydamo.structures.rweirs_df = pd.DataFrame({"id": ["ST1"]})

    fm = FMModel()
    fm.time.refdate = 20160601
    fm.time.tstop = 3600

    drtc = DRTCModel(hydamo, fm, output_path=tmp_path)
    assert drtc._resolve_management_structure(hydamo.management.iloc[0]) == ("weir", "ST1")


def test_load_from_gpkg_with_validation_warn_generates_rules_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)
    captured: dict[str, object] = {}

    class _ResultSummary:
        success = True
        status = "finished"
        error = []

    def _validator_factory(**kwargs):
        captured["factory_kwargs"] = kwargs

        def _run(directory, raise_error=False):
            rules = json.loads((Path(directory) / "validationrules.json").read_text())
            captured["rules"] = rules
            captured["dataset_files"] = sorted(
                p.name for p in (Path(directory) / "datasets").iterdir()
            )
            return None, None, _ResultSummary()

        return _run

    fake_module = types.SimpleNamespace(validator=_validator_factory)
    monkeypatch.setitem(sys.modules, "hydamo_validation", fake_module)

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file,
        hydamo_version="2.5",
        validate=True,
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
    assert captured["rules"]["hydamo_version"] == "2.5"
    assert captured["rules"]["schema"] == "1.5"
    assert any(obj["object"] == "hydroobject" for obj in captured["rules"]["objects"])


def test_load_from_gpkg_with_validation_strict_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    class _ResultSummary:
        success = False
        status = "failed"
        error = ["bad data"]

    def _validator_factory(**kwargs):
        def _run(directory, raise_error=False):
            return None, None, _ResultSummary()

        return _run

    fake_module = types.SimpleNamespace(validator=_validator_factory)
    monkeypatch.setitem(sys.modules, "hydamo_validation", fake_module)

    with pytest.raises(HyDAMOValidationError):
        HyDAMO().load_from_gpkg(
            gpkg_file,
            hydamo_version="2.5",
            validation_mode="strict",
        )


def test_load_from_gpkg_raises_for_unsupported_version():
    with pytest.raises(ValueError, match="Unsupported"):
        HyDAMO().load_from_gpkg("irrelevant.gpkg", hydamo_version="9.9")


def test_load_from_gpkg_v22_postprocess_copies_soortregelmiddel(tmp_path: Path):
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
    assert hydamo.management_device.at[0, "soortafsluitmiddel"] == "schuif"


def test_load_from_gpkg_validate_true_promotes_mode_to_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    class _ResultSummary:
        success = True
        status = "finished"
        error = []

    def _validator_factory(**kwargs):
        def _run(directory, raise_error=False):
            return None, None, _ResultSummary()

        return _run

    monkeypatch.setitem(
        sys.modules, "hydamo_validation", types.SimpleNamespace(validator=_validator_factory)
    )

    # validate=True with default validation_mode="off" should promote to "warn"
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.5", validate=True)

    assert hydamo.validation_result is not None
    assert hydamo.validation_result.success is True


def test_load_from_gpkg_validation_warn_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    class _ResultSummary:
        success = False
        status = "failed"
        error = ["bad data"]

    def _validator_factory(**kwargs):
        def _run(directory, raise_error=False):
            return None, None, _ResultSummary()

        return _run

    monkeypatch.setitem(
        sys.modules, "hydamo_validation", types.SimpleNamespace(validator=_validator_factory)
    )

    hydamo = HyDAMO().load_from_gpkg(
        gpkg_file, hydamo_version="2.5", validation_mode="warn"
    )

    assert hydamo.validation_result is not None
    assert hydamo.validation_result.success is False
    assert hydamo.validation_result.errors == ["bad data"]


def test_validate_gpkg_returns_result_and_stores_on_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gpkg_file = _make_minimal_25_gpkg(tmp_path)

    class _ResultSummary:
        success = True
        status = "finished"
        error = []

    def _validator_factory(**kwargs):
        def _run(directory, raise_error=False):
            return None, None, _ResultSummary()

        return _run

    monkeypatch.setitem(
        sys.modules, "hydamo_validation", types.SimpleNamespace(validator=_validator_factory)
    )

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
def test_load_from_gpkg_real_validation_succeeds():
    """Integration test: runs the real hydamo_validation library against Example_model.gpkg.

    Only runs on Python 3.12 where hydamo-validation is installed via pip install hydrolib[validation].
    Verifies that our wrapper correctly calls the library and that the minimal auto-generated
    rules pass for the bundled example model.
    """
    hydamo = HyDAMO().load_from_gpkg(
        DATA_PATH / "Example_model.gpkg",
        hydamo_version="2.2",
        validation_mode="warn",
    )

    assert hydamo.validation_result is not None
    assert hydamo.validation_result.success is True
    assert hydamo.validation_result.status is not None
