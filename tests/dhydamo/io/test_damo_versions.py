"""The DAMO version steps, chained by `source_layers`."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from hydrolib.dhydamo.io.damo_versions import (
    SUPPORTED_HYDAMO_VERSIONS,
    SourceLayer,
    source_layers,
    steps,
)
from shapely.geometry import Point

# The step in which DAMO merged Regelmiddel into Afsluitmiddel
STEP_24_25 = next(i for i, step in enumerate(steps.STEPS) if step.new == "2.5")


def test_source_layers_chains_the_renames_of_every_step_per_table() -> None:
    sources = source_layers("2.5")

    # typeruwheid renamed in 2.4, vormkoker in 2.5
    assert sources["culverts"] == SourceLayer(
        "DuikerSifonHevel",
        {"typemethoderuwheid": "typeruwheid", "vormduikersifonhevel": "vormkoker"},
    )
    # Moved from Regelmiddel to Afsluitmiddel in 2.5, with Regelmiddel's old names.
    assert sources["management_device"] == SourceLayer(
        "Afsluitmiddel",
        {
            "typeafsluitmiddel": "soortregelmiddel",
            "maximalehoogteonderkant": "hoogteopening",
        },
    )


def test_source_layers_of_damo_22_renames_nothing() -> None:
    sources = source_layers("2.2")

    assert sources["management_device"] == SourceLayer("Regelmiddel", {})
    assert all(not source.renames for source in sources.values())


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_no_two_tables_read_the_same_layer(version: str) -> None:
    layers = [source.layer.lower() for source in source_layers(version).values()]

    assert len(layers) == len(set(layers))


def test_source_layers_rejects_two_tables_reading_the_same_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The steps are keyed by table, so each table must have its own layer.
    broken = list(steps.STEPS)
    broken[STEP_24_25] = dataclasses.replace(
        broken[STEP_24_25],
        layers={**broken[STEP_24_25].layers, "weirs": "Afsluitmiddel"},
    )
    monkeypatch.setattr(steps, "STEPS", tuple(broken))

    with pytest.raises(
        ValueError, match='"weirs" and "management_device" both read layer'
    ):
        source_layers("2.5")


@pytest.mark.parametrize("field", ["layers", "columns"])
def test_source_layers_rejects_a_step_naming_a_table_that_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    # e.g. a typo, which would otherwise be ignored (layers) or raise a bare KeyError
    broken = list(steps.STEPS)
    if field == "layers":
        broken[STEP_24_25] = dataclasses.replace(
            broken[STEP_24_25], layers={"managment_device": "Afsluitmiddel"}
        )
    else:
        broken[STEP_24_25] = dataclasses.replace(
            broken[STEP_24_25], columns={"managment_device": {"old": "new"}}
        )
    monkeypatch.setattr(steps, "STEPS", tuple(broken))

    with pytest.raises(ValueError, match=r'step 2\.4 -> 2\.5 .* "managment_device"'):
        source_layers("2.5")


def test_source_layers_rejects_an_unsupported_version() -> None:
    with pytest.raises(ValueError, match='Unsupported HyDAMO DAMO version "9.9"'):
        source_layers("9.9")


def test_loading_does_not_import_hydamo_validation(tmp_path: Path) -> None:
    # The steps and the data model are plain data; only validation and the contract
    # test use the validatiemodule, which exists only for Python 3.12.
    gpkg_file = tmp_path / "weir.gpkg"
    gpd.GeoDataFrame(
        {
            "code": ["ST1"],
            "globalid": ["st1-guid"],
            "afvoercoefficient": [1.0],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:28992",
    ).to_file(gpkg_file, layer="Stuw", driver="GPKG", engine="pyogrio")
    code = (
        "import sys\n"
        "from hydrolib.dhydamo.core.hydamo import HyDAMO\n"
        f"hydamo = HyDAMO().load_from_gpkg({str(gpkg_file)!r}, hydamo_version='2.6')\n"
        "assert hydamo.weirs.index.tolist() == ['ST1']\n"
        "assert 'hydamo_validation' not in sys.modules\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
