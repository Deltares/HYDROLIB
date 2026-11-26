"""Write small DAMO GeoPackages, and build from them, for the DAMO loading tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.dhydamo import validation
from hydrolib.dhydamo.core.hydamo import HyDAMO
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

    _write_branch(gpkg_path, categorieoppervlaktewater=["primair"])
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
    _write_devices(
        gpkg_path,
        _damo_names("2.5"),
        [
            {
                "code": "AF1",
                "globalid": "afsluit-guid",
                "kunstwerkopeningid": "opening-guid",
                "overlaatonderlaat": "onderlaat",
                "soortregelmiddel": "schuif",
                "typeregelbaarheid": "Regelbaar - lokaal automatisch - elektronisch",
                "maximalehoogteonderkant": 0.7,
                "afvoercoefficient": 0.8,
            }
        ],
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


def _damo_names(version: str) -> dict[str, str]:
    """Return ``{2.2 name: name in *version*}`` of the columns DAMO renamed.

    The management device is Regelmiddel up to DAMO 2.4 and Afsluitmiddel from 2.5;
    its type column is ``soortregelmiddel`` on Regelmiddel and ``typeafsluitmiddel``
    on the 2.5 Afsluitmiddel. Its gate height is not a rename: see `_gate_height`.

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
        "richting": "richting",
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
        names["richting"] = "stroomrichting"
        names["onderlaat"] = "Onderlaat"
    return names


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
