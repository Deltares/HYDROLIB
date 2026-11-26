"""Weirs, orifices and culverts built from loaded DAMO data."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.dhydamo.core.hydamo import HyDAMO
from shapely.geometry import LineString, Point

from tests.dhydamo.io.damo_gpkg import (
    _convert_weirs,
    _damo_names,
    _write_branch,
    _write_devices,
    _write_layer,
    _write_stuw,
)

# The structure tests run on the versions on either side of the 2.4 -> 2.5 step, the
# step that changes the management device; the steps and the contract test cover the
# other versions.
STEP_VERSIONS = ("2.4", "2.5")


def _write_weir_with_opening(
    gpkg_path: Path,
    crests: tuple[float, ...] = (0.4,),
    widths: tuple[float, ...] = (1.2,),
) -> None:
    """Write branch H1 and weir ST1 with an opening per crest and width.

    The first opening is ``opening-guid``, the next ``opening-guid-2`` and so on. By
    default one opening, 1.2 m wide with its crest at 0.4 m.
    """
    globalids = ["opening-guid"]
    globalids += [f"opening-guid-{i}" for i in range(2, len(crests) + 1)]
    _write_branch(gpkg_path)
    _write_stuw(gpkg_path)
    _write_layer(
        gpkg_path,
        "Kunstwerkopening",
        gpd.GeoDataFrame(
            {
                "globalid": globalids,
                "stuwid": ["stuw-guid"] * len(crests),
                "laagstedoorstroombreedte": list(widths),
                "laagstedoorstroomhoogte": list(crests),
                "afvoercoefficient": [1.0] * len(crests),
                "geometry": [Point(5, 0)] * len(crests),
            },
            crs="EPSG:28992",
        ),
    )


def _write_culvert(
    gpkg_path: Path, names: dict[str, str], upstream_invert: float = 0.1
) -> None:
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
                "hoogtebinnenonderkantbov": [upstream_invert],
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


def _gate_height(version: str, level: float, height: float) -> dict[str, float]:
    """Return the device columns of a gate *height* m above *level* m NAP in *version*.

    Up to DAMO 2.4 ``hoogteopening`` is the height above the opening's crest (or the
    culvert's invert); from 2.5 ``maximalehoogteonderkant`` is the gate's level.
    """
    if version >= "2.5":
        return {"maximalehoogteonderkant": level + height}
    return {"hoogteopening": height}


def _convert_culverts(hydamo: HyDAMO) -> pd.DataFrame:
    """Build the D-Flow FM culverts on branch H1 and return them, by id."""
    hydamo.culverts["branch_id"] = "H1"
    hydamo.culverts["branch_offset"] = 3.5
    hydamo.structures.convert.culverts(
        hydamo.culverts, management_device=hydamo.management_device
    )
    return hydamo.structures.culverts_df.set_index("id")


@pytest.mark.parametrize("version", STEP_VERSIONS)
def test_load_from_gpkg_gives_an_orifice_the_gate_of_its_device(
    tmp_path: Path, version: str
) -> None:
    # Up to DAMO 2.4 the device has the gate's height above the crest, from 2.5 the
    # gate's level in m NAP.
    names = _damo_names(version)
    gpkg_file = tmp_path / "orifice.gpkg"
    _write_weir_with_opening(gpkg_file, crests=(-1.0,))
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
                **_gate_height(version, level=-1.0, height=0.5),
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    _convert_weirs(hydamo)

    assert hydamo.management_device.loc[0, "hoogteopening"] == pytest.approx(0.5)
    orifice = hydamo.structures.orifices_df.set_index("id").loc["ST1"]
    assert orifice["crestlevel"] == pytest.approx(-1.0)
    assert orifice["gateloweredgelevel"] == pytest.approx(-0.5)


@pytest.mark.parametrize("version", STEP_VERSIONS)
def test_load_from_gpkg_gives_a_culvert_the_valve_of_its_device(
    tmp_path: Path, version: str
) -> None:
    names = _damo_names(version)
    gpkg_file = tmp_path / "valve.gpkg"
    _write_branch(gpkg_file)
    _write_culvert(gpkg_file, names, upstream_invert=-1.2)
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
                # 0.4 m above the upstream invert
                **_gate_height(version, level=-1.2, height=0.4),
                "afvoercoefficient": 0.8,
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    culvert = _convert_culverts(hydamo).loc["DU1"]

    assert culvert["valveonoff"] == 1
    assert culvert["valveopeningheight"] == pytest.approx(0.4)
    # Relative to the culvert's opening height of 2 m.
    assert culvert["relopening"] == pytest.approx([0.2])
    assert culvert["allowedflowdir"] == "both"


def _non_return_culvert(
    tmp_path: Path, version: str, direction: str | None
) -> pd.Series:
    """Build culvert DU1 with non-return valve AF1 of *direction*, and return it."""
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
                "richting": direction,
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    culvert = _convert_culverts(hydamo).loc["DU1"]
    assert isinstance(culvert, pd.Series)
    return culvert


@pytest.mark.parametrize("version", STEP_VERSIONS)
@pytest.mark.parametrize(
    ("direction", "allowedflowdir"),
    [("Positief", "positive"), ("Negatief", "negative")],
)
def test_load_from_gpkg_gives_a_culvert_the_non_return_valve_of_its_device(
    tmp_path: Path, version: str, direction: str, allowedflowdir: str
) -> None:
    # richting up to DAMO 2.4, stroomrichting from 2.5: relative to the HydroObject,
    # whose direction the branch keeps.
    culvert = _non_return_culvert(tmp_path, version, direction)

    assert culvert["allowedflowdir"] == allowedflowdir
    assert culvert["valveonoff"] == 0


@pytest.mark.parametrize("version", STEP_VERSIONS)
@pytest.mark.parametrize("direction", [None, "Beide"])
def test_load_from_gpkg_gives_a_non_return_valve_without_direction_positive_flow(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    version: str,
    direction: str | None,
) -> None:
    with caplog.at_level(logging.WARNING):
        culvert = _non_return_culvert(tmp_path, version, direction)

    assert culvert["allowedflowdir"] == "positive"
    assert "culvert DU1" in caplog.text
    assert "AF1" in caplog.text


def test_load_from_gpkg_leaves_closing_devices_of_locks_coupures_and_tunnels_out_of_the_model(
    tmp_path: Path,
) -> None:
    # From DAMO 2.5, Afsluitmiddel also holds the closing devices of locks, coupures
    # and tunnels. They load as management devices, but nothing joins on them.
    names = _damo_names("2.5")
    gpkg_file = tmp_path / "closing_devices.gpkg"
    _write_weir_with_opening(gpkg_file)
    _write_culvert(gpkg_file, names)
    structure_device = {"overlaatonderlaat": "Nvt", "maximalehoogteonderkant": 2.0}
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
                "maximalehoogteonderkant": 0.7,
            },
            {
                "code": "AF-DU1",
                "globalid": "culvert-device-guid",
                "duikersifonhevelid": "duiker-guid",
                "overlaatonderlaat": "Nvt",
                "soortregelmiddel": "schuif",
                "maximalehoogteonderkant": 0.4,
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
    # The gate height isn't renamed: maximalehoogteonderkant (0.7 for the weir's
    # device, 0.4 for the culvert's) wins over a stale hoogteopening when deriving it.
    stale = {"hoogteopening": 0.1, "soortregelmiddel": "terugslagklep"}
    current = {"typeafsluitmiddel": "schuif"}
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
                    "maximalehoogteonderkant": 0.7,
                    "geometry": Point(5, 0),
                },
                {
                    "code": "AF-DU1",
                    "globalid": "culvert-device-guid",
                    "duikersifonhevelid": "duiker-guid",
                    "overlaatonderlaat": "Nvt",
                    **device_columns,
                    "maximalehoogteonderkant": 0.4,
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
        'HyDAMO layer "Afsluitmiddel" has both "typeafsluitmiddel" and '
        '"soortregelmiddel"' in caplog.text
    )


@pytest.mark.parametrize("version", STEP_VERSIONS)
def test_load_from_gpkg_skips_an_orifice_without_gate_height(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, version: str
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
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    with caplog.at_level(logging.WARNING):
        _convert_weirs(hydamo)

    assert hydamo.structures.orifices_df.empty
    assert "weir ST1" in caplog.text
    assert "AF1" in caplog.text
    assert "no gate height" in caplog.text


@pytest.mark.parametrize("version", STEP_VERSIONS)
def test_load_from_gpkg_builds_a_culvert_without_the_valve_of_a_gate_without_height(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, version: str
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
                "afvoercoefficient": 0.8,
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    with caplog.at_level(logging.WARNING):
        culvert = _convert_culverts(hydamo).loc["DU1"]

    assert culvert["valveonoff"] == 0
    assert "culvert DU1" in caplog.text
    assert "AF1" in caplog.text
    assert "no gate height" in caplog.text


@pytest.mark.parametrize("version", STEP_VERSIONS)
def test_load_from_gpkg_gives_each_underflow_opening_its_own_crest(
    tmp_path: Path, version: str
) -> None:
    names = _damo_names(version)
    gpkg_file = tmp_path / "compound.gpkg"
    _write_weir_with_opening(gpkg_file, crests=(0.4, 0.6), widths=(1.2, 2.0))
    devices = []
    for i, (opening, crest) in enumerate(
        [("opening-guid", 0.4), ("opening-guid-2", 0.6)], start=1
    ):
        devices.append(
            {
                "code": f"AF{i}",
                "globalid": f"device-guid-{i}",
                "kunstwerkopeningid": opening,
                "overlaatonderlaat": names["onderlaat"],
                "soortregelmiddel": "schuif",
                **_gate_height(version, level=crest, height=0.3),
            }
        )
    _write_devices(gpkg_file, names, devices)

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    _convert_weirs(hydamo)

    orifices = hydamo.structures.orifices_df.set_index("id")
    assert orifices.loc["ST1_1", "crestlevel"] == pytest.approx(0.4)
    assert orifices.loc["ST1_1", "crestwidth"] == pytest.approx(1.2)
    assert orifices.loc["ST1_1", "gateloweredgelevel"] == pytest.approx(0.7)
    assert orifices.loc["ST1_2", "crestlevel"] == pytest.approx(0.6)
    assert orifices.loc["ST1_2", "crestwidth"] == pytest.approx(2.0)
    assert orifices.loc["ST1_2", "gateloweredgelevel"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("version", "device_type"),
    [("2.4", "onbekend"), ("2.5", "onbekend"), ("2.5", "balg")],
)
def test_load_from_gpkg_builds_a_culvert_without_a_device_of_another_type(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    version: str,
    device_type: str,
) -> None:
    names = _damo_names(version)
    gpkg_file = tmp_path / "culvert.gpkg"
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
                "soortregelmiddel": device_type,
            }
        ],
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version=version)
    with caplog.at_level(logging.WARNING):
        culvert = _convert_culverts(hydamo).loc["DU1"]

    assert culvert["valveonoff"] == 0
    assert culvert["allowedflowdir"] == "both"
    assert "culvert DU1" in caplog.text
    assert "AF1" in caplog.text
    assert f'"{device_type}"' in caplog.text


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
