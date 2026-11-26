"""Real-time control built from loaded DAMO Sturing data."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.drtc import CREST_LEVEL, DRTCModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.damo_versions import SUPPORTED_HYDAMO_VERSIONS
from shapely.geometry import LineString, Point

from tests.dhydamo.io.damo_gpkg import (
    _convert_weirs,
    _damo_names,
    _make_minimal_25_gpkg,
    _sturing,
    _write_layer,
    _write_stuw,
)


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
