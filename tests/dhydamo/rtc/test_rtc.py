import logging
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, r".")
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter
from tests.dhydamo.io.test_to_hydrolibcore import setup_model


def _add_default_simple_control(data_path, hydamo, drtcmodel):
    pid_settings = {}
    pid_settings["global"] = {
        "ki": -0.05,
        "kp": -0.03,
        "kd": 0.0,
        "maxspeed": 0.00033,
    }

    if not hydamo.management.typecontroller.empty:
        timeseries = pd.read_csv(data_path / "timecontrollers.csv")
        timeseries.index = timeseries.Time

        pid_settings["kst_pid"] = {
            "ki": -0.03,
            "kp": -0.0,
            "kd": 0.0,
            "maxspeed": 0.00033,
        }

        drtcmodel.from_hydamo(pid_settings=pid_settings, timeseries=timeseries)

    drtcmodel.add_pid_controller(
        structure_id="S_96544",
        steering_variable="Crest level (s)",
        target_variable="Water level (op)",
        setpoint=18.2,
        observation_location="ObsS_96544",
        lower_bound=18.0,
        upper_bound=18.4,
        ki=0.001,
        kp=0.0,
        kd=0,
        max_speed=0.00033,
        interpolation_option="LINEAR",
        extrapolation_option="BLOCK",
    )

    drtcmodel.add_time_controller(
        structure_id="S_96548",
        steering_variable="Crest level (s)",
        data=timeseries.iloc[:, 1],
    )

    drtcmodel.add_interval_controller(
        structure_id="orifice_test",
        observation_location="ObsO_test",
        steering_variable="Gate lower edge level (s)",
        target_variable="Discharge (op)",
        setpoint=13.2,
        setting_below=12.8,
        setting_above=13.4,
        max_speed=0.00033,
        deadband=0.1,
        interpolation_option="LINEAR",
        extrapolation_option="BLOCK",
    )

def _setup_rtc_model(hydamo=None, fm=None, output_path=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()

    if output_path is None:
        output_path = Path("hydrolib/tests/model").resolve()

    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    drtcmodel = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=data_path / "complex_controllers_1",
        id_limit_complex_controllers=["S_96684", "ObsS_96684"],
        rtc_timestep=60.0,
    )
    _add_default_simple_control(data_path, hydamo, drtcmodel)

    return drtcmodel


def test_setup_rtc_model(hydamo=None):
    rtcd = _setup_rtc_model(hydamo=hydamo)
    rtcd.write_xml_v1()
    assert len(rtcd.pid_controllers) == 5
    assert len(rtcd.time_controllers) == 2
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 2
    assert len(rtcd.all_controllers) == 8

def test_complex_controller_already_present(caplog, hydamo=None):
    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        rtcd = _setup_rtc_model(hydamo=hydamo)
        rtcd.add_time_controller(
            structure_id="S_96684",
            steering_variable="Crest level (s)",
            data=pd.Series(np.zeros(10)),
        )
        rtcd.write_xml_v1()

    expected_msg = "RtcToolsConfig.xml: Skipped writing Time control for S_96684, complex controller already present"
    assert expected_msg in caplog.messages

    assert len(rtcd.pid_controllers) == 5
    assert len(rtcd.time_controllers) == 3
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 2
    assert len(rtcd.all_controllers) == 8

def test_complex_controller_multiple_folders(hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.structures.add_uweir(
        id="uweir_test",
        branchid="W_242213_0",
        chainage=2.0,
        crestlevel=18.00,
        crestwidth=7.5,
        dischargecoeff=1.0,
        numlevels=4,
        yvalues="0.0 1.0 2.0 3.0",
        zvalues="19.0 18.0 18.2 19",
    )

    id_limit_complex_controllers = [
        "S_96684",
        "ObsS_96684",
        "uweir_test"
    ]

    rtcd1 = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=data_path / "complex_controllers_1",
        id_limit_complex_controllers=id_limit_complex_controllers,
        rtc_timestep=60.0,
    )
    rtcd2 = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder= data_path / "complex_controllers_2",
        id_limit_complex_controllers=id_limit_complex_controllers,
        rtc_timestep=60.0,
    )

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=id_limit_complex_controllers,
        rtc_timestep=60.0,
    )

    for key in rtcd.complex_controllers.keys():
        if key == "dimr_config":
            assert len(rtcd.complex_controllers[key]) == 1
        else:
            assert len(rtcd1.complex_controllers[key]) + len(rtcd2.complex_controllers[key]) == len(rtcd.complex_controllers[key])

def test_complex_controller_multiple_folders_dimr_merged(hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.structures.add_uweir(
        id="uweir_test",
        branchid="W_242213_0",
        chainage=2.0,
        crestlevel=18.00,
        crestwidth=7.5,
        dischargecoeff=1.0,
        numlevels=4,
        yvalues="0.0 1.0 2.0 3.0",
        zvalues="19.0 18.0 18.2 19",
    )

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=["S_96684", "ObsS_96684", "uweir_test"],
        rtc_timestep=60.0,
    )
    root = rtcd.complex_controllers["dimr_config"][0]
    targets = {
        el.text
        for el in root.findall(".//{*}coupler[@name='rtc_to_flow']/{*}item/{*}targetName")
    }
    assert "weirs/S_96684/crestLevel" in targets
    assert "weirs/uweir_test/crestLevel" in targets


def test_dimrwriter_deduplicates_coupler_items(hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.structures.add_uweir(
        id="uweir_test",
        branchid="W_242213_0",
        chainage=2.0,
        crestlevel=18.00,
        crestwidth=7.5,
        dischargecoeff=1.0,
        numlevels=4,
        yvalues="0.0 1.0 2.0 3.0",
        zvalues="19.0 18.0 18.2 19",
    )

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=["S_96684", "ObsS_96684", "uweir_test"],
        rtc_timestep=60.0,
    )
    _add_default_simple_control(data_path, hydamo, rtcd)
    rtcd.write_xml_v1()

    dimrwriter = DIMRWriter(output_path=output_path)
    dimrwriter.write_dimrconfig(fm, rtc_model=rtcd)

    root = ET.parse(output_path / "dimr_config.xml").getroot()
    flow_to_rtc_pairs = []
    for coupler_name in ("flow_to_rtc", "rtc_to_flow"):
        pairs = []
        for item in root.findall(f".//{{*}}coupler[@name='{coupler_name}']/{{*}}item"):
            source = item.find("./{*}sourceName")
            target = item.find("./{*}targetName")
            if source is None or target is None:
                continue
            pairs.append((source.text, target.text))
        if coupler_name == "flow_to_rtc":
            flow_to_rtc_pairs = pairs

        assert len(pairs) == len(set(pairs))

    assert (
        "observations/ObsO_test/discharge",
        "[Input]ObsO_test/Discharge (op)",
    ) in flow_to_rtc_pairs


def test_drtc_deduplicates_complex_fragments(hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.structures.add_uweir(
        id="uweir_test",
        branchid="W_242213_0",
        chainage=2.0,
        crestlevel=18.00,
        crestwidth=7.5,
        dischargecoeff=1.0,
        numlevels=4,
        yvalues="0.0 1.0 2.0 3.0",
        zvalues="19.0 18.0 18.2 19",
    )

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=["S_96684", "ObsS_96684", "uweir_test"],
        rtc_timestep=60.0,
    )
    _add_default_simple_control(data_path, hydamo, rtcd)

    # Inject duplicate complex snippets explicitly.
    rtcd.complex_controllers["dataconfig_import"].append(
        rtcd.complex_controllers["dataconfig_import"][0]
    )
    rtcd.complex_controllers["dataconfig_export"].append(
        rtcd.complex_controllers["dataconfig_export"][0]
    )
    rtcd.complex_controllers["timeseries"].append(
        rtcd.complex_controllers["timeseries"][0]
    )
    rtcd.complex_controllers["state"].append(rtcd.complex_controllers["state"][0])

    rtcd.write_xml_v1()

    rtc_data_root = ET.parse(output_path / "rtc" / "rtcDataConfig.xml").getroot()
    dataconfig_ids = [
        el.attrib["id"]
        for el in rtc_data_root.findall(".//{*}timeSeries")
        if "id" in el.attrib
    ]
    assert len(dataconfig_ids) == len(set(dataconfig_ids))

    ts_root = ET.parse(output_path / "rtc" / "timeseries_import.xml").getroot()
    ts_keys = []
    for series in ts_root.findall("./{*}series"):
        location = series.find("./{*}header/{*}locationId")
        parameter = series.find("./{*}header/{*}parameterId")
        if location is None or parameter is None:
            continue
        ts_keys.append(f"{location.text}|{parameter.text}")
    assert len(ts_keys) == len(set(ts_keys))

    state_root = ET.parse(output_path / "rtc" / "state_import.xml").getroot()
    state_ids = [
        leaf.attrib["id"]
        for leaf in state_root.findall(".//{*}treeVectorLeaf")
        if "id" in leaf.attrib
    ]
    assert len(state_ids) == len(set(state_ids))


def test_complex_controller_fourtypes(caplog, hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.observationpoints.add_points(
        [Point(200064,395087), Point(200775,395209), Point(201585,395162), Point(199877,393954)],
        ["ObsS_98143", "ObsS_96840", "ObsS_96547", "ObsS_96789"],
        locationTypes=["1d", "1d", "1d", "1d"],
        snap_distance=10.0,
    )

    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        rtcd = DRTCModel(
            hydamo,
            fm,
            output_path=output_path,
            complex_controllers_folder=data_path / "complex_controllers_4types",
            id_limit_complex_controllers=[
                "S_98143",
                "S_96789",
                "S_96547",
                "S_96840",
            ],
            rtc_timestep=60.0,
        )
        _add_default_simple_control(data_path, hydamo, rtcd)
        rtcd.write_xml_v1()

    check_msg = "RtcToolsConfig.xml: Skipped writing Time control for S_96840, complex controller already present"
    assert check_msg in caplog.messages

    assert len(rtcd.pid_controllers) == 5
    assert len(rtcd.time_controllers) == 2
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 6
    assert len(rtcd.all_controllers) == 7


def test_complex_controller_fourtypes_limit(caplog, hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.observationpoints.add_points(
        [Point(200064,395087), Point(200775,395209), Point(201585,395162), Point(199877,393954)],
        ["ObsS_98143", "ObsS_96840", "ObsS_96547", "ObsS_96789"],
        locationTypes=["1d", "1d", "1d", "1d"],
        snap_distance=10.0,
    )

    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        rtcd = DRTCModel(
            hydamo,
            fm,
            output_path=output_path,
            complex_controllers_folder=data_path / "complex_controllers_4types",
            id_limit_complex_controllers=[
                "S_98143",
                "S_96789",
                "S_96547",
                # removed "S_96840"
            ],
            rtc_timestep=60.0,
        )
        _add_default_simple_control(data_path, hydamo, rtcd)
        rtcd.write_xml_v1()

    check_msg1 = "dimr_config.xml: Skipped rtc_to_flow element with 'targetName' 'weirs/S_96840/crestLevel' (not allowed by complex controller filter)."
    check_msg2 = "rtcDataConfig.xml: Skipped exportSeries item for elementId 'S_96840' (not allowed by complex controller filter)."
    check_msg3 = "rtcToolsConfig.xml: Skipped rule element '[TimeRule]reuseObsPoint/Time Rule' (not allowed by complex controller filter)."
    assert check_msg1 in caplog.messages
    assert check_msg2 in caplog.messages
    assert check_msg3 in caplog.messages

    assert len(rtcd.pid_controllers) == 5
    assert len(rtcd.time_controllers) == 2
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 6
    assert len(rtcd.all_controllers) == 8

def test_complex_controller_wrong(caplog, hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/tests/model").resolve()
    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    hydamo.observationpoints.add_points(
        [Point(200064,395087), Point(200775,395209), Point(201585,395162), Point(199877,393954)],
        ["ObsS_98143", "ObsS_96840", "ObsS_96547", "ObsS_96789"],
        locationTypes=["1d", "1d", "1d", "1d"],
        snap_distance=10.0,
    )

    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        rtcd = DRTCModel(
            hydamo,
            fm,
            output_path=output_path,
            complex_controllers_folder=data_path / "complex_controllers_fout",
            id_limit_complex_controllers=[
                "S_98143",
                "S_96789",
                "S_96547",
                "S_96840",
            ],
            rtc_timestep=60.0,
        )
        _add_default_simple_control(data_path, hydamo, rtcd)
        rtcd.write_xml_v1()

    check_msg1 = "dimr_config.xml: Skipped rtc_to_flow element with 'targetName' 'weirs/S_96548/crestLevel' (not allowed by complex controller filter)."
    check_msg2 = "rtcDataConfig.xml: Skipped exportSeries item for elementId 'S_96548' (not allowed by complex controller filter)."
    check_msg3 = "rtcToolsConfig.xml: Skipped rule element '[PID]PIDfout/PID Rule' (not allowed by complex controller filter)."
    check_msg4 = "RtcToolsConfig.xml: Skipped writing Time control for S_96840, complex controller already present"

    assert check_msg1 in caplog.messages
    assert check_msg2 in caplog.messages
    assert check_msg3 in caplog.messages
    assert check_msg4 in caplog.messages
