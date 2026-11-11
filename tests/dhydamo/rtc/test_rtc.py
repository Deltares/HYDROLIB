import copy
import functools
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter
from netCDF4 import Dataset, chartostring
from shapely.geometry import Point
from tests.dhydamo.io.test_to_hydrolibcore import setup_model

# repository root, so the tests do not depend on the working directory
REPOSITORY = Path(__file__).resolve().parents[3]
RTC_STATE = REPOSITORY / "hydrolib" / "sample_data" / "data" / "rtc_state"
REFERENCE_HIS = REPOSITORY / "tests" / "reference_model" / "dflowfm" / "output" / "DFM_his.nc"


@functools.cache
def _build_full_model():
    return setup_model(full_test=True)


def _full_model():
    """Build the full test model once per session and return a fresh copy of hydamo.

    Building the model (mainly the 2D mesh) takes ~20s, so it is cached. Tests mutate
    hydamo (adding observation points, dropping structures), so each gets a deep copy.
    fm is only read by the RTC code and holds a MeshKernel instance that cannot be
    deep-copied, so it is shared.
    """
    hydamo, fm = _build_full_model()
    return copy.deepcopy(hydamo), fm


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

    # pid controller to weir
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

    # time controller to weir
    drtcmodel.add_time_controller(
        structure_id="S_96548",
        steering_variable="Crest level (s)",
        data=timeseries.iloc[:, 1],
    )

    # interval controller to orifice
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

    # pid controller to culvert valve


def _setup_rtc_model(hydamo=None, fm=None, output_path=None):
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()

    if output_path is None:
        output_path = Path("hydrolib/sample_data/model").resolve()

    if hydamo is None:
        hydamo, fm = _full_model()

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


def test_setup_rtc_model():
    rtcd = _setup_rtc_model()
    rtcd.write_xml_v1()
    assert len(rtcd.pid_controllers) == 6
    assert len(rtcd.time_controllers) == 2
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 2
    assert len(rtcd.all_controllers) == 9


def test_complex_controller_already_present(caplog):
    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        rtcd = _setup_rtc_model()
        rtcd.add_time_controller(
            structure_id="S_96684",
            steering_variable="Crest level (s)",
            data=pd.Series(np.zeros(10)),
        )
        rtcd.write_xml_v1()

    expected_msg = "RtcToolsConfig.xml: Skipped writing Time control for S_96684, complex controller already present"
    assert expected_msg in caplog.messages

    assert len(rtcd.pid_controllers) == 6
    assert len(rtcd.time_controllers) == 3
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 2
    assert len(rtcd.all_controllers) == 9


def test_complex_controller_multiple_folders():
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    id_limit_complex_controllers = ["S_96684", "ObsS_96684", "ObsS_96789"]

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
        complex_controllers_folder=data_path / "complex_controllers_2",
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

    for key in rtcd.complex_controllers:
        if key == "dimr_config":
            assert len(rtcd.complex_controllers[key]) == 1
        else:
            assert len(rtcd1.complex_controllers[key]) + len(
                rtcd2.complex_controllers[key]
            ) == len(rtcd.complex_controllers[key])


def test_complex_controller_multiple_folders_dimr_merged():
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=["S_96684", "S_96789", "ObsS_96684", "ObsS_96789"],
        rtc_timestep=60.0,
    )
    root = rtcd.complex_controllers["dimr_config"][0]
    targets = {
        el.text
        for el in root.findall(
            ".//{*}coupler[@name='rtc_to_flow']/{*}item/{*}targetName"
        )
    }
    assert "weirs/S_96684/crestLevel" in targets
    assert "weirs/S_96789/crestLevel" in targets


def test_dimrwriter_deduplicates_coupler_items():
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=["S_96684", "ObsS_96684", "ObsS_96789"],
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


def test_dimrwriter_flow_to_rtc_components_for_complex_only():
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=data_path / "complex_controllers_1",
        id_limit_complex_controllers=["S_96684", "ObsS_96684"],
        rtc_timestep=60.0,
    )
    rtcd.write_xml_v1()

    dimrwriter = DIMRWriter(output_path=output_path)
    dimrwriter.write_dimrconfig(fm, rtc_model=rtcd)

    root = ET.parse(output_path / "dimr_config.xml").getroot()
    # the full coupler block (with source/targetComponent) is a direct child of root
    coupler = root.find("./{*}coupler[@name='flow_to_rtc']")
    assert coupler is not None

    source_component = coupler.find("./{*}sourceComponent")
    target_component = coupler.find("./{*}targetComponent")
    assert source_component is not None
    assert target_component is not None
    assert source_component.text == "DFM"
    assert target_component.text == "Real_Time_Control"

    # the control block must reference flow_to_rtc too, otherwise DIMR never
    # invokes the coupler even though it's defined below
    control_coupler = root.find(
        "./{*}control/{*}parallel/{*}startGroup/{*}coupler[@name='flow_to_rtc']"
    )
    assert control_coupler is not None


def test_drtc_deduplicates_complex_fragments():
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()

    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ],
        id_limit_complex_controllers=["S_96684", "S_96789", "ObsS_96684"],
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


def _build_fourtypes_rtc(caplog, folder, id_limit):
    """Build an RTC model with the four observation points used by the
    complex-controller fixtures, then add default simple control and write XML."""
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    hydamo.observationpoints.add_points(
        [
            Point(200064, 395087),
            Point(200775, 395209),
            Point(201585, 395162),
            Point(199877, 393954),
        ],
        ["ObsS_98143", "ObsS_96840", "ObsS_96547", "ObsS_96789"],
        locationTypes=["1d"] * 4,
        snap_distance=10.0,
    )

    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        rtcd = DRTCModel(
            hydamo,
            fm,
            output_path=output_path,
            complex_controllers_folder=data_path / folder,
            id_limit_complex_controllers=id_limit,
            rtc_timestep=60.0,
        )
        _add_default_simple_control(data_path, hydamo, rtcd)
        rtcd.write_xml_v1()
    return rtcd


def test_complex_controller_fourtypes(caplog):
    rtcd = _build_fourtypes_rtc(
        caplog,
        "complex_controllers_4types",
        ["S_98143", "S_96789", "S_96547", "S_96840"],
    )

    check_msg = "RtcToolsConfig.xml: Skipped writing Time control for S_96840, complex controller already present"
    assert check_msg in caplog.messages

    assert len(rtcd.pid_controllers) == 6
    assert len(rtcd.time_controllers) == 2
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 6
    assert len(rtcd.all_controllers) == 8


def test_complex_controller_fourtypes_limit(caplog):
    rtcd = _build_fourtypes_rtc(
        caplog,
        "complex_controllers_4types",
        ["S_98143", "S_96789", "S_96547"],  # removed "S_96840"
    )

    check_msg1 = "dimr_config.xml: Skipped rtc_to_flow element with 'targetName' 'weirs/S_96840/crestLevel': excluded by id_limit_complex_controllers."
    check_msg2 = "rtcDataConfig.xml: Skipped exportSeries item for elementId 'S_96840': excluded by id_limit_complex_controllers."
    check_msg3 = "rtcToolsConfig.xml: Skipped rule element '[TimeRule]reuseObsPoint/Time Rule': excluded by id_limit_complex_controllers."
    assert check_msg1 in caplog.messages
    assert check_msg2 in caplog.messages
    assert check_msg3 in caplog.messages

    assert len(rtcd.pid_controllers) == 6
    assert len(rtcd.time_controllers) == 2
    assert len(rtcd.interval_controllers) == 1
    assert len(rtcd.cc_ids) == 6
    assert len(rtcd.all_controllers) == 9


def test_complex_controller_filters_dimr_items_missing_from_hydamo(caplog):
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    # Reuse an existing complex-controller fixture, but make one referenced
    # structure invalid by removing it from the in-memory HyDAMO model. This
    # reproduces cases where user-supplied RTC XML still contains an old/foreign
    # structure id, such as a DIMR target that is no longer present in HyDAMO.
    missing_structure_id = "S_96840"
    hydamo.structures.rweirs_df = hydamo.structures.rweirs_df[
        hydamo.structures.rweirs_df.id != missing_structure_id
    ]
    hydamo.structures.uweirs_df = hydamo.structures.uweirs_df[
        hydamo.structures.uweirs_df.id != missing_structure_id
    ]

    hydamo.observationpoints.add_points(
        [Point(200064, 395087), Point(199996, 393899), Point(199877, 393954)],
        ["ObsS_98143", "ObsS_96789", "ObsS_96547"],
        locationTypes=["1d", "1d", "1d"],
        snap_distance=10.0,
    )

    # Keep the missing id in the whitelist deliberately. The bug is not that it
    # is disallowed by configuration; the bug is that it is referenced by complex
    # RTC XML but fails HyDAMO validation and must therefore be removed everywhere.
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
                missing_structure_id,
            ],
            rtc_timestep=60.0,
        )

    check_msg = (
        "dimr_config.xml: Skipped rtc_to_flow element with 'targetName' "
        "'weirs/S_96840/crestLevel': "
        "referenced by complex-controller XML but missing from HyDAMO."
    )
    assert check_msg in caplog.messages

    # rtcDataConfig.xml filtering already used validated complex-controller ids,
    # so the invalid structure disappeared there. This assertion documents that
    # baseline and makes the DIMR assertion below compare like with like.
    assert all(
        f"<elementId>{missing_structure_id}</elementId>" not in xml_text
        for xml_text in rtcd.complex_controllers["dataconfig_export"]
    )

    # DIMR filtering must make the same decision. A referenced-but-invalid
    # complex-controller id must not survive as an rtc_to_flow coupling target,
    # otherwise DIMR can reference an RTC output that no RTC XML defines.
    dimr_root = rtcd.complex_controllers["dimr_config"][0]
    dimr_targets = {
        el.text
        for el in dimr_root.findall(
            ".//{*}coupler[@name='rtc_to_flow']/{*}item/{*}targetName"
        )
    }
    assert f"weirs/{missing_structure_id}/crestLevel" not in dimr_targets


def test_complex_controller_logs_items_not_referenced_by_dimr(caplog):
    data_path = Path("hydrolib/sample_data/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/sample_data/model").resolve()
    hydamo, fm = _full_model()

    rtcd = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=data_path / "complex_controllers_1",
        id_limit_complex_controllers=["S_96684", "S_96544"],
        rtc_timestep=60.0,
    )

    # Reuse the complex_controllers_1 fixture, whose dimr_config.xml references
    # only S_96684. Inject one extra RTC data fragment for a valid HyDAMO weir
    # that is not part of those DIMR-discovered complex-controller references.
    # This exercises the reason before the whitelist decision is reached.
    root = ET.fromstring(
        """
        <rtcDataConfig xmlns="http://www.wldelft.nl/fews">
          <exportSeries>
            <timeSeries id="[Output]S_96544/Crest level (s)">
            <OpenMIExchangeItem>
                <elementId>S_96544</elementId>
                <quantityId>Crest level (s)</quantityId>
                <unit>m</unit>
            </OpenMIExchangeItem>
            </timeSeries>
          </exportSeries>
        </rtcDataConfig>
        """
    )

    with caplog.at_level(logging.INFO, logger="hydrolib.dhydamo.core.drtc"):
        savedict = {
            "dataconfig_import": [],
            "dataconfig_export": [],
            "toolsconfig_rules": [],
            "toolsconfig_triggers": [],
            "timeseries": [],
            "state": [],
            "dimr_config": [],
        }
        rtcd._parse_cc_rtc_dataconfig(root, savedict)

    check_msg = (
        "rtcDataConfig.xml: Skipped exportSeries item for elementId 'S_96544': "
        "not part of complex-controller references in dimr_config.xml."
    )
    assert check_msg in caplog.messages
    assert savedict["dataconfig_export"] == []


def test_complex_controller_wrong(caplog):
    _build_fourtypes_rtc(
        caplog,
        "complex_controllers_fout",
        ["S_98143", "S_96789", "S_96547", "S_96840"],
    )

    check_msg1 = "dimr_config.xml: Skipped rtc_to_flow element with 'targetName' 'weirs/S_96548/crestLevel': excluded by id_limit_complex_controllers."
    check_msg2 = "rtcDataConfig.xml: Skipped exportSeries item for elementId 'S_96548': excluded by id_limit_complex_controllers."
    check_msg3 = "rtcToolsConfig.xml: Skipped rule element '[PID]PIDfout/PID Rule': excluded by id_limit_complex_controllers."
    check_msg4 = "RtcToolsConfig.xml: Skipped writing Time control for S_96840, complex controller already present"

    assert check_msg1 in caplog.messages
    assert check_msg2 in caplog.messages
    assert check_msg3 in caplog.messages
    assert check_msg4 in caplog.messages


def test_state_from_file(tmp_path, caplog):
    state_path = RTC_STATE
    rtcd = _setup_rtc_model(output_path=tmp_path)
    with caplog.at_level(logging.WARNING, logger="hydrolib.dhydamo.core.drtc"):
        rtcd.set_state_from_file(state_path)
        rtcd.write_xml_v1()

    state = DRTCModel._read_state_leaves(tmp_path / "rtc" / "state_import.xml")
    exported = DRTCModel._read_state_leaves(state_path / "state_export.xml")
    valid_ids = rtcd._dataconfig_ids()

    # all exported items that are part of this RTC model are copied, the others are skipped
    copied = {key: value for key, value in exported.items() if key in valid_ids}
    assert len(copied) > 0
    assert all(state[key] == value for key, value in copied.items())
    assert all(key in valid_ids for key in state)
    assert any("not part of this RTC model and are skipped" in msg for msg in caplog.messages)
    # the state is valid for the end of the earlier run, which differs from the start of this model
    assert any("The RTC state is valid for 2016-06-03" in msg for msg in caplog.messages)


def test_set_initial_state(tmp_path, caplog):
    rtcd = _setup_rtc_model(output_path=tmp_path)
    structure_id = next(iter(rtcd.pid_controllers))
    with caplog.at_level(logging.WARNING, logger="hydrolib.dhydamo.core.drtc"):
        rtcd.set_initial_state({structure_id: 12.34, "not_a_structure": 1.0})
        rtcd.write_xml_v1()

    state = DRTCModel._read_state_leaves(tmp_path / "rtc" / "state_import.xml")
    steering_variable = rtcd.pid_controllers[structure_id]["steering_variable"]
    assert state[f"[Output]{structure_id}/{steering_variable}"] == "12.34"
    assert any("no controlled structure not_a_structure" in msg for msg in caplog.messages)


def test_initial_pid_setting_written_to_state(tmp_path):
    # the start value of each PID controller ends up in state_import.xml; the value itself is tested with fixed
    # numbers in test_initial_pid_setting_crest_level
    rtcd = _setup_rtc_model(output_path=tmp_path)
    rtcd.write_xml_v1()
    state = DRTCModel._read_state_leaves(tmp_path / "rtc" / "state_import.xml")

    checked = 0
    for structure_id, controller in rtcd.pid_controllers.items():
        leaf_id = f"[Output]{structure_id}/{controller['steering_variable']}"
        if leaf_id not in state:
            continue  # replaced by a complex controller
        assert float(state[leaf_id]) == rtcd._initial_pid_setting(structure_id, controller)
        checked += 1
    assert checked > 0


def test_state_from_modelrun(tmp_path, caplog):
    rtcd = _setup_rtc_model(output_path=tmp_path)
    his_path = REFERENCE_HIS
    with caplog.at_level(logging.INFO):
        values = rtcd.set_state_from_modelrun(his_path, time="2016-06-02 12:00")
    assert values
    assert rtcd.initial_state == values

    # compare one weir with the his-file at the same time step
    with Dataset(his_path) as his:
        index = int(np.argmin(np.abs(his["time"][:] - 36 * 3600)))
        names = [n.strip() for n in chartostring(his["weirgen_name"][:])]
        weir = next(sid for sid in values if sid in names)
        assert values[weir] == float(his["weirgen_crest_level"][index, names.index(weir)])

    rtcd.write_xml_v1()
    state = DRTCModel._read_state_leaves(tmp_path / "rtc" / "state_import.xml")
    controller = {**rtcd.time_controllers, **rtcd.interval_controllers, **rtcd.pid_controllers}[weir]
    assert float(state[f"[Output]{weir}/{controller['steering_variable']}"]) == values[weir]


def test_state_from_modelrun_time_options(tmp_path):
    rtcd = _setup_rtc_model(output_path=tmp_path)
    his_path = REFERENCE_HIS
    # without a time, the last time step is used
    last = rtcd.set_state_from_modelrun(his_path)
    assert last == rtcd.set_state_from_modelrun(his_path, netcdf_index=-1)
    with pytest.raises(ValueError):
        rtcd.set_state_from_modelrun(his_path, time="2016-06-02", netcdf_index=-1)


@pytest.fixture
def minimal_drtc(tmp_path):
    """Minimal DRTCModel built from a bare HyDAMO instance — no GeoPackage load."""
    hydamo = HyDAMO()
    fm = FMModel()
    fm.time.refdate = 20160601
    fm.time.tstop = 3600
    drtc = DRTCModel(hydamo, fm, output_path=tmp_path)
    return drtc, hydamo


def test_resolve_management_structure_pump(minimal_drtc):
    drtc, hydamo = minimal_drtc
    hydamo.pumps = pd.DataFrame({"globalid": ["test-pump-guid"], "code": ["P_TEST"]})

    row = pd.Series({"pompid": "test-pump-guid", "regelmiddelid": "some-id"})
    assert drtc._resolve_management_structure(row) == ("pump", "P_TEST")


def test_resolve_management_structure_pump_not_in_hydamo_returns_none(minimal_drtc):
    drtc, hydamo = minimal_drtc
    hydamo.pumps = pd.DataFrame({"globalid": [], "code": []})

    row = pd.Series({"pompid": "missing-pump", "regelmiddelid": "some-id"})
    assert drtc._resolve_management_structure(row) is None


def test_resolve_management_structure_null_regelmiddelid_raises(minimal_drtc):
    drtc, _ = minimal_drtc

    row = pd.Series({"pompid": pd.NA, "regelmiddelid": pd.NA})
    with pytest.raises(ValueError, match="Only management_devices"):
        drtc._resolve_management_structure(row)


def test_resolve_management_structure_device_not_found_returns_none(minimal_drtc, caplog):
    drtc, hydamo = minimal_drtc
    hydamo.management_device = pd.DataFrame({"globalid": ["other-guid"], "duikersifonhevelid": [None]})

    row = pd.Series({"pompid": pd.NA, "regelmiddelid": "does-not-exist-guid"})
    with caplog.at_level(logging.WARNING, logger="hydrolib.dhydamo.core.drtc"):
        result = drtc._resolve_management_structure(row)

    assert result is None
    assert "does-not-exist-guid" in caplog.text


def test_resolve_management_structure_culvert(minimal_drtc):
    drtc, hydamo = minimal_drtc
    hydamo.management_device = pd.DataFrame({
        "globalid": ["dev-guid"],
        "duikersifonhevelid": ["culv-guid"],
    })
    hydamo.culverts = pd.DataFrame({"globalid": ["culv-guid"], "code": ["C_TEST"]})

    row = pd.Series({"pompid": pd.NA, "regelmiddelid": "dev-guid"})
    assert drtc._resolve_management_structure(row) == ("culvert", "C_TEST")


@pytest.mark.parametrize(
    ("attribute", "crestlevel", "expected"),
    [
        ("rweirs_df", 15.2, 15.2),  # the structure's own crest level
        ("uweirs_df", 15.2, 15.2),  # also for universal weirs
        ("uweirs_df", 18.0, 16.0),  # limited to the upper bound of the controller
        ("uweirs_df", 12.0, 14.0),  # limited to the lower bound of the controller
    ],
)
def test_initial_pid_setting_crest_level(minimal_drtc, attribute, crestlevel, expected):
    drtc, hydamo = minimal_drtc
    setattr(hydamo.structures, attribute, pd.DataFrame({"id": ["S_TEST"], "crestlevel": [crestlevel]}))
    controller = {"steering_variable": "Crest level (s)", "lower_bound": 14.0, "upper_bound": 16.0}
    assert drtc._initial_pid_setting("S_TEST", controller) == expected


def test_initial_pid_setting_without_structure_setting(minimal_drtc):
    # no setting available for the structure: start at the upper bound, as before
    drtc, _ = minimal_drtc
    controller = {"steering_variable": "Crest level (s)", "lower_bound": 14.0, "upper_bound": 16.0}
    assert drtc._initial_pid_setting("S_UNKNOWN", controller) == 16.0
