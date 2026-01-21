import logging
import sys

sys.path.insert(0, r".")
from pathlib import Path

import numpy as np
import pandas as pd

from hydrolib.dhydamo.core.drtc import DRTCModel
from tests.dhydamo.io.test_to_hydrolibcore import setup_model


def _setup_rtc_model(hydamo=None, fm=None, output_path=None, multiple_folders=False):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    
    if output_path is None:
        output_path = Path("hydrolib/tests/model").resolve()

    if hydamo is None:
        hydamo, fm = setup_model(hydamo=hydamo, full_test=True)

    if multiple_folders:
        complex_controllers_folder=[
            data_path / "complex_controllers_1",
            data_path / "complex_controllers_2",
        ]
    else:
        complex_controllers_folder=data_path / "complex_controllers_1"

    drtcmodel = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=complex_controllers_folder,
        id_limit_complex_controllers=["S_96684", "ObsS_96684"],
        rtc_timestep=60.0,
    )

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

    return drtcmodel


def test_setup_rtc_model(hydamo=None):
    drtcmodel = _setup_rtc_model(hydamo=hydamo)
    assert len(drtcmodel.pid_controllers) == 3
    assert len(drtcmodel.time_controllers) == 2
    assert len(drtcmodel.interval_controllers) == 1


def test_complex_controller_already_present(caplog, hydamo=None):
    rtcd = _setup_rtc_model(hydamo=hydamo)
    rtcd.add_time_controller(
        structure_id="S_96684",
        steering_variable="Crest level (s)",
        data=pd.Series(np.zeros(10)),
    )
    with caplog.at_level(logging.WARNING, logger="hydrolib.dhydamo.core.drtc"):
        rtcd.write_xml_v1()
    
    assert any("Skipped writing" in message for message in caplog.messages)
