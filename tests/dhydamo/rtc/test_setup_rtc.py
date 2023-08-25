import sys

sys.path.insert(0, r".")
import os
import pandas as pd
from pathlib import Path
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from tests.dhydamo.io import test_from_hydamo
from hydrolib.dhydamo.core.drtc import DRTCModel
from tests.dhydamo.io.test_to_hydrolibcore import setup_model


def test_setup_rtc_model(hydamo=None):
    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    output_path = Path("hydrolib/tests/model").resolve()
    assert output_path.exists()

    hydamo, fm = setup_model(hydamo=hydamo)

    drtcmodel = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=data_path / "complex_controllers",
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

    assert len(drtcmodel.pid_controllers) == 1
    drtcmodel.add_pid_controller(
        structure_id="S_96544",
        steering_variable="Crest level (s)",
        target_variable="Water level (op)",
        setpoint=18.2,
        observation_location="ObsS_96544",
        lower_bound=18.0,
        upper_bound=18.4,
        pid_settings=pid_settings["global"],
    )

    assert len(drtcmodel.pid_controllers) == 2

    drtcmodel.add_time_controller(
        structure_id="S_96548",
        steering_variable="Crest level (s)",
        data=timeseries.iloc[:, 1],
    )

    assert len(drtcmodel.time_controllers) == 2

    return drtcmodel
