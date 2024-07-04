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

    assert len(drtcmodel.pid_controllers) == 3
    drtcmodel.add_pid_controller(
        structure_id="S_96544",
        steering_variable="Crest level (s)",
        target_variable="Water level (op)",
        setpoint=18.2,
        observation_location="ObsS_96544",
        lower_bound=18.0,
        upper_bound=18.4,
        ki = 0.001,
        kp =0.,
        kd =0,
        max_speed = 0.00033,
        interpolation_option = 'LINEAR',
        extrapolation_option = "BLOCK"
    )

    assert len(drtcmodel.pid_controllers) == 4

    drtcmodel.add_time_controller(
        structure_id="S_96548",
        steering_variable="Crest level (s)",
        data=timeseries.iloc[:, 1],
    )

    assert len(drtcmodel.time_controllers) == 2
    
    drtcmodel.add_interval_controller(structure_id='orifice_test', 
                                observation_location='ObsO_test', 
                                steering_variable='Gate lower edge level (s)', 
                                target_variable='Discharge (op)', 
                                setpoint=13.2,
                                setting_below=12.8,
                                setting_above=13.4,
                                max_speed=0.00033,
                                deadband=0.1,
                                interpolation_option = 'LINEAR',
                                extrapolation_option = "BLOCK"
                                )
    assert len(drtcmodel.interval_controllers) == 1


    return drtcmodel
