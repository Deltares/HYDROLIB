import logging
import shutil
import os
from pathlib import Path
from typing import Union
import pandas as pd
from pydantic import validate_arguments
from rasterio.transform import from_origin

logger = logging.getLogger(__name__)


class DRTCModel:
    """Main data structure for DRTC-module in DflowFM."""

    def __init__(self, hydamo, complex_controllers=None, output_path=None):
        self.hydamo = hydamo

        self.settings = {
            "starttime": pd.to_datetime("2021-02-10 0:00:00"),
            "endtime": pd.to_datetime("2021-02-15 0:00:00"),
            "timestep": 600.0,
        }

        self.pid_controllers = {}
        self.time_controllers = {}
        if complex_controllers is not None:
            self.complex_controllers = complex_controllers

        if output_path is None:
            self.output_path = Path(".")

        self.output_path = output_path / "rtc"
        self.output_path.makedirs(parents=True, exist_ok=True)

    def from_hydamo(
        self, pid_settings={"global": {"kp": 0.01, "kr": 0.01}}, time_series=None
    ):
        for idx, management in self.hydamo.management.iterrows():
            if management.regelmiddelid is not None:
                opening = self.hydamo.regelmiddel[
                    self.hydamo.regelmiddel.globalid == management.regelmiddelid
                ].kunstwerkopeningid

                if management.typecontroller == "PID":
                    controller = {"fsd": "dfs"}
                elif management.typecontroller == "time":
                    if time_series is None:
                        raise ValueError(
                            "No time series were provided for time controlers"
                        )
                    controller = {"sdfs": "sdfs"}
                else:
                    raise ValueError(
                        "Only PID and time controllers are implemented at this moment."
                    )
            elif management.pompid is not None:
                print("Controllers for pump capacities are not implemented yet.")

    def add_pid_controller(
        self,
        structure=None,
        targetlevel=None,
        pid_settings=None,
        variable=None,
        observation_point=None,
    ):
        pass

    def add_time_controller(self, structure=None, timeseries=None):
        pass

    def add_complex_controller(self, xml_file=None):
        pass

    def copy_files(self, output_path=None):
        srcRTC = os.path.join(os.path.dirname(__file__), "..", "resources", "RTC")
        targetRTC = os.path.join(self.output_path)
        shutil.copytree(srcRTC, targetRTC)
        return True

    def write_xml(self):
        pass
