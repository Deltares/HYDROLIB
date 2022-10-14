import logging
import shutil
import os
from pathlib import Path
from typing import Union
import pandas as pd
from pydantic import validate_arguments
from datetime import datetime as dt
import xml.etree.ElementTree as ET

from hydrolib.core.io.mdu.models import FMModel
from hydrolib.dhydamo.core.hydamo import HyDAMO

# TODO: these classes are generated from XSD-files, but still to figure out how to use them
# from hydrolib.core.io.rtc.rtcruntimeconfig.models import *
# from hydrolib.core.io.rtc.rtctoolsconfig.models import *
# from hydrolib.core.io.rtc.rtcdataconfig.models import *
# from hydrolib.core.io.rtc.pi_timeseries.models import *
# from hydrolib.core.io.rtc.pi_state.models import *
# from hydrolib.core.io.rtc.rtcobjectiveconfig.models import *

logger = logging.getLogger(__name__)


class DRTCModel:
    """Main class to generate RTC-module files."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        hydamo: HyDAMO,
        fm: FMModel,
        output_path: Union[str, Path] = None,
        complex_controllers_folder: Union[str, Path] = None,
        rtc_timestep: Union[int, float] = 60,
    ) -> None:
        """Initialization of the DRTCModel class. (empty) lists/dicts and filepaths are initialized. Also, complex controllers are parsed. Template files are (already) copied to the output folder.

        Args:
            hydamo (instance of HyDAMO): data structure containing the HyDAMO DAMO2.2
            fm (instance of FMModel): model structure set up for Hydrolib-core
            output_path (str or Windows-path, optional): path where the rtc-files are placed. Defaults to None.
            complex_controllers_folder (Path or str, optional): Path where users can put xml-files that will be imported in the RTC-model. Defaults to None.
            rtc_timestep (Union[int, float], optional): Time step of the RTC model. Defaults to 60 seconds.
        """
        self.hydamo = hydamo

        self.time_settings = {
            "start": pd.to_datetime(fm.time.refdate, format="%Y%m%d"),
            "end": pd.to_datetime(fm.time.refdate, format="%Y%m%d")
            + pd.to_timedelta(fm.time.tstop, unit="s"),
            "step": rtc_timestep,
        }

        self.pid_controllers = {}
        self.time_controllers = {}

        # set up the output path
        if output_path is None:
            self.output_path = Path(".")

        self.output_path = output_path / "rtc"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # parse user-provided controllers
        if complex_controllers_folder is not None:
            self.complex_controllers = self.parse_complex_controller(
                complex_controllers_folder
            )
        else:
            self.complex_controllers = None

        # copy files from the template RTC-folder
        self.template_dir = Path(os.path.dirname(__file__)) / ".." / "resources" / "RTC"

        generic_files = os.listdir(self.template_dir)
        generic_files = [
            file
            for file in generic_files
            if file.endswith(".xsd") or file.endswith("json")
        ]
        for file in generic_files:
            shutil.copy(self.template_dir / file, self.output_path / file)

    @staticmethod
    @validate_arguments
    def parse_complex_controller(xml_folder: Union[Path, str]) -> dict:
        """Method to parse user-specified 'complex' controllers

        Args:
            xml_folder (Union[Path, str]): Folder where the user located the custom XML files.s

        Returns:
            dict: dict of list with the data in the files. Every key is a RTC-file, including the DIMR-config.
        """
        files = os.listdir(xml_folder)
        files = [file for file in files if file.endswith("xml")]
        savedict = {}
        savedict["dataconfig_import"] = []
        savedict["dataconfig_export"] = []
        savedict["toolsconfig"] = []
        savedict["timeseries"] = []
        savedict["dimr_config"] = []
        for file in files:
            tree = ET.parse(xml_folder / file)
            root = tree.getroot()
            if file == "rtcDataConfig.xml":
                for i in range(len(root[0])):
                    savedict["dataconfig_import"].append(
                        ET.tostring(root[0][i]).decode()
                    )
                for i in range(2, len(root[1])):
                    savedict["dataconfig_export"].append(
                        ET.tostring(root[1][i]).decode()
                    )
            elif file == "rtcToolsConfig.xml":
                for i in range(1, len(root)):
                    savedict["toolsconfig"].append(ET.tostring(root[i][0]).decode())
            elif file == "timeseries_import.xml":
                savedict["timeseries"].append(ET.tostring(root[0]).decode())
            elif file == "dimr_config.xml":
                savedict["dimr_config"].append(root)
        return savedict

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_hydamo(
        self, pid_settings: dict, timeseries: Union[pd.DataFrame, pd.Series]
    ) -> None:
        """Function to convert HyDAMO management data to controller-dictionaries. So far only time- and PID-controllers are implemented. PID settings can be specified globally or per structdure.

        Args:
            pid_settings (dict): RTC settings (for PID controllers) that are not in the HyDAMO format.
            timeseries (pandas.Series): timeseries that are input to timecontrollers.

        Raises:
            ValueError: errors are raised for inconsistent input data.

        """
        for _, management in self.hydamo.management.iterrows():
            # first get the structure ID through the coupled items. It can so far be three different structure types.
            if management.regelmiddelid is not None:
                opening_id = self.hydamo.management_device[
                    self.hydamo.management_device.globalid == management.regelmiddelid
                ].kunstwerkopeningid.values[0]

                weir_id = self.hydamo.opening[
                    self.hydamo.opening.globalid == opening_id
                ].stuwid.values[0]
                weir_code = self.hydamo.weirs[
                    self.hydamo.weirs.globalid == weir_id
                ].code.values[0]
                if weir_code in list(self.hydamo.structures.rweirs_df.id):
                    weir = self.hydamo.structures.rweirs_df[
                        self.hydamo.structures.rweirs_df.id == weir_code
                    ]
                elif weir_code in list(self.hydamo.structures.uweirs_df.id):
                    weir = self.hydamo.structures.uweirs_df[
                        self.hydamo.structures.uweirs.id == weir_code
                    ]
                elif weir_code in list(self.hydamo.structures.orifices_df.id):
                    weir = self.hydamo.structures.orifices_df[
                        self.hydamo.structures.orifices.id == weir_code
                    ]
                else:
                    raise ValueError(
                        f"Management with id {management.id} could not be connnected to a structure."
                    )
                struc_id = weir.id.values[0]
            elif management.pompid is not None:
                logger.info(
                    f"{management.pompid} is a regular pump - controllers for pump capacities are not yet implemented."
                )
                continue
            else:
                raise ValueError(
                    "Only management_devices and pumps can be connected to a management object."
                )
            if management.stuurvariabele == "bovenkant afsluitmiddel":
                steering_variable = "Crest level (s)"
            if management.doelvariabele == "waterstand":
                target_variable = "Water level (op)"
            #  if the ID is not specified separately, use the global settings
            if management.id not in pid_settings:
                settings = pid_settings["global"]
            else:
                settings = pid_settings[management.id]

            if management.typecontroller == "PID":
                self.add_pid_controller(
                    structure_id=struc_id,
                    steering_variable=steering_variable,
                    target_variable=target_variable,
                    pid_settings=settings,
                    setpoint=management.streefwaarde,
                    lower_bound=management.ondergrens,
                    upper_bound=management.bovengrens,
                    observation_location=management.meetlocatieid,
                )

            elif management.typecontroller == "time":
                if timeseries is None:
                    raise ValueError(
                        "No time series were provided for time controllers"
                    )
                else:
                    data = timeseries.loc[:, management.id]
                    self.add_time_controller(
                        structure_id=struc_id,
                        steering_variable=steering_variable,
                        data=data,
                    )
            else:
                raise ValueError(
                    "Only PID and time controllers are implemented at this moment."
                )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_time_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        data: pd.Series = None,
    ) -> None:
        """Functon to add a time controller to a certain structure.

        Args:
            structure_id (str): structure id.
            steering_variable (str): variable that is controlled, usually crest level.
            data (pd.Series): timeseries.
        """
        self.time_controllers[structure_id] = {
            "data": data,
            "steering_variable": steering_variable,
        }

    @validate_arguments
    def add_pid_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        target_variable: str = None,
        pid_settings: dict = None,
        setpoint: Union[float, str] = None,
        lower_bound: Union[float, str] = None,
        upper_bound: Union[float, str] = None,
        observation_location: str = None,
    ) -> None:
        """Function a add PID controller.

        Args:
            structure_id (str): structure iD.
            steering_variable (str): variable to be controlled, usually crest level.
            target_variable (str): target variable (usually water level)
            pid_settings (dict): settings of the controller (ki, kp, kd, max_speed)
            setpoint (Union[float, str]): setpoint value
            lower_bound (Union[float, str]): lowest value to be allowed
            upper_bound (Union[float, str]): highest value to be allowed
            observation_location (str): id of the observation point
        """
        self.pid_controllers[structure_id] = {
            "settings": pid_settings,
            "steering_variable": steering_variable,
            "target_variable": target_variable,
            "setpoint": setpoint,
            "observation_point": observation_location,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    @staticmethod
    @validate_arguments
    def finish_file(xmlroot, configfile, filename: Union[Path, str]) -> None:
        """Method to finish a XML file in the required namespace and format.

        Args:
            xmlroot: Xml Tree
            configfile : Xml file object
            filename (Union[Path, str]): filepath of the file to be written
        """
        configfile.write(filename)
        xml = (
            bytes(
                '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n',
                encoding="utf-8",
            )
            + ET.tostring(xmlroot)
        )
        xml = xml.decode("utf-8")
        with open(filename, "w+") as f:
            f.write(xml)

    def write_xml_v1(self) -> None:
        """Wrapper function to write individual XML files."""
        self.write_runtimeconfig()
        self.write_toolsconfig()
        self.write_dataconfig()
        self.write_timeseries_import()

    # def write_runtimeconfig_2(self):
    #     timing = RtcUserDefinedRuntimeComplexType(tartDate = '2016-01-01 00:00:00', rtc:endDate='2016-01-03 00:00:00', rtc:timeStep='3600')

    def write_runtimeconfig(self) -> None:
        """Function to write RtcRunTimeConfig.xml from the created dictionaries. They are built from empty files in the template directory using the Etree-package."""

        # namespaces for all other xml files
        generalname = "http://www.wldelft.nl/fews"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # parsing xml file to python and get the root of the existing xml file
        configfile = ET.parse(
            os.path.join(self.template_dir, "rtcRuntimeConfig_empty.xml")
        )
        myroot = configfile.getroot()

        # convert date and runtime to required input for runtimeconfig file

        # replace start/stop dates and times in xml file
        for x in myroot.iter(gn_brackets + "startDate"):
            x.set("date", dt.strftime(self.time_settings["start"], format="%Y-%m-%d"))
            x.set("time", dt.strftime(self.time_settings["start"], format="%H:%M:%S"))
        for x in myroot.iter(gn_brackets + "endDate"):
            x.set("date", dt.strftime(self.time_settings["end"], format="%Y-%m-%d"))
            x.set("time", dt.strftime(self.time_settings["end"], format="%H:%M:%S"))
        for x in myroot.iter(gn_brackets + "timeStep"):
            x.set("unit", "second")
            x.set("divider", "1")
            x.set("multiplier", str(int(self.time_settings["step"])))

        # write new xml file
        self.finish_file(myroot, configfile, self.output_path / "rtcRuntimeConfig.xml")

    def write_toolsconfig(self) -> None:
        """Function to write RtcToolsConfig.xml from the created dictionaries. They are built from empty files in the template directory using the Etree-package."""
        generalname = "http://www.wldelft.nl/fews"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # parsing xml file
        configfile = ET.parse(self.template_dir / "rtcToolsConfig_empty.xml")
        myroot = configfile.getroot()

        self.all_controllers = self.time_controllers
        self.all_controllers.update(self.pid_controllers)

        for ikey, key in enumerate(self.all_controllers.keys()):

            controller = self.all_controllers[key]

            a = ET.SubElement(myroot[1], gn_brackets + "rule")
            a.tail = "\n    "
            if ikey == len(self.all_controllers) - 1:
                a.tail = "\n"
            a.text = "\n"
            myroot[1].tail = "\n"
            myroot[1].text = "\n"

            if "settings" in controller.keys():

                settings = controller["settings"]

                # rule type (PID)
                b = ET.SubElement(a, gn_brackets + "pid")
                b.tail = "\n    "
                b.text = "\n        "
                b.set("id", "[PID]" + "Control group " + str(key) + "/PID Rule")

                # standard settings
                c = ET.SubElement(b, gn_brackets + "mode")
                c.text = "PIDVEL"
                c.tail = "\n"

                d = ET.SubElement(b, gn_brackets + "settingMin")
                d.tail = "\n        "
                d.text = str(controller["lower_bound"])

                e = ET.SubElement(b, gn_brackets + "settingMax")
                e.tail = "\n        "
                e.text = str(controller["upper_bound"])

                f = ET.SubElement(b, gn_brackets + "settingMaxSpeed")
                f.tail = "\n        "
                f.text = str(settings["maxspeed"])

                g = ET.SubElement(b, gn_brackets + "kp")
                g.tail = "\n        "
                g.text = str(settings["kp"])

                h = ET.SubElement(b, gn_brackets + "ki")
                h.tail = "\n        "
                h.text = str(settings["ki"])

                i = ET.SubElement(b, gn_brackets + "kd")
                i.tail = "\n        "
                i.text = str(settings["kd"])

                # input
                j = ET.SubElement(b, gn_brackets + "input")
                j.tail = "\n        "
                j.text = "\n          "

                k = ET.SubElement(j, gn_brackets + "x")
                k.tail = "\n          "
                k.text = (
                    "[Input]"
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"]
                )

                l = ET.SubElement(j, gn_brackets + "setpointValue")
                l.tail = "\n        "
                l.text = str(controller["setpoint"])

                # output
                m = ET.SubElement(b, gn_brackets + "output")
                m.tail = "\n      "
                m.text = "\n          "

                o = ET.SubElement(m, gn_brackets + "y")
                o.tail = "\n          "
                o.text = "[Output]" + str(key) + "/" + controller["steering_variable"]

                p = ET.SubElement(m, gn_brackets + "integralPart")
                p.tail = "\n          "
                p.text = "[IP]" + "Control group " + str(key) + "/PID Rule"

                q = ET.SubElement(m, gn_brackets + "differentialPart")
                q.tail = "\n        "
                q.text = "[DP]" + "Control group " + str(key) + "/PID Rule"
            else:
                # rule type (timeabsolujte)
                b = ET.SubElement(a, gn_brackets + "timeAbsolute")
                b.tail = "\n    "
                b.text = "\n        "
                b.set("id", "[TimeRule]" + "Control group " + str(key) + "/Time Rule")

                # input
                c = ET.SubElement(b, gn_brackets + "input")
                c.tail = "\n        "
                c.text = "\n          "

                d = ET.SubElement(c, gn_brackets + "x")
                d.tail = "\n          "
                d.text = "Control group " + str(key) + "/Time Rule"

                e = ET.SubElement(b, gn_brackets + "output")
                e.tail = "\n        "
                e.text = "\n          "

                f = ET.SubElement(e, gn_brackets + "y")
                f.tail = "\n          "
                f.text = "[Output]" + str(key) + "/" + controller["steering_variable"]

        # elements that are parsed ffrom user specified files should be inserted at the right place.
        if self.complex_controllers is not None:
            for ctl in self.complex_controllers["toolsconfig"]:
                if ctl.startswith("<ns0:rule"):
                    myroot[1].append(ET.fromstring(ctl))
                elif ctl.startswith("<ns0:trigger"):
                    # no trigger block present yet
                    if len(myroot) == 2:
                        trigger = ET.Element(gn_brackets + "triggers")
                        trigger.text = ""
                        trigger.tail = "\n"
                        myroot.append(trigger)
                        myroot[2].append(ET.fromstring(ctl))
                else:
                    print("Only rules and triggers allowed for rtctoolsconfig.xml.")
        self.finish_file(myroot, configfile, self.output_path / "rtcToolsConfig.xml")

    def write_dataconfig(self) -> None:
        """Function to write RtcDataConfig.xml from the created dictionaries. They are built from empty files in the template directory using the Etree-package."""
        generalname = "http://www.wldelft.nl/fews"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # Parsing xml file
        configfile = ET.parse(self.template_dir / "rtcDataConfig_empty.xml")
        myroot = configfile.getroot()

        # implementing standard settings import and exportdata
        a0 = ET.SubElement(myroot[1], gn_brackets + "CSVTimeSeriesFile")
        a0.set("decimalSeparator", ".")
        a0.set("delimiter", ",")
        a0.set("adjointOutput", "false")
        a0.tail = "\n    "
        a0.text = ""

        a1 = ET.SubElement(myroot[1], gn_brackets + "PITimeSeriesFile")
        a1.tail = "\n     "
        a1.text = "\n      "

        a2 = ET.SubElement(a1, gn_brackets + "timeSeriesFile")
        a2.text = "timeseries_export.xml"
        a2.tail = "\n      "

        a3 = ET.SubElement(a1, gn_brackets + "useBinFile")
        a3.text = "false"
        a3.tail = "\n    "

        # implementing standard settings import and exportdata
        a4 = ET.SubElement(myroot[0], gn_brackets + "PITimeSeriesFile")
        a4.tail = "\n     "
        a4.text = "\n      "

        a5 = ET.SubElement(a4, gn_brackets + "timeSeriesFile")
        a5.text = "timeseries_import.xml"
        a5.tail = "\n      "

        a6 = ET.SubElement(a4, gn_brackets + "useBinFile")
        a6.text = "false"
        a6.tail = "\n    "

        # weir dependable data
        for ikey, key in enumerate(self.all_controllers.keys()):

            controller = self.all_controllers[key]

            # te importeren data

            if "settings" in controller.keys():
                a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                a.tail = "\n    "
                if ikey == len(self.all_controllers) - 1:
                    a.tail = "\n  "
                a.text = "\n      "
                myroot[0].text = "\n    "
                a.set(
                    "id",
                    "[Input]"
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"],
                )

                b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")
                b.tail = "\n    "
                b.text = "\n        "

                c = ET.SubElement(b, gn_brackets + "elementId")
                c.text = controller["observation_point"]
                c.tail = "\n        "

                d = ET.SubElement(b, gn_brackets + "quantityId")
                d.text = controller["target_variable"]
                d.tail = "\n        "

                e = ET.SubElement(b, gn_brackets + "unit")
                e.text = "m"
                e.tail = "\n      "
            else:
                a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                a.tail = "\n    "
                if ikey == len(self.all_controllers) - 1:
                    a.tail = "\n  "
                a.text = "\n      "
                myroot[0].text = "\n    "
                a.set("id", "Control group " + str(key) + "/Time Rule")
                b = ET.SubElement(a, gn_brackets + "PITimeSeries")
                b.tail = "\n    "
                b.text = "\n        "

                c = ET.SubElement(b, gn_brackets + "locationId")
                c.text = f"[TimeRule]Control group {key}/Time Rule"
                c.tail = "\n        "

                d = ET.SubElement(b, gn_brackets + "parameterId")
                d.text = "TimeSeries"
                d.tail = "\n        "

                e = ET.SubElement(b, gn_brackets + "interpolationOption")
                e.text = "LINEAR"
                e.tail = "\n      "

                e = ET.SubElement(b, gn_brackets + "extrapolationOption")
                e.text = "BLOCK"
                e.tail = "\n      "

            # te exporteren data:
            f = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
            f.tail = "\n    "
            if ikey == len(self.all_controllers) - 1:
                f.tail = "\n    "
            f.text = "\n      "
            myroot[1][1].tail = "\n    "
            f.set("id", "[Output]" + str(key) + "/" + controller["steering_variable"])

            g = ET.SubElement(f, gn_brackets + "OpenMIExchangeItem")
            g.tail = "\n    "
            g.text = "\n        "

            h = ET.SubElement(g, gn_brackets + "elementId")
            h.text = str(key)
            h.tail = "\n        "

            j = ET.SubElement(g, gn_brackets + "quantityId")
            j.text = controller["steering_variable"]
            j.tail = "\n        "

            k = ET.SubElement(g, gn_brackets + "unit")
            k.text = "m"
            k.tail = "\n      "

        for ikey, key in enumerate(self.all_controllers.keys()):
            controller = self.all_controllers[key]

            if "settings" in controller:
                i = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                i.set("id", "[IP]Control group " + str(key) + "/PID Rule")
                i.tail = "\n    "

                j = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                j.set("id", "[DP]Control group " + str(key) + "/PID Rule")
                j.tail = "\n    "
                if ikey == len(self.all_controllers):
                    j.tail = "\n  "

        # the parsed complex controllers should be inserted at the right place
        if self.complex_controllers is not None:
            for ctl in self.complex_controllers["dataconfig_import"]:
                myroot[0].append(ET.fromstring(ctl))
            for ctl in self.complex_controllers["dataconfig_export"]:
                myroot[1].append(ET.fromstring(ctl))
        self.finish_file(myroot, configfile, self.output_path / "rtcDataConfig.xml")

    def write_timeseries_import(self) -> None:
        """Function to write timeseries_import.xml from the created dictionaries. They are built from empty files in the template directory using the Etree-package."""
        generalname = "http://www.wldelft.nl/fews/PI"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # Parsing xml file
        configfile = ET.parse(self.template_dir / "timeseries_import_empty.xml")
        myroot = configfile.getroot()

        for key in self.all_controllers.keys():

            controller = self.all_controllers[key]

            if "settings" not in controller.keys():
                # te importeren data
                dates = pd.to_datetime(controller["data"].index).strftime("%Y-%m-%d")
                times = pd.to_datetime(controller["data"].index).strftime("%H:%M:%S")
                timestep = (
                    pd.to_datetime(controller["data"].index)[1]
                    - pd.to_datetime(controller["data"].index)[0]
                ).total_seconds()
                a = ET.SubElement(myroot, gn_brackets + "series")
                a.text = ""
                a.tail = "\n "
                b = ET.SubElement(a, gn_brackets + "header")
                b.text = ""
                b.tail = "\n "
                c = ET.SubElement(b, gn_brackets + "type")
                c.text = "instantaneous"
                c.tail = "\n"
                d = ET.SubElement(b, gn_brackets + "locationId")
                d.text = f"[TimeRule]Control Group {key}/Time Rule"
                d.tail = "\n"
                e = ET.SubElement(b, gn_brackets + "parameterId")
                e.text = "TimeSeries"
                e.tail = "\n"
                f = ET.SubElement(b, gn_brackets + "timeStep")
                f.attrib = {
                    "unit": "minute",
                    "multiplier": str(int(timestep / 60.0)),
                    "divider": str(1),
                }
                f.tail = "\n"
                g = ET.SubElement(b, gn_brackets + "startDate")
                g.attrib = {"date": dates[0], "time": times[0]}
                g.tail = "\n"
                h = ET.SubElement(b, gn_brackets + "endDate")
                h.attrib = {"date": dates[-1], "time": times[-1]}
                h.tail = "\n"
                i = ET.SubElement(b, gn_brackets + "missVal")
                i.text = "-999.0"
                i.tail = "\n"
                j = ET.SubElement(b, gn_brackets + "stationName")
                j.text = ""
                j.tail = "\n"
                for i in range(len(controller["data"])):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": dates[i],
                        "time": times[i],
                        "value": str(controller["data"].values[i]),
                    }
                    k.tail = "\n"

        self.finish_file(myroot, configfile, self.output_path / "timeseries_import.xml")
