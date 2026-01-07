import logging
import os
import shutil
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import datetime as dt
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import pandas as pd
from pydantic.v1 import validate_arguments

from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.hydamo import HyDAMO

logger = logging.getLogger(__name__)


@dataclass
class DRTCStructure:
    "Internal dataclass for flow structures referenced in complex controllers"
    struct_type: str
    struct_name: str
    struct_id: int
    struct_property: str

class DRTCModel:
    """Main class to generate RTC-module files."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        hydamo: HyDAMO,
        fm: FMModel,
        output_path: Union[str, Path] = None,
        rtc_onlytimeseries: bool = False,
        rtc_timeseriesdata: pd.DataFrame=None,
        complex_controllers_folder: Union[list[Union[str, Path]], str, Path] = None,
        id_limit_complex_controllers: Union[list[str], None] = None,
        rtc_timestep: Union[int, float] = 60,

    ) -> None:
        """Initialization of the DRTCModel class. (empty) lists/dicts and filepaths are initialized. Also, complex controllers are parsed. Template files are (already) copied to the output folder.

        Args:
            hydamo (instance of HyDAMO): data structure containing the HyDAMO DAMO2.2
            fm (instance of FMModel): model structure set up for Hydrolib-core
            output_path (str or Windows-path, optional): path where the rtc-files are placed. Defaults to None.
            rtc_onlytimeseries (bool): defines the rtc mode: only time series or actual controllers. Defaults to False (actual controllers)
            rtc_timeseriesdata (pd.DataFrame): timeseries in case they are used instead of controllers. Defaults to None.
            complex_controllers_folder (list[Path or str] or Path or str, optional): Path where users can put xml-files that will be imported in the RTC-model. Defaults to None.
            id_limit_complex_controllers (list[str], optional): whitelist of id's to couple to the complex controller logic. Defaults to None (all complex controller logic will be imported)
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
        self.interval_controllers = {}

        # set up the output path
        if output_path is None:
            self.output_path = Path(".")

        self.output_path = output_path / "rtc"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # parse user-provided controllers
        self.complex_controllers = None
        self.complex_controller_structs = None
        self.complex_controller_ids = None
        self.id_limit_complex_controllers = None
        if not rtc_onlytimeseries:
            if complex_controllers_folder is not None:
                (
                    self.complex_controllers,
                    self.complex_controller_structs,
                    self.complex_controller_ids,
                    self.id_limit_complex_controllers,
                ) = self._load_complex_controllers(
                    complex_controllers_folder,
                    id_limit_complex_controllers,
                )

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

        if rtc_onlytimeseries:
            for name, data in rtc_timeseriesdata.items():
                if name in hydamo.structures.rweirs_df.id.to_list():
                    steering_var = "Crest level (s)"
                if name in hydamo.structures.orifices_df.id.to_list():
                    steering_var = 'Gate lower edge level (s)'
                if name in hydamo.structures.pumps_df.id.to_list():
                    steering_var = 'Capacity (p)'
                self.add_time_controller(
                    structure_id=name, steering_variable=steering_var, data=data
                )
            self.check_timeseries(rtc_timeseriesdata)
            self.complex_controllers  = None

    @validate_arguments
    def check_timeseries(self, timeseries):
        hydamo_controllers = self.hydamo.management[~self.hydamo.management.regelmiddelid.isnull()].regelmiddelid
        for controller in hydamo_controllers:
            mandev = self.hydamo.management_device[self.hydamo.management_device.globalid ==controller]
            if ~mandev.kunstwerkopeningid.isnull().values[0]:
                ko = self.hydamo.opening[self.hydamo.opening.globalid ==mandev.kunstwerkopeningid.values[0]]
                weir = self.hydamo.weirs[self.hydamo.weirs.globalid ==ko.stuwid.values[0]].code.values[0]
                if weir not in timeseries.columns:
                    logger.warning(f'For {weir} a controller is defined in hydamo.management, but no timeseries is provided for it.')
            elif ~mandev.duikersifonhevelid.isnull().values[0]:
                dsh = self.hydamo.culvert[self.hydamo.culvert.globalid ==mandev.duikersifonhevelid.values[0]].code
                if dsh not in timeseries.columns:
                    logger.warning(f'For {dsh} a controller is defined in hydamo.management, but no timeseries is provided for it.')
            else:
                logger.warning(f'{mandev.code} is not associated with a management_device or culvert.')
        hydamo_pumps = self.hydamo.management[~self.hydamo.management.pompid.isnull()].pompid
        for pump in hydamo_pumps:
            pmp = self.hydamo.pumps[self.hydamo.pumps.globalid ==pump].code.values[0]
            if pmp not in timeseries.columns:
                logger.warning(f'For {pmp} a controller is defined in hydamo.management, but no timeseries is provided for it.')

    @staticmethod
    @validate_arguments
    def parse_complex_controller(xml_folder: Union[Path, str]) -> tuple[dict[str, ET.Element], list[DRTCStructure]]:
        """Method to parse user-specified 'complex' controllers

        Args:
            xml_folder (Union[Path, str]): Folder where the user located the custom XML files.s

        Returns:
            dict: dict of list with the data in the files. Every key is a RTC-file, including the DIMR-config.
        """
        files = os.listdir(xml_folder)
        files = [file for file in files if file.endswith("xml")]
        flow_structures = None
        savedict = {}
        savedict["dataconfig_import"] = []
        savedict["dataconfig_export"] = []
        savedict["toolsconfig_rules"] = []
        savedict["toolsconfig_triggers"] = []
        savedict["timeseries"] = []
        savedict["state"] = []
        savedict["dimr_config"] = []
        for file in files:
            tree = ET.parse(xml_folder / file)
            root = tree.getroot()
            if file == "rtcDataConfig.xml":
                children = DRTCModel._parse_unique_children(root)
                if "importSeries" in children:
                    for num, el in enumerate(children["importSeries"]):#[1:]:
                        if 'PITimeSeries' in ET.tostring(el).decode() and num == 0:
                            continue
                        savedict["dataconfig_import"].append(ET.tostring(el).decode())
                if "exportSeries" in children:
                    for el in children["exportSeries"]:#[2:]:
                        if ('PITimeSeries' not in ET.tostring(el).decode()) and ('CSVTimeSeries' not in ET.tostring(el).decode()):
                            savedict["dataconfig_export"].append(ET.tostring(el).decode())
            elif file == "rtcToolsConfig.xml":
                children = DRTCModel._parse_unique_children(root)
                if "rules" in children:
                    for el in children["rules"]:
                        savedict["toolsconfig_rules"].append(ET.tostring(el).decode())
                if "triggers" in children:
                    for el in children["triggers"]:
                        savedict["toolsconfig_triggers"].append(ET.tostring(el).decode())
            elif file == "timeseries_import.xml":
                for el in root:
                    savedict["timeseries"].append(ET.tostring(el).decode())
            elif file == "state_import.xml":
                for el in root[0]:
                    savedict["state"].append(ET.tostring(el).decode())
            elif file == "dimr_config.xml":
                savedict["dimr_config"].append(root)
                flow_structures = DRTCModel._parse_referenced_structures(root)

        return savedict, flow_structures

    def _load_complex_controllers(
        self,
        complex_controllers_folder: Union[list[Union[str, Path]], str, Path],
        id_limit_complex_controllers: list[str],
    ) -> tuple[dict[str, list[str]], list[DRTCStructure], set[str], set[str]]:
        """Normalize input folders, merge parsed controllers, and validate unique IDs."""
        if isinstance(complex_controllers_folder, list):
            combined_controllers = None
            combined_structs = []
            for folder in complex_controllers_folder:
                # Merge controller XML fragments from multiple folders.
                controllers, structs = DRTCModel.parse_complex_controller(Path(folder))
                if combined_controllers is None:
                    combined_controllers = {key: list(items) for key, items in controllers.items()}
                else:
                    for key, items in controllers.items():
                        # Append controller XML fragments across folders by section key.
                        combined_controllers.setdefault(key, []).extend(items)
                if structs:
                    combined_structs.extend(structs)
            complex_controllers = combined_controllers or {}
            complex_controller_structs = combined_structs
        else:
            complex_controllers, complex_controller_structs = DRTCModel.parse_complex_controller(
                Path(complex_controllers_folder)
            )
            if complex_controller_structs is None:
                complex_controller_structs = []

        # Validate unique controller ids.
        complex_controller_names = [fs.struct_name for fs in complex_controller_structs]
        name_counts = {}
        for name in complex_controller_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            logger.error("Duplicate complex controller ids found: %s", duplicates)
            raise ValueError("Duplicate complex controller ids found in complex controllers")

        # Build a set of unique controller ids.
        complex_controller_ids = set(complex_controller_names)

        # Validate that referenced structures exist in HyDAMO.
        struct_ids_by_type = {
            "observations": set(self.hydamo.observationpoints.observation_points.get("name", [])),
            "weirs": set(self.hydamo.structures.rweirs_df.get("id", [])) | set(self.hydamo.structures.uweirs_df.get("id", [])),
            "orifices": set(self.hydamo.structures.orifices_df.get("id", [])),
            "pumps": set(self.hydamo.structures.pumps_df.get("id", [])),
            "generalstructures": set(self.hydamo.structures.generalstructures_df.get("id", [])),
            "culverts": set(self.hydamo.structures.culverts_df.get("id", [])),
            "bridges": set(self.hydamo.structures.bridges_df.get("id", [])),
        }
        missing_structs = []
        unknown_types = set()
        for fs in complex_controller_structs:
            if fs.struct_type in struct_ids_by_type:
                if fs.struct_name not in struct_ids_by_type[fs.struct_type]:
                    missing_structs.append(f"{fs.struct_type}/{fs.struct_name}")
            else:
                unknown_types.add(fs.struct_type)
        if unknown_types:
            logger.warning("Skipping HyDAMO complex controller validation for unsupported structure types: %s", sorted(unknown_types))
        if missing_structs:
            logger.error("Complex controller structures not found in HyDAMO: %s", missing_structs)
            raise ValueError(f"Complex controller structures not found in HyDAMO: {missing_structs}")

        # Build whitelist set of allowed controller ids.
        if id_limit_complex_controllers is None:
            raise ValueError("Explicit list of allowed complex controller structures is required (id_limit_complex_controllers)")
        id_limit_complex_controllers = set(id_limit_complex_controllers)

        return (
            complex_controllers,
            complex_controller_structs,
            complex_controller_ids,
            id_limit_complex_controllers,
        )

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _parse_referenced_structures(root: ET.Element) -> list[DRTCStructure]:
        structures = []
        rtc_to_flow = root.findall(".//{*}coupler[@name='rtc_to_flow']/{*}item/{*}targetName")
        flow_to_rtc = root.findall(".//{*}coupler[@name='flow_to_rtc']/{*}item/{*}sourceName")
        for item in rtc_to_flow + flow_to_rtc:
            stype, sname, sprop = item.text.split("/")
            sid = int(sname.split("_")[1])
            structures.append(DRTCStructure(stype, sname, sid, sprop))

        return structures

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _parse_unique_children(root: ET.Element):
        # Store xml sections by tag, this function assumes unique children.
        children = {}
        for child in root:
            # remove namespace from tag
            tag = child.tag
            if tag.startswith("{"):
                tag = tag.split("}")[1]
            # Sanity check: ensure that tag does not exist yet
            if tag in children:
                raise KeyError(f"Duplicate tag '{tag}'")
            children[tag] = child

        return children

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_hydamo(
        self, pid_settings: Optional[dict]=None, interval_settings: Optional[dict]=None, timeseries: Optional[pd.DataFrame]=None
    ) -> None:
        """Function to convert HyDAMO management data to controller-dictionaries. So far only time- and PID-controllers are implemented. PID settings can be specified globally or per structdure.

        Args:
            pid_settings (dict): RTC settings for PID controllers that are not in the HyDAMO format.
            interval_settings (dict): RTC settings for interval controllers that are not in the HyDAMO format.
            timeseries (pandas.Series): timeseries that are input to timecontrollers.

        Raises:
            ValueError: errors are raised for inconsistent input data.

        """
        for _, management in self.hydamo.management.iterrows():
            # first get the structure ID through the coupled items. It can so far be three different structure types.
            if not pd.isnull(management.regelmiddelid):
                weir_code = management.stuwid

                if weir_code in list(self.hydamo.structures.rweirs_df.id):
                    weir = self.hydamo.structures.rweirs_df[
                        self.hydamo.structures.rweirs_df.id == weir_code
                    ]
                elif not self.hydamo.structures.uweirs_df.empty and weir_code in list(self.hydamo.structures.uweirs_df.id):
                    weir = self.hydamo.structures.uweirs_df[
                        self.hydamo.structures.uweirs_df.id == weir_code
                    ]
                elif not self.hydamo.structures.orifices_df.empty and  weir_code in list(self.hydamo.structures.orifices_df.id):
                    weir = self.hydamo.structures.orifices_df[
                        self.hydamo.structures.orifices_df.id == weir_code
                    ]
                else:
                    logger.warning(
                        f"Management for management_device {management.regelmiddelid} could not be connnected to a structure. Skipping it."
                    )
                    continue
                struc_id = weir.id.values[0]
            elif not pd.isnull(management.pompid):
                if not self.hydamo.pumps.empty and management.pompid in list(self.hydamo.pumps.globalid):
                    struc_id = self.hydamo.pumps[self.hydamo.pumps.globalid == management.pompid].code.values[0]
            else:
                raise ValueError(
                    "Only management_devices and pumps can be connected to a management object."
                )
            if management.stuurvariabele == "bovenkant afsluitmiddel":
                steering_variable = "Crest level (s)"
            elif management.stuurvariabele == "hoogte opening":
                steering_variable = "Gate lower edge level (s)"
            elif management.stuurvariabele == "pompdebiet":
                steering_variable = "Capacity (p)"
            else:
                raise ValueError(
                    f"Invalid value for steering variable of {struc_id}: {management.stuurvariabele}."
                )

            if management.doelvariabele == "waterstand":
                target_variable = "Water level (op)"
            elif management.doelvariabele == "debiet":
                target_variable = "Discharge (op)"
            else:
                raise ValueError(
                    f"Invalid value for target variable of {struc_id}: {management.doelvariabele}."
                )

            if management.typecontroller == "PID":
                #  if the ID is not specified separately, use the global settings
                if pid_settings is None:
                    raise ValueError(f'{management.code} contains a PID controller, but no pid_settings are provided. Please do so.')
                if struc_id not in pid_settings:
                    ki = pid_settings["global"]["ki"]
                    kp = pid_settings["global"]["kp"]
                    kd = pid_settings["global"]["kd"]
                    max_speed = pid_settings["global"]["maxspeed"]
                else:
                    ki = pid_settings[struc_id]['ki']
                    kp = pid_settings[struc_id]['kp']
                    kd = pid_settings[struc_id]['kd']
                    max_speed = pid_settings[struc_id]['maxspeed']

                self.add_pid_controller(
                    structure_id=struc_id,
                    steering_variable=steering_variable,
                    target_variable=target_variable,
                    ki=ki,
                    kp=kp,
                    kd=kd,
                    max_speed=max_speed,
                    setpoint=management.streefwaarde,
                    lower_bound=management.ondergrens,
                    upper_bound=management.bovengrens,
                    observation_location=management.meetlocatieid,
                )

            elif management.typecontroller == "interval":
                if interval_settings is None:
                    raise ValueError(f'{management.code} contains an interval controller, but no interval_settings are provided. Please do so.')

                if struc_id not in interval_settings:
                    deadband = interval_settings["global"]["deadband"]
                    max_speed = interval_settings["global"]["maxspeed"]
                else:
                    deadband = interval_settings[struc_id]['deadband']
                    max_speed = interval_settings[struc_id]['maxspeed']

                self.add_interval_controller(
                    structure_id=struc_id,
                    steering_variable=steering_variable,
                    target_variable=target_variable,
                    deadband=deadband,
                    setting_above=management.bovengrens,
                    setting_below=management.ondergrens,
                    max_speed=max_speed,
                    setpoint=management.streefwaarde,
                    observation_location=management.meetlocatieid,
                )

            elif management.typecontroller == "time":
                if timeseries is None:
                     raise ValueError(f'{management.code} contains a time controller, but no time series are provided. Please do so.')
                else:
                    data = timeseries.loc[:, struc_id]
                    self.add_time_controller(
                        structure_id=struc_id,
                        steering_variable=steering_variable,
                        data=data,
                    )
            else:
                logger.warning(
                    f"{management.typecontroller} is not a valid controller type - skipped."
                )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_time_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        data: pd.Series = None,
        interpolation_option: str = 'LINEAR',
        extrapolation_option: str = 'BLOCK',
    ) -> None:
        """Functon to add a time controller to a certain structure.

        Args:
            structure_id (str): structure id.
            steering_variable (str): variable that is controlled, usually crest level.
            data (pd.Series): timeseries.
            interpolation_option (str): interpolation option used.
            extrapolation_option (str): extrapolation option used.
        """
        self.time_controllers[structure_id] = {
            "type": "Time",
            "data": data,
            "steering_variable": steering_variable,
            "interpolation_option": interpolation_option,
            "extrapolation_option": extrapolation_option,
        }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_pid_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        target_variable: str = None,
        setpoint: Union[float, str, pd.Series] = None,
        lower_bound: Union[float, str] = None,
        upper_bound: Union[float, str] = None,
        observation_location: str = None,
        ki: float = 0.001,
        kp: float = 0.0,
        kd: float = 0.0,
        max_speed: float=0.00033,
        interpolation_option: str = 'LINEAR',
        extrapolation_option: str = 'BLOCK',
    ) -> None:
        """Function a add PID controller.

        Args:
            structure_id (str): structure iD.
            steering_variable (str): variable to be controlled, usually crest level.
            target_variable (str): target variable (usually water level)
            setpoint (Union[float, str, pd.Series]): setpoint value or timeseries of setpointvalue
            lower_bound (Union[float, str]): lowest value to be allowed
            upper_bound (Union[float, str]): highest value to be allowed
            observation_location (str): id of the observation point
            ki (float): gain factor ki
            kp (float): faimn factor kp
            kd (float): gain factor kd
            max_speed (float): maximum speed to change target variable
            interpolation_option (str): interpolation option used
            extrapolation_option (str): extrapolation option used
        """
        self.pid_controllers[structure_id] = {
            "type": "PID",
            "steering_variable": steering_variable,
            "target_variable": target_variable,
            "setpoint": setpoint,
            "observation_point": observation_location,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "ki": ki,
            "kp": kp,
            "kd": kd,
            'max_speed': max_speed,
            "interpolation_option": interpolation_option,
            "extrapolation_option": extrapolation_option,
        }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_interval_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        target_variable: str = None,
        deadband: Union[float,str] = None,
        setpoint: Union[float, str, pd.Series] = None,
        setting_below: Union[float, str] = None,
        setting_above: Union[float, str] = None,
        max_speed: Union[float, str] = None,
        observation_location: str = None,
        interpolation_option: str = 'LINEAR',
        extrapolation_option: str = 'BLOCK',
    ) -> None:
        """Function to add an Interval controller.

        Args:
            structure_id (str): structure iD.
            steering_variable (str): variable to be controlled, usually crest level.
            target_variable (str): target variable (usually water level)
            deadband (float): deadband around the setpoint
            setpoint (Union[float, str, pd.Series]): setpoint value (or timeseries of setpointvalue)
            setting_below (Union[float,str]): value of target variable below setpoint
            setting_above (Union[float, str]): value of target variable above setpoint
            max_speed (Union[float,str]): maximum speed to change target variable
            observation_location (str): id of the observation point
            interpolation_option (str): interpolation option used
            extrapolation_option (str): extrapolation option used
        """
        self.interval_controllers[structure_id] = {
            "type": 'Interval',
            "steering_variable": steering_variable,
            "target_variable": target_variable,
            "setpoint": setpoint,
            "observation_point": observation_location,
            "setting_below": setting_below,
            "setting_above": setting_above,
            "max_speed": max_speed,
            "deadband": deadband,
            "interpolation_option": interpolation_option,
            "extrapolation_option": extrapolation_option,
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
        xmlstring = (
            bytes(
                '<?xml version="1.0" encoding="utf-8" standalone="yes" ?>',
                encoding="utf-8",
            )
            + ET.tostring(xmlroot)
        )
        xmlstring = xmlstring.decode("utf-8").replace('\n', '').replace("  ","")
        with open(filename, "w+") as f:
            f.write(xmlstring)
        with open(filename, "r") as f:
            temp = xml.dom.minidom.parseString(f.read())
        with open(filename, "w+") as f:
            f.write(temp.toprettyxml())

    def write_xml_v1(self) -> None:
        """Wrapper function to write individual XML files."""
        self.write_runtimeconfig()
        self.write_toolsconfig()
        self.write_timeseries_import()
        self.write_dataconfig()
        self.write_state_import()


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

        self.all_controllers = self.time_controllers.copy()
        self.all_controllers.update(self.pid_controllers)
        self.all_controllers.update(self.interval_controllers)

        for key in self.all_controllers.keys():

            controller = self.all_controllers[key]
            if key in self.complex_controller_ids and key in self.id_limit_complex_controllers:
                logger.warning(f"Skipped writing {controller['type']} control for {key}, complex controller already present")
                continue

            a = ET.SubElement(myroot[1], gn_brackets + "rule")
            if controller['type'] == "PID":

                # rule type (PID)
                b = ET.SubElement(a, gn_brackets + "pid")
                b.set("id", "[PID]" + "Control group " + str(key) + "/PID Rule")

                # standard settings
                c = ET.SubElement(b, gn_brackets + "mode")
                c.text = "PIDVEL"

                d = ET.SubElement(b, gn_brackets + "settingMin")
                d.text = str(controller["lower_bound"])

                e = ET.SubElement(b, gn_brackets + "settingMax")
                e.text = str(controller["upper_bound"])

                f = ET.SubElement(b, gn_brackets + "settingMaxSpeed")
                f.text = str(controller["max_speed"])

                g = ET.SubElement(b, gn_brackets + "kp")
                g.text = str(controller["kp"])

                h = ET.SubElement(b, gn_brackets + "ki")
                h.text = str(controller["ki"])

                i = ET.SubElement(b, gn_brackets + "kd")
                i.text = str(controller["kd"])

                # input
                j = ET.SubElement(b, gn_brackets + "input")

                k = ET.SubElement(j, gn_brackets + "x")
                k.text = (
                    "[Input]"
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"]
                )

                # If setpoint varies in time
                if isinstance(controller["setpoint"], pd.Series):
                    ll = ET.SubElement(j, gn_brackets + "setpointSeries")
                    ll.text = "[SP]" + "Control group " + str(key) + "/PID Rule"
                # Else fixed setpoint
                else:
                    ll = ET.SubElement(j, gn_brackets + "setpointValue")
                    ll.text = str(controller["setpoint"])

                # output
                m = ET.SubElement(b, gn_brackets + "output")

                o = ET.SubElement(m, gn_brackets + "y")
                o.text = "[Output]" + str(key) + "/" + controller["steering_variable"]

                p = ET.SubElement(m, gn_brackets + "integralPart")
                p.text = "[IP]" + "Control group " + str(key) + "/PID Rule"

                q = ET.SubElement(m, gn_brackets + "differentialPart")
                q.text = "[DP]" + "Control group " + str(key) + "/PID Rule"

            elif controller['type'] == 'Interval':
                # Interval RTC
                # rule type (Interval)
                b = ET.SubElement(a, gn_brackets + "interval")
                b.set("id", "[IntervalRule]" + "Control group " + str(key) + "/Interval Rule")

                # standard settings
                d = ET.SubElement(b, gn_brackets + "settingBelow")
                d.text = str(controller["setting_below"])

                e = ET.SubElement(b, gn_brackets + "settingAbove")
                e.text = str(controller["setting_above"])

                f = ET.SubElement(b, gn_brackets + "settingMaxSpeed")
                f.text = str(controller["max_speed"])

                g = ET.SubElement(b, gn_brackets + "deadbandSetpointAbsolute")
                g.text = str(controller["deadband"])

                # input
                j = ET.SubElement(b, gn_brackets + "input")

                k = ET.SubElement(j, gn_brackets + "x") # leave ref = "EXPLICIT" out for now
                k.text = (
                    "[Input]"
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"]
                )
                # If setpoint varies in time
                ll = ET.SubElement(j, gn_brackets + "setpoint")
                ll.text = "[SP]" + "Control group " + str(key) + "/Interval Rule"

                # output
                m = ET.SubElement(b, gn_brackets + "output")

                o = ET.SubElement(m, gn_brackets + "y")
                o.text = "[Output]" + str(key) + "/" + controller["steering_variable"]

                p = ET.SubElement(m, gn_brackets + "status")
                p.text = "[Status]" + "Control group " + str(key) + "/Interval Rule"
            # Add time rule
            else:
                # rule type (timeabsolute)
                b = ET.SubElement(a, gn_brackets + "timeAbsolute")
                b.set("id", "[TimeRule]" + "Control group " + str(key) + "/Time Rule")

                # input
                c = ET.SubElement(b, gn_brackets + "input")

                d = ET.SubElement(c, gn_brackets + "x")
                d.text = "Control group " + str(key) + "/Time Rule"

                e = ET.SubElement(b, gn_brackets + "output")

                f = ET.SubElement(e, gn_brackets + "y")
                f.text = "[Output]" + str(key) + "/" + controller["steering_variable"]

        # elements that are parsed from user specified files should be inserted at the right place.
        if self.complex_controllers is not None:
            for ctl in self.complex_controllers["toolsconfig_rules"]:
                myroot[1].append(ET.fromstring(ctl))
            for ctl in self.complex_controllers["toolsconfig_triggers"]:
                # no trigger block present yet
                if len(myroot) == 2:
                    trigger = ET.Element(gn_brackets + "triggers")
                    myroot.append(trigger)
                    myroot[2].append(ET.fromstring(ctl))
                else:
                    myroot[2].append(ET.fromstring(ctl))

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

        timeseries_length = len(ET.parse(self.output_path / 'timeseries_import.xml').getroot())


        # implementing standard settings import and exportdata
        a0 = ET.SubElement(myroot[1], gn_brackets + "CSVTimeSeriesFile")
        a0.set("decimalSeparator", ".")
        a0.set("delimiter", ",")
        a0.set("adjointOutput", "false")

        a1 = ET.SubElement(myroot[1], gn_brackets + "PITimeSeriesFile")

        a2 = ET.SubElement(a1, gn_brackets + "timeSeriesFile")
        a2.text = "timeseries_export.xml"

        a3 = ET.SubElement(a1, gn_brackets + "useBinFile")
        a3.text = "false"

       # implementing standard settings import and exportdata
        if timeseries_length > 0:
            # only if timeseries are written to the import
            a4 = ET.SubElement(myroot[0], gn_brackets + "PITimeSeriesFile")
            a5 = ET.SubElement(a4, gn_brackets + "timeSeriesFile")
            a5.text = "timeseries_import.xml"
            a6 = ET.SubElement(a4, gn_brackets + "useBinFile")
            a6.text = "false"

          # weir dependable data
        for ikey, key in enumerate(self.all_controllers.keys()):

            controller = self.all_controllers[key]

            # te importeren data
            if controller['type'] == 'PID':
                a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                a.set(
                    "id",
                    "[Input]"
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"],
                )

                b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")

                c = ET.SubElement(b, gn_brackets + "elementId")
                c.text = controller["observation_point"]

                d = ET.SubElement(b, gn_brackets + "quantityId")
                d.text = controller["target_variable"]

                e = ET.SubElement(b, gn_brackets + "unit")
                e.text = "m"

                # If a time dependent setpoint is required, add the Time Rule
                if type(controller['setpoint']) is pd.Series:
                    a2 = ET.SubElement(myroot[0], gn_brackets + "timeSeries")

                    if controller['type'] =='PID':
                        a2.set("id", f"[SP]Control group {key}/PID Rule")
                        b2 = ET.SubElement(a2, gn_brackets + "PITimeSeries")

                        c2 = ET.SubElement(b2, gn_brackets + "locationId")
                        c2.text = f"[PID]Control group {key}/PID Rule"

                    elif controller['type'] == 'Interval':
                        a2.set("id", "[SP] Interval Rule")
                        b2 = ET.SubElement(a2, gn_brackets + "PITimeSeries")

                        c2 = ET.SubElement(b2, gn_brackets + "locationId")
                        c2.text = f"[IntervalRule]Control group {key}/Interval Rule"

                    d2 = ET.SubElement(b2, gn_brackets + "parameterId")
                    d2.text = "SP"

                    e2 = ET.SubElement(b2, gn_brackets + "interpolationOption")
                    e2.text = controller['interpolation_option']

                    e2 = ET.SubElement(b2, gn_brackets + "extrapolationOption")
                    e2.text = controller['extrapolation_option'] # Changed from Block: HL
            elif controller['type'] == 'Interval':
                a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                a.set(
                    "id",
                    "[Input]"
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"],
                )

                b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")

                c = ET.SubElement(b, gn_brackets + "elementId")
                c.text = controller["observation_point"]

                d = ET.SubElement(b, gn_brackets + "quantityId")
                d.text = controller["target_variable"]

                e = ET.SubElement(b, gn_brackets + "unit")
                e.text = "m"

                a2 = ET.SubElement(myroot[0], gn_brackets + "timeSeries")

                a2.set("id", f"[SP]Control group {key}/Interval Rule")
                b3 = ET.SubElement(a2, gn_brackets + "PITimeSeries")

                c3 = ET.SubElement(b3, gn_brackets + "locationId")
                c3.text = f"[IntervalRule]Control group {key}/Interval Rule"

                d3 = ET.SubElement(b3, gn_brackets + "parameterId")
                d3.text = "SP"

                e3 = ET.SubElement(b3, gn_brackets + "interpolationOption")
                e3.text = controller['interpolation_option']

                f3 = ET.SubElement(b3, gn_brackets + "extrapolationOption")
                f3.text = controller['extrapolation_option'] # Changed from Block: HL

            else:
                a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                a.set("id", "Control group " + str(key) + "/Time Rule")
                b = ET.SubElement(a, gn_brackets + "PITimeSeries")

                c = ET.SubElement(b, gn_brackets + "locationId")
                c.text = f"[TimeRule]Control group {key}/Time Rule"

                d = ET.SubElement(b, gn_brackets + "parameterId")
                d.text = "TimeSeries"

                e = ET.SubElement(b, gn_brackets + "interpolationOption")
                e.text = controller['interpolation_option']

                e = ET.SubElement(b, gn_brackets + "extrapolationOption")
                e.text = controller['extrapolation_option'] # Changed from Block: HL

            # te exporteren data:
            f = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
            f.set("id", "[Output]" + str(key) + "/" + controller["steering_variable"])

            g = ET.SubElement(f, gn_brackets + "OpenMIExchangeItem")

            h = ET.SubElement(g, gn_brackets + "elementId")
            h.text = str(key)

            j = ET.SubElement(g, gn_brackets + "quantityId")
            j.text = controller["steering_variable"]

            k = ET.SubElement(g, gn_brackets + "unit")
            k.text = "m"

        for ikey, key in enumerate(self.all_controllers.keys()):
            controller = self.all_controllers[key]

            if controller['type'] == 'PID':
                i = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                i.set("id", "[IP]Control group " + str(key) + "/PID Rule")

                j = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                j.set("id", "[DP]Control group " + str(key) + "/PID Rule")

            elif controller['type'] == 'Interval': # Change slightly when working with Interval rule
                j = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                j.set("id", "[Status]Control group " + str(key) + "/Interval Rule")

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

            if controller['type'] == 'Time':
                # te importeren data
                dates = pd.to_datetime(controller["data"].index).strftime("%Y-%m-%d")
                times = pd.to_datetime(controller["data"].index).strftime("%H:%M:%S")
                timestep = (
                    pd.to_datetime(controller["data"].index)[1]
                    - pd.to_datetime(controller["data"].index)[0]
                ).total_seconds()
                a = ET.SubElement(myroot, gn_brackets + "series")
                b = ET.SubElement(a, gn_brackets + "header")
                c = ET.SubElement(b, gn_brackets + "type")
                c.text = "instantaneous"
                d = ET.SubElement(b, gn_brackets + "locationId")
                d.text = f"[TimeRule]Control group {key}/Time Rule"
                e = ET.SubElement(b, gn_brackets + "parameterId")
                e.text = "TimeSeries"
                f = ET.SubElement(b, gn_brackets + "timeStep")
                f.attrib = {
                    "unit": "minute",
                    "multiplier": str(int(timestep / 60.0)),
                    "divider": str(1),
                }
                g = ET.SubElement(b, gn_brackets + "startDate")
                g.attrib = {"date": dates[0], "time": times[0]}
                h = ET.SubElement(b, gn_brackets + "endDate")
                h.attrib = {"date": dates[-1], "time": times[-1]}
                i = ET.SubElement(b, gn_brackets + "missVal")
                i.text = "-999.0"
                j = ET.SubElement(b, gn_brackets + "stationName")
                j.text = ""
                for i in range(len(controller["data"])):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": dates[i],
                        "time": times[i],
                        "value": str(controller["data"].values[i]),
                    }
            elif controller['type'] == "Interval":
                if isinstance(controller['setpoint'], float):
                    controller['setpoint'] = pd.Series([controller['setpoint'],controller['setpoint']], index=[self.time_settings['start'],self.time_settings['end']])

                # te importeren data
                dates = pd.to_datetime( controller["setpoint"].index).strftime("%Y-%m-%d")
                times = pd.to_datetime(controller["setpoint"].index).strftime("%H:%M:%S")
                timestep = (pd.to_datetime(f'{dates[1]} {times[1]}') - pd.to_datetime(f'{dates[0]} {times[0]}')).total_seconds()

                a = ET.SubElement(myroot, gn_brackets + "series")
                b = ET.SubElement(a, gn_brackets + "header")
                c = ET.SubElement(b, gn_brackets + "type")
                c.text = "instantaneous"

                d = ET.SubElement(b, gn_brackets + "locationId")
                d.text = f"[IntervalRule]Control group {key}/Interval Rule"

                e = ET.SubElement(b, gn_brackets + "parameterId")
                e.text = "SP"
                f = ET.SubElement(b, gn_brackets + "timeStep")
                f.attrib = {
                    "unit": "minute",
                    "multiplier": str(int(timestep / 60.0)),
                    "divider": str(1),
                }
                g = ET.SubElement(b, gn_brackets + "startDate")
                g.attrib = {"date": dates[0], "time": times[0]}
                h = ET.SubElement(b, gn_brackets + "endDate")
                h.attrib = {"date": dates[-1], "time": times[-1]}
                i = ET.SubElement(b, gn_brackets + "missVal")
                i.text = "-999.0"
                for i in range(len(controller["setpoint"])):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": dates[i],
                        "time": times[i],
                        "value": str(controller["setpoint"].values[i]),
                    }

            # Create a timeseries import if a time-dependent setpoint is used
            elif controller['type'] == 'PID' and isinstance(controller['setpoint'], pd.Series):
                # te importeren data
                dates = pd.to_datetime(controller["setpoint"].index).strftime("%Y-%m-%d")
                times = pd.to_datetime(controller["setpoint"].index).strftime("%H:%M:%S")
                timestep = (
                    pd.to_datetime(controller["setpoint"].index)[1]
                    - pd.to_datetime(controller["setpoint"].index)[0]
                ).total_seconds()
                a = ET.SubElement(myroot, gn_brackets + "series")
                b = ET.SubElement(a, gn_brackets + "header")
                c = ET.SubElement(b, gn_brackets + "type")
                c.text = "instantaneous"

                if controller['type'] =='PID':
                    d = ET.SubElement(b, gn_brackets + "locationId")
                    d.text = f"[PID]Control group {key}/PID Rule"
                elif controller['type'] == 'Interval':
                    d = ET.SubElement(b, gn_brackets + "locationId")
                    d.text = f"[IntervalRule]Control group {key}/Interval Rule"

                e = ET.SubElement(b, gn_brackets + "parameterId")
                e.text = "SP"
                f = ET.SubElement(b, gn_brackets + "timeStep")
                f.attrib = {
                    "unit": "minute",
                    "multiplier": str(int(timestep / 60.0)),
                    "divider": str(1),
                }
                g = ET.SubElement(b, gn_brackets + "startDate")
                g.attrib = {"date": dates[0], "time": times[0]}
                h = ET.SubElement(b, gn_brackets + "endDate")
                h.attrib = {"date": dates[-1], "time": times[-1]}
                i = ET.SubElement(b, gn_brackets + "missVal")
                i.text = "-999.0"
                j = ET.SubElement(b, gn_brackets + "stationName")
                j.text = ""
                for i in range(len(controller["setpoint"])):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": dates[i],
                        "time": times[i],
                        "value": str(controller["setpoint"].values[i]),
                    }

        if self.complex_controllers is not None:
            for ctl in self.complex_controllers["timeseries"]:
                myroot.append(ET.fromstring(ctl))

        self.finish_file(myroot, configfile, self.output_path / "timeseries_import.xml")

    def write_state_import(self) -> None:
        """Function to write state_import.xml from the created dictionaries. They are built from empty files in the template directory using the Etree-package."""
        generalname = "http://www.openda.org"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # Parsing xml file
        configfile = ET.parse(self.template_dir / "state_import_empty.xml")
        myroot = configfile.getroot()

        a0 = ET.SubElement(myroot, gn_brackets + "treeVector")

        for key in self.all_controllers.keys():

            controller = self.all_controllers[key]

            # te importeren data
            a = ET.SubElement(a0, gn_brackets + "treeVectorLeaf")
            a.attrib = {"id": "[Output]" + key + "/" + controller["steering_variable"]}
            b = ET.SubElement(a, gn_brackets + "vector")
            if controller['type'] == 'PID':
                b.text = str(controller["upper_bound"])
            elif controller['type'] == 'Interval':
                b.text = str(max(controller['setting_above'], controller['setting_below'])) # Take the maximum value as a starting value
            else:
                b.text = str(controller["data"].values[0])

        # the parsed complex controllers should be inserted at the right place
        if self.complex_controllers is not None:
            for ctl in self.complex_controllers["state"]:
                myroot[0].append(ET.fromstring(ctl))

        self.finish_file(myroot, configfile, self.output_path / "state_import.xml")
