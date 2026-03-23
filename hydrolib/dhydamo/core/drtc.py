import copy
import logging
import os
import shutil
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path

import pandas as pd
from pydantic.v1 import ConfigDict, validate_arguments

from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.hydamo import HyDAMO

logger = logging.getLogger(__name__)

TIMESERIES_IMPORT_XML = "timeseries_import.xml"
RTC_DATA_CONFIG_XML = "rtcDataConfig.xml"
RTC_TOOLS_CONFIG_XML = "rtcToolsConfig.xml"
STATE_IMPORT_XML = "state_import.xml"
INPUT_PREFIX = "[Input]"
OUTPUT_PREFIX = "[Output]"


@dataclass
class DRTCStructure:
    "Internal dataclass for flow structures referenced in complex controllers"
    struct_type: str
    struct_name: str
    struct_property: str

class DRTCModel:
    """Main class to generate RTC-module files."""

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        hydamo: HyDAMO,
        fm: FMModel,
        output_path: str | Path = None,
        rtc_onlytimeseries: bool = False,
        rtc_timeseriesdata: pd.DataFrame=None,
        complex_controllers_folder: list[str | Path] | str | Path = None,
        id_limit_complex_controllers: list[str] | None = None,
        rtc_timestep: int | float = 60,

    ) -> None:
        """Initialize the DRTCModel.

        Internal controller dictionaries and output paths are initialized, optional
        complex-controller XML is parsed, and RTC template files are copied to the
        output folder.

        Args:
            hydamo (instance of HyDAMO): data structure containing the HyDAMO DAMO2.2
            fm (instance of FMModel): model structure setup for Hydrolib-core
            output_path (str or Path, optional): base path where an `rtc` subfolder
                is created for generated RTC files. Defaults to the current working
                directory.
            rtc_onlytimeseries (bool): if True, build RTC control from
                `rtc_timeseriesdata` only. If True, `complex_controllers_folder` is
                ignored. Defaults to False.
            rtc_timeseriesdata (pd.DataFrame, optional): time series data used when
                `rtc_onlytimeseries=True`. Column names are expected to match
                structure IDs. Defaults to None.
            complex_controllers_folder (list[Path or str] or Path or str, optional):
                folder(s) with custom RTC XML files to import when
                `rtc_onlytimeseries=False`. Defaults to None.
            id_limit_complex_controllers (list[str], optional): whitelist of IDs that
                may be coupled to complex controller logic. Required when
                `complex_controllers_folder` is provided. An empty list means no IDs
                are allowed.
            rtc_timestep (Union[int, float], optional): Time step of the RTC model.
                Defaults to 60 seconds.
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
        base_output_path = Path(".") if output_path is None else Path(output_path)
        self.output_path = base_output_path / "rtc"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save object id by type
        self.struct_ids_by_type = DRTCModel._get_struct_ids_by_type(self.hydamo)

        # parse user-provided controllers
        self.complex_controllers = None
        self.cc_structs = None
        self.cc_ids = None
        self.cc_id_limit = None
        if rtc_onlytimeseries and complex_controllers_folder is not None:
            # User supplied controllers in timeseries_only mode, emit warning
            logger.warning(
                "`complex_controllers_folder` is ignored because `rtc_onlytimeseries=True`. "
                "Set `rtc_onlytimeseries=False` to enable complex controllers."
            )
        elif not rtc_onlytimeseries and complex_controllers_folder is not None:
            if id_limit_complex_controllers is None:
                # When complex_controllers_folder is supplied, the whitelist
                # needs to be supplied as well
                raise SyntaxError(
                    "Missing required `id_limit_complex_controllers` while "
                    "`complex_controllers_folder` is provided. Supply a list of "
                    "allowed IDs to couple to complex controller logic."
                )

            # Discover all complex controller related structures and id's
            self.cc_structs, self.cc_ids = self._load_complex_controller_structs(
                complex_controllers_folder,
                self.struct_ids_by_type,
                log_validation=True,
            )
            logger.info(
                "Found %d complex controller structures referenced in XML: %s",
                len(self.cc_structs),
                self.cc_ids,
            )

            # Save whitelist of allowed controller ids.
            self.cc_id_limit = set(id_limit_complex_controllers)
            if len(self.cc_id_limit) == 0:
                logger.warning(
                    "`id_limit_complex_controllers` is empty. No IDs are allowed, "
                    "so all complex controller references will be filtered out."
                )
            else:
                logger.info(
                    "Applying complex controller ID filter with %d allowed IDs: %s",
                    len(self.cc_id_limit),
                    self.cc_id_limit,
                )

            # Load complex controllers
            self.complex_controllers = self._load_complex_controllers(complex_controllers_folder)

        # copy files from the template RTC-folder
        self.template_dir = Path(__file__).resolve().parent / ".." / "resources" / "RTC"

        generic_files = [p for p in self.template_dir.iterdir() if p.suffix in {".xsd", ".json"}]
        for filepath in generic_files:
            shutil.copy(filepath, self.output_path / filepath.name)

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
    def allow_struct(self, cc_id: str, allow_observations: bool = False) -> bool:
        """Return whether a structure id is allowed by the complex-controller filter."""
        if allow_observations and cc_id in self.struct_ids_by_type["observations"]:
            return True

        if self.cc_ids is None or self.cc_id_limit is None:
            return True

        # The structure should exist in the HyDAMO model and be allowed
        present_and_allowed = cc_id in self.cc_ids and cc_id in self.cc_id_limit
        return present_and_allowed

    @validate_arguments
    def check_timeseries(self, timeseries):
        hydamo_controllers = self.hydamo.management[~self.hydamo.management.regelmiddelid.isna()].regelmiddelid
        for controller in hydamo_controllers:
            mandev = self.hydamo.management_device[self.hydamo.management_device.globalid ==controller]
            if ~mandev.kunstwerkopeningid.isna().to_numpy()[0]:
                ko = self.hydamo.opening[self.hydamo.opening.globalid ==mandev.kunstwerkopeningid.to_numpy()[0]]
                weir = self.hydamo.weirs[self.hydamo.weirs.globalid ==ko.stuwid.to_numpy()[0]].code.to_numpy()[0]
                if weir not in timeseries.columns:
                    logger.warning(f'For {weir} a controller is defined in hydamo.management, but no timeseries is provided for it.')
            elif ~mandev.duikersifonhevelid.isna().to_numpy()[0]:
                dsh = self.hydamo.culvert[self.hydamo.culvert.globalid ==mandev.duikersifonhevelid.to_numpy()[0]].code
                if dsh not in timeseries.columns:
                    logger.warning(f'For {dsh} a controller is defined in hydamo.management, but no timeseries is provided for it.')
            else:
                logger.warning(f'{mandev.code} is not associated with a management_device or culvert.')
        hydamo_pumps = self.hydamo.management[~self.hydamo.management.pompid.isna()].pompid
        for pump in hydamo_pumps:
            pmp = self.hydamo.pumps[self.hydamo.pumps.globalid ==pump].code.to_numpy()[0]
            if pmp not in timeseries.columns:
                logger.warning(f'For {pmp} a controller is defined in hydamo.management, but no timeseries is provided for it.')

    @validate_arguments
    def parse_complex_controller(
        self, xml_folder: Path | str
    ) -> dict[str, list[str | ET.Element]]:
        """Method to parse user-specified 'complex' controllers

        Args:
            xml_folder (Union[Path, str]): Folder where the user located the custom XML files

        Returns:
            dict: dict of list with the data in the files. Every key is a RTC-file, including the DIMR-config.
        """
        files = [p for p in Path(xml_folder).iterdir() if p.suffix == ".xml"]
        savedict = {
            "dataconfig_import": [],
            "dataconfig_export": [],
            "toolsconfig_rules": [],
            "toolsconfig_triggers": [],
            "timeseries": [],
            "state": [],
            "dimr_config": [],
        }

        handlers = {
            RTC_DATA_CONFIG_XML: self._parse_cc_rtc_dataconfig,
            RTC_TOOLS_CONFIG_XML: self._parse_cc_rtc_toolsconfig,
            TIMESERIES_IMPORT_XML: self._parse_cc_timeseries,
            STATE_IMPORT_XML: self._parse_cc_state,
            "dimr_config.xml": self._parse_cc_dimr_config,
        }

        for filepath in files:
            handler = handlers.get(filepath.name)
            if handler is None:
                continue
            root = ET.parse(filepath).getroot()
            savedict = handler(root, savedict)

        return savedict

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_cc_rtc_dataconfig(
        self, root: ET.Element, savedict: dict[str, list[str | ET.Element]]
    ) -> dict[str, list[str | ET.Element]]:
        children = self._parse_unique_children(root)
        import_series = children.get("importSeries")
        if import_series is not None:
            for num, el in enumerate(import_series):
                xml_text = ET.tostring(el).decode()
                if "PITimeSeries" in xml_text and num == 0:
                    continue
                allow, el_text = self._parse_dataconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_DATA_CONFIG_XML}: Skipped importSeries item for elementId '%s' (not allowed by complex controller filter).",
                        el_text,
                    )
                    continue
                savedict["dataconfig_import"].append(xml_text)

        export_series = children.get("exportSeries")
        if export_series is not None:
            for el in export_series:
                xml_text = ET.tostring(el).decode()
                if "PITimeSeries" in xml_text or "CSVTimeSeries" in xml_text:
                    continue
                allow, el_text = self._parse_dataconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_DATA_CONFIG_XML}: Skipped exportSeries item for elementId '%s' (not allowed by complex controller filter).",
                        el_text,
                    )
                    continue
                savedict["dataconfig_export"].append(xml_text)

        return savedict

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_cc_rtc_toolsconfig(
        self, root: ET.Element, savedict: dict[str, list[str | ET.Element]]
    ) -> dict[str, list[str | ET.Element]]:
        children = self._parse_unique_children(root)

        rules = children.get("rules")
        if rules is not None:
            for el in rules:
                allow, el_text = self._parse_toolsconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_TOOLS_CONFIG_XML}: Skipped rule element '%s' (not allowed by complex controller filter).",
                        el_text,
                    )
                    continue
                savedict["toolsconfig_rules"].append(ET.tostring(el).decode())

        triggers = children.get("triggers")
        if triggers is not None:
            for el in triggers:
                allow, el_text = self._parse_toolsconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_TOOLS_CONFIG_XML}: Skipped trigger element '%s' (not allowed by complex controller filter).",
                        el_text,
                    )
                    continue
                savedict["toolsconfig_triggers"].append(ET.tostring(el).decode())

        return savedict

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_cc_timeseries(
        self, root: ET.Element, savedict: dict[str, list[str | ET.Element]]
    ) -> dict[str, list[str | ET.Element]]:
        for el in root:
            savedict["timeseries"].append(ET.tostring(el).decode())

        return savedict

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_cc_state(
        self, root: ET.Element, savedict: dict[str, list[str | ET.Element]]
    ) -> dict[str, list[str | ET.Element]]:
        for el in root[0]:
            savedict["state"].append(ET.tostring(el).decode())

        return savedict

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_cc_dimr_config(
        self, root: ET.Element, savedict: dict[str, list[str | ET.Element]]
    ) -> dict[str, list[str | ET.Element]]:
        red_root = copy.deepcopy(root)
        for coupler_name, coupler_target in (
            ("rtc_to_flow", "targetName"),
            ("flow_to_rtc", "sourceName"),
        ):
            self._filter_dimr_coupler_items(red_root, coupler_name, coupler_target)
        savedict["dimr_config"].append(red_root)

        return savedict

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _filter_dimr_coupler_items(
        self, red_root: ET.Element, coupler_name: str, coupler_target: str
    ) -> None:
        for coupler in list(red_root):
            if coupler.attrib.get("name") != coupler_name:
                continue
            for sub_el in list(coupler):
                target = sub_el.find(".//{*}" + coupler_target)
                allow, el_text = self._parse_dimr_item(target)
                if allow:
                    continue
                logger.info(
                    "dimr_config.xml: Skipped %s element with '%s' '%s' (not allowed by complex controller filter).",
                    coupler_name,
                    coupler_target,
                    el_text,
                )
                coupler.remove(sub_el)

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def find_complex_controller_ids(
        complex_controllers_folder: list[str | Path] | str | Path,
        hydamo: HyDAMO,
    ) -> set[str]:
        # Do not log validation warnings in this method
        struct_ids_by_type = DRTCModel._get_struct_ids_by_type(hydamo)
        cc_structs, _ = DRTCModel._load_complex_controller_structs(complex_controllers_folder, struct_ids_by_type, log_validation=False)
        # Do not return observation point IDs in this method
        cc_ids = set([cc.struct_name for cc in cc_structs if cc.struct_type != "observations"])

        return cc_ids


    @staticmethod
    @validate_arguments
    def _load_complex_controller_structs(
        complex_controllers_folder: list[str | Path] | str | Path,
        struct_ids_by_type: dict[str, set[str]],
        log_validation: bool = True,
    ) -> tuple[list[DRTCStructure], set[str]]:
        folders = DRTCModel._as_folder_list(complex_controllers_folder)
        cc_structs = DRTCModel._collect_complex_controller_structs(folders)
        cc_structs, cc_ids = DRTCModel._deduplicate_complex_controller_structs(cc_structs)
        cc_structs, cc_ids = DRTCModel._validate_complex_controller_structs(cc_structs, struct_ids_by_type, log_validation)

        return cc_structs, cc_ids

    @staticmethod
    @validate_arguments
    def _as_folder_list(
        complex_controllers_folder: list[str | Path] | str | Path
    ) -> list[str | Path]:
        if isinstance(complex_controllers_folder, list):
            return complex_controllers_folder
        return [complex_controllers_folder]

    @staticmethod
    @validate_arguments
    def _collect_complex_controller_structs(
        folders: list[str | Path]
    ) -> list[DRTCStructure]:
        # Find complex controller structs and referred observation points.
        complex_controller_structs = []
        for folder in folders:
            for filepath in Path(folder).iterdir():
                if filepath.name != "dimr_config.xml":
                    continue
                root = ET.parse(filepath).getroot()
                complex_controller_structs.extend(DRTCModel._parse_referenced_structures(root))
        return complex_controller_structs

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _deduplicate_complex_controller_structs(
        complex_controller_structs: list[DRTCStructure]
    ) -> tuple[list[DRTCStructure], set[str]]:
        # Observation points can be referenced multiple times, but we keep one entry.
        # Other structures can only be defined once.
        duplicates = []
        unique_structs = {}
        for fs in complex_controller_structs:
            if fs.struct_name not in unique_structs:
                unique_structs[fs.struct_name] = fs
                continue
            if fs.struct_type != "observations":
                duplicates.append(fs.struct_name)

        if duplicates:
            msg = f"Duplicate complex controller ids found: {duplicates}"
            logger.error(msg)
            raise ValueError(msg)

        deduplicated_structs = list(unique_structs.values())
        complex_controller_ids = set(unique_structs.keys())
        return deduplicated_structs, complex_controller_ids

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _validate_complex_controller_structs(
        complex_controller_structs: list[DRTCStructure],
        struct_ids_by_type: dict[str, set[str]],
        log_validation: bool
    ) -> tuple[list[DRTCStructure], set[str]]:
        validated_cc_structs = []
        for fs in complex_controller_structs:
            if fs.struct_type not in struct_ids_by_type or fs.struct_name not in struct_ids_by_type[fs.struct_type]:
                msg = f"Complex controller structure not found in HyDAMO, will not be used: {fs.struct_type}/{fs.struct_name}"
                if log_validation:
                    logger.warning(msg)
            else:
                validated_cc_structs.append(fs)
        validated_cc_ids = set([fs.struct_name for fs in validated_cc_structs])

        return validated_cc_structs, validated_cc_ids

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _get_struct_ids_by_type(hydamo: HyDAMO) -> dict[str, set[str]]:
        return {
            "observations": set(hydamo.observationpoints.observation_points.get("name", [])),
            "weirs": set(hydamo.structures.rweirs_df.get("id", []))
            | set(hydamo.structures.uweirs_df.get("id", [])),
            "orifices": set(hydamo.structures.orifices_df.get("id", [])),
            "pumps": set(hydamo.structures.pumps_df.get("id", [])),
            "generalstructures": set(hydamo.structures.generalstructures_df.get("id", [])),
            "culverts": set(hydamo.structures.culverts_df.get("id", [])),
            "bridges": set(hydamo.structures.bridges_df.get("id", [])),
        }

    @validate_arguments
    def _load_complex_controllers(
        self, complex_controllers_folder: list[str | Path] | str | Path,
    ) -> dict[str, list[str | ET.Element]]:
        """Normalize input folders, merge parsed controllers, and validate unique IDs."""
        if isinstance(complex_controllers_folder, list):
            complex_controllers = {}
            for folder in complex_controllers_folder:
                # Merge controller XML fragments from multiple folders.
                controllers = self.parse_complex_controller(Path(folder))
                for key, items in controllers.items():
                    # Append controller XML fragments across folders by section key.
                    complex_controllers.setdefault(key, []).extend(items)
        else:
            complex_controllers = self.parse_complex_controller(Path(complex_controllers_folder))

        # Keep a single merged DIMR root so downstream writers can consume index 0.
        complex_controllers["dimr_config"] = self._merge_dimr_config_roots(
            complex_controllers.get("dimr_config", [])
        )

        return complex_controllers

    @staticmethod
    def get_item_pair(item: ET.Element) -> tuple[str, str] | None:
        source = item.find(".//{*}sourceName")
        target = item.find(".//{*}targetName")
        if source is None or target is None or source.text is None or target.text is None:
            return None
        return source.text, target.text

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _merge_dimr_config_roots(dimr_roots: list[ET.Element]) -> list[ET.Element]:
        """Merge multiple dimr_config roots into one by combining coupler items."""
        if len(dimr_roots) <= 1:
            return dimr_roots

        # Use the first root as canonical structure/template for the merged result.
        # Remove all couplers first
        merged_root = copy.deepcopy(dimr_roots[0])
        for coupler in merged_root.findall("./{*}coupler"):
            merged_root.remove(coupler)

        # Merge couplers and items
        seen_items_by_coupler = {}
        for root in dimr_roots:
            for coupler in root.findall("./{*}coupler"):
                coupler_name = coupler.attrib.get("name")
                if coupler_name is None:
                    continue

                if coupler_name not in seen_items_by_coupler:
                    # Add this coupler without items
                    mc = copy.deepcopy(coupler)
                    for item in mc.findall("./{*}item"):
                        mc.remove(item)
                    merged_root.append(mc)

                    # Initialize tracking reference
                    seen_items_by_coupler[coupler_name] = {
                        "reference": mc,
                        "items": set(),
                    }

                # Only add unseen coupler items
                for item in coupler.findall("./{*}item"):
                    mitem = copy.deepcopy(item)
                    pair = DRTCModel.get_item_pair(mitem)
                    if pair is None:
                        continue

                    if pair not in seen_items_by_coupler[coupler_name]["items"]:
                        seen_items_by_coupler[coupler_name]["items"].add(pair)
                        seen_items_by_coupler[coupler_name]["reference"].append(mitem)

        return [merged_root]

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_dataconfig_item(self, el: ET.Element) -> tuple[bool, str | None]:
        allow = True
        el_text = None

        # Check if this is a complex controller but not in the whitelist
        # Always allow observation points
        el_id = el.find(".//{*}elementId")
        if el_id is not None:
            el_text = el_id.text
            allow = self.allow_struct(el_text, allow_observations=True)

        return allow, el_text

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_toolsconfig_item(self, el: ET.Element) -> tuple[bool, str | None]:
        allow = True
        el_firstchild_text = None

        el_firstchild = next(iter(el), None)
        el_tags = []
        for tag in ["input", "output", "trigger", "condition"]:
            el_tags += el.findall(".//{*}" + tag)
        for el_tag in el_tags:
            for child in el_tag:
                if child.text.startswith(INPUT_PREFIX) or child.text.startswith(OUTPUT_PREFIX):
                    child_text = child.text.replace(INPUT_PREFIX, "").replace(OUTPUT_PREFIX, "")
                    child_text = child_text.split("/")[0]

                    # Check if this is a complex controller but not in the whitelist
                    # Always allow observation points
                    if allow:
                        allow = self.allow_struct(child_text, allow_observations=True)
                        el_firstchild_text = el_firstchild.get("id")

        return allow, el_firstchild_text


    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_dimr_item(self, target: ET.Element | None) -> tuple[bool, str | None]:
        allow = True
        el_text = None
        if target is None or not target.text:
            return allow, el_text

        parts = target.text.split("/")
        if len(parts) < 3:
            return allow, target.text

        struct_type, struct_id, _ = parts

        # Check if this is a complex controller but not in the whitelist
        # Always allow observation points
        if (
            struct_type != "observations"
            and self.cc_ids is not None
            and self.cc_id_limit is not None
            and struct_id in self.cc_ids
            and struct_id not in self.cc_id_limit
        ):
            allow = False
            el_text = target.text

        return allow, el_text


    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _parse_referenced_structures(root: ET.Element) -> list[DRTCStructure]:
        structures = []
        rtc_to_flow = root.findall(".//{*}coupler[@name='rtc_to_flow']/{*}item/{*}targetName")
        flow_to_rtc = root.findall(".//{*}coupler[@name='flow_to_rtc']/{*}item/{*}sourceName")
        for item in rtc_to_flow + flow_to_rtc:
            structures.append(DRTCStructure(*item.text.split("/")))

        return structures

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
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

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _strip_namespace(tag: str) -> str:
        return tag.split("}", 1)[-1] if tag.startswith("{") else tag

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _dataconfig_timeseries_key(el: ET.Element) -> str | None:
        if DRTCModel._strip_namespace(el.tag) != "timeSeries":
            return None
        return el.attrib.get("id")

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _timeseries_series_key(el: ET.Element) -> str | None:
        if DRTCModel._strip_namespace(el.tag) != "series":
            return None
        location = el.find("./{*}header/{*}locationId")
        parameter = el.find("./{*}header/{*}parameterId")
        if location is None or parameter is None:
            return None
        if location.text is None or parameter.text is None:
            return None
        return f"{location.text}|{parameter.text}"

    @staticmethod
    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _state_leaf_key(el: ET.Element) -> str | None:
        if DRTCModel._strip_namespace(el.tag) != "treeVectorLeaf":
            return None
        return el.attrib.get("id")

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def _append_unique_elements(
        self,
        parent: ET.Element,
        elements: list[str | ET.Element],
        key_getter: Callable[[ET.Element], str | None],
        file_label: str,
    ) -> None:
        seen_keys = set()
        seen_raw_xml = set()

        for child in list(parent):
            key = key_getter(child)
            if key is not None:
                seen_keys.add(key)
            else:
                seen_raw_xml.add(ET.tostring(child, encoding="unicode"))

        for item in elements:
            element = ET.fromstring(item) if isinstance(item, str) else copy.deepcopy(item)
            key = key_getter(element)
            if key is not None:
                if key in seen_keys:
                    logger.warning("%s: Skipped writing %s, id already present", file_label, key)
                    continue
                seen_keys.add(key)
                parent.append(element)
                continue

            raw_xml = ET.tostring(element, encoding="unicode")
            if raw_xml in seen_raw_xml:
                logger.warning("%s: Skipped writing duplicate XML fragment", file_label)
                continue
            seen_raw_xml.add(raw_xml)
            parent.append(element)

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def from_hydamo(
        self, pid_settings: dict | None=None, interval_settings: dict | None=None, timeseries: pd.DataFrame | None=None
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
            if not pd.isna(management.regelmiddelid):
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
                struc_id = weir.id.to_numpy()[0]
            elif not pd.isna(management.pompid):
                if not self.hydamo.pumps.empty and management.pompid in list(self.hydamo.pumps.globalid):
                    struc_id = self.hydamo.pumps[self.hydamo.pumps.globalid == management.pompid].code.to_numpy()[0]
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

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
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

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def add_pid_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        target_variable: str = None,
        setpoint: float | str | pd.Series = None,
        lower_bound: float | str = None,
        upper_bound: float | str = None,
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

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def add_interval_controller(
        self,
        structure_id: str = None,
        steering_variable: str = None,
        target_variable: str = None,
        deadband: float | str = None,
        setpoint: float | str | pd.Series = None,
        setting_below: float | str = None,
        setting_above: float | str = None,
        max_speed: float | str = None,
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
    def finish_file(xmlroot, configfile, filename: Path | str) -> None:
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
        with open(filename) as f:
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

        to_remove = []
        for key in self.all_controllers.keys():

            controller = self.all_controllers[key]
            if self.cc_ids is not None and self.cc_id_limit is not None:
                if key in self.cc_ids and key in self.cc_id_limit:
                    logger.warning(f"RtcToolsConfig.xml: Skipped writing {controller['type']} control for {key}, complex controller already present")
                    to_remove.append(key)
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
                    INPUT_PREFIX
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
                o.text = OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]

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
                    INPUT_PREFIX
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
                o.text = OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]

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
                f.text = OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]

        # remove controllers that have complex controllers
        for key in to_remove:
            del self.all_controllers[key]

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

        self.finish_file(myroot, configfile, self.output_path / RTC_TOOLS_CONFIG_XML)


    def write_dataconfig(self) -> None:
        """Function to write RtcDataConfig.xml from the created dictionaries. They are built from empty files in the template directory using the Etree-package."""
        generalname = "http://www.wldelft.nl/fews"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"
        m3unit = "m^3/s"
        munit = "m"

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
            a5.text = TIMESERIES_IMPORT_XML
            a6 = ET.SubElement(a4, gn_brackets + "useBinFile")
            a6.text = "false"

          # weir dependable data
        for ikey, key in enumerate(self.all_controllers.keys()):

            controller = self.all_controllers[key]
            if self.cc_ids is not None and self.cc_id_limit is not None:
                if key in self.cc_ids and key in self.cc_id_limit:
                    logger.warning(f"{RTC_DATA_CONFIG_XML}: Skipped writing {controller['type']} control for {key}, complex controller already present")
                    continue

            # te importeren data
            if controller['type'] == 'PID':

                input_id = INPUT_PREFIX + controller["observation_point"] + "/" +  controller["target_variable"]

                if myroot[0].find(f".//*[@id='{input_id}']") is None:
                    a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                    a.set("id", input_id)

                    b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")

                    c = ET.SubElement(b, gn_brackets + "elementId")
                    c.text = controller["observation_point"]

                    d = ET.SubElement(b, gn_brackets + "quantityId")
                    d.text = controller["target_variable"]

                    e = ET.SubElement(b, gn_brackets + "unit")
                    e.text = munit if controller['target_variable'] == 'Water level (op)' else m3unit
                   
                else:
                    logger.warning(f"{RTC_DATA_CONFIG_XML}: Skipped writing {input_id}, observation point already present")
                        
                # If a time dependent setpoint is required, add the Time Rule
                if type(controller['setpoint']) is pd.Series:
                    a2 = ET.SubElement(myroot[0], gn_brackets + "timeSeries")

                    a2.set("id", f"[SP]Control group {key}/PID Rule")
                    b2 = ET.SubElement(a2, gn_brackets + "PITimeSeries")

                    c2 = ET.SubElement(b2, gn_brackets + "locationId")
                    c2.text = f"[PID]Control group {key}/PID Rule"

                    d2 = ET.SubElement(b2, gn_brackets + "parameterId")
                    d2.text = "SP"

                    e2 = ET.SubElement(b2, gn_brackets + "interpolationOption")
                    e2.text = controller['interpolation_option']

                    e2 = ET.SubElement(b2, gn_brackets + "extrapolationOption")  
                    e2.text = controller['extrapolation_option'] # Changed from Block: HL

            elif controller['type'] == 'Interval':               

                input_id = INPUT_PREFIX + controller["observation_point"] + "/" +  controller["target_variable"]
                if myroot.find(f".//*[@id='{input_id}']") is None:
                    a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                    
                    a.set("id", input_id)

                    b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")

                    c = ET.SubElement(b, gn_brackets + "elementId")
                    c.text = controller["observation_point"]

                    d = ET.SubElement(b, gn_brackets + "quantityId")
                    d.text = controller["target_variable"]

                    e = ET.SubElement(b, gn_brackets + "unit")
                    e.text = munit if controller['target_variable'] == 'Water level (op)' else m3unit
                else:
                    logger.warning(f"{RTC_DATA_CONFIG_XML}: Skipped writing {input_id}, observation point already present")
                
                if type(controller['setpoint']) is pd.Series:
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
            f.set("id", OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"])

            g = ET.SubElement(f, gn_brackets + "OpenMIExchangeItem")

            h = ET.SubElement(g, gn_brackets + "elementId")
            h.text = str(key)

            j = ET.SubElement(g, gn_brackets + "quantityId")
            j.text = controller["steering_variable"]

            k = ET.SubElement(g, gn_brackets + "unit")
            k.text = m3unit if controller["steering_variable"] == 'Capacity (p)' else munit

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
            self._append_unique_elements(
                parent=myroot[0],
                elements=self.complex_controllers["dataconfig_import"],
                key_getter=self._dataconfig_timeseries_key,
                file_label=RTC_DATA_CONFIG_XML,
            )
            self._append_unique_elements(
                parent=myroot[1],
                elements=self.complex_controllers["dataconfig_export"],
                key_getter=self._dataconfig_timeseries_key,
                file_label=RTC_DATA_CONFIG_XML,
            )

        self.finish_file(myroot, configfile, self.output_path / RTC_DATA_CONFIG_XML)

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
            if self.cc_ids is not None and self.cc_id_limit is not None:
                if key in self.cc_ids and key in self.cc_id_limit:
                    logger.warning(f"{TIMESERIES_IMPORT_XML}: Skipped writing {controller['type']} control for {key}, complex controller already present")
                    continue

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
                        "value": str(controller["data"].to_numpy()[i]),
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
                        "value": str(controller["setpoint"].to_numpy()[i]),
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
                        "value": str(controller["setpoint"].to_numpy()[i]),
                    }

        if self.complex_controllers is not None:
            self._append_unique_elements(
                parent=myroot,
                elements=self.complex_controllers["timeseries"],
                key_getter=self._timeseries_series_key,
                file_label=TIMESERIES_IMPORT_XML,
            )

        self.finish_file(myroot, configfile, self.output_path / TIMESERIES_IMPORT_XML)

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
            if self.cc_ids is not None and self.cc_id_limit is not None:
                if key in self.cc_ids and key in self.cc_id_limit:
                    logger.warning(f"{STATE_IMPORT_XML}: Skipped writing {controller['type']} control for {key}, complex controller already present")
                    continue

            # te importeren data
            a = ET.SubElement(a0, gn_brackets + "treeVectorLeaf")
            a.attrib = {"id": OUTPUT_PREFIX + key + "/" + controller["steering_variable"]}
            b = ET.SubElement(a, gn_brackets + "vector")
            if controller['type'] == 'PID':
                b.text = str(controller["upper_bound"])
            elif controller['type'] == 'Interval':
                b.text = str(max(controller['setting_above'], controller['setting_below'])) # Take the maximum value as a starting value
            else:
                b.text = str(controller["data"].to_numpy()[0])

        # the parsed complex controllers should be inserted at the right place
        if self.complex_controllers is not None:
            self._append_unique_elements(
                parent=myroot[0],
                elements=self.complex_controllers["state"],
                key_getter=self._state_leaf_key,
                file_label=STATE_IMPORT_XML,
            )

        self.finish_file(myroot, configfile, self.output_path / STATE_IMPORT_XML)
