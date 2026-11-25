"""The D-RTC model: time, PID and interval controllers, complex controllers and their XML files."""

import copy
import logging
import os
import shutil
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring
from pydantic import ConfigDict, Field, validate_call
from typing_extensions import TypedDict

from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.damo_converters.data_model import DamoRelations
from hydrolib.dhydamo.io.netcdfreader import time_index

logger = logging.getLogger(__name__)

TIMESERIES_IMPORT_XML = "timeseries_import.xml"
RTC_DATA_CONFIG_XML = "rtcDataConfig.xml"
RTC_TOOLS_CONFIG_XML = "rtcToolsConfig.xml"
STATE_IMPORT_XML = "state_import.xml"

# RTC steering variables (output of a controller to a structure)
CREST_LEVEL = "Crest level (s)"
GATE_LOWER_EDGE_LEVEL = "Gate lower edge level (s)"
VALVE_OPENING = "Valve opening (s)"
CAPACITY = "Capacity (p)"
INPUT_PREFIX = "[Input]"
OUTPUT_PREFIX = "[Output]"

# Where the setting of a controlled structure is found, per steering variable. The first match is used.
# - D-Flow FM his-file output: (structure type, quantity), read from the variable f"{type}_{quantity}" at
#   the position of the structure in f"{type}_name". These names are defined by D-Flow FM, not by DAMO.
HIS_SETTING_VARIABLES: dict[str, list[tuple[str, str]]] = {
    CREST_LEVEL: [
        ("weirgen", "crest_level"),
        ("orifice", "crest_level"),
        ("uniweir", "crest_level"),
    ],
    GATE_LOWER_EDGE_LEVEL: [
        ("orifice", "gate_lower_edge_level"),
        ("culvert", "gate_lower_edge_level"),
    ],
    VALVE_OPENING: [
        ("culvert", "gate_opening_height"),
    ],
    CAPACITY: [
        ("pump", "capacity"),
    ],
}
# - structure dataframes of HyDAMO.structures: (dataframe attribute, column), matched on the column 'id'.
STRUCTURE_SETTING_COLUMNS: dict[str, list[tuple[str, str]]] = {
    CREST_LEVEL: [
        ("rweirs_df", "crestlevel"),
        ("orifices_df", "crestlevel"),
        ("uweirs_df", "crestlevel"),
    ],
    GATE_LOWER_EDGE_LEVEL: [
        ("orifices_df", "gateloweredgelevel"),
    ],
    VALVE_OPENING: [
        ("culverts_df", "valveopeningheight"),
    ],
    CAPACITY: [
        ("pumps_df", "capacity"),
    ],
}


class ComplexControllers(TypedDict):
    """XML of the complex controllers: fragments per RTC file, and the parsed DIMR config roots.

    pydantic needs ``typing_extensions.TypedDict`` (not ``typing.TypedDict``) on Python < 3.12.
    """

    dataconfig_import: list[str]
    dataconfig_export: list[str]
    toolsconfig_rules: list[str]
    toolsconfig_triggers: list[str]
    timeseries: list[str]
    state: list[str]
    dimr_config: list[ET.Element]


class TimeSettings(TypedDict):
    """Start and end time of the model, and the RTC time step in seconds."""

    start: pd.Timestamp
    end: pd.Timestamp
    step: float


class TimeController(TypedDict):
    """Time controller of a structure, added with ``DRTCModel.add_time_controller``."""

    type: Literal["Time"]
    data: pd.Series
    steering_variable: str
    interpolation_option: str
    extrapolation_option: str


class PIDController(TypedDict):
    """PID controller of a structure, added with ``DRTCModel.add_pid_controller``."""

    type: Literal["PID"]
    steering_variable: str
    target_variable: str
    setpoint: float | str | pd.Series
    observation_point: str
    lower_bound: float | str | None
    upper_bound: float | str | None
    ki: float
    kp: float
    kd: float
    max_speed: float
    interpolation_option: str
    extrapolation_option: str


class IntervalController(TypedDict):
    """Interval controller of a structure, added with ``DRTCModel.add_interval_controller``."""

    type: Literal["Interval"]
    steering_variable: str
    target_variable: str
    setpoint: float | str | pd.Series
    observation_point: str
    setting_below: float | str
    setting_above: float | str
    max_speed: float | str | None
    deadband: float | str | None
    interpolation_option: str
    extrapolation_option: str


@dataclass
class DRTCStructure:
    """A flow structure referenced in a complex controller."""

    struct_type: str
    struct_name: str
    struct_property: str


def _warn_unknown_settings_keys(
    name: str, settings: dict | None, controlled_structures: set[str]
) -> None:
    """Warn once about settings keys that match no controlled structure.

    Such a key is ignored, so a stale or misspelt structure id would otherwise go
    unnoticed and its controller would silently get the ``"global"`` settings.

    Parameters
    ----------
    name : str
        Name of the settings argument, for the message.
    settings : dict or None
        The settings, keyed by structure id and ``"global"``.
    controlled_structures : set of str
        Ids of the structures with this type of controller.
    """
    if settings is None:
        return
    # dict order, not sorted: keys of different types cannot be compared
    unknown = [
        key for key in settings if key != "global" and key not in controlled_structures
    ]
    if unknown:
        logger.warning(
            "%s keys %s match no structure with this type of controller, so they "
            "are ignored. Keys are D-Flow FM structure ids.",
            name,
            unknown,
        )


def _settings_for(name: str, settings: dict, structure_id: str) -> dict:
    """Return the settings of a structure, or the ``"global"`` settings.

    Parameters
    ----------
    name : str
        Name of the settings argument, for the message.
    settings : dict
        The settings, keyed by structure id and ``"global"``.
    structure_id : str
        D-Flow FM id of the controlled structure.

    Returns
    -------
    dict
        The settings under ``structure_id``, otherwise those under ``"global"``.

    Raises
    ------
    ValueError
        If ``settings`` has neither key, e.g. because the structure's key is
        misspelt and there is no ``"global"`` fallback.
    """
    if structure_id in settings:
        return settings[structure_id]
    if "global" in settings:
        return settings["global"]
    raise ValueError(
        f"{name} has no key '{structure_id}' and no 'global' fallback. "
        "Keys are D-Flow FM structure ids."
    )


class DRTCModel:
    """Main class to generate RTC-module files."""

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def __init__(
        self,
        hydamo: HyDAMO,
        fm: FMModel,
        output_path: str | Path | None = None,
        rtc_onlytimeseries: bool = False,
        rtc_timeseriesdata: pd.DataFrame | None = None,
        complex_controllers_folder: list[str | Path] | str | Path | None = None,
        id_limit_complex_controllers: list[str] | None = None,
        rtc_timestep: float = 60,
    ) -> None:
        """Initialize the DRTCModel.

        Internal controller dictionaries and output paths are initialized, optional
        complex-controller XML is parsed, and RTC template files are copied to the
        output folder.

        Parameters
        ----------
        hydamo : HyDAMO
            The HyDAMO data, with the structures and management.
        fm : FMModel
            The FM model; its reference date and times set the RTC period.
        output_path : str or Path, optional
            Folder in which an ``rtc`` subfolder is created for the RTC files. Defaults to
            None: the current working directory.
        rtc_onlytimeseries : bool, optional
            Build the RTC control from ``rtc_timeseriesdata`` only;
            ``complex_controllers_folder`` is then ignored. Defaults to False.
        rtc_timeseriesdata : pd.DataFrame, optional
            Time series used with ``rtc_onlytimeseries``, with the structure ids as
            columns. Defaults to None.
        complex_controllers_folder : list[Path or str], Path or str, optional
            Folder(s) with custom RTC XML files to import, without
            ``rtc_onlytimeseries``. Defaults to None.
        id_limit_complex_controllers : list[str], optional
            Ids that may be coupled to complex controllers. Required with
            ``complex_controllers_folder``; an empty list allows no ids. Defaults to None.
        rtc_timestep : float, optional
            Time step of the RTC model (s). Defaults to 60.
        """
        self.hydamo = hydamo

        self.time_settings: TimeSettings = {
            "start": pd.to_datetime(fm.time.refdate, format="%Y%m%d"),
            "end": pd.to_datetime(fm.time.refdate, format="%Y%m%d")
            + pd.to_timedelta(fm.time.tstop, unit="s"),
            "step": rtc_timestep,
        }

        self.pid_controllers: dict[str, PIDController] = {}
        self.time_controllers: dict[str, TimeController] = {}
        self.interval_controllers: dict[str, IntervalController] = {}

        # initial state: from an earlier model run (set_state_from_file) and/or per structure (set_initial_state)
        self.state_from_file: dict[str, str] | None = None
        self.initial_state: dict[str, float] = {}
        # ids of the time series in rtcDataConfig.xml, set by write_dataconfig
        self._written_dataconfig_ids: set[str] | None = None

        # set up the output path
        base_output_path = Path(".") if output_path is None else Path(output_path)
        self.output_path = base_output_path / "rtc"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save object id by type
        self.struct_ids_by_type = DRTCModel._get_struct_ids_by_type(self.hydamo)

        # parse user-provided controllers
        self.complex_controllers: ComplexControllers | None = None
        self.cc_structs = None
        self.cc_ids = None
        self.cc_referenced_ids = None
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

            # Discover complex-controller references before and after HyDAMO validation.
            #
            # These are intentionally two separate concepts:
            # - cc_referenced_ids: all structure ids mentioned in the user-provided
            #   complex-controller dimr_config.xml files.
            # - cc_ids: only the referenced ids that also exist in the HyDAMO model.
            #
            # The distinction matters when filtering DIMR couplers. A referenced id
            # that is missing from HyDAMO must be removed from the complex-controller
            # XML. If we only kept the validated set, that invalid id would look like
            # an unrelated/non-complex DIMR item and could accidentally pass through.
            (
                self.cc_structs,
                self.cc_ids,
                self.cc_referenced_ids,
            ) = self._load_complex_controller_structs(
                complex_controllers_folder,
                self.struct_ids_by_type,
                log_validation=True,
            )
            logger.info(
                "Complex controllers: found %d unique structure references in dimr_config.xml: %s",
                len(self.cc_referenced_ids),
                self.cc_referenced_ids,
            )
            logger.info(
                "Complex controllers: matched %d references to HyDAMO structures: %s",
                len(self.cc_ids),
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
                    "Complex controllers: applying id_limit_complex_controllers with %d allowed IDs: %s",
                    len(self.cc_id_limit),
                    self.cc_id_limit,
                )

            # Load complex controllers
            self.complex_controllers = self._load_complex_controllers(
                complex_controllers_folder
            )

        # copy files from the template RTC-folder
        self.template_dir = Path(__file__).resolve().parent / ".." / "resources" / "RTC"
        for filepath in self.template_dir.iterdir():
            if filepath.suffix in {".xsd", ".json"}:
                shutil.copy(filepath, self.output_path / filepath.name)

        if rtc_onlytimeseries:
            if rtc_timeseriesdata is None:
                raise ValueError(
                    "`rtc_timeseriesdata` is required when `rtc_onlytimeseries=True`."
                )
            for name, data in rtc_timeseriesdata.items():
                if name in hydamo.structures.rweirs_df.id.to_list():
                    steering_var = CREST_LEVEL
                if name in hydamo.structures.orifices_df.id.to_list():
                    steering_var = GATE_LOWER_EDGE_LEVEL
                if name in hydamo.structures.pumps_df.id.to_list():
                    steering_var = CAPACITY
                self.add_time_controller(
                    structure_id=str(name), steering_variable=steering_var, data=data
                )
            self.check_timeseries(rtc_timeseriesdata)
            self.complex_controllers = None

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def allow_struct(
        self,
        cc_id: str,
        allow_observations: bool = False,
        allow_if_filter_inactive: bool = True,
        allow_if_not_referenced: bool = False,
    ) -> bool:
        """Return whether a structure id is allowed by the complex-controller filter.

        Parameters
        ----------
        cc_id : str
            Structure or observation id to evaluate.
        allow_observations : bool, optional
            Pass through observation ids that exist in the HyDAMO model. Defaults to
            False.
        allow_if_filter_inactive : bool, optional
            Return value when no complex controller filter is active
            (``cc_referenced_ids`` or ``cc_id_limit`` is None). Defaults to True.
        allow_if_not_referenced : bool, optional
            Return value when filtering is active, but ``cc_id`` is not in
            ``cc_referenced_ids``. Defaults to False.

        Returns
        -------
        bool
            Whether the id is allowed.
        """
        allow, _ = self._allow_struct_with_reason(
            cc_id=cc_id,
            allow_observations=allow_observations,
            allow_if_filter_inactive=allow_if_filter_inactive,
            allow_if_not_referenced=allow_if_not_referenced,
        )
        return allow

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _allow_struct_with_reason(
        self,
        cc_id: str,
        allow_observations: bool = False,
        allow_if_filter_inactive: bool = True,
        allow_if_not_referenced: bool = False,
    ) -> tuple[bool, str | None]:
        """Return whether an id is allowed and why it is rejected.

        The reason is only populated for rejected ids, so skip log messages can
        distinguish "not whitelisted" from "referenced in RTC XML but missing
        from HyDAMO".
        """
        if allow_observations and cc_id in self.struct_ids_by_type["observations"]:
            return True, None

        if (
            self.cc_referenced_ids is None
            or self.cc_ids is None
            or self.cc_id_limit is None
        ):
            return allow_if_filter_inactive, (
                None
                if allow_if_filter_inactive
                else "complex-controller filtering is inactive"
            )

        # IDs not mentioned in the DIMR couplers are outside the complex-controller
        # reference set. Some callers, such as DIMR coupler filtering, choose to
        # keep those unrelated items; callers parsing RTC fragments reject them.
        if cc_id not in self.cc_referenced_ids:
            return allow_if_not_referenced, (
                None
                if allow_if_not_referenced
                else "not part of complex-controller references in dimr_config.xml"
            )

        # The id is referenced by complex-controller DIMR XML but was not validated
        # against HyDAMO. It should never be coupled, even when the caller keeps
        # unrelated/non-referenced DIMR items.
        if cc_id not in self.cc_ids:
            return (
                False,
                "referenced by complex-controller XML but missing from HyDAMO",
            )

        if cc_id not in self.cc_id_limit:
            return False, "excluded by id_limit_complex_controllers"

        return True, None

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def check_timeseries(self, timeseries: pd.DataFrame) -> None:
        """Warn for each structure with a controller in ``hydamo.management`` but no time series.

        Parameters
        ----------
        timeseries : pd.DataFrame
            Time series, with the structure ids as columns.
        """
        all_management = self.hydamo.management
        # The resolver raises for rows referencing neither a pump nor a management device.
        hydamo_controllers = all_management[
            all_management[DamoRelations.MANAGEMENT_PUMP.foreign_key].notna()
            | all_management[DamoRelations.MANAGEMENT_DEVICE.foreign_key].notna()
        ]
        for _, management in hydamo_controllers.iterrows():
            resolved = self._resolve_management_structure(management)
            if resolved is None:
                continue
            _, structure_id = resolved
            if structure_id not in timeseries.columns:
                logger.warning(
                    "For %s a controller is defined in hydamo.management, but no timeseries is provided for it.",
                    structure_id,
                )

    def _resolve_management_structure(
        self, management: pd.Series
    ) -> tuple[str, str] | None:
        """Resolve a `hydamo.management` row to the structure it steers.

        Dispatches on whether the row references a pump (`pompid`) or a
        management device (`regelmiddelid`), then, for management devices,
        resolves the underlying weir/orifice (via `_resolve_weir_via_opening`
        and `hydamo.structures.compound_weir_structure_ids`) or culvert (via
        `management_device.duikersifonhevelid`).

        Architecture note: this method, and the weir/orifice resolution it
        delegates to `_resolve_weir_via_opening`, intentionally lives here and
        not in `hydrolib.dhydamo.io.damo_converters`. The converters package
        only normalizes differences between DAMO schema versions (column
        renames, layer name changes) into one canonical HyDAMO column set;
        once `hydamo.management` reaches this method, the DAMO version is
        irrelevant. Mapping a management record to a controllable structure is
        RTC-specific business logic, so it belongs in DRTCModel.

        Parameters
        ----------
        management : pd.Series
            A single row from ``hydamo.management``, in canonical
            (post-``damo_converters``) column form. Must have ``pompid`` and
            ``regelmiddelid``, where exactly one of the two is expected to be non-null.

        Returns
        -------
        tuple[str, str] or None
            ``(structure_type, structure_id)``, where ``structure_type`` is one of
            ``"pump"``, ``"weir"`` or ``"culvert"`` and ``structure_id`` is the id of the
            built D-FlowFM structure. None when the referenced pump, management device,
            weir or culvert cannot be resolved or is not present in the built D-FlowFM
            structures; a warning is logged in that case.

        Raises
        ------
        ValueError
            If neither ``pompid`` nor ``regelmiddelid`` is set on ``management``, since a
            management record must reference one of the two.
        """
        management_pump = DamoRelations.MANAGEMENT_PUMP
        management_device = DamoRelations.MANAGEMENT_DEVICE
        management_rows = management.to_frame().T
        pump_id = management[management_pump.foreign_key]
        if not pd.isna(pump_id):
            pumps = management_pump.parents_of(management_rows, self.hydamo.pumps)
            if not pumps.empty:
                struc_id = pumps["code"].to_numpy()[0]
                return "pump", struc_id
            return None

        management_device_id = management[management_device.foreign_key]
        if pd.isna(management_device_id):
            raise ValueError(
                "Only management_devices and pumps can be connected to a management object."
            )

        mandev = management_device.parents_of(
            management_rows,
            self.hydamo.management_device,
        )
        if mandev.empty:
            logger.warning(
                "Management for management_device %s could not be found.",
                management_device_id,
            )
            return None

        weir_code = self._resolve_weir_via_opening(mandev)
        if weir_code is None:
            # Fall back to ids assigned at D-FlowFM structure-build time for
            # weirs/orifices with more than one opening (compound structures).
            # See the comment on Structures.compound_weir_structure_ids in
            # core/hydamo.py for why this lookup cannot be made statically.
            weir_code = self.hydamo.structures.compound_weir_structure_ids.get(
                management_device_id
            )
        if weir_code is not None:
            if weir_code in list(self.hydamo.structures.rweirs_df.id) or (
                not self.hydamo.structures.orifices_df.empty
                and weir_code in list(self.hydamo.structures.orifices_df.id)
            ):
                return "weir", weir_code
            # Resolved a HyDAMO/structure-build reference, but the D-FlowFM
            # structures haven't been built (yet), or building it was skipped
            # (e.g. it was filtered out, or had no valid opening data).
            logger.warning(
                "Management for management_device %s resolved to structure '%s', "
                "but it is not present in the built D-FlowFM structures.",
                management_device_id,
                weir_code,
            )
            return None

        culvert_relation = DamoRelations.MANAGEMENT_DEVICE_CULVERT
        if pd.notna(mandev[culvert_relation.foreign_key]).any():
            struc = culvert_relation.parents_of(
                mandev.iloc[:1],
                self.hydamo.culverts,
            )
            if struc.empty:
                logger.warning(
                    "Management for management_device %s could not be connected to a culvert.",
                    management_device_id,
                )
                return None
            return "culvert", struc.code.to_numpy()[0]

        logger.warning(
            "Management for management_device %s could not be connected to a culvert or weir.",
            management_device_id,
        )
        return None

    def _resolve_weir_via_opening(self, mandev: pd.DataFrame) -> str | None:
        """Resolve a single-opening weir/orifice structure id from static HyDAMO relations.

        Walks the FK chain that already exists on the HyDAMO model, independent
        of D-FlowFM structure conversion having run:

            management_device.kunstwerkopeningid
                -> opening.globalid
                -> opening.stuwid
                -> weirs.globalid -> weirs.code

        For a weir/orifice with exactly one opening, StructuresIO.weirs() (in
        converters/hydamo2df.py) uses `weir.code` as the D-FlowFM structure id,
        i.e. exactly the code this chain resolves to - so this is safe to call
        before structures have been built.

        It deliberately returns None (rather than raising or guessing) for
        compound, multi-opening weirs (i.e. more than one `opening` row shares
        the resolved `stuwid`): their structure id gets a `_N` suffix assigned
        during structure conversion (enumeration order of openings in
        StructuresIO.weirs()), which cannot be derived from HyDAMO data alone.
        Those are looked up separately via
        `hydamo.structures.compound_weir_structure_ids` by the caller.

        Parameters
        ----------
        mandev : pd.DataFrame
            The ``hydamo.management_device`` row(s) for a single management record's
            ``regelmiddelid`` (as selected by ``_resolve_management_structure``).
            Expected to contain at most one row; only the first row is used if there are
            more.

        Returns
        -------
        str or None
            The weir's or orifice's HyDAMO ``code`` (equal to its D-FlowFM structure id),
            or None if ``mandev`` has no ``kunstwerkopeningid``, the chain to ``weirs``
            cannot be resolved, or the weir turns out to be compound (multi-opening).
        """
        opening_relation = DamoRelations.MANAGEMENT_DEVICE_OPENING
        if opening_relation.foreign_key not in mandev.columns:
            return None
        opening_id = mandev[opening_relation.foreign_key].to_numpy()[0]
        if pd.isna(opening_id):
            return None

        opening_row = opening_relation.parents_of(
            mandev.iloc[:1],
            self.hydamo.opening,
        )
        weir_relation = DamoRelations.OPENING_WEIR
        if opening_row.empty or weir_relation.foreign_key not in opening_row.columns:
            return None
        weir_row = weir_relation.parents_of(opening_row, self.hydamo.weirs)
        if weir_row.empty:
            return None

        # A weir with more than one opening is "compound": StructuresIO builds
        # one structure per opening with a generated suffix (weir.code_N), not
        # weir.code itself. We cannot tell here which suffix belongs to this
        # particular opening, so defer to compound_weir_structure_ids instead
        # of returning the (wrong, unsuffixed) weir code.
        sibling_openings = weir_relation.children_of(
            weir_row.iloc[:1], self.hydamo.opening
        )
        if len(sibling_openings) > 1:
            return None

        return weir_row["code"].to_numpy()[0]

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def parse_complex_controller(self, xml_folder: Path | str) -> ComplexControllers:
        """Read user-specified 'complex' controllers from the RTC XML files in a folder.

        Items that refer to structures that are not allowed (see `allow_struct`) are
        skipped and logged.

        Parameters
        ----------
        xml_folder : Path or str
            Folder with the custom XML files.

        Returns
        -------
        ComplexControllers
            The XML fragments per RTC file, and the parsed DIMR config.
        """
        files = [p for p in Path(xml_folder).iterdir() if p.suffix == ".xml"]
        savedict: ComplexControllers = {
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

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_cc_rtc_dataconfig(
        self, root: ET.Element, savedict: ComplexControllers
    ) -> ComplexControllers:
        """Add the allowed import and export series of rtcDataConfig.xml to ``savedict``."""
        children = self._parse_unique_children(root)
        import_series = children.get("importSeries")
        if import_series is not None:
            for num, el in enumerate(import_series):
                xml_text = ET.tostring(el).decode()
                if "PITimeSeries" in xml_text and num == 0:
                    continue
                allow, el_text, reason = self._parse_dataconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_DATA_CONFIG_XML}: Skipped importSeries item for elementId '%s': %s.",
                        el_text,
                        reason,
                    )
                    continue
                savedict["dataconfig_import"].append(xml_text)

        export_series = children.get("exportSeries")
        if export_series is not None:
            for el in export_series:
                xml_text = ET.tostring(el).decode()
                if "PITimeSeries" in xml_text or "CSVTimeSeries" in xml_text:
                    continue
                allow, el_text, reason = self._parse_dataconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_DATA_CONFIG_XML}: Skipped exportSeries item for elementId '%s': %s.",
                        el_text,
                        reason,
                    )
                    continue
                savedict["dataconfig_export"].append(xml_text)

        return savedict

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_cc_rtc_toolsconfig(
        self, root: ET.Element, savedict: ComplexControllers
    ) -> ComplexControllers:
        """Add the allowed rules and triggers of rtcToolsConfig.xml to ``savedict``."""
        children = self._parse_unique_children(root)

        rules = children.get("rules")
        if rules is not None:
            for el in rules:
                allow, el_text, reason = self._parse_toolsconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_TOOLS_CONFIG_XML}: Skipped rule element '%s': %s.",
                        el_text,
                        reason,
                    )
                    continue
                savedict["toolsconfig_rules"].append(ET.tostring(el).decode())

        triggers = children.get("triggers")
        if triggers is not None:
            for el in triggers:
                allow, el_text, reason = self._parse_toolsconfig_item(el)
                if not allow:
                    logger.info(
                        f"{RTC_TOOLS_CONFIG_XML}: Skipped trigger element '%s': %s.",
                        el_text,
                        reason,
                    )
                    continue
                savedict["toolsconfig_triggers"].append(ET.tostring(el).decode())

        return savedict

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_cc_timeseries(
        self, root: ET.Element, savedict: ComplexControllers
    ) -> ComplexControllers:
        """Add all elements of timeseries_import.xml to ``savedict``."""
        for el in root:
            savedict["timeseries"].append(ET.tostring(el).decode())

        return savedict

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_cc_state(
        self, root: ET.Element, savedict: ComplexControllers
    ) -> ComplexControllers:
        """Add the elements of the tree vector of state_import.xml to ``savedict``."""
        for el in root[0]:
            savedict["state"].append(ET.tostring(el).decode())

        return savedict

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_cc_dimr_config(
        self, root: ET.Element, savedict: ComplexControllers
    ) -> ComplexControllers:
        """Add dimr_config.xml to ``savedict``, without the coupler items that are not allowed."""
        red_root = copy.deepcopy(root)
        for coupler_name, coupler_target in (
            ("rtc_to_flow", "targetName"),
            ("flow_to_rtc", "sourceName"),
        ):
            self._filter_dimr_coupler_items(red_root, coupler_name, coupler_target)
        savedict["dimr_config"].append(red_root)

        return savedict

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _filter_dimr_coupler_items(
        self, red_root: ET.Element, coupler_name: str, coupler_target: str
    ) -> None:
        """Remove the items of a coupler that refer to structures that are not allowed, in place."""
        for coupler in list(red_root):
            if coupler.attrib.get("name") != coupler_name:
                continue
            for sub_el in list(coupler):
                target = sub_el.find(".//{*}" + coupler_target)
                allow, el_text, reason = self._parse_dimr_item(target)
                if allow:
                    continue
                logger.info(
                    "dimr_config.xml: Skipped %s element with '%s' '%s': %s.",
                    coupler_name,
                    coupler_target,
                    el_text,
                    reason,
                )
                coupler.remove(sub_el)

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def find_complex_controller_ids(
        complex_controllers_folder: list[str | Path] | str | Path,
        hydamo: HyDAMO,
    ) -> set[str]:
        """Return the ids of the structures that the complex controllers refer to.

        Only structures that exist in the HyDAMO model are returned; observation points
        are left out.

        Parameters
        ----------
        complex_controllers_folder : list[str or Path], str or Path
            Folder(s) with the complex controllers (their ``dimr_config.xml``).
        hydamo : HyDAMO
            The HyDAMO data, with the structures.

        Returns
        -------
        set[str]
            Structure ids.
        """
        # Do not log validation warnings in this method
        struct_ids_by_type = DRTCModel._get_struct_ids_by_type(hydamo)
        cc_structs, _, _ = DRTCModel._load_complex_controller_structs(
            complex_controllers_folder,
            struct_ids_by_type,
            log_validation=False,
        )
        # Do not return observation point IDs in this method
        cc_ids = {
            cc.struct_name for cc in cc_structs if cc.struct_type != "observations"
        }

        return cc_ids

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _load_complex_controller_structs(
        complex_controllers_folder: list[str | Path] | str | Path,
        struct_ids_by_type: dict[str, set[str]],
        log_validation: bool = True,
    ) -> tuple[list[DRTCStructure], set[str], set[str]]:
        """Return the structures referenced by the complex controllers that exist in HyDAMO.

        Returns the valid structures, their ids, and the ids of all referenced
        structures. Raises a ValueError for duplicate structure ids.
        """
        folders = DRTCModel._as_folder_list(complex_controllers_folder)
        referenced_cc_structs = DRTCModel._collect_complex_controller_structs(folders)
        referenced_cc_structs, referenced_cc_ids = (
            DRTCModel._deduplicate_complex_controller_structs(referenced_cc_structs)
        )
        # Keep both the pre-validation references and the validated subset. The
        # pre-validation ids are needed to recognize XML references that are invalid
        # because they do not exist in HyDAMO; otherwise they are indistinguishable
        # from unrelated DIMR coupler items.
        cc_structs, cc_ids = DRTCModel._validate_complex_controller_structs(
            referenced_cc_structs, struct_ids_by_type, log_validation
        )

        return cc_structs, cc_ids, referenced_cc_ids

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _as_folder_list(
        complex_controllers_folder: list[str | Path] | str | Path,
    ) -> list[str | Path]:
        """Return one folder, or a list of folders, as a list."""
        if isinstance(complex_controllers_folder, list):
            return complex_controllers_folder
        return [complex_controllers_folder]

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _collect_complex_controller_structs(
        folders: list[str | Path],
    ) -> list[DRTCStructure]:
        """Return the structures referenced in the dimr_config.xml of each folder."""
        # Find complex controller structs and referred observation points.
        complex_controller_structs = []
        for folder in folders:
            for filepath in Path(folder).iterdir():
                if filepath.name != "dimr_config.xml":
                    continue
                root = ET.parse(filepath).getroot()
                complex_controller_structs.extend(
                    DRTCModel._parse_referenced_structures(root)
                )
        return complex_controller_structs

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _deduplicate_complex_controller_structs(
        complex_controller_structs: list[DRTCStructure],
    ) -> tuple[list[DRTCStructure], set[str]]:
        """Return each structure once, and their ids; raise a ValueError for duplicate structures.

        Observation points may occur more than once.
        """
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
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _validate_complex_controller_structs(
        complex_controller_structs: list[DRTCStructure],
        struct_ids_by_type: dict[str, set[str]],
        log_validation: bool,
    ) -> tuple[list[DRTCStructure], set[str]]:
        """Return the structures that exist in HyDAMO, and their ids; log the others."""
        validated_cc_structs = []
        for fs in complex_controller_structs:
            if (
                fs.struct_type not in struct_ids_by_type
                or fs.struct_name not in struct_ids_by_type[fs.struct_type]
            ):
                msg = f"Complex controllers: reference missing from HyDAMO; skipping {fs.struct_type}/{fs.struct_name}"
                if log_validation:
                    logger.warning(msg)
            else:
                validated_cc_structs.append(fs)
        validated_cc_ids = {fs.struct_name for fs in validated_cc_structs}

        return validated_cc_structs, validated_cc_ids

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _get_struct_ids_by_type(hydamo: HyDAMO) -> dict[str, set[str]]:
        """Return the ids of the observation points and structures of HyDAMO, per type."""
        return {
            "observations": set(
                hydamo.observationpoints.observation_points.get("name", [])
            ),
            "weirs": set(hydamo.structures.rweirs_df.get("id", []))
            | set(hydamo.structures.uweirs_df.get("id", [])),
            "orifices": set(hydamo.structures.orifices_df.get("id", [])),
            "pumps": set(hydamo.structures.pumps_df.get("id", [])),
            "generalstructures": set(
                hydamo.structures.generalstructures_df.get("id", [])
            ),
            "culverts": set(hydamo.structures.culverts_df.get("id", [])),
            "bridges": set(hydamo.structures.bridges_df.get("id", [])),
        }

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _load_complex_controllers(
        self,
        complex_controllers_folder: list[str | Path] | str | Path,
    ) -> ComplexControllers:
        """Normalize input folders, merge parsed controllers, and validate unique IDs."""
        if isinstance(complex_controllers_folder, list):
            folders = complex_controllers_folder
        else:
            folders = [complex_controllers_folder]
        parsed = [self.parse_complex_controller(Path(folder)) for folder in folders]
        # Append the XML fragments of all folders per section, and keep a single merged
        # DIMR root so downstream writers can consume index 0.
        return {
            "dataconfig_import": [x for c in parsed for x in c["dataconfig_import"]],
            "dataconfig_export": [x for c in parsed for x in c["dataconfig_export"]],
            "toolsconfig_rules": [x for c in parsed for x in c["toolsconfig_rules"]],
            "toolsconfig_triggers": [
                x for c in parsed for x in c["toolsconfig_triggers"]
            ],
            "timeseries": [x for c in parsed for x in c["timeseries"]],
            "state": [x for c in parsed for x in c["state"]],
            "dimr_config": self._merge_dimr_config_roots(
                [x for c in parsed for x in c["dimr_config"]]
            ),
        }

    @staticmethod
    def get_item_pair(item: ET.Element) -> tuple[str, str] | None:
        """Return the source and target name of a DIMR coupler item.

        Parameters
        ----------
        item : ET.Element
            An ``item`` element of a coupler.

        Returns
        -------
        tuple[str, str] or None
            ``(sourceName, targetName)``, or None if one of them is missing.
        """
        source = item.find(".//{*}sourceName")
        target = item.find(".//{*}targetName")
        if (
            source is None
            or target is None
            or source.text is None
            or target.text is None
        ):
            return None
        return source.text, target.text

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
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
        merged_couplers: dict[str, ET.Element] = {}
        seen_items: dict[str, set[tuple[str, str]]] = {}
        for root in dimr_roots:
            for coupler in root.findall("./{*}coupler"):
                coupler_name = coupler.attrib.get("name")
                if coupler_name is None:
                    continue

                if coupler_name not in merged_couplers:
                    # Add this coupler without items
                    mc = copy.deepcopy(coupler)
                    for item in mc.findall("./{*}item"):
                        mc.remove(item)
                    merged_root.append(mc)

                    # Initialize tracking reference
                    merged_couplers[coupler_name] = mc
                    seen_items[coupler_name] = set()

                # Only add unseen coupler items
                for item in coupler.findall("./{*}item"):
                    mitem = copy.deepcopy(item)
                    pair = DRTCModel.get_item_pair(mitem)
                    if pair is None:
                        continue

                    if pair not in seen_items[coupler_name]:
                        seen_items[coupler_name].add(pair)
                        merged_couplers[coupler_name].append(mitem)

        return [merged_root]

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_dataconfig_item(
        self, el: ET.Element
    ) -> tuple[bool, str | None, str | None]:
        """Return whether a data config item is allowed, its elementId, and why it is not."""
        allow = True
        el_text = None
        reason = None

        # Check if this is a complex controller but not in the whitelist
        # Always allow observation points
        el_id = el.find(".//{*}elementId")
        if el_id is not None and el_id.text is not None:
            el_text = el_id.text
            # In complex-controller fragments: keep observation ids, keep all when no filter is configured,
            # but reject ids that are not part of referenced/validated complex-controller structures.
            allow, reason = self._allow_struct_with_reason(
                cc_id=el_text,
                allow_observations=True,
                allow_if_filter_inactive=True,
                allow_if_not_referenced=False,
            )

        return allow, el_text, reason

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_toolsconfig_item(
        self, el: ET.Element
    ) -> tuple[bool, str | None, str | None]:
        """Return whether a rule or trigger is allowed, its id, and why it is not.

        It is allowed if all structures in its inputs, outputs, triggers and conditions
        are allowed.
        """
        allow = True
        el_firstchild_text = None
        reason = None

        el_firstchild = next(iter(el), None)
        if el_firstchild is None:
            return allow, el_firstchild_text, reason
        el_tags = []
        for tag in ["input", "output", "trigger", "condition"]:
            el_tags += el.findall(".//{*}" + tag)
        for el_tag in el_tags:
            for child in el_tag:
                if child.text is None:
                    continue
                if child.text.startswith(INPUT_PREFIX) or child.text.startswith(
                    OUTPUT_PREFIX
                ):
                    child_text = child.text.replace(INPUT_PREFIX, "").replace(
                        OUTPUT_PREFIX, ""
                    )
                    child_text = child_text.split("/")[0]

                    # Check if this is a complex controller but not in the whitelist
                    # Always allow observation points
                    if allow:
                        # In tools fragments: keep observation ids, keep all when no filter is configured,
                        # but reject ids that are not referenced by validated complex-controller structures.
                        allow, reason = self._allow_struct_with_reason(
                            cc_id=child_text,
                            allow_observations=True,
                            allow_if_filter_inactive=True,
                            allow_if_not_referenced=False,
                        )
                        el_firstchild_text = el_firstchild.get("id")

        return allow, el_firstchild_text, reason

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_dimr_item(
        self, target: ET.Element | None
    ) -> tuple[bool, str | None, str | None]:
        """Return whether a coupler item is allowed, its name if not, and why it is not."""
        allow = True
        el_text = None
        reason = None
        if target is None or not target.text:
            return allow, el_text, reason

        parts = target.text.split("/")
        if len(parts) < 3:
            return allow, target.text, reason

        _, struct_id, _ = parts

        # Check if this is a complex controller but not in the whitelist
        # For DIMR coupler items: keep observation ids and keep non-complex/non-referenced ids,
        # and only filter out referenced complex-controller ids that are not in the whitelist.
        allow, reason = self._allow_struct_with_reason(
            cc_id=struct_id,
            allow_observations=True,
            allow_if_filter_inactive=True,
            allow_if_not_referenced=True,
        )
        if not allow:
            el_text = target.text

        return allow, el_text, reason

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_referenced_structures(root: ET.Element) -> list[DRTCStructure]:
        """Return the structures in the rtc_to_flow and flow_to_rtc couplers of a DIMR config."""
        structures = []
        rtc_to_flow = root.findall(
            ".//{*}coupler[@name='rtc_to_flow']/{*}item/{*}targetName"
        )
        flow_to_rtc = root.findall(
            ".//{*}coupler[@name='flow_to_rtc']/{*}item/{*}sourceName"
        )
        for item in rtc_to_flow + flow_to_rtc:
            if item.text is None:
                continue
            structures.append(DRTCStructure(*item.text.split("/")))

        return structures

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _parse_unique_children(root: ET.Element) -> dict[str, ET.Element]:
        """Return the children of an element by tag, without namespace; raise a KeyError for duplicates."""
        # Store xml sections by tag, this function assumes unique children.
        children: dict[str, ET.Element] = {}
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
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _strip_namespace(tag: str) -> str:
        """Return a tag without its ``{namespace}``."""
        return tag.split("}", 1)[-1] if tag.startswith("{") else tag

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _dataconfig_timeseries_key(el: ET.Element) -> str | None:
        """Return the id of a timeSeries element, or None for other elements."""
        if DRTCModel._strip_namespace(el.tag) != "timeSeries":
            return None
        return el.attrib.get("id")

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _timeseries_series_key(el: ET.Element) -> str | None:
        """Return ``location|parameter`` of a series element, or None for other elements."""
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
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _state_leaf_key(el: ET.Element) -> str | None:
        """Return the id of a treeVectorLeaf element, or None for other elements."""
        if DRTCModel._strip_namespace(el.tag) != "treeVectorLeaf":
            return None
        return el.attrib.get("id")

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def _append_unique_elements(
        self,
        parent: ET.Element,
        elements: Sequence[str | ET.Element],
        key_getter: Callable[[ET.Element], str | None],
        file_label: str,
    ) -> None:
        """Append elements to a parent, skipping ids or XML that the parent already has, in place.

        Elements with a key (from ``key_getter``) are compared by key, others by their XML.
        Skipped elements are logged with ``file_label``.
        """
        seen_keys = set()
        seen_raw_xml = set()

        for child in list(parent):
            key = key_getter(child)
            if key is not None:
                seen_keys.add(key)
            else:
                seen_raw_xml.add(ET.tostring(child, encoding="unicode"))

        for item in elements:
            element = (
                ET.fromstring(item) if isinstance(item, str) else copy.deepcopy(item)
            )
            key = key_getter(element)
            if key is not None:
                if key in seen_keys:
                    logger.warning(
                        "%s: Skipped writing %s, id already present", file_label, key
                    )
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

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def from_hydamo(
        self,
        pid_settings: dict | None = None,
        interval_settings: dict | None = None,
        timeseries: pd.DataFrame | None = None,
    ) -> None:
        """Add the time, PID and interval controllers of ``hydamo.management``.

        The settings that HyDAMO lacks are keyed by D-Flow FM structure id: the weir,
        orifice, pump or culvert code, or for a weir with several openings the id of
        the opening's structure (e.g. ``S_98740_3``). Structures without their own key
        get the settings under ``"global"``. Keys that match no structure with that
        type of controller are listed in one warning per settings dict.

        A structure gets one controller: when several Sturing rows with a supported
        controller type (PID, interval or time) resolve to the same structure, the
        first of them is used and the others are skipped with a warning. Rows with
        another type are skipped with a warning before any other check.

        Parameters
        ----------
        pid_settings : dict, optional
            ``ki``, ``kp``, ``kd`` and ``maxspeed`` of the PID controllers, per structure
            id and under ``"global"``. Defaults to None.
        interval_settings : dict, optional
            ``deadband`` and ``maxspeed`` of the interval controllers, per structure id
            and under ``"global"``. Defaults to None.
        timeseries : pd.DataFrame, optional
            Time series of the time controllers, with the structure ids as columns.
            Defaults to None.

        Raises
        ------
        ValueError
            For inconsistent input, e.g. a PID controller without ``pid_settings``.
        """
        # the Sturing code of the row that controls each structure: the first wins
        controlled_by: dict[str, str] = {}
        pid_structures: set[str] = set()
        interval_structures: set[str] = set()
        for _, management in self.hydamo.management.iterrows():
            resolved = self._resolve_management_structure(management)
            if resolved is None:
                continue
            structype, struc_id = resolved
            # only rows that create a controller compete for a structure
            if management.typecontroller not in ("PID", "interval", "time"):
                logger.warning(
                    f"{management.typecontroller} is not a valid controller type - skipped."
                )
                continue
            if struc_id in controlled_by:
                logger.warning(
                    "Structure %s is controlled by Sturing %s and %s. Using %s.",
                    struc_id,
                    controlled_by[struc_id],
                    management.code,
                    controlled_by[struc_id],
                )
                continue
            if (
                structype == "weir"
                and management.stuurvariabele == "bovenkant afsluitmiddel"
            ):
                steering_variable = CREST_LEVEL
            elif structype == "weir" and management.stuurvariabele == "hoogte opening":
                steering_variable = GATE_LOWER_EDGE_LEVEL
            elif structype == "pump" and management.stuurvariabele == "pompdebiet":
                steering_variable = CAPACITY
            elif (
                structype == "culvert" and management.stuurvariabele == "hoogte opening"
            ):
                steering_variable = VALVE_OPENING
            else:
                raise ValueError(
                    f"Invalid value for steering variable of {struc_id}: {management.stuurvariabele}. Allowed values are 'bovenkant afsluitmiddel' for weirs, 'hoogte opening' for oroficies and culverts, and 'pompdebiet' for pumps."
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
                if pid_settings is None:
                    raise ValueError(
                        f"{management.code} contains a PID controller, but no pid_settings are provided. Please do so."
                    )
                pid_structures.add(struc_id)
                pid = _settings_for("pid_settings", pid_settings, struc_id)
                self.add_pid_controller(
                    structure_id=struc_id,
                    steering_variable=steering_variable,
                    target_variable=target_variable,
                    ki=pid["ki"],
                    kp=pid["kp"],
                    kd=pid["kd"],
                    max_speed=pid["maxspeed"],
                    setpoint=management.streefwaarde,
                    lower_bound=management.ondergrens,
                    upper_bound=management.bovengrens,
                    observation_location=management.meetlocatieid,
                )

            elif management.typecontroller == "interval":
                if interval_settings is None:
                    raise ValueError(
                        f"{management.code} contains an interval controller, but no interval_settings are provided. Please do so."
                    )
                interval_structures.add(struc_id)
                interval = _settings_for(
                    "interval_settings", interval_settings, struc_id
                )
                self.add_interval_controller(
                    structure_id=struc_id,
                    steering_variable=steering_variable,
                    target_variable=target_variable,
                    deadband=interval["deadband"],
                    setting_above=management.bovengrens,
                    setting_below=management.ondergrens,
                    max_speed=interval["maxspeed"],
                    setpoint=management.streefwaarde,
                    observation_location=management.meetlocatieid,
                )

            elif management.typecontroller == "time":
                if timeseries is None:
                    raise ValueError(
                        f"{management.code} contains a time controller, but no time series are provided. Please do so."
                    )
                else:
                    data = timeseries.loc[:, struc_id]
                    self.add_time_controller(
                        structure_id=struc_id,
                        steering_variable=steering_variable,
                        data=data,
                    )
            controlled_by[struc_id] = management.code

        _warn_unknown_settings_keys("pid_settings", pid_settings, pid_structures)
        _warn_unknown_settings_keys(
            "interval_settings", interval_settings, interval_structures
        )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_time_controller(
        self,
        structure_id: str,
        steering_variable: str,
        data: pd.Series,
        interpolation_option: str = "LINEAR",
        extrapolation_option: str = "BLOCK",
    ) -> None:
        """Add a time controller to a structure.

        Parameters
        ----------
        structure_id : str
            Structure id.
        steering_variable : str
            Variable that is controlled, usually the crest level.
        data : pd.Series
            Time series of the setting.
        interpolation_option : str, optional
            Interpolation option. Defaults to ``"LINEAR"``.
        extrapolation_option : str, optional
            Extrapolation option. Defaults to ``"BLOCK"``.
        """
        self.time_controllers[structure_id] = {
            "type": "Time",
            "data": data,
            "steering_variable": steering_variable,
            "interpolation_option": interpolation_option,
            "extrapolation_option": extrapolation_option,
        }

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_pid_controller(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        structure_id: str,
        steering_variable: str,
        target_variable: str,
        setpoint: Annotated[float | str | pd.Series, Field(union_mode="left_to_right")],
        lower_bound: Annotated[
            float | str | None, Field(union_mode="left_to_right")
        ] = None,
        upper_bound: Annotated[
            float | str | None, Field(union_mode="left_to_right")
        ] = None,
        observation_location: str,
        ki: float = 0.001,
        kp: float = 0.0,
        kd: float = 0.0,
        max_speed: float = 0.00033,
        interpolation_option: str = "LINEAR",
        extrapolation_option: str = "BLOCK",
    ) -> None:
        """Add a PID controller to a structure.

        All arguments are keyword-only.

        Parameters
        ----------
        structure_id : str
            Structure id.
        steering_variable : str
            Variable that is controlled, usually the crest level.
        target_variable : str
            Target variable, usually the water level.
        setpoint : float, str or pd.Series
            Setpoint, or a time series of the setpoint.
        lower_bound, upper_bound : float or str, optional
            Lowest and highest allowed setting. Default to None.
        observation_location : str
            Id of the observation point.
        ki, kp, kd : float, optional
            Gain factors. Default to 0.001, 0.0 and 0.0.
        max_speed : float, optional
            Maximum speed of change of the setting. Defaults to 0.00033.
        interpolation_option : str, optional
            Interpolation option. Defaults to ``"LINEAR"``.
        extrapolation_option : str, optional
            Extrapolation option. Defaults to ``"BLOCK"``.
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
            "max_speed": max_speed,
            "interpolation_option": interpolation_option,
            "extrapolation_option": extrapolation_option,
        }

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_interval_controller(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        structure_id: str,
        steering_variable: str,
        target_variable: str,
        deadband: Annotated[
            float | str | None, Field(union_mode="left_to_right")
        ] = None,
        setpoint: Annotated[float | str | pd.Series, Field(union_mode="left_to_right")],
        setting_below: Annotated[float | str, Field(union_mode="left_to_right")],
        setting_above: Annotated[float | str, Field(union_mode="left_to_right")],
        max_speed: Annotated[
            float | str | None, Field(union_mode="left_to_right")
        ] = None,
        observation_location: str,
        interpolation_option: str = "LINEAR",
        extrapolation_option: str = "BLOCK",
    ) -> None:
        """Add an interval controller to a structure.

        All arguments are keyword-only.

        Parameters
        ----------
        structure_id : str
            Structure id.
        steering_variable : str
            Variable that is controlled, usually the crest level.
        target_variable : str
            Target variable, usually the water level.
        deadband : float or str, optional
            Dead band around the setpoint. Defaults to None.
        setpoint : float, str or pd.Series
            Setpoint, or a time series of the setpoint.
        setting_below, setting_above : float or str
            Setting when the target variable is below or above the setpoint.
        max_speed : float or str, optional
            Maximum speed of change of the setting. Defaults to None.
        observation_location : str
            Id of the observation point.
        interpolation_option : str, optional
            Interpolation option. Defaults to ``"LINEAR"``.
        extrapolation_option : str, optional
            Extrapolation option. Defaults to ``"BLOCK"``.
        """
        self.interval_controllers[structure_id] = {
            "type": "Interval",
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

    def set_state_from_file(
        self, state_export: str | Path, check_time: bool = True
    ) -> None:
        """Use the RTC state of an earlier model run as initial state (warm start).

        The state_export.xml of the earlier run is written as state_import.xml, including the internal
        state of the rules (e.g. the integral term of PID controllers). This assumes that the RTC configuration
        is the same as in the earlier run: state items that do not occur in the new rtcDataConfig.xml are
        skipped, and controllers that are missing in the state keep their default initial setting. Both
        are reported as a warning.

        Parameters
        ----------
        state_export : str or Path
            Path to state_export.xml, or the rtc folder of the earlier run.
        check_time : bool, optional
            Compare the time of the state (from statePI.xml next to the state file) with
            the start time of this model, and warn if they differ. Defaults to True.
        """
        path = Path(state_export)
        if path.is_dir():
            path = path / "state_export.xml"
        if not path.exists():
            raise FileNotFoundError(f"RTC state file {path} not found.")

        leaves = self._read_state_leaves(path)
        if not leaves:
            raise ValueError(f"No state items (treeVectorLeaf) found in {path}.")
        self.state_from_file = leaves
        logger.info(f"Read {len(leaves)} RTC state items from {path}.")

        if not check_time:
            return
        statepi = path.parent / "statePI.xml"
        state_time = self._read_state_time(statepi) if statepi.exists() else None
        if state_time is None:
            logger.info(
                f"No time found for the RTC state in {statepi}; the time of the state is not checked."
            )
        elif state_time != self.time_settings["start"]:
            logger.warning(
                f"The RTC state is valid for {state_time}, but the model starts at {self.time_settings['start']}."
            )

    def set_state_from_modelrun(
        self,
        his_path: str | Path,
        time: dt | pd.Timestamp | str | None = None,
        netcdf_index: int | None = None,
    ) -> dict[str, float]:
        """Use the structure settings of an earlier model run (his-file) as initial setting of the controllers.

        For every controlled structure, the steered variable (crest level, gate lower edge level, valve
        opening or pump capacity) is read from the his-file at the given time and passed to set_initial_state.
        Unlike set_state_from_file, any time step can be used, but the internal state of the rules (e.g. the
        integral term of PID controllers) is not restored.

        Parameters
        ----------
        his_path : str or Path
            Path to the his file of the earlier run.
        time : datetime, pd.Timestamp or str, optional
            Time of the settings; the nearest time step is used. Defaults to None.
        netcdf_index : int, optional
            Time index instead of a time. Defaults to None: without a time, the last
            time step.

        Returns
        -------
        dict[str, float]
            Initial setting per structure id.
        """
        if time is not None and netcdf_index is not None:
            raise ValueError("Provide either time or netcdf_index, not both.")
        if time is None and netcdf_index is None:
            logger.info(f"No time given; the last time step in {his_path} is used.")
        controllers = {
            **self.time_controllers,
            **self.interval_controllers,
            **self.pid_controllers,
        }
        kinds = {
            kind for sources in HIS_SETTING_VARIABLES.values() for kind, _ in sources
        }

        values, missing = {}, []
        with Dataset(his_path) as his:
            if time is not None:
                index = time_index(his, time)
            else:
                index = -1 if netcdf_index is None else netcdf_index
            names = {
                kind: [n.strip() for n in chartostring(his[f"{kind}_name"][:])]
                for kind in kinds
                if f"{kind}_name" in his.variables
            }
            for structure_id, controller in controllers.items():
                value = self._read_structure_setting(
                    his, names, structure_id, controller["steering_variable"], index
                )
                if value is None:
                    missing.append(structure_id)
                else:
                    values[structure_id] = value

        if missing:
            logger.warning(
                f"No setting found in {his_path} for {len(missing)} controlled structure(s), "
                f"these keep their default initial setting: {missing}"
            )
        logger.info(
            f"Initial setting of {len(values)} controlled structures read from {his_path}."
        )
        self.set_initial_state(values)
        return values

    @staticmethod
    def _read_structure_setting(
        his: Dataset,
        names: dict[str, list[str]],
        structure_id: str,
        steering_variable: str,
        index: int,
    ) -> float | None:
        """Read the setting of a structure for a steering variable from a his-file.

        The first structure type in HIS_SETTING_VARIABLES that contains the structure is used. If its value is
        missing (masked or not finite), None is returned; other structure types are not tried.

        Parameters
        ----------
        his : Dataset
            The opened his-file.
        names : dict[str, list[str]]
            Structure names per structure type in the his-file, e.g. {"weirgen": ["S_1", ...]}.
        structure_id : str
            Id of the structure.
        steering_variable : str
            Steering variable of the controller, e.g. CREST_LEVEL.
        index : int
            Time index.

        Returns
        -------
        float | None
            The setting, or None if the structure or its value is not found.
        """
        for kind, quantity in HIS_SETTING_VARIABLES.get(steering_variable, []):
            variable = f"{kind}_{quantity}"
            if (
                kind in names
                and structure_id in names[kind]
                and variable in his.variables
            ):
                value = his[variable][index, names[kind].index(structure_id)]
                if np.ma.is_masked(value) or not np.isfinite(value):
                    return None
                return float(value)
        return None

    def set_initial_state(self, values: dict[str, float]) -> None:
        """Set the initial setting of controlled structures, e.g. {"S_96840": 16.2}.

        The value is written to state_import.xml for the steered variable of the structure. It overrides the
        default initial setting and a state read with set_state_from_file.

        Parameters
        ----------
        values : dict[str, float]
            Initial setting per structure id.
        """
        self.initial_state.update(
            {str(key): float(value) for key, value in values.items()}
        )

    @staticmethod
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def finish_file(xmlroot: ET.Element, filename: Path | str) -> None:
        """Write an XML tree to a file, with an XML declaration, pretty-printed.

        Parameters
        ----------
        xmlroot : ET.Element
            Root of the XML tree.
        filename : Path or str
            File to write.
        """
        xmlbytes = bytes(
            '<?xml version="1.0" encoding="utf-8" standalone="yes" ?>',
            encoding="utf-8",
        ) + ET.tostring(xmlroot)
        xmlstring = xmlbytes.decode("utf-8").replace("\n", "").replace("  ", "")
        with open(filename, "w+") as f:
            f.write(xmlstring)
        with open(filename) as f:
            temp = xml.dom.minidom.parseString(f.read())
        with open(filename, "w+") as f:
            f.write(temp.toprettyxml())

    def write_xml_v1(self) -> None:
        """Write all RTC XML files."""
        self.write_runtimeconfig()
        self.write_toolsconfig()
        self.write_timeseries_import()
        self.write_dataconfig()
        self.write_state_import()

    def write_runtimeconfig(self) -> None:
        """Write rtcRuntimeConfig.xml, from the template, with the model period and time step."""
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
            x.set("date", self.time_settings["start"].strftime("%Y-%m-%d"))
            x.set("time", self.time_settings["start"].strftime("%H:%M:%S"))
        for x in myroot.iter(gn_brackets + "endDate"):
            x.set("date", self.time_settings["end"].strftime("%Y-%m-%d"))
            x.set("time", self.time_settings["end"].strftime("%H:%M:%S"))
        for x in myroot.iter(gn_brackets + "timeStep"):
            x.set("unit", "second")
            x.set("divider", "1")
            x.set("multiplier", str(int(self.time_settings["step"])))

        # write new xml file
        self.finish_file(myroot, self.output_path / "rtcRuntimeConfig.xml")

    def write_toolsconfig(self) -> None:
        """Write rtcToolsConfig.xml, from the template, with the rules of all controllers."""
        generalname = "http://www.wldelft.nl/fews"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # parsing xml file
        configfile = ET.parse(self.template_dir / "rtcToolsConfig_empty.xml")
        myroot = configfile.getroot()

        self.all_controllers: dict[
            str, TimeController | PIDController | IntervalController
        ] = {
            **self.time_controllers,
            **self.pid_controllers,
            **self.interval_controllers,
        }

        to_remove = []
        for key in self.all_controllers:
            controller = self.all_controllers[key]
            # For simple controllers: skip only when this id has an allowed complex controller.
            # If filtering is inactive or the id is not a referenced complex id, do not skip.
            if self.allow_struct(
                cc_id=key,
                allow_observations=False,
                allow_if_filter_inactive=False,
                allow_if_not_referenced=False,
            ):
                logger.warning(
                    f"RtcToolsConfig.xml: Skipped writing {controller['type']} control for {key}, complex controller already present"
                )
                to_remove.append(key)
                continue

            a = ET.SubElement(myroot[1], gn_brackets + "rule")
            if controller["type"] == "PID":
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
                o.text = (
                    OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]
                )

                p = ET.SubElement(m, gn_brackets + "integralPart")
                p.text = "[IP]" + "Control group " + str(key) + "/PID Rule"

                q = ET.SubElement(m, gn_brackets + "differentialPart")
                q.text = "[DP]" + "Control group " + str(key) + "/PID Rule"

            elif controller["type"] == "Interval":
                # Interval RTC
                # rule type (Interval)
                b = ET.SubElement(a, gn_brackets + "interval")
                b.set(
                    "id",
                    "[IntervalRule]" + "Control group " + str(key) + "/Interval Rule",
                )

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

                k = ET.SubElement(
                    j, gn_brackets + "x"
                )  # leave ref = "EXPLICIT" out for now
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
                o.text = (
                    OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]
                )

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
                f.text = (
                    OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]
                )

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

        self.finish_file(myroot, self.output_path / RTC_TOOLS_CONFIG_XML)

    def write_dataconfig(self) -> None:
        """Write rtcDataConfig.xml, from the template, with the series of all controllers."""
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

        timeseries_length = len(
            ET.parse(self.output_path / "timeseries_import.xml").getroot()
        )

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
            # For simple controllers: skip only when this id has an allowed complex controller.
            # If filtering is inactive or the id is not a referenced complex id, do not skip.
            if self.allow_struct(
                cc_id=key,
                allow_observations=False,
                allow_if_filter_inactive=False,
                allow_if_not_referenced=False,
            ):
                logger.warning(
                    f"{RTC_DATA_CONFIG_XML}: Skipped writing {controller['type']} control for {key}, complex controller already present"
                )
                continue

            # te importeren data
            if controller["type"] == "PID":
                input_id = (
                    INPUT_PREFIX
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"]
                )

                if myroot[0].find(f".//*[@id='{input_id}']") is None:
                    a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                    a.set("id", input_id)

                    b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")

                    c = ET.SubElement(b, gn_brackets + "elementId")
                    c.text = controller["observation_point"]

                    d = ET.SubElement(b, gn_brackets + "quantityId")
                    d.text = controller["target_variable"]

                    e = ET.SubElement(b, gn_brackets + "unit")
                    e.text = (
                        munit
                        if controller["target_variable"] == "Water level (op)"
                        else m3unit
                    )

                else:
                    logger.warning(
                        f"{RTC_DATA_CONFIG_XML}: Skipped writing {input_id}, observation point already present"
                    )

                # If a time dependent setpoint is required, add the Time Rule
                if type(controller["setpoint"]) is pd.Series:
                    a2 = ET.SubElement(myroot[0], gn_brackets + "timeSeries")

                    a2.set("id", f"[SP]Control group {key}/PID Rule")
                    b2 = ET.SubElement(a2, gn_brackets + "PITimeSeries")

                    c2 = ET.SubElement(b2, gn_brackets + "locationId")
                    c2.text = f"[PID]Control group {key}/PID Rule"

                    d2 = ET.SubElement(b2, gn_brackets + "parameterId")
                    d2.text = "SP"

                    e2 = ET.SubElement(b2, gn_brackets + "interpolationOption")
                    e2.text = controller["interpolation_option"]

                    e2 = ET.SubElement(b2, gn_brackets + "extrapolationOption")
                    e2.text = controller[
                        "extrapolation_option"
                    ]  # Changed from Block: HL

            elif controller["type"] == "Interval":
                input_id = (
                    INPUT_PREFIX
                    + controller["observation_point"]
                    + "/"
                    + controller["target_variable"]
                )
                if myroot.find(f".//*[@id='{input_id}']") is None:
                    a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")

                    a.set("id", input_id)

                    b = ET.SubElement(a, gn_brackets + "OpenMIExchangeItem")

                    c = ET.SubElement(b, gn_brackets + "elementId")
                    c.text = controller["observation_point"]

                    d = ET.SubElement(b, gn_brackets + "quantityId")
                    d.text = controller["target_variable"]

                    e = ET.SubElement(b, gn_brackets + "unit")
                    e.text = (
                        munit
                        if controller["target_variable"] == "Water level (op)"
                        else m3unit
                    )
                else:
                    logger.warning(
                        f"{RTC_DATA_CONFIG_XML}: Skipped writing {input_id}, observation point already present"
                    )

                if type(controller["setpoint"]) is pd.Series:
                    a2 = ET.SubElement(myroot[0], gn_brackets + "timeSeries")

                    a2.set("id", f"[SP]Control group {key}/Interval Rule")
                    b3 = ET.SubElement(a2, gn_brackets + "PITimeSeries")

                    c3 = ET.SubElement(b3, gn_brackets + "locationId")
                    c3.text = f"[IntervalRule]Control group {key}/Interval Rule"

                    d3 = ET.SubElement(b3, gn_brackets + "parameterId")
                    d3.text = "SP"

                    e3 = ET.SubElement(b3, gn_brackets + "interpolationOption")
                    e3.text = controller["interpolation_option"]

                    f3 = ET.SubElement(b3, gn_brackets + "extrapolationOption")
                    f3.text = controller[
                        "extrapolation_option"
                    ]  # Changed from Block: HL

            else:
                a = ET.SubElement(myroot[0], gn_brackets + "timeSeries")
                a.set("id", "Control group " + str(key) + "/Time Rule")
                b = ET.SubElement(a, gn_brackets + "PITimeSeries")

                c = ET.SubElement(b, gn_brackets + "locationId")
                c.text = f"[TimeRule]Control group {key}/Time Rule"

                d = ET.SubElement(b, gn_brackets + "parameterId")
                d.text = "TimeSeries"

                e = ET.SubElement(b, gn_brackets + "interpolationOption")
                e.text = controller["interpolation_option"]

                e = ET.SubElement(b, gn_brackets + "extrapolationOption")
                e.text = controller["extrapolation_option"]  # Changed from Block: HL

            # te exporteren data:
            f = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
            f.set(
                "id", OUTPUT_PREFIX + str(key) + "/" + controller["steering_variable"]
            )

            g = ET.SubElement(f, gn_brackets + "OpenMIExchangeItem")

            h = ET.SubElement(g, gn_brackets + "elementId")
            h.text = str(key)

            j = ET.SubElement(g, gn_brackets + "quantityId")
            j.text = controller["steering_variable"]

            k = ET.SubElement(g, gn_brackets + "unit")
            k.text = m3unit if controller["steering_variable"] == CAPACITY else munit

        for ikey, key in enumerate(self.all_controllers.keys()):
            controller = self.all_controllers[key]

            if controller["type"] == "PID":
                i = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                i.set("id", "[IP]Control group " + str(key) + "/PID Rule")

                j = ET.SubElement(myroot[1], gn_brackets + "timeSeries")
                j.set("id", "[DP]Control group " + str(key) + "/PID Rule")

            elif (
                controller["type"] == "Interval"
            ):  # Change slightly when working with Interval rule
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

        self.finish_file(myroot, self.output_path / RTC_DATA_CONFIG_XML)
        self._written_dataconfig_ids = self._element_ids(myroot)

    def write_timeseries_import(self) -> None:
        """Write timeseries_import.xml, from the template, with the time series of the controllers."""
        generalname = "http://www.wldelft.nl/fews/PI"
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # Parsing xml file
        configfile = ET.parse(self.template_dir / "timeseries_import_empty.xml")
        myroot = configfile.getroot()

        for key in self.all_controllers:
            controller = self.all_controllers[key]
            # For simple controllers: skip only when this id has an allowed complex controller.
            # If filtering is inactive or the id is not a referenced complex id, do not skip.
            if self.allow_struct(
                cc_id=key,
                allow_observations=False,
                allow_if_filter_inactive=False,
                allow_if_not_referenced=False,
            ):
                logger.warning(
                    f"{TIMESERIES_IMPORT_XML}: Skipped writing {controller['type']} control for {key}, complex controller already present"
                )
                continue

            if controller["type"] == "Time":
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
                for event_date, event_time, value in zip(
                    dates, times, controller["data"].to_numpy()
                ):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": event_date,
                        "time": event_time,
                        "value": str(value),
                    }
            elif controller["type"] == "Interval":
                setpoint = controller["setpoint"]
                if isinstance(setpoint, float):
                    setpoint = pd.Series(
                        [setpoint, setpoint],
                        index=[self.time_settings["start"], self.time_settings["end"]],
                    )
                    controller["setpoint"] = setpoint
                if isinstance(setpoint, str):
                    raise TypeError(
                        f"Setpoint {setpoint!r} of interval controller {key} must be a number or a pd.Series."
                    )

                # te importeren data
                dates = pd.to_datetime(setpoint.index).strftime("%Y-%m-%d")
                times = pd.to_datetime(setpoint.index).strftime("%H:%M:%S")
                timestep = (
                    pd.to_datetime(f"{dates[1]} {times[1]}")
                    - pd.to_datetime(f"{dates[0]} {times[0]}")
                ).total_seconds()

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
                for event_date, event_time, value in zip(
                    dates, times, setpoint.to_numpy()
                ):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": event_date,
                        "time": event_time,
                        "value": str(value),
                    }

            # Create a timeseries import if a time-dependent setpoint is used
            elif controller["type"] == "PID" and isinstance(
                controller["setpoint"], pd.Series
            ):
                # te importeren data
                dates = pd.to_datetime(controller["setpoint"].index).strftime(
                    "%Y-%m-%d"
                )
                times = pd.to_datetime(controller["setpoint"].index).strftime(
                    "%H:%M:%S"
                )
                timestep = (
                    pd.to_datetime(controller["setpoint"].index)[1]
                    - pd.to_datetime(controller["setpoint"].index)[0]
                ).total_seconds()
                a = ET.SubElement(myroot, gn_brackets + "series")
                b = ET.SubElement(a, gn_brackets + "header")
                c = ET.SubElement(b, gn_brackets + "type")
                c.text = "instantaneous"

                d = ET.SubElement(b, gn_brackets + "locationId")
                d.text = f"[PID]Control group {key}/PID Rule"

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
                for event_date, event_time, value in zip(
                    dates, times, controller["setpoint"].to_numpy()
                ):
                    k = ET.SubElement(a, gn_brackets + "event")
                    k.attrib = {
                        "date": event_date,
                        "time": event_time,
                        "value": str(value),
                    }

        if self.complex_controllers is not None:
            self._append_unique_elements(
                parent=myroot,
                elements=self.complex_controllers["timeseries"],
                key_getter=self._timeseries_series_key,
                file_label=TIMESERIES_IMPORT_XML,
            )

        self.finish_file(myroot, self.output_path / TIMESERIES_IMPORT_XML)

    def write_state_import(self) -> None:
        """Write state_import.xml, from the template, with the initial state of the controllers."""
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

        for key in self.all_controllers:
            controller = self.all_controllers[key]
            # For simple controllers: skip only when this id has an allowed complex controller.
            # If filtering is inactive or the id is not a referenced complex id, do not skip.
            if self.allow_struct(
                cc_id=key,
                allow_observations=False,
                allow_if_filter_inactive=False,
                allow_if_not_referenced=False,
            ):
                logger.warning(
                    f"{STATE_IMPORT_XML}: Skipped writing {controller['type']} control for {key}, complex controller already present"
                )
                continue

            # te importeren data
            a = ET.SubElement(a0, gn_brackets + "treeVectorLeaf")
            a.attrib = {
                "id": OUTPUT_PREFIX + key + "/" + controller["steering_variable"]
            }
            b = ET.SubElement(a, gn_brackets + "vector")
            if controller["type"] == "PID":
                b.text = str(self._initial_pid_setting(key, controller))
            elif controller["type"] == "Interval":
                # Take the maximum value as a starting value
                b.text = str(
                    max(
                        float(controller["setting_above"]),
                        float(controller["setting_below"]),
                    )
                )
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

        # a state from an earlier model run and user-defined initial settings override the defaults
        self._apply_state_overrides(myroot[0], gn_brackets)

        self.finish_file(myroot, self.output_path / STATE_IMPORT_XML)

    def _initial_pid_setting(
        self, structure_id: str, controller: PIDController
    ) -> float | str | None:
        """Return the initial setting of a PID controller.

        That is the current setting of the structure, limited to the bounds of the
        controller. Falls back to the upper bound if the structure setting is not available.
        """
        setting = self._structure_setting(structure_id, controller["steering_variable"])
        if setting is None:
            return controller["upper_bound"]
        lower_bound, upper_bound = controller["lower_bound"], controller["upper_bound"]
        if lower_bound is None or upper_bound is None:
            return setting
        try:
            lower, upper = float(lower_bound), float(upper_bound)
        except ValueError:
            return setting
        return min(max(setting, lower), upper)

    def _structure_setting(
        self, structure_id: str, steering_variable: str
    ) -> float | None:
        """Get the setting of a structure for a steering variable from the structure dataframes.

        These are the columns of the HyDAMO structure tables; set_state_from_modelrun reads the corresponding
        his-file variables, which also include values that are not in these tables (e.g. culvert gate levels).
        """
        for attribute, column in STRUCTURE_SETTING_COLUMNS.get(steering_variable, []):
            df = getattr(self.hydamo.structures, attribute, None)
            if (
                df is None
                or df.empty
                or "id" not in df.columns
                or column not in df.columns
            ):
                continue
            values = df.loc[df["id"] == structure_id, column]
            if len(values) > 0 and pd.notna(values.iloc[0]):
                try:
                    return float(values.iloc[0])
                except (TypeError, ValueError):
                    return None
        return None

    def _apply_state_overrides(self, tree_vector: ET.Element, gn_brackets: str) -> None:
        """Apply the state from set_state_from_file and the values from set_initial_state to the tree vector."""
        leaf_tag = gn_brackets + "treeVectorLeaf"
        vector_tag = gn_brackets + "vector"
        if self.state_from_file is not None:
            self._apply_state_from_file(
                self.state_from_file, tree_vector, leaf_tag, vector_tag
            )
        self._apply_initial_state(tree_vector, leaf_tag, vector_tag)

    def _apply_state_from_file(
        self,
        state: dict[str, str],
        tree_vector: ET.Element,
        leaf_tag: str,
        vector_tag: str,
    ) -> None:
        """Write the state read with set_state_from_file to the tree vector.

        Items that are not part of this RTC model are skipped; leaves that are not in the state keep
        their default value.

        Parameters
        ----------
        state : dict[str, str]
            Vector (as text) per leaf id, from set_state_from_file.
        tree_vector : ET.Element
            The treeVector element of state_import.xml.
        leaf_tag : str
            Tag of the treeVectorLeaf elements, including the namespace.
        vector_tag : str
            Tag of the vector elements, including the namespace.
        """
        valid_ids = self._dataconfig_ids()
        current = self._leaves_by_id(tree_vector, leaf_tag)
        skipped = []
        for leaf_id, vector in state.items():
            if valid_ids is not None and leaf_id not in valid_ids:
                skipped.append(leaf_id)
                continue
            if leaf_id not in current:
                current[leaf_id] = ET.SubElement(tree_vector, leaf_tag, {"id": leaf_id})
            self._leaf_vector(current[leaf_id], vector_tag).text = vector

        missing = [leaf_id for leaf_id in current if leaf_id not in state]
        if skipped:
            logger.warning(
                f"{STATE_IMPORT_XML}: {len(skipped)} state item(s) are not part of this RTC model and are skipped: {skipped[:10]}"
            )
        if missing:
            logger.warning(
                f"{STATE_IMPORT_XML}: {len(missing)} item(s) are not in the state file and keep their default value: {missing[:10]}"
            )

    def _apply_initial_state(
        self, tree_vector: ET.Element, leaf_tag: str, vector_tag: str
    ) -> None:
        """Write the values from set_initial_state to the output leaves of the controlled structures.

        Parameters
        ----------
        tree_vector : ET.Element
            The treeVector element of state_import.xml.
        leaf_tag : str
            Tag of the treeVectorLeaf elements, including the namespace.
        vector_tag : str
            Tag of the vector elements, including the namespace.
        """
        current = self._leaves_by_id(tree_vector, leaf_tag)
        for structure_id, value in self.initial_state.items():
            prefix = f"{OUTPUT_PREFIX}{structure_id}/"
            matches = [leaf_id for leaf_id in current if leaf_id.startswith(prefix)]
            if not matches:
                logger.warning(
                    f"{STATE_IMPORT_XML}: no controlled structure {structure_id}, initial state not set."
                )
            for leaf_id in matches:
                self._leaf_vector(current[leaf_id], vector_tag).text = str(value)

    @staticmethod
    def _leaves_by_id(tree_vector: ET.Element, leaf_tag: str) -> dict[str, ET.Element]:
        """Return the treeVectorLeaf elements of the tree vector by id; leaves without an id are skipped.

        Parameters
        ----------
        tree_vector : ET.Element
            The treeVector element of state_import.xml.
        leaf_tag : str
            Tag of the treeVectorLeaf elements, including the namespace.

        Returns
        -------
        dict[str, ET.Element]
            treeVectorLeaf element per id.
        """
        leaves = {}
        for leaf in tree_vector.iter(leaf_tag):
            leaf_id = leaf.get("id")
            if leaf_id is not None:
                leaves[leaf_id] = leaf
        return leaves

    @staticmethod
    def _leaf_vector(leaf: ET.Element, vector_tag: str) -> ET.Element:
        """Return the vector element of a treeVectorLeaf, created if it is missing.

        Parameters
        ----------
        leaf : ET.Element
            A treeVectorLeaf element.
        vector_tag : str
            Tag of the vector elements, including the namespace.

        Returns
        -------
        ET.Element
            The (new) vector element of the leaf.
        """
        vector = leaf.find(vector_tag)
        if vector is None:
            vector = ET.SubElement(leaf, vector_tag)
        return vector

    def _dataconfig_ids(self) -> set[str] | None:
        """Return the ids of all time series in rtcDataConfig.xml, or None if it has not been written yet.

        Uses the ids from write_dataconfig of this model; reads the file only if the data config was not written
        by this model (e.g. write_state_import called on its own).
        """
        if self._written_dataconfig_ids is not None:
            return self._written_dataconfig_ids
        path = self.output_path / RTC_DATA_CONFIG_XML
        if not path.exists():
            return None
        return self._element_ids(ET.parse(path).getroot())

    @staticmethod
    def _element_ids(root: ET.Element) -> set[str]:
        """Ids of all elements in an XML tree that have an id attribute."""
        ids = set()
        for el in root.iter():
            el_id = el.get("id")
            if el_id is not None:
                ids.add(el_id)
        return ids

    @staticmethod
    def _read_state_leaves(path: Path) -> dict[str, str]:
        """Read the id and vector of all treeVectorLeaf elements of an RTC state file."""
        leaves = {}
        for el in ET.parse(path).getroot().iter():
            if DRTCModel._strip_namespace(el.tag) != "treeVectorLeaf":
                continue
            vector = next(
                (c for c in el if DRTCModel._strip_namespace(c.tag) == "vector"), None
            )
            leaf_id = el.get("id")
            if leaf_id is not None and vector is not None and vector.text is not None:
                leaves[leaf_id] = " ".join(vector.text.split())
        return leaves

    @staticmethod
    def _read_state_time(path: Path) -> pd.Timestamp | None:
        """Read the time of an RTC state from a statePI.xml file."""
        for el in ET.parse(path).getroot().iter():
            if (
                DRTCModel._strip_namespace(el.tag) == "dateTime"
                and el.get("date") is not None
            ):
                return pd.Timestamp(f"{el.get('date')} {el.get('time', '00:00:00')}")
        return None
