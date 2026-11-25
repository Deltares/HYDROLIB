"""Convert HyDAMO (DAMO) tables to the cross sections, structures, forcings and storage nodes of HyDAMO."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, no_type_check

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring
from pydantic import ConfigDict, validate_call
from shapely.geometry import Point

from hydrolib.dhydamo.core.relations import RelationsHyDAMO
from hydrolib.dhydamo.geometry.mesh import Network
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame

if TYPE_CHECKING:
    from hydrolib.dhydamo.core.hydamo import (
        CrossSections,
        ExternalForcings,
        StorageNodes,
        Structures,
    )

logger = logging.getLogger(__name__)


class RoughnessVariant(Enum):
    """Which roughness value of a DAMO profile to use: the high or the low value."""

    HIGH = "High"
    LOW = "Low"


class CrossSectionsIO:
    """Add cross sections from DAMO tables.

    Parameters
    ----------
    crosssections : CrossSections
        The cross sections of a HyDAMO object, filled in place.
    """

    def __init__(self, crosssections: CrossSections) -> None:
        self.crosssections = crosssections

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def from_datamodel(
        self, crsdefs: pd.DataFrame | None = None, crslocs: pd.DataFrame | None = None
    ) -> None:
        """Add cross sections from parsed data models (legacy from delft3dfmpy, untested)."""
        if crslocs is not None:
            for _, crsloc in crslocs.iterrows():
                # add location
                self.crosssections.add_crosssection_location(
                    branchid=crsloc["branch_id"],
                    chainage=crsloc["branch_offset"],
                    shift=crsloc["shift"],
                    definition=crsloc["crosssectiondefinitionid"],
                )

        if crsdefs is not None:
            crsdefs = crsdefs.drop_duplicates(subset=["crosssectiondefinitionid"])
            for _, crsdef in crsdefs.iterrows():
                # Set roughness value on default if cross-section has non defined (e.g. culverts)
                if isinstance(crsdef["frictionid"], str):
                    roughtype = crsdef["frictionid"].split("_")[0]
                else:
                    roughtype = "Chezy"

                if isinstance(crsdef["frictionid"], str):
                    roughval = float(crsdef["frictionid"].split("_")[-1])
                else:
                    roughval = 45

                # add definition
                if crsdef["type"] == "circle":
                    self.crosssections.add_circle_definition(
                        diameter=crsdef["diameter"],
                        roughnesstype=roughtype,
                        roughnessvalue=roughval,
                        name=crsdef["crosssectiondefinitionid"],
                    )
                elif crsdef["type"] == "rectangle":
                    self.crosssections.add_rectangle_definition(
                        height=crsdef["height"],
                        width=crsdef["width"],
                        closed=crsdef["closed"],
                        roughnesstype=roughtype,
                        roughnessvalue=roughval,
                        name=crsdef["crosssectiondefinitionid"],
                    )
                elif crsdef["type"] == "trapezium":
                    self.crosssections.add_trapezium_definition(
                        slope=(crsdef["t_width"] - crsdef["width"])
                        / 2
                        / crsdef["height"],
                        maximumflowwidth=crsdef["t_width"],
                        bottomwidth=crsdef["width"],
                        closed=crsdef["closed"],
                        roughnesstype=roughtype,
                        roughnessvalue=roughval,
                        name=crsdef["crosssectiondefinitionid"],
                    )
                elif crsdef["type"] == "zw":
                    self.crosssections.add_zw_definition(
                        numLevels=crsdef["numlevels"],
                        levels=crsdef["levels"],
                        flowWidths=crsdef["flowwidths"],
                        totalWidths=crsdef["totalwidths"],
                        roughnesstype=roughtype,
                        roughnessvalue=roughval,
                        name=crsdef["crosssectiondefinitionid"],
                    )

                else:
                    raise NotImplementedError

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def profiles(
        self,
        branches: ExtendedGeoDataFrame,  # = None,
        roughness_variant: RoughnessVariant,  # =None,
        crosssections: ExtendedGeoDataFrame | None = None,
        crosssection_roughness: ExtendedDataFrame | None = None,
        profile_groups: ExtendedDataFrame | None = None,
        profile_lines: ExtendedGeoDataFrame | None = None,
        param_profile: ExtendedDataFrame | None = None,
        param_profile_values: ExtendedDataFrame | None = None,
    ) -> None:
        """Add the cross sections of the measured and the parametrised profiles.

        Measured profiles (dwarsprofiel, with profile groups and lines) come first; branches
        without one get the parametrised profile (normgeparametriseerd profiel), if any.
        Branches without either have no cross section; give them a default profile, see
        ``CrossSections.set_default_definition``.

        Parameters
        ----------
        branches : ExtendedGeoDataFrame
            The branches.
        roughness_variant : RoughnessVariant
            Use the high or the low roughness value of the profiles.
        crosssections : ExtendedGeoDataFrame, optional
            Measured profile points (profielpunt). Defaults to None.
        crosssection_roughness : ExtendedDataFrame, optional
            Roughness of the measured profiles. Defaults to None.
        profile_groups : ExtendedDataFrame, optional
            Profile groups, required with ``crosssections``. Defaults to None.
        profile_lines : ExtendedGeoDataFrame, optional
            Profile lines, required with ``crosssections``. Defaults to None.
        param_profile : ExtendedDataFrame, optional
            Parametrised profiles. Defaults to None.
        param_profile_values : ExtendedDataFrame, optional
            Values of the parametrised profiles. Defaults to None.
        """
        dp_branches = None
        dp_structures = None
        if crosssections is not None and profile_groups is not None:
            if profile_lines is None:
                raise ValueError("profile_lines is required with profile_groups.")
            profile_group_bridge = RelationsHyDAMO.PROFILE_GROUP_BRIDGE
            profile_group_weir = RelationsHyDAMO.PROFILE_GROUP_WEIR
            # check for profile_groups items with valid brugid or stuwid. They need to be droppped from profiles.
            structure_group_mask = pd.Series(False, index=profile_groups.index)
            for relation in (profile_group_bridge, profile_group_weir):
                if relation.foreign_key in profile_groups.columns:
                    structure_group_mask |= profile_groups[relation.foreign_key].notna()

            structure_groups = profile_groups.loc[structure_group_mask]
            structure_lines = RelationsHyDAMO.PROFILE_LINE_GROUP.children_of(
                structure_groups,
                profile_lines,
            )
            structure_profiles = RelationsHyDAMO.PROFILE_LINE.children_of(
                structure_lines,
                crosssections,
            )
            # make a copy and drop the profiles corresponding to a structure
            dp_branches = crosssections.drop(structure_profiles.index, axis=0)

            dp_structures = gpd.GeoDataFrame(structure_profiles, copy=True)
        elif crosssections is not None:
            dp_branches = crosssections.copy(deep=True)

        # Assign cross-sections to branches
        nnocross = len(self.crosssections.get_branches_without_crosssection())
        logger.info(
            f"Before adding the number of branches without cross section is: {nnocross}."
        )

        if dp_branches is not None:
            if crosssection_roughness is None:
                raise ValueError(
                    "crosssection_roughness is required with crosssections."
                )
            # 1. Collect cross sections from 'dwarsprofielen'
            yz_profiles = self.crosssections.crosssection_to_yzprofiles(
                dp_branches,
                crosssection_roughness,
                branches,
                roughness_variant=roughness_variant,
            )

            for name, css in yz_profiles.items():
                # Add definition
                self.crosssections.add_yz_definition(
                    yz=css["yz"],
                    thalweg=css["thalweg"],
                    name=name,
                    roughnesstype=css["typeruwheid"],
                    roughnessvalue=css["ruwheid"],
                )
                # Add location
                self.crosssections.add_crosssection_location(
                    branchid=css["branchid"], chainage=css["chainage"], definition=name
                )

        # Check the number of branches with cross sections
        no_crosssection_id = self.crosssections.get_branches_without_crosssection()
        no_crosssection = [
            b for b in branches.itertuples() if b.code in no_crosssection_id
        ]

        nnocross = len(no_crosssection)
        logger.info(
            f"After adding 'dwarsprofielen' the number of branches without cross section is: {nnocross}."
        )
        if nnocross == 0:
            logger.info("No further branches without a profile.")
        elif param_profile is None:
            logger.info("No parametrised crossections available for branches.")
        else:
            if param_profile_values is None:
                raise ValueError("param_profile_values is required with param_profile.")
            # Derive norm cross sections for norm parametrised
            param_profiles_converted = self.crosssections.parametrised_to_profiles(
                param_profile,
                param_profile_values,
                no_crosssection,
                roughness_variant=roughness_variant,
            )
            # Get branch information
            branchdata = self.crosssections.hydamo.branches.loc[
                list(param_profiles_converted.keys())
            ]
            branchdata["chainage_upper"] = 0.05 * branchdata.length
            branchdata["chainage_lower"] = 0.95 * branchdata.length

            # Add cross sections
            for branchid, css in param_profiles_converted.items():
                chainage_upper = float(branchdata.loc[branchid, "chainage_upper"])
                chainage_lower = float(branchdata.loc[branchid, "chainage_lower"])

                if css["type"] == "rectangle":
                    name = self.crosssections.add_rectangle_definition(
                        height=css["height"],
                        width=css["width"],
                        closed=css["closed"],
                        roughnesstype=css["typeruwheid"],
                        roughnessvalue=css["ruwheid"],
                    )

                if css["type"] == "trapezium":
                    name = self.crosssections.add_trapezium_definition(
                        slope=css["slope"],
                        maximumflowwidth=css["maximumflowwidth"],
                        bottomwidth=css["bottomwidth"],
                        closed=css["closed"],
                        roughnesstype=css["typeruwheid"],
                        roughnessvalue=css["ruwheid"],
                    )

                # Add location
                self.crosssections.add_crosssection_location(
                    branchid=branchid,
                    chainage=chainage_upper,
                    definition=name,
                    shift=css["bottomlevel_upper"],
                )

                self.crosssections.add_crosssection_location(
                    branchid=branchid,
                    chainage=chainage_lower,
                    definition=name,
                    shift=css["bottomlevel_lower"],
                )

        nnocross = len(self.crosssections.get_branches_without_crosssection())
        logger.info(
            f"After adding 'normgeparametriseerd' the number of branches without cross section is: {nnocross}."
        )

        if dp_structures is not None:
            if crosssection_roughness is None:
                raise ValueError(
                    "crosssection_roughness is required with crosssections."
                )
            # 1. Collect cross sections from 'dwarsprofielen'
            yz_profiles = self.crosssections.crosssection_to_yzprofiles(
                dp_structures,
                crosssection_roughness,
                None,
                roughness_variant=roughness_variant,
            )

            for name, css in yz_profiles.items():
                # Add definition
                self.crosssections.add_yz_definition(
                    yz=css["yz"],
                    thalweg=css["thalweg"],
                    name=name,
                    roughnesstype=css["typeruwheid"],
                    roughnessvalue=css["ruwheid"],
                )


class ExternalForcingsIO:
    """Add boundary conditions and laterals from DAMO tables.

    Parameters
    ----------
    external_forcings : ExternalForcings
        The external forcings of a HyDAMO object, filled in place.
    """

    def __init__(self, external_forcings: ExternalForcings) -> None:
        self.external_forcings = external_forcings

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def boundaries(
        self, boundary_conditions: ExtendedGeoDataFrame, mesh1d: Network | None = None
    ) -> None:
        """Add constant boundary conditions from the DAMO table randvoorwaarden.

        The table has no time series; to add one, use
        ``ExternalForcings.add_boundary_condition``.

        Parameters
        ----------
        boundary_conditions : ExtendedGeoDataFrame
            Locations and properties of the boundary conditions: ``code``,
            ``typerandvoorwaarde`` (water level or discharge), ``waterstand`` and ``debiet``.
        mesh1d : Network, optional
            Network with the 1D mesh, to find the node of each boundary. Defaults to None.
        """
        # Read from Hydamo
        bcdct = {}

        boundary_columns = zip(
            boundary_conditions["code"],
            boundary_conditions["typerandvoorwaarde"],
            boundary_conditions["waterstand"],
            boundary_conditions["debiet"],
            boundary_conditions.geometry,
        )
        for code, kind, waterstand, debiet, geometry in boundary_columns:
            if "waterstand" in kind:
                quantity = "waterlevelbnd"
            elif "debiet" in kind:
                quantity = "dischargebnd"

            # Add boundary condition
            bcdct[code] = {
                "code": code,
                "quantity": quantity,
                "value": waterstand if not np.isnan(waterstand) else debiet,
                "time": None,
                "geometry": geometry,
            }

        # Add all items
        for key, item in bcdct.items():
            self.external_forcings.add_boundary_condition(
                key, item["geometry"], item["quantity"], item["value"], mesh1d=mesh1d
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def laterals(
        self,
        locations: ExtendedGeoDataFrame,
        overflows: ExtendedGeoDataFrame | None = None,
        greenhouse_laterals: ExtendedGeoDataFrame | None = None,
        lateral_discharges: pd.DataFrame | pd.Series | None = None,
        rr_boundaries: dict | None = None,
    ) -> None:
        """Add the laterals, with a discharge or coupled to RR.

        Laterals in ``rr_boundaries`` get their discharge from the RR model (``realtime``);
        the others from ``lateral_discharges``. Laterals without either are logged and left
        out.

        Parameters
        ----------
        locations : ExtendedGeoDataFrame
            Lateral points, with the columns ``code``, ``branch_id`` and ``branch_offset``.
        overflows : ExtendedGeoDataFrame, optional
            Overflows, added as laterals as well. Defaults to None.
        greenhouse_laterals : ExtendedGeoDataFrame, optional
            Greenhouse outlets, added as laterals as well. Defaults to None.
        lateral_discharges : pd.DataFrame or pd.Series, optional
            A time series per lateral (columns by code, a time index), or a constant
            discharge per lateral (index by code). Defaults to None.
        rr_boundaries : dict, optional
            Codes of the laterals that are coupled to RR boundary nodes. Defaults to None.
        """
        if rr_boundaries is None:
            rr_boundaries = {}

        locations.geometry = locations.geometry.force_2d()

        latdct: dict = {}
        all_locations: pd.DataFrame = locations
        if overflows is not None:
            all_locations = pd.concat([all_locations, overflows], ignore_index=True)
        if greenhouse_laterals is not None:
            all_locations = pd.concat(
                [all_locations, greenhouse_laterals], ignore_index=True
            )

        # Get time series and add to dictionary
        # for nidx, lateral in zip(nearest_idx, locations.itertuples()):
        lateral_columns = zip(
            all_locations["code"],
            all_locations["branch_id"],
            all_locations["branch_offset"],
        )
        for code, branch_id, branch_offset in lateral_columns:
            # Check if a time is provided for the lateral
            if code in rr_boundaries:
                # Add to dictionary
                latdct[code] = {
                    "branchid": branch_id,
                    "chainage": str(branch_offset),
                    "discharge": "realtime",
                }

            else:
                if lateral_discharges is None:
                    logger.warning(
                        "No lateral_discharges provided. %s expects them. Skipping.",
                        code,
                    )
                else:
                    if isinstance(lateral_discharges, pd.Series):
                        series = lateral_discharges.loc[code]

                        # Add to dictionary
                        latdct[code] = {
                            "branchid": branch_id,
                            "chainage": str(branch_offset),
                            "discharge": series,
                        }

                    else:
                        if code not in lateral_discharges.columns:
                            logger.warning(
                                "No data found for %s. Skipping.",
                                code,
                            )
                            continue

                        # Get timeseries
                        series = lateral_discharges.loc[:, code]

                        # Add to dictionary
                        latdct[code] = {
                            "branchid": branch_id,
                            "chainage": str(branch_offset),
                            "discharge": series,
                        }

        # Add all items
        for key, item in latdct.items():
            self.external_forcings.add_lateral(
                id=key,
                branchid=item["branchid"],
                chainage=item["chainage"],
                discharge=item["discharge"],
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def timeseries_from_other_model(
        self,
        his_file: Path | str,
        location_type: str,
        location_id: str,
        variable: str,
        starttime: str | datetime | None = None,
        endtime: str | datetime | None = None,
    ) -> tuple[Point | None, pd.Series]:
        """Read a time series at a location from the his file of another model.

        Parameters
        ----------
        his_file : Path or str
            The his file (``*_his.nc``) of the other model.
        location_type : str
            Type of location: ``"weir"``, ``"observation_point"``, ``"pump"``, ``"uweir"``,
            ``"bridge"``, ``"culvert"`` or ``"compound"``.
        location_id : str
            Id of the location in the his file.
        variable : str
            ``"discharge"``, ``"waterlevel"``, ``"waterlevel_upstream"`` or
            ``"waterlevel_downstream"``.
        starttime, endtime : str or datetime, optional
            Period of the output. Default to None: the whole period.

        Returns
        -------
        tuple[Point or None, pd.Series]
            The location (if the file has its coordinates) and the time series.

        Raises
        ------
        ValueError
            If the file has no data for the location and variable.
        """
        # if timesteps are strings, convert them to datetime objects
        if starttime is not None:
            if isinstance(starttime, str):
                starttime = pd.to_datetime(starttime)
            if isinstance(endtime, str):
                endtime = pd.to_datetime(endtime)
        # read the contents of the his file
        his = Dataset(his_file, "r")

        his_time = his.variables["time"][:]
        time_unit = his["time"].units.split(" ")
        refdate = pd.to_datetime(time_unit[2] + " " + time_unit[3])
        interval_seconds = his_time[1] - his_time[0]
        times = pd.date_range(
            start=refdate,
            end=refdate + pd.Timedelta(hours=np.max(his_time) / 3600.0),
            freq=f"{interval_seconds}S",
        )

        if location_type == "weir":
            dtype = "weirgen"
            loc_ids = chartostring(his[f"{dtype}_name"][:])
            coord_string_x = "weir_input_geom_node_coordx"
            coord_string_y = "weir_input_geom_node_coordy"

        elif location_type == "observation_point":
            dtype = "station"
            loc_ids = chartostring(his[f"{dtype}_name"][:])
            coord_string_x = "station_x_coordinate"
            coord_string_y = "station_y_coordinate"
        elif location_type == "pump":
            coord_string_x = f"{location_type}_input_geom_node_coordx"
            coord_string_y = f"{location_type}_input_geom_node_coordy"
            loc_ids = chartostring(his[f"{location_type}_name"][:])
            if variable == "discharge":
                dtype = "pump_structure"
            else:
                dtype = location_type
        elif location_type == "uweir":
            dtype = "uniweir"
            loc_ids = chartostring(his[f"{dtype}_name"][:])
            coord_string_x = f"{dtype}_input_geom_node_coordx"
            coord_string_y = f"{dtype}_input_geom_node_coordy"
        elif location_type == "compound":
            dtype = "cmpstru"
            loc_ids = chartostring(his[f"{dtype}_name"][:])
        else:
            dtype = location_type
            loc_ids = chartostring(his[f"{dtype}_name"][:])
            coord_string_x = f"{dtype}_input_geom_node_coordx"
            coord_string_y = f"{dtype}_input_geom_node_coordy"
        try:
            loc_index = np.nonzero(loc_ids == location_id)[0][0]
        except IndexError as err:
            raise ValueError(
                f"Location ID {location_id} of type {location_type} not found in {his_file}. Available IDs: {loc_ids}"
            ) from err

        if variable == "waterlevel_upstream":
            variable = "s1up"
        elif variable == "waterlevel_downstream":
            variable = "s1dn"
        if dtype != "cmpstru":
            loc_x = his[coord_string_x][:][loc_index]
            loc_y = his[coord_string_y][:][loc_index]

            loc_geom = Point(loc_x, loc_y)
        else:
            logger.warning(
                "Geometry for compound structures not supported yet. Get the location of one of its structures."
            )
            loc_geom = None

        if location_type == "observation_point":
            timeseries = his["waterlevel"][:, loc_index]
        else:
            timeseries = his[f"{dtype}_{variable}"][:, loc_index]
        loc_series = pd.Series(timeseries, index=times, name=location_id)
        if starttime is not None:
            loc_series = loc_series.loc[starttime:endtime]
            if loc_series.empty:
                logger.error(
                    f"No data available for {location_id} between {starttime} and {endtime}."
                )
        return loc_geom, loc_series


class StructuresIO:
    """Add structures from DAMO tables.

    Parameters
    ----------
    structures : Structures
        The structures of a HyDAMO object, filled in place.
    """

    def __init__(self, structures: Structures) -> None:
        self.structures = structures

    def _name_or_nan(self, row: pd.Series) -> Hashable:
        """Return the index label of the row if it has a name column, otherwise NaN."""
        return row.name if "name" in row.index else np.nan

    def _common_structure_kwargs(
        self, row: pd.Series, id_attr: str = "id"
    ) -> dict[str, object]:
        """Return the id, name, branch id and chainage of a structure row."""
        return {
            "id": getattr(row, id_attr),
            "name": self._name_or_nan(row),
            "branchid": row.branch_id,
            "chainage": row.branch_offset,
        }

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def generalstructures_from_datamodel(self, generalstructures: pd.DataFrame) -> None:
        """Add general structures from a parsed data model (legacy from delft3dfmpy, untested)."""
        for generalstructure_idx, generalstructure in generalstructures.iterrows():
            base = self._common_structure_kwargs(generalstructure)
            self.structures.add_generalstructure(
                **base,
                allowedflowdir="both",
                upstream1width=generalstructure.upstream1width
                if "upstream1width" in generalstructure.index
                else np.nan,
                upstream1level=generalstructure.upstream1level
                if "upstream1level" in generalstructure.index
                else np.nan,
                upstream2width=generalstructure.upstream2width
                if "upstream2width" in generalstructure.index
                else np.nan,
                upstream2level=generalstructure.upstream2level
                if "upstream2level" in generalstructure.index
                else np.nan,
                crestwidth=generalstructure.crestwidth
                if "crestwidth" in generalstructure.index
                else np.nan,
                crestlevel=generalstructure.crestlevel
                if "crestlevel" in generalstructure.index
                else np.nan,
                crestlength=generalstructure.crestlength
                if "crestlength" in generalstructure.index
                else np.nan,
                downstream1width=generalstructure.downstream1width
                if "downstream1width" in generalstructure.index
                else np.nan,
                downstream1level=generalstructure.downstream1level
                if "downstream1level" in generalstructure.index
                else np.nan,
                downstream2width=generalstructure.downstream2width
                if "downstream2width" in generalstructure.index
                else np.nan,
                downstream2level=generalstructure.downstream2level
                if "downstream2level" in generalstructure.index
                else np.nan,
                gateloweredgelevel=generalstructure.gateloweredgelevel
                if "gateloweredgelevel" in generalstructure.index
                else np.nan,
                posfreegateflowcoeff=generalstructure.posfreegateflowcoeff
                if "posfreegateflowcoeff" in generalstructure.index
                else np.nan,
                posdrowngateflowcoeff=generalstructure.posdrowngateflowcoeff
                if "posdrowngateflowcoeff" in generalstructure.index
                else np.nan,
                posfreeweirflowcoeff=generalstructure.posfreeweirflowcoeff
                if "posfreeweirflowcoeff" in generalstructure.index
                else np.nan,
                posdrownweirflowcoeff=generalstructure.posdrownweirflowcoeff
                if "posdrownweirflowcoeff" in generalstructure.index
                else np.nan,
                poscontrcoeffreegate=generalstructure.poscontrcoeffreegate
                if "poscontrcoeffreegate" in generalstructure.index
                else np.nan,
                negfreegateflowcoeff=generalstructure.negfreegateflowcoeff
                if "negfreegateflowcoeff" in generalstructure.index
                else np.nan,
                negdrowngateflowcoeff=generalstructure.negdrowngateflowcoeff
                if "negdrowngateflowcoeff" in generalstructure.index
                else np.nan,
                negfreeweirflowcoeff=generalstructure.negfreeweirflowcoeff
                if "negfreeweirflowcoeff" in generalstructure.index
                else np.nan,
                negdrownweirflowcoeff=generalstructure.negdrownweirflowcoeff
                if "negdrownweirflowcoeff" in generalstructure.index
                else np.nan,
                negcontrcoeffreegate=generalstructure.negcontrcoeffreegate
                if "negcontrcoeffreegate" in generalstructure.index
                else np.nan,
                extraresistance=generalstructure.extraresistance
                if "extraresistance" in generalstructure.index
                else np.nan,
                gateheight=generalstructure.gateheight
                if "gateheight" in generalstructure.index
                else np.nan,
                gateopeningwidth=generalstructure.gateopeningwidth
                if "gateopeningwidth" in generalstructure.index
                else np.nan,
                gateopeninghorizontaldirection=generalstructure.gateopeninghorizontaldirection
                if "gateopeninghorizontaldirection" in generalstructure.index
                else np.nan,
                usevelocityheight=generalstructure.usevelocityheight
                if "usevelocityheight" in generalstructure.index
                else np.nan,
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def weirs(
        self,
        weirs: ExtendedGeoDataFrame | None = None,
        profile_groups: ExtendedDataFrame | None = None,
        profile_lines: ExtendedGeoDataFrame | None = None,
        profiles: ExtendedGeoDataFrame | None = None,
        opening: ExtendedDataFrame | None = None,
        management_device: ExtendedDataFrame | None = None,
        usevelocityheight: bool = True,
    ) -> None:
        """Add the DAMO weirs as regular weirs, orifices or universal weirs.

        A weir with a profile group (``stuwid``) becomes a universal weir. A weir whose
        management device has ``overlaatonderlaat`` set to ``"onderlaat"`` becomes an
        orifice. Other weirs become regular (rectangular) weirs. A weir with more than one
        opening becomes a compound structure.

        Parameters
        ----------
        weirs : ExtendedGeoDataFrame, optional
            DAMO weirs (stuw). Defaults to None.
        profile_groups, profile_lines, profiles : optional
            DAMO profile groups, lines and points, for universal weirs. Default to None.
        opening : ExtendedDataFrame, optional
            DAMO openings (kunstwerkopening). Defaults to None.
        management_device : ExtendedDataFrame, optional
            DAMO management devices (regelmiddel). Defaults to None.
        usevelocityheight : bool, optional
            D-Flow FM keyword ``usevelocityheight``. Defaults to True.
        """
        opening_weir = RelationsHyDAMO.OPENING_WEIR
        management_device_opening = RelationsHyDAMO.MANAGEMENT_DEVICE_OPENING
        profile_group_weir = RelationsHyDAMO.PROFILE_GROUP_WEIR
        profile_line_group = RelationsHyDAMO.PROFILE_LINE_GROUP
        profile_line = RelationsHyDAMO.PROFILE_LINE
        if weirs is None or opening is None or management_device is None:
            raise ValueError("weirs, opening and management_device are required.")
        index = np.zeros(len(weirs.code))
        if (
            profile_groups is not None
            and profile_group_weir.foreign_key in profile_groups
        ):
            index[
                np.isin(
                    weirs[opening_weir.parent_key],
                    np.asarray(profile_groups[profile_group_weir.foreign_key]),
                )
            ] = 1

        rweirs = weirs[index == 0]
        for weir in rweirs.itertuples():
            weir_rows = rweirs.loc[[weir.Index]]
            weir_opening = opening_weir.children_of(weir_rows, opening)

            # the name, or the code when there is none
            if "naam" in weirs.columns and pd.notna(weir.naam):
                name = str(weir.naam)
            else:
                name = str(weir.code)

            if weir_opening.shape[0] > 1:
                logger.info(
                    "Weir %s contains %d openings. Creating a compound structure with a fictional weir for each one.",
                    weir.code,
                    weir_opening.shape[0],
                )
                cmp_list = []
                for num_op, (op_index, op_row) in enumerate(weir_opening.iterrows()):
                    opening_rows = weir_opening.loc[[op_index]]
                    weir_mandev = management_device_opening.children_of(
                        opening_rows,
                        management_device,
                    )

                    if weir_mandev.empty:
                        logger.warning(
                            "Skipping opening %s (weir %s) because there is no associated management device.",
                            op_row.code,
                            weir.code,
                        )
                        continue
                    if weir_mandev.shape[0] > 1:
                        logger.warning(
                            "Multiple management devices associated to opening %s (weir %s). Taking the first one.",
                            op_row.code,
                            weir.code,
                        )
                        weir_mandev = weir_mandev.iloc[[0]]

                    # This weir has multiple openings, so its structure id gets an
                    # enumeration suffix that is invented right here and is not
                    # present anywhere in the HyDAMO data. Record it so DRTCModel
                    # can later resolve a management record's regelmiddelid to
                    # this structure id; see Structures.compound_weir_structure_ids
                    # in core/hydamo.py.
                    weir_id = f"{weir.code}_{num_op + 1}"
                    self.structures.compound_weir_structure_ids[
                        str(weir_mandev["globalid"].iloc[0])
                    ] = weir_id
                    kind = str(weir_mandev["overlaatonderlaat"].iloc[0]).lower()
                    if kind == "overlaat":
                        cmp_list.append(weir_id)
                        self.structures.add_rweir(
                            id=weir_id,
                            name=name,
                            branchid=weir.branch_id,
                            chainage=weir.branch_offset,
                            crestlevel=op_row.laagstedoorstroomhoogte,
                            crestwidth=op_row.laagstedoorstroombreedte,
                            corrcoeff=weir.afvoercoefficient,
                            allowedflowdir="both",
                            usevelocityheight=usevelocityheight,
                        )
                    elif kind == "onderlaat":
                        cmp_list.append(weir_id)
                        if "maximaaldebiet" not in weir_mandev or pd.isna(
                            weir_mandev.maximaaldebiet.to_numpy()[0]
                        ):
                            limitflow = False
                            maxq = 0.0
                        else:
                            limitflow = True
                            maxq = float(weir_mandev.maximaaldebiet.to_numpy()[0])
                        self.structures.add_orifice(
                            id=weir_id,
                            name=name,
                            branchid=weir.branch_id,
                            chainage=weir.branch_offset,
                            crestlevel=float(
                                weir_opening.laagstedoorstroomhoogte.to_numpy()[0]
                            ),
                            crestwidth=float(
                                weir_opening.laagstedoorstroombreedte.to_numpy()[0]
                            ),
                            corrcoeff=weir.afvoercoefficient,
                            allowedflowdir="both",
                            usevelocityheight=usevelocityheight,
                            gateloweredgelevel=float(
                                weir_opening.laagstedoorstroomhoogte.to_numpy()[0]
                            )
                            + float(weir_mandev.hoogteopening.to_numpy()[0]),
                            uselimitflowpos=limitflow,
                            limitflowpos=maxq,
                            uselimitflowneg=limitflow,
                            limitflowneg=maxq,
                        )
                    else:
                        logger.warning(
                            'Skipping weir %s - from "overlaatonderlaat" %s the type of structure could not be determined.',
                            weir.code,
                            weir_mandev.overlaatonderlaat,
                        )
                self.structures.add_compound(
                    id=f"cmp_{weir.code}", structureids=cmp_list
                )
                # self.structures.rweirs_df.drop(weir.code)
            else:
                if weir_opening.empty:
                    logger.warning(
                        "Skipping weir %s because there is no associated opening.",
                        weir.code,
                    )
                    continue

                weir_id = weir.code
                # Unlike the compound case above, no need to record this id in
                # compound_weir_structure_ids: this weir has a single opening, so
                # its structure id is just weir.code, which
                # DRTCModel._resolve_weir_via_opening derives directly from
                # HyDAMO's static relations.
                weir_mandev = management_device_opening.children_of(
                    weir_opening,
                    management_device,
                )

                if weir_mandev.empty:
                    logger.warning(
                        "Skipping opening %s (weir %s) because there is no associated management device.",
                        weir_opening.code.squeeze(),
                        weir_id,
                    )
                    continue
                if weir_mandev.shape[0] > 1:
                    logger.warning(
                        "Multiple management devices associated to opening %s (weir %s). Taking the first one.",
                        weir_opening.code.squeeze(),
                        weir_id,
                    )
                    weir_mandev = weir_mandev.iloc[[0]]

                overlaatonderlaat = str(weir_mandev["overlaatonderlaat"].iloc[0])

                if overlaatonderlaat.lower() == "overlaat":
                    self.structures.add_rweir(
                        id=weir_id,
                        name=name,
                        branchid=weir.branch_id,
                        chainage=weir.branch_offset,
                        crestlevel=weir_opening.laagstedoorstroomhoogte.to_numpy()[0],
                        crestwidth=weir_opening.laagstedoorstroombreedte.to_numpy()[0],
                        corrcoeff=weir.afvoercoefficient,
                        allowedflowdir="both",
                        usevelocityheight=usevelocityheight,
                    )

                elif overlaatonderlaat.lower() == "onderlaat":
                    if "maximaaldebiet" not in weir_mandev or pd.isna(
                        weir_mandev.maximaaldebiet.to_numpy()[0]
                    ):
                        limitflow = False
                        maxq = 0.0
                    else:
                        limitflow = True
                        maxq = float(weir_mandev.maximaaldebiet.to_numpy()[0])
                    self.structures.add_orifice(
                        id=weir_id,
                        name=name,
                        branchid=weir.branch_id,
                        chainage=weir.branch_offset,
                        crestlevel=float(
                            weir_opening.laagstedoorstroomhoogte.to_numpy()[0]
                        ),
                        crestwidth=float(
                            weir_opening.laagstedoorstroombreedte.to_numpy()[0]
                        ),
                        corrcoeff=weir.afvoercoefficient,
                        allowedflowdir="both",
                        usevelocityheight=usevelocityheight,
                        gateloweredgelevel=float(
                            weir_opening.laagstedoorstroomhoogte.to_numpy()[0]
                        )
                        + float(weir_mandev.hoogteopening.to_numpy()[0]),
                        uselimitflowpos=limitflow,
                        limitflowpos=maxq,
                        uselimitflowneg=limitflow,
                        limitflowneg=maxq,
                    )
                else:
                    logger.warning(
                        'Skipping %s - from "overlaatonderlaat" %s the type of structure could not be determined.',
                        weir.code,
                        weir_mandev.overlaatonderlaat,
                    )

        uweirs = weirs[index == 1]
        for uweir in uweirs.itertuples():
            uweir_rows = uweirs.loc[[uweir.Index]]
            # the name, or the code when there is none
            if "naam" in uweirs.columns and pd.notna(uweir.naam):
                name = str(uweir.naam)
            else:
                name = str(uweir.code)

            prof = pd.DataFrame()
            if (
                profiles is not None
                and profile_groups is not None
                and profile_lines is not None
                and profile_group_weir.foreign_key in profile_groups
            ):
                group = profile_group_weir.children_of(uweir_rows, profile_groups)
                line = profile_line_group.children_of(group, profile_lines)
                prof = profile_line.children_of(line, profiles)
                if not prof.empty:
                    counts = len(prof.geometry.iloc[0].coords[:])
                    xyz = np.vstack(prof.geometry.iloc[0].coords[:])
                    length = np.r_[
                        0,
                        np.cumsum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1]))),
                    ]
                    yzvalues = np.c_[length, xyz[:, -1] - np.min(xyz[:, -1])]

            if len(prof) == 0:
                # return an error it is still not found
                raise ValueError(f"{uweir.code} is not found in any cross-section.")

            if not hasattr(uweir, "laagstedoorstroomhoogte") or pd.isna(
                uweir.laagstedoorstroomhoogte
            ):
                kruinhoogte = np.min(xyz[:, -1])
            else:
                kruinhoogte = uweir.laagstedoorstroomhoogte
            self.structures.add_uweir(
                id=uweir.code,
                name=name,
                branchid=uweir.branch_id,
                chainage=uweir.branch_offset,
                crestlevel=kruinhoogte,
                dischargecoeff=uweir.afvoercoefficient,
                allowedflowdir="both",
                numlevels=counts,
                yvalues=" ".join([f"{yz[0]:7.3f}" for yz in yzvalues]),
                zvalues=" ".join([f"{yz[1]:7.3f}" for yz in yzvalues]),
            )

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def weirs_from_datamodel(self, weirs: pd.DataFrame) -> None:
        """Add weirs from a parsed data model (legacy from delft3dfmpy, untested)."""
        for weir_idx, weir in weirs.iterrows():
            base = self._common_structure_kwargs(weir)
            self.structures.add_weir(
                **base,
                crestlevel=weir.crestlevel,
                crestwidth=weir.crestwidth,
                corrcoeff=weir.corrcoeff,
            )

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def orifices_from_datamodel(self, orifices: pd.DataFrame) -> None:
        """Add orifices from a parsed data model (legacy from delft3dfmpy, untested)."""
        for orifice_idx, orifice in orifices.iterrows():
            base = self._common_structure_kwargs(orifice)
            self.structures.add_orifice(
                **base,
                allowedflowdir="both",
                crestlevel=orifice.crestlevel,
                crestwidth=orifice.crestwidth,
                gateloweredgelevel=orifice.gateloweredgelevel,
                corrcoeff=orifice.corrcoef,
                uselimitflowpos=orifice.uselimitflowpos,
                limitflowpos=orifice.limitflowpos,
                uselimitflowneg=orifice.uselimitflowneg,
                limitflowneg=orifice.limitflowneg,
            )

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def uweirs_from_datamodel(self, uweirs: pd.DataFrame) -> None:
        """Add universal weirs from a parsed data model (legacy from delft3dfmpy, untested)."""
        for uweir_idx, uweir in uweirs.iterrows():
            base = self._common_structure_kwargs(uweir)
            self.structures.add_uweir(
                **base,
                crestlevel=uweir.crestlevel,
                yvalues=uweir.yvalues,
                zvalues=uweir.zvalues,
                allowedflowdir="both",
                dischargecoeff=uweir.dischargecoeff,
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def bridges(
        self,
        bridges: ExtendedGeoDataFrame,
        profile_groups: ExtendedDataFrame | None = None,
        profile_lines: ExtendedGeoDataFrame | None = None,
        profiles: ExtendedGeoDataFrame | None = None,
    ) -> None:
        """Add the DAMO bridges.

        Every bridge needs a YZ profile, through ``profile_groups`` (``brugid``),
        ``profile_lines`` and ``profiles``.

        Parameters
        ----------
        bridges : ExtendedGeoDataFrame
            DAMO bridges (brug).
        profile_groups, profile_lines, profiles : optional
            DAMO profile groups, lines and points. Default to None.
        """
        profile_group_bridge = RelationsHyDAMO.PROFILE_GROUP_BRIDGE
        profile_line_group = RelationsHyDAMO.PROFILE_LINE_GROUP
        profile_line = RelationsHyDAMO.PROFILE_LINE
        if profile_groups is None or profile_lines is None or profiles is None:
            raise ValueError(
                "profile_groups, profile_lines and profiles are required for bridges."
            )
        for bridge in bridges.itertuples():
            bridge_rows = bridges.loc[[bridge.Index]]
            # first search in yz-profiles
            group = profile_group_bridge.children_of(bridge_rows, profile_groups)
            line = profile_line_group.children_of(group, profile_lines)
            prof = profile_line.children_of(line, profiles)

            if len(prof) == 0:
                raise ValueError(f"{bridge.code} is not found in any cross-section.")

            # the name, if present, otherwise the code
            name = str(bridge.code)
            if "naam" in bridges and pd.notna(bridge.naam):
                name = str(bridge.naam)

            profile_id = prof.code.to_numpy()[0]
            self.structures.add_bridge(
                id=str(bridge.code),
                name=name,
                branchid=str(bridge.branch_id),
                chainage=pd.to_numeric(bridge.branch_offset),
                csdefid=profile_id,
                shift=0.0,
                allowedflowdir="both",
                inletlosscoeff=pd.to_numeric(bridge.intreeverlies),
                outletlosscoeff=pd.to_numeric(bridge.uittreeverlies),
                length=pd.to_numeric(bridge.lengte),
                frictiontype=str(bridge.typeruwheid),
                friction=pd.to_numeric(bridge.ruwheid),
            )

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def bridges_from_datamodel(self, bridges: pd.DataFrame) -> None:
        """Add bridges from a parsed data model (legacy from delft3dfmpy, untested)."""
        for bridge_idx, bridge in bridges.iterrows():
            base = self._common_structure_kwargs(bridge, id_attr="code")
            self.structures.add_bridge(
                **base,
                csdefid=bridge.csdefid,
                shift=0.0,
                allowedflowdir="both",
                inletlosscoeff=bridge.intreeverlies,
                outletlosscoeff=bridge.uittreeverlies,
                length=bridge.lengte,
                frictiontype=bridge.typeruwheid,
                friction=bridge.ruwheid,
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def culverts(
        self,
        culverts: ExtendedGeoDataFrame,
        management_device: ExtendedDataFrame | None = None,
    ) -> None:
        """Add the DAMO culverts.

        A valve or a slide can be schematized from ``management_device``. DAMO 2.2 also has
        closing devices (afsluitmiddel), which are not supported.

        Parameters
        ----------
        culverts : ExtendedGeoDataFrame
            DAMO culverts (duikersifonhevel).
        management_device : ExtendedDataFrame, optional
            DAMO management devices (regelmiddel). Defaults to None.
        """
        management_device_culvert = RelationsHyDAMO.MANAGEMENT_DEVICE_CULVERT
        if (
            management_device is not None
            and "soortafsluitmiddel" not in management_device.columns
        ):
            management_device["soortafsluitmiddel"] = management_device[
                "soortregelmiddel"
            ]

        for culvert in culverts.itertuples():
            culvert_rows = culverts.loc[[culvert.Index]]
            # Generate cross section definition name
            vormkoker = str(culvert.vormkoker).lower()
            if vormkoker in {"rond", "ellipsvormig"}:
                crosssection = {"shape": "circle", "diameter": culvert.hoogteopening}
            elif vormkoker in {
                "rechthoekig",
                "onbekend",
                "eivormig",
                "muilprofiel",
                "heulprofiel",
            }:
                crosssection = {
                    "shape": "rectangle",
                    "height": culvert.hoogteopening,
                    "width": culvert.breedteopening,
                    "closed": 1,
                }
            else:
                crosssection = {"shape": "circle", "diameter": 0.40}
                logger.info(
                    f"Culvert {culvert.code} has an unknown shape: {culvert.vormkoker}. Applying a default profile (round - 40cm)"
                )

            # check whether an afsluitmiddel is present and take action dependent on its settings
            if management_device is not None:
                mandev = management_device_culvert.children_of(
                    culvert_rows,
                    management_device,
                )
                if "soortafsluitmiddel" not in mandev:
                    mandev.loc[mandev.index, "soortafsluitmiddel"] = mandev[
                        "soortregelmiddel"
                    ]
            else:
                mandev = pd.DataFrame()

            # default settings
            allowedflowdir = "both"
            valveonoff = 0
            numlosscoeff = None
            valveopeningheight: float = 0
            relopening = None
            losscoeff = None
            if not mandev.empty:
                for _, i in mandev.iterrows():
                    if i["soortafsluitmiddel"] == "terugslagklep":
                        allowedflowdir = "positive"
                    elif i["soortafsluitmiddel"] == "schuif":
                        valveonoff = 1
                        valveopeningheight = float(i["hoogteopening"])
                        numlosscoeff = 1
                        relopening = [
                            float(i["hoogteopening"])
                            / float(culvert_rows["hoogteopening"].to_numpy()[0])
                        ]
                        losscoeff = [float(i["afvoercoefficient"])]
                    else:
                        raise NotImplementedError(
                            f'Type of management device for culvert {culvert.code} is not implemented; only "schuif" and "terugslagklep" are allowed.'
                        )

            # check if a separate name field is present
            # the name, if present, otherwise the code
            name = str(culvert.code)
            if "naam" in culverts and pd.notna(culvert.naam):
                name = str(culvert.naam)

            self.structures.add_culvert(
                id=str(culvert.code),
                name=name,
                branchid=str(culvert.branch_id),
                chainage=pd.to_numeric(culvert.branch_offset),
                leftlevel=pd.to_numeric(culvert.hoogtebinnenonderkantbov),
                rightlevel=pd.to_numeric(culvert.hoogtebinnenonderkantbene),
                length=pd.to_numeric(culvert.lengte),
                inletlosscoeff=pd.to_numeric(culvert.intreeverlies),
                outletlosscoeff=pd.to_numeric(culvert.uittreeverlies),
                crosssection=crosssection,
                allowedflowdir=allowedflowdir,
                valveonoff=valveonoff,
                numlosscoeff=numlosscoeff,
                valveopeningheight=valveopeningheight,
                relopening=relopening,
                losscoeff=losscoeff,
                bedfrictiontype=str(culvert.typeruwheid),
                bedfriction=pd.to_numeric(culvert.ruwheid),
            )

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def culverts_from_datamodel(self, culverts: pd.DataFrame) -> None:
        """Add culverts from a parsed data model (legacy from delft3dfmpy, untested)."""
        # Add to dict
        for culvert_idx, culvert in culverts.iterrows():
            base = self._common_structure_kwargs(culvert)
            self.structures.add_culvert(
                **base,
                leftlevel=culvert.leftlevel,
                rightlevel=culvert.rightlevel,
                crosssection=culvert.crosssectiondefinitionid,
                length=culvert.geometry.length
                if "geometry" in culvert.index
                else culvert.length,
                inletlosscoeff=culvert.inletlosscoeff,
                outletlosscoeff=culvert.outletlosscoeff,
                allowedflowdir="both",
                valveonoff=0,
                numlosscoeff=0,
                valveopeningheight=np.nan,
                relopening=np.nan,
                losscoeff=np.nan,
                frictiontype=culvert.frictiontype,
                frictionvalue=culvert.frictionvalue,
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def pumps(
        self,
        pumpstations: ExtendedGeoDataFrame,
        pumps: ExtendedDataFrame | None = None,
        management: ExtendedDataFrame | None = None,
    ) -> None:
        """Add the DAMO pumps.

        A pump station with more than one pump becomes a compound structure.

        Parameters
        ----------
        pumpstations : ExtendedGeoDataFrame
            DAMO pump stations (gemaal).
        pumps : ExtendedDataFrame
            DAMO pumps (pomp). Required.
        management : ExtendedDataFrame
            DAMO control (sturing). Required.

        Raises
        ------
        ValueError
            If ``pumps`` or ``management`` is missing.
        """
        pump_pumpstation = RelationsHyDAMO.PUMP_PUMPSTATION
        management_pump = RelationsHyDAMO.MANAGEMENT_PUMP
        if pumps is None or management is None:
            raise ValueError("pumps and management are required.")

        # Add sturing to pumps
        for pumpstation in pumpstations.itertuples():
            pumpstation_rows = pumpstations.loc[[pumpstation.Index]]

            # find pumps for gemaal
            pumps_subset = pump_pumpstation.children_of(pumpstation_rows, pumps)
            if pumps_subset.empty:
                logger.warning(
                    "Skipping %s because there is no associated pump.",
                    pumpstation.code,
                )
                continue

            # the name, or the code when there is none
            if "naam" in pumpstations.columns and pd.notna(pumpstation.naam):
                name = str(pumpstation.naam)
            else:
                name = str(pumpstation.code)
            is_compound = len(pumps_subset) > 1
            cmp_list = []
            if is_compound:
                logger.info(
                    "Pumpstation %s contains %d openings. Creating a compound structure with a fictional pump for each one.",
                    pumpstation.code,
                    pumps_subset.shape[0],
                )

            for ipump, (pump_index, pump) in enumerate(pumps_subset.iterrows()):
                pump_rows = pumps_subset.loc[[pump_index]]
                pump_control = management_pump.children_of(
                    pump_rows,
                    management,
                )
                if pump_control.empty:
                    if is_compound:
                        logger.warning("No management found for %s", pump.code)
                    else:
                        logger.warning(
                            "Skipping %s because there is no associated management.",
                            pumpstation.code,
                        )
                    continue
                if len(pump_control) > 1:
                    raise ValueError(
                        f"Pump station {pumpstation.code}: expected one management row "
                        f"per pump, found {len(pump_control)}."
                    )
                startlevelsuctionside = [pump_control["bovengrens"].iloc[0]]
                stoplevelsuctionside = [pump_control["ondergrens"].iloc[0]]

                if is_compound:
                    pumpid = f"{pumpstation.code}_{ipump + 1}"
                else:
                    pumpid = str(pumpstation.code)
                cmp_list.append(pumpid)
                self.structures.add_pump(
                    id=pumpid,
                    name=str(name),
                    branchid=str(pumpstation.branch_id),
                    chainage=pd.to_numeric(pumpstation.branch_offset),
                    orientation="positive",
                    numstages=1,
                    controlside="suctionside",
                    capacity=pump.maximalecapaciteit / 60.0,
                    startlevelsuctionside=startlevelsuctionside,
                    stoplevelsuctionside=stoplevelsuctionside,
                    startleveldeliveryside=startlevelsuctionside,
                    stopleveldeliveryside=stoplevelsuctionside,
                )

            if is_compound:
                self.structures.add_compound(
                    id=f"cmp_{pumpstation.code}", structureids=cmp_list
                )

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def pumps_from_datamodel(self, pumps: pd.DataFrame) -> None:
        """Add pumps from a parsed data model (legacy from delft3dfmpy, untested)."""
        for pump_idx, pump in pumps.iterrows():
            base = self._common_structure_kwargs(pump)
            self.structures.add_pump(
                **base,
                orientation="positive",
                numstages=1,
                controlside=pump.controlside,
                capacity=pump.maximumcapacity,
                startlevelsuctionside=pump.startlevelsuctionside,
                stoplevelsuctionside=pump.stoplevelsuctionside,
                startleveldeliveryside=pump.startleveldeliveryside,
                stopleveldeliveryside=pump.stopleveldeliveryside,
            )

    @staticmethod
    def move_structure(
        struc: str,
        struc_dict: dict[str, dict[str, str | float]],
        branch: str,
        offset: float,
    ) -> dict[str, dict[str, str | float]]:
        """Move a structure to the location of the first structure of its compound, in place.

        Parameters
        ----------
        struc : str
            Id of the structure to move.
        struc_dict : dict
            All structures of one type, by id.
        branch : str
            Branch id of the first structure in the compound.
        offset : float
            Chainage of the first structure in the compound.

        Returns
        -------
        dict
            ``struc_dict``, with the moved structure.
        """
        branch2 = struc_dict[struc]["branchid"]
        if branch2 != branch:
            logger.warning(
                f"Structures of not on the same branche. Moving structure {struc} to branch {branch}."
            )
        struc_dict[struc]["branchid"] = branch
        struc_dict[struc]["chainage"] = offset
        return struc_dict

    def _structure_frames(self) -> tuple[pd.DataFrame, ...]:
        """Return the tables of all structure types that can be part of a compound."""
        return (
            self.structures.pumps_df,
            self.structures.rweirs_df,
            self.structures.uweirs_df,
            self.structures.culverts_df,
            self.structures.bridges_df,
            self.structures.orifices_df,
        )

    def _get_structure_location(
        self, structure_id: str
    ) -> tuple[str | None, float | None]:
        """Return the branch id and chainage of a structure, or None if it is not found."""
        branch = None
        offset = None
        for df in self._structure_frames():
            if df.empty:
                continue
            match = df[df.id == structure_id]
            if not match.empty:
                branch = match.branchid.to_numpy()[0]
                offset = match.chainage.to_numpy()[0]
        return branch, offset

    def _set_structure_location(
        self, structure_id: str, branch: str, offset: float
    ) -> None:
        """Set the branch id and chainage of a structure, in place."""
        for df in self._structure_frames():
            if df.empty:
                continue
            idx = df.index[df.id == structure_id]
            if not idx.empty:
                df.loc[idx, "branchid"] = branch
                df.loc[idx, "chainage"] = offset

    def compound_structures(
        self, idlist: list[str], structurelist: list[list[str]]
    ) -> None:
        """Add compound structures, and move their structures to the location of the first.

        Parameters
        ----------
        idlist : list[str]
            Id of each compound structure.
        structurelist : list[list[str]]
            Ids of the structures of each compound structure.
        """
        # probably the coordinates should all be set to those of the first structure (still to do)
        for c_i, c_id in enumerate(idlist):
            branch = None
            offset = None
            # check the substructure coordinates. If they do not coincide, move subsequent structures to the coordinates of the first
            for s_i, struc in enumerate(structurelist[c_i]):
                if s_i == 0:
                    # find out what type the first structure it is and get its coordinates
                    branch, offset = self._get_structure_location(struc)
                else:
                    # move a subsequent structure to the location of the first
                    if branch is not None and offset is not None:
                        self._set_structure_location(struc, branch, offset)

            self.structures.add_compound(id=c_id, structureids=structurelist[c_i])


class StorageNodesIO:
    """Add storage nodes from tables.

    Parameters
    ----------
    storagenodes : StorageNodes
        The storage nodes of a HyDAMO object, filled in place.
    """

    def __init__(self, storagenodes: StorageNodes) -> None:
        self.storagenodes = storagenodes

    # legacy from delft3dfmpy, not used or tested here (some are broken): mypy skips it
    @no_type_check
    def storagenodes_from_datamodel(self, storagenodes):
        """Add storage nodes from a parsed data model (legacy from delft3dfmpy, untested)."""
        for storagenode_idx, storagenode in storagenodes.iterrows():
            self.add_storagenode(
                id=storagenode.id,
                name=storagenode.name if "name" in storagenode.index else np.nan,
                usestreetstorage=storagenode.usestreetstorage,
                nodetype="unspecified",
                nodeid=storagenode.nodeid,
                usetable="false",
                bedlevel=storagenode.bedlevel,
                area=storagenode.area,
                streetlevel=storagenode.streetlevel,
                streetstoragearea=storagenode.streetstoragearea,
                storagetype=storagenode.storagetype,
            )

    def storagenodes_from_input(
        self,
        storagenodes: gpd.GeoDataFrame | None = None,
        nodeid: str | None = None,
        xy: tuple[float, float] | None = None,
        storagedata: pd.DataFrame | None = None,
        usestreetstorage: bool = True,
        network: Network | None = None,
    ) -> None:
        """Add storage nodes with a level-area table.

        Parameters
        ----------
        storagenodes : gpd.GeoDataFrame
            Storage nodes, indexed by code, with ``branch_id`` and ``branch_offset``. Required.
        nodeid : str, optional
            Node id for all storage nodes. Defaults to None: the node at the chainage.
        xy : tuple[float, float], optional
            Location for all storage nodes. Defaults to None.
        storagedata : pd.DataFrame
            Level-area table, with the columns ``code``, ``level`` and ``area``. Required.
        usestreetstorage : bool, optional
            D-Flow FM keyword ``useStreetStorage``. Defaults to True.
        network : Network, optional
            Network with the 1D mesh, to find the node at the chainage. Defaults to None.

        Raises
        ------
        ValueError
            If ``storagenodes`` or ``storagedata`` is missing.
        """
        if storagenodes is None or storagedata is None:
            raise ValueError("Provide storagenodes and storagedata.")

        for storagenode_idx, storagenode in storagenodes.iterrows():
            data = storagedata[storagedata.code == storagenode_idx]
            # the index label of the row, or the code when that is missing
            name = storagenode.name
            if name is None or (isinstance(name, float) and np.isnan(name)):
                name = storagenode.code
            self.storagenodes.add_storagenode(
                id=str(storagenode_idx),
                name=str(name),
                usestreetstorage=usestreetstorage,
                nodetype="unspecified",
                nodeid=nodeid,
                xy=xy,
                branchid=storagenode.branch_id,
                chainage=storagenode.branch_offset,
                usetable="true",
                storagearea=" ".join(data.area.astype(str)),
                levels=" ".join(data.level.astype(str)),
                network=network,
            )
