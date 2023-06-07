import logging
from enum import Enum
from typing import Union
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from shapely.geometry import Point

from hydrolib.dhydamo.geometry.mesh import Network
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame

logger = logging.getLogger(__name__)


class RoughnessVariant(Enum):
    HIGH = "High"
    LOW = "Low"


class CrossSectionsIO:
    def __init__(self, crosssections):
        self.crosssections = crosssections

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_datamodel(
        self, crsdefs: pd.DataFrame = None, crslocs: pd.DataFrame = None
    ) -> None:
        """ "
        From parsed data models of crsdefs and crslocs
        """

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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def profiles(
        self,
        crosssections: ExtendedGeoDataFrame = None,
        crosssection_roughness: ExtendedDataFrame = None,
        profile_groups: ExtendedDataFrame = None,
        profile_lines: ExtendedGeoDataFrame = None,
        param_profile: ExtendedDataFrame = None,
        param_profile_values: ExtendedDataFrame = None,
        branches: Union[ExtendedGeoDataFrame, None] = None,
        roughness_variant: RoughnessVariant = None,
    ) -> None:
        """
        Method to add cross section from hydamo files. Two files
        can be handed to the function, the cross section file (dwarsprofiel) and the
        parametrised file (normgeparametriseerd). The
        hierarchical order is 1. dwarsprofiel, 2. normgeparametriseerd.
        Each branch will be assigned a profile following this order. If parametrised
        and standard are not given, branches can be without cross section. In that case
        a standard profile should be assigned
        """
        dp_branches = None
        dp_structures = None
        if profile_groups is not None:
            # check for profile_groups items with valid brugid or stuwid. They need to be droppped from profiles.
            groupidx = [
                idx
                for idx, group in profile_groups.iterrows()
                if ("brugid" in profile_groups.columns) & (group.brugid is not None)
            ]

            groupidx = groupidx + [
                idx
                for idx, group in profile_groups.iterrows()
                if ("stuwid" in profile_groups.columns) & (group.stuwid is not None)
            ]

            # index of the lines that are associated to these groups
            lineidx = [
                profile_lines[
                    profile_lines["profielgroepid"]
                    == profile_groups.loc[grindex, "globalid"]
                ].index.values[0]
                for grindex in groupidx
            ]
            # index of the profiles associated to these lines
            profidx = [
                crosssections[
                    crosssections["profiellijnid"]
                    == profile_lines.loc[lindex, "globalid"]
                ].index.values[0]
                for lindex in lineidx
            ]
            # make a copy and drop the profiles corresponding to a structure
            dp_branches = crosssections.copy(deep=True)
            dp_branches.drop(profidx, axis=0, inplace=True)

            dp_structures = crosssections.copy(deep=True)
            dp_structures = dp_structures.loc[profidx, :]
        else:
            dp_branches = crosssections.copy(deep=True)

        # Assign cross-sections to branches
        nnocross = len(self.crosssections.get_branches_without_crosssection())
        print(
            f"Before adding the number of branches without cross section is: {nnocross}."
        )

        if dp_branches is not None:
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
        print(
            f"After adding 'dwarsprofielen' the number of branches without cross section is: {nnocross}."
        )
        if nnocross == 0:
            print("No further branches without a profile.")
        elif param_profile is None:
            print("No parametrised crossections available for branches.")
        else:
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
            branchdata["chainage"] = branchdata.length / 2.0

            # Add cross sections
            for branchid, css in param_profiles_converted.items():
                chainage = branchdata.at[branchid, "chainage"]

                if css["type"] == "rectangle":
                    name = self.add_rectangle_definition(
                        height=css["height"],
                        width=css["width"],
                        closed=css["closed"],
                        roughnesstype=css["typeruwheid"],
                        roughnessvalue=css["ruwheid"],
                    )

                if css["type"] == "trapezium":
                    name = self.add_trapezium_definition(
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
                    chainage=chainage,
                    definition=name,
                    shift=css["bottomlevel"],
                )

        nnocross = len(self.crosssections.get_branches_without_crosssection())
        print(
            f"After adding 'normgeparametriseerd' the number of branches without cross section is: {nnocross}."
        )

        if dp_structures is not None:
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
    def __init__(self, external_forcings):
        self.external_forcings = external_forcings

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def boundaries(
        self, boundary_conditions: ExtendedGeoDataFrame, mesh1d: Network = None
    ) -> None:
        """
        Generate boundary conditions from hydamo 'randvoorwaarden' file. The file format does not allow for timeseries.

        To add a time series, use 'add_boundary' from the workflow.

        Parameters
        ----------
        boundary_conditions: gpd.GeoDataFrame
            geodataframe with the locations and properties of the boundary conditions

        Returns
        -------
        dictionary
            Dictionary with attributes of boundary conditions, usable for dflowfm
        """
        # Read from Hydamo
        bcdct = {}

        for bndcnd in boundary_conditions.itertuples():
            if "waterstand" in bndcnd.typerandvoorwaarde:
                quantity = "waterlevelbnd"
            elif "debiet" in bndcnd.typerandvoorwaarde:
                quantity = "dischargebnd"

            # Add boundary condition
            bcdct[bndcnd.code] = {
                "code": bndcnd.code,
                "quantity": quantity,
                "value": bndcnd.waterstand
                if not np.isnan(bndcnd.waterstand)
                else bndcnd.debiet,
                "time": None,
                "geometry": bndcnd.geometry,
            }

        # Add all items
        for key, item in bcdct.items():
            self.external_forcings.add_boundary_condition(
                key, item["geometry"], item["quantity"], item["value"], mesh1d=mesh1d
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def laterals(
        self,
        locations: ExtendedGeoDataFrame,
        overflows: ExtendedGeoDataFrame = None,
        lateral_discharges=None,
        rr_boundaries: dict = None,
    ) -> None:
        """
        Process laterals

        Parameters
        ----------
        locations: gpd.GeoDataFrame
            GeoDataFrame with at least 'geometry' (Point) and the column 'code'
        lateral_discharges: pd.DataFrame
            DataFrame with lateral discharges. The index should be a time object (datetime or similar).
        rr_boundaries: pd.DataFrame
            DataFrame with RR-catchments that are coupled
        """

        if rr_boundaries is None:
            rr_boundaries = []

        # in case of 3d points, remove the 3rd dimension
        locations["geometry2"] = [
            Point([point.geometry.x, point.geometry.y])
            for _, point in locations.iterrows()
        ]
        locations.drop("geometry", inplace=True, axis=1)
        locations.rename(columns={"geometry2": "geometry"}, inplace=True)

        latdct = {}
        if overflows is not None:
            locations = locations.append(overflows)

        # Get time series and add to dictionary
        # for nidx, lateral in zip(nearest_idx, locations.itertuples()):
        for lateral in locations.itertuples():
            # Check if a time is provided for the lateral
            if lateral.code in rr_boundaries:
                # Add to dictionary
                latdct[lateral.code] = {
                    "branchid": lateral.branch_id,
                    "chainage": str(lateral.branch_offset),
                    "discharge": "realtime",
                }

            else:
                if lateral_discharges is None:
                    logger.warning(
                        f"No lateral_discharges provided. {lateral.code} expects them. Skipping."
                    )
                    continue
                else:
                    if type(lateral_discharges) == pd.Series:
                        series = lateral_discharges.loc[lateral.code]

                        # Add to dictionary
                        latdct[lateral.code] = {
                            "branchid": lateral.branch_id,
                            "chainage": str(lateral.branch_offset),
                            "discharge": series,
                        }

                    else:
                        if lateral.code not in lateral_discharges.columns:
                            logger.warning(
                                f"No data found for {lateral.code}. Skipping."
                            )
                            continue

                        # Get timeseries
                        series = lateral_discharges.loc[:, lateral.code]

                        # Add to dictionary
                        latdct[lateral.code] = {
                            "branchid": lateral.branch_id,
                            "chainage": str(lateral.branch_offset),
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


class StructuresIO:
    def __init__(self, structures):
        self.structures = structures

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generalstructures_from_datamodel(self, generalstructures: pd.DataFrame) -> None:
        """From parsed data model of orifices

        Args:
            generalstructures (pd.DataFrame): dataframe containing the data
        """

        for generalstructure_idx, generalstructure in generalstructures.iterrows():
            self.structures.add_generalstructure(
                id=generalstructure.id,
                name=generalstructure.name
                if "name" in generalstructure.index
                else np.nan,
                branchid=generalstructure.branch_id,
                chainage=generalstructure.branch_offset,
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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def weirs(
        self,
        weirs: ExtendedGeoDataFrame = None,
        profile_groups: ExtendedDataFrame = None,
        profile_lines: ExtendedGeoDataFrame = None,
        profiles: ExtendedGeoDataFrame = None,
        opening: ExtendedDataFrame = None,
        management_device: ExtendedDataFrame = None,
        usevelocityheight: str = "true",
    ) -> None:
        """
        Method to convert HyDAMO weirs to DFlowFM structures: regular weirs, orifices and universal weirs.
        If a weir ID corresponds to a 'stuwid' in the profile_groups object, a universal weir is created.
        If the management_device corresponding to a weir has the field "overlaatonderlaat" set to "onderlaat', an orifice is schematized. In all other cases, a regular (rectangular) weir is created.

        Parameters corrspond to the HyDAMO DAMO2.2 objects. DFlowFM keyword 'usevelocityheight' can be specificied as a string, default is 'true'.
        """

        index = np.zeros((len(weirs.code)))
        if (profile_groups is not None) & ("stuwid" in profile_groups):
            index[np.isin(weirs.globalid, np.asarray(profile_groups.stuwid))] = 1

        rweirs = weirs[index == 0]
        for weir in rweirs.itertuples():
            weir_opening = opening[opening.stuwid == weir.globalid]
            weir_mandev = management_device[
                management_device.kunstwerkopeningid
                == weir_opening.globalid.to_string(index=False)
            ]

            # check if a separate name field is present
            if "naam" in weirs:
                name = weir.naam
            else:
                name = weir.code

            if (
                weir_mandev.overlaatonderlaat.to_string(index=False).lower()
                == "overlaat"
            ):
                self.structures.add_rweir(
                    id=weir.code,
                    name=name,
                    branchid=weir.branch_id,
                    chainage=weir.branch_offset,
                    crestlevel=weir_opening.laagstedoorstroomhoogte.values[0],
                    crestwidth=weir_opening.laagstedoorstroombreedte.values[0],
                    corrcoeff=weir.afvoercoefficient,
                    allowedflowdir="both",
                    usevelocityheight=usevelocityheight,
                )

            elif (
                weir_mandev.overlaatonderlaat.to_string(index=False).lower()
                == "onderlaat"
            ):
                if "maximaaldebiet" not in weir_mandev:
                    limitflow = "false"
                    maxq = 0.0
                else:
                    limitflow = "true"
                self.structures.add_orifice(
                    id=weir.code,
                    name=name,
                    branchid=weir.branch_id,
                    chainage=weir.branch_offset,
                    crestlevel=weir_opening.laagstedoorstroomhoogte.values[0],
                    crestwidth=weir_opening.laagstedoorstroombreedte.values[0],
                    corrcoeff=weir.afvoercoefficient,
                    allowedflowdir="both",
                    usevelocityheight=usevelocityheight,
                    gateloweredgelevel=weir_opening.laagstedoorstroomhoogte.values[0]
                    + weir_mandev.hoogteopening.values[0],
                    uselimitflowpos=limitflow,
                    limitflowpos=maxq,
                    uselimitflowneg=limitflow,
                    limitflowneg=maxq,
                )

        uweirs = weirs[index == 1]
        for uweir in uweirs.itertuples():
            # check if a separate name field is present
            if "naam" in uweirs:
                name = uweir.naam
            else:
                name = uweir.code

            prof = np.empty(0)
            if (profiles is not None) & ("stuwid" in profile_groups):
                group = profile_groups[profile_groups["stuwid"] == uweir.globalid]
                line = profile_lines[
                    profile_lines["profielgroepid"] == group["globalid"].values[0]
                ]
                prof = profiles[profiles["profiellijnid"] == line["globalid"].values[0]]
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
            self.structures.add_uweir(
                id=uweir.code,
                name=name,
                branchid=uweir.branch_id,
                chainage=uweir.branch_offset,
                crestlevel=uweir.laagstedoorstroomhoogte,
                crestwidth=uweir.kruinbreedte,
                dischargecoeff=uweir.afvoercoefficient,
                allowedflowdir="both",
                usevelocityheight=usevelocityheight,
                numlevels=counts,
                yvalues=" ".join([f"{yz[0]:7.3f}" for yz in yzvalues]),
                zvalues=" ".join([f"{yz[1]:7.3f}" for yz in yzvalues]),
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def weirs_from_datamodel(self, weirs: pd.DataFrame) -> None:
        """ "From parsed data model of weirs"""
        for weir_idx, weir in weirs.iterrows():
            self.structures.add_weir(
                id=weir.id,
                name=weir.name if "name" in weir.index else np.nan,
                branchid=weir.branch_id,
                chainage=weir.branch_offset,
                crestlevel=weir.crestlevel,
                crestwidth=weir.crestwidth,
                corrcoeff=weir.corrcoeff,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def orifices_from_datamodel(self, orifices: pd.DataFrame) -> None:
        """ "From parsed data model of orifices"""
        for orifice_idx, orifice in orifices.iterrows():
            self.structures.add_orifice(
                id=orifice.id,
                name=orifice.name if "name" in orifice.index else np.nan,
                branchid=orifice.branch_id,
                chainage=orifice.branch_offset,
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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def uweirs_from_datamodel(self, uweirs: pd.DataFrame) -> None:
        """ "From parsed data model of universal weirs"""
        for uweir_idx, uweir in uweirs.iterrows():
            self.structures.add_uweir(
                id=uweir.id,
                name=uweir.name if "name" in uweir.index else np.nan,
                branchid=uweir.branch_id,
                chainage=uweir.branch_offset,
                crestlevel=uweir.crestlevel,
                yvalues=uweir.yvalues,
                zvalues=uweir.zvalues,
                allowedflowdir="both",
                dischargecoeff=uweir.dischargecoeff,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def bridges(
        self,
        bridges: ExtendedGeoDataFrame,
        profile_groups: ExtendedDataFrame = None,
        profile_lines: ExtendedGeoDataFrame = None,
        profiles: ExtendedGeoDataFrame = None,
    ) -> None:
        """
        Method to convert HyDAMO bridges to DFlowFM bridges. Every bridge needs an associated YZ-profile through profile_groups ('brugid'), profile_lines and profiles.

        Parameters corrspond to the HyDAMO DAMO2.2 objects.
        """
        for bridge in bridges.itertuples():
            # first search in yz-profiles
            group = profile_groups[profile_groups["brugid"] == bridge.globalid]
            line = profile_lines[
                profile_lines["profielgroepid"] == group["globalid"].values[0]
            ]
            prof = profiles[profiles["profiellijnid"] == line["globalid"].values[0]]

            if len(prof) == 0:
                raise ValueError(f"{bridge.code} is not found in any cross-section.")

            if "naam" in bridges:
                name = bridge.naam
            else:
                name = bridge.code

            profile_id = prof.code.values[0]
            self.structures.add_bridge(
                id=bridge.code,
                name=name,
                branchid=bridge.branch_id,
                chainage=bridge.branch_offset,
                csdefid=profile_id,
                shift=0.0,
                allowedflowdir="both",
                inletlosscoeff=bridge.intreeverlies,
                outletlosscoeff=bridge.uittreeverlies,
                length=bridge.lengte,
                frictiontype=bridge.typeruwheid,
                friction=bridge.ruwheid,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def bridges_from_datamodel(self, bridges: pd.DataFrame) -> None:
        """ "From parsed data model of bridges"""
        for bridge_idx, bridge in bridges.iterrows():
            self.structures.add_bridge(
                id=bridge.code,
                name=bridge.name if "name" in bridge.index else np.nan,
                branchid=bridge.branch_id,
                chainage=bridge.branch_offset,
                csdefid=bridge.csdefid,
                shift=0.0,
                allowedflowdir="both",
                inletlosscoeff=bridge.intreeverlies,
                outletlosscoeff=bridge.uittreeverlies,
                length=bridge.lengte,
                frictiontype=bridge.typeruwheid,
                friction=bridge.ruwheid,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def culverts(
        self,
        culverts: ExtendedGeoDataFrame,
        management_device: ExtendedDataFrame = None,
    ) -> None:
        """
        Method to convert HyDAMO culverts to DFlowFM culverts. Devices like a valve and a slide can be schematized from the management_device object.
        According to HyDAMO DAMO2.2 a closing_device ('afsluitmiddel') could also be used but this is not supported.

        Parameters corrspond to the HyDAMO DAMO2.2 objects.
        """
        for culvert in culverts.itertuples():
            # Generate cross section definition name
            if culvert.vormkoker == "Rond" or culvert.vormkoker == "Ellipsvormig":
                crosssection = {"shape": "circle", "diameter": culvert.hoogteopening}
            elif (
                culvert.vormkoker == "Rechthoekig"
                or culvert.vormkoker == "Onbekend"
                or culvert.vormkoker == "Eivormig"
                or culvert.vormkoker == "Muilprofiel"
                or culvert.vormkoker == "Heulprofiel"
            ):
                crosssection = {
                    "shape": "rectangle",
                    "height": culvert.hoogteopening,
                    "width": culvert.breedteopening,
                    "closed": 1,
                }
            else:
                crosssection = {"shape": "circle", "diameter": 0.40}
                print(
                    f"Culvert {culvert.code} has an unknown shape: {culvert.vormkoker}. Applying a default profile (round - 40cm)"
                )

            # check whether an afsluitmiddel is present and take action dependent on its settings
            mandev = management_device[
                management_device.duikersifonhevelid == culvert.globalid
            ]
            if mandev.empty:
                allowedflowdir = "both"
                valveonoff = 0
                numlosscoeff = None
                valveopeningheight = 0
                relopening = None
                losscoeff = None
            else:
                for _, i in mandev.iterrows():
                    if i["soortregelmiddel"] == "terugslagklep":
                        allowedflowdir = "positive"
                    elif i["soortregelmiddel"] == "schuif":
                        allowedflowdir = "positive"
                        valveonoff = 1
                        valveopeningheight = float(i["hoogteopening"])
                        numlosscoeff = 1
                        relopening = [float(i["hoogteopening"]) / culvert.hoogteopening]
                        losscoeff = [float(i["afvoercoefficient"])]
                    else:
                        print(
                            f'Type of closing device for culvert {culvert.code} is not implemented; only "schuif" and "terugslagklep" are allowed.'
                        )

            # check if a separate name field is present
            if "naam" in culverts:
                name = culvert.naam
            else:
                name = culvert.code

            self.structures.add_culvert(
                id=culvert.code,
                name=name,
                branchid=culvert.branch_id,
                chainage=culvert.branch_offset,
                leftlevel=culvert.hoogtebinnenonderkantbov,
                rightlevel=culvert.hoogtebinnenonderkantbene,
                length=culvert.lengte,
                inletlosscoeff=culvert.intreeverlies,
                outletlosscoeff=culvert.uittreeverlies,
                crosssection=crosssection,
                allowedflowdir=allowedflowdir,
                valveonoff=valveonoff,
                numlosscoeff=numlosscoeff,
                valveopeningheight=valveopeningheight,
                relopening=relopening,
                losscoeff=losscoeff,
                bedfrictiontype=culvert.typeruwheid,
                bedfriction=culvert.ruwheid,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def culverts_from_datamodel(self, culverts: pd.DataFrame) -> None:
        """
        From parsed model of culverts
        """

        # Add to dict
        for culvert_idx, culvert in culverts.iterrows():
            self.structures.add_culvert(
                id=culvert.id,
                name=culvert.name if "name" in culvert.index else np.nan,
                branchid=culvert.branch_id,
                chainage=culvert.branch_offset,
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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pumps(
        self,
        pumpstations: ExtendedGeoDataFrame,
        pumps: ExtendedDataFrame = None,
        management: ExtendedDataFrame = None,
    ) -> None:
        """
        Method to convert HyDAMO pumps to DFlowFM pumps. Three objects are required: pumpstations, pumps and management ('sturing').

        Parameters corrspond to the HyDAMO DAMO2.2 objects.
        """

        # DAMO contains m3/min, while D-Hydro needs m3/s
        pumps["maximalecapaciteit"] /= 60

        # Add sturing to pumps
        for pump in pumps.itertuples():
            # Find sturing for pump
            sturingidx = (management.pompid == pump.globalid).values

            # find gemaal for pump
            gemaalidx = (pumpstations.globalid == pump.gemaalid).values

            # so first check if there are multiple pumps with one 'sturing'
            if sum(sturingidx) != 1:
                raise IndexError(
                    f"Multiple or no management rules found in hydamo.management for pump {pump.code}."
                )

            # If there als multiple pumping stations connected to one pump, raise an error
            if sum(gemaalidx) != 1:
                raise IndexError(
                    f"Multiple or no pump stations (gemalen) found for pump {pump.code}."
                )

            # Find the idx if the pumping station connected to the pump
            # gemaalidx = gemalen.iloc[np.where(gemaalidx)[0][0]]['code']
            # Find the control for the pumping station (and thus for the pump)
            # @sturingidx = (sturing.codegerelateerdobject == gemaalidx).values

            # assert sum(sturingidx) == 1

            branch_id = pumpstations.iloc[np.where(gemaalidx)[0][0]]["branch_id"]
            branch_offset = pumpstations.iloc[np.where(gemaalidx)[0][0]][
                "branch_offset"
            ]
            # Get the control by index
            pump_control = management.iloc[np.where(sturingidx)[0][0]]

            # if (
            #     pump_control.doelvariabele != 1
            #     and pump_control.doelvariabele != "waterstand"
            # ):
            #     raise NotImplementedError(
            #         "Sturing not implemented for anything else than water level (1)."
            #     )

            # Add levels for suction side
            startlevelsuctionside = [pump_control["bovengrens"]]
            stoplevelsuctionside = [pump_control["ondergrens"]]

            if "naam" in pumps:
                name = pump.name
            else:
                name = pump.code
            self.structures.add_pump(
                id=pump.code,
                name=name,
                branchid=branch_id,
                chainage=branch_offset,
                orientation="positive",
                numstages=1,
                controlside="suctionside",
                capacity=pump.maximalecapaciteit,
                startlevelsuctionside=startlevelsuctionside,
                stoplevelsuctionside=stoplevelsuctionside,
                startleveldeliveryside=startlevelsuctionside,
                stopleveldeliveryside=stoplevelsuctionside,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pumps_from_datamodel(self, pumps: pd.DataFrame) -> None:
        """From parsed data model of pumps"""

        for pump_idx, pump in pumps.iterrows():
            self.structures.add_pump(
                id=pump.id,
                name=pump.name if "name" in pump.index else np.nan,
                branchid=pump.branch_id,
                chainage=pump.branch_offset,
                orientation="positive",
                numstages=1,
                controlside=pump.controlside,
                capacity=pump.maximumcapacity,
                startlevelsuctionside=pump.startlevelsuctionside,
                stoplevelsuctionside=pump.stoplevelsuctionside,
                startleveldeliveryside=pump.startleveldeliveryside,
                stopleveldeliveryside=pump.stopleveldeliveryside,
            )

    def move_structure(struc, struc_dict, branch, offset):
        """
        Function the move a structure if needed for a compound structure.

        Parameters
        ----------
        struc : string
            current sub-structure id
        struc_dict : dict
            dict with all structures of a certain type
        branch : string
            branch id of the first structure in the compound
        offset : float
            chainage of the first structure in the compound

        Returns
        -------
        Dict with shifted coordinates.

        """
        branch2 = struc_dict[struc]["branchid"]
        if branch2 != branch:
            logger.warning(
                f"Structures of not on the same branche. Moving structure {struc} to branch {branch}."
            )
        struc_dict[struc]["branchid"] = branch
        struc_dict[struc]["chainage"] = offset
        return struc_dict

    def compound_structures(self, idlist, structurelist):
        # probably the coordinates should all be set to those of the first structure (still to do)
        # self.compounds_df = ExtendedDataFrame(
        #     required_columns=["code", "structurelist"]
        # )
        # self.compounds_df.set_data(
        #     pd.DataFrame(
        #         np.zeros((len(idlist), 3)),
        #         columns=["code", "numstructures", "structurelist"],
        #         dtype="str",
        #     ),
        #     index_col="code",
        # )
        # self.compounds_df.index = idlist
        # for ii, compound in enumerate(self.compounds_df.itertuples()):
        for c_i, c_id in enumerate(idlist):
            # self.compounds_df.at[compound.Index, "code"] = idlist[ii]
            # self.compounds_df.at[compound.Index, "numstructures"] = len(
            #     structurelist[ii]
            # )

            # check the substructure coordinates. If they do not coincide, move subsequent structures to the coordinates of the first
            for s_i, struc in enumerate(structurelist[c_i]):
                if s_i == 0:
                    # find out what type the first structure it is and get its coordinates                    
                    if not self.structures.pumps_df.empty:
                        if struc in list(self.structures.pumps_df.id):
                            branch = self.structures.pumps_df[
                                self.structures.pumps_df.id == struc
                            ].branchid.values[0]
                            offset = self.structures.pumps_df[
                                self.structures.pumps_df.id == struc
                            ].chainage.values[0]
                    elif not self.structures.rweirs_df.empty:
                        if struc in list(self.structures.rweirs_df.id):
                            branch = self.structures.rweirs_df[
                                self.structures.rweirs_df.id == struc
                            ].branchid.values[0]
                            offset = self.structures.rweirs_df[
                                self.structures.rweirs_df.id == struc
                            ].chainage.values[0]
                    elif not self.structures.uweirs_df.empty:
                        if struc in list(self.structures.uweirs_df.id):
                            branch = self.structures.uweirs_df[
                                self.structures.uweirs_df.id == struc
                            ].branchid.values[0]
                            offset = self.structures.uweirs_df[
                                self.structures.uweirs_df.id == struc
                            ].chainage.values[0]
                    elif not self.structures.culverts_df.empty:
                        if struc in list(self.structures.culverts_df.id):
                            branch = self.structures.culverts_df[
                                self.structures.culverts_df.id == struc
                            ].branchid.values[0]
                            offset = self.structures.culverts_df[
                                self.structures.culverts_df.id == struc
                            ].chainage.values[0]
                    elif not self.structures.bridges_df.empty:
                        if struc in list(self.structures.bridges_df.id):
                            branch = self.structures.bridges_df[
                                self.structures.bridges_df.id == struc
                            ].branchid.values[0]
                            offset = self.structures.bridges_df[
                                self.structures.pumpbridges_dfs_df.id == struc
                            ].chainage.values[0]
                    elif not self.structures.orifices_df.empty:
                        if struc in list(self.structures.orifices_df.id):
                            branch = self.structures.orifices_df[
                                self.structures.orifices_df.id == struc
                            ].branchid.values[0]
                            offset = self.structures.orifices_df[
                                self.structures.orifices_df.id == struc
                            ].chainage.values[0]                    
                    else:
                        raise ValueError('ALl structure dataframes are empty. Convert individual structures first.')
                else:
                    # move a subsequent structure to the location of the first
                    if not self.structures.pumps_df.empty:
                        if struc in list(self.structures.pumps_df.id):
                            self.structures.pumps_df.loc[
                                self.structures.pumps_df[
                                    self.structures.pumps_df.id == struc
                                ].index,
                                "branchid",
                            ] = branch
                            self.structures.pumps_df.loc[
                                self.structures.pumps_df[
                                    self.structures.pumps_df.id == struc
                                ].index,
                                "chainage",
                            ] = offset
                    elif not self.structures.rweirs_df.empty:
                        if struc in list(self.structures.rweirs_df.id):
                            self.structures.rweirs_df.loc[
                                self.structures.rweirs_df[
                                    self.structures.rweirs_df.id == struc
                                ].index,
                                "branchid",
                            ] = branch
                            self.structures.rweirs_df.loc[
                                self.structures.rweirs_df[
                                    self.structures.rweirs_df.id == struc
                                ].index,
                                "chainage",
                            ] = offset
                    elif not self.structures.uweirs_df.empty:
                        if struc in list(self.structures.uweirs_df.id):
                            self.structures.uweirs_df.loc[
                                self.structures.uweirs_df[
                                    self.structures.uweirs_df.id == struc
                                ].index,
                                "branchid",
                            ] = branch
                            self.structures.uweirs_df.loc[
                                self.structures.uweirs_df[
                                    self.structures.uweirs_df.id == struc
                                ].index,
                                "chainage",
                            ] = offset
                    elif not self.structures.culverts_df.empty:
                        if struc in list(self.structures.culverts_df.id):
                            self.structures.culverts_df.loc[
                                self.structures.culverts_df[
                                    self.structures.culverts_df.id == struc
                                ].index,
                                "branchid",
                            ] = branch
                            self.structures.culverts_df.loc[
                                self.structures.culverts_df[
                                    self.structures.culverts_df.id == struc
                                ].index,
                                "chainage",
                            ] = offset
                    elif not self.structures.bridges_df.empty:
                        if struc in list(self.structures.bridges_df.id):
                            self.structures.bridges_df.loc[
                                self.structures.bridges_df[
                                    self.structures.bridges_df.id == struc
                                ].index,
                                "branchid",
                            ] = branch
                            self.structures.bridges_df.loc[
                                self.structures.bridges_df[
                                    self.structures.bridges_df.id == struc
                                ].index,
                                "chainage",
                            ] = offset
                    elif not self.structures.orifices_df.empty:
                        if struc in list(self.structures.orifices_df.id):
                            self.structures.orifices_df.loc[
                                self.structures.orifices_df[
                                    self.structures.orifices_df.id == struc
                                ].index,
                                "branchid",
                            ] = branch
                            self.structures.orifices_df.loc[
                                self.structures.orifices_df[
                                    self.structures.orifices_df.id == struc
                                ].index,
                                "chainage",
                            ] = offset

            self.structures.add_compound(id=c_id, structureids=structurelist[c_i])
            # self.structures.compounds_df.at[compound.Index, "structurelist"] = ";".join(
            #     [f"{s}" for s in structurelist[ii]]
            # )


class StorageNodesIO:
    def __init__(self, storage_nodes):
        self.storage_nodes = {}

    def storagenodes_from_datamodel(self, storagenodes):
        """ "From parsed data model of storage nodes"""
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
