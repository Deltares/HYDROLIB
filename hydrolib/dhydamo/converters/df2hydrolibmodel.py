import logging
import pandas as pd
import numpy as np
from pathlib import Path
from hydrolib.core.dflowfm.structure.models import (
    Weir,
    UniversalWeir,
    Orifice,
    Bridge,
    Pump,
    Culvert,
    Compound,
)
from hydrolib.core.dflowfm.crosssection.models import (
    CircleCrsDef,
    RectangleCrsDef,
    YZCrsDef,
    CrossSection,
    ZWCrsDef,
)
from hydrolib.core.dflowfm.ext.models import Boundary, Lateral
from hydrolib.core.dflowfm.bc.models import (
    ForcingModel,
    TimeSeries,
    Constant,
    QuantityUnitPair,
)
from hydrolib.core.dflowfm.friction.models import FrictGlobal
from hydrolib.core.dflowfm.obs.models import ObservationPoint
from hydrolib.core.dflowfm.storagenode.models import StorageNode
from hydrolib.core.dflowfm.inifield.models import InitialField
from hydrolib.core.dflowfm.onedfield.models import OneDFieldGlobal

logger = logging.getLogger(__name__)


class Df2HydrolibModel:
    def __init__(self, hydamo, assign_default_profiles=False):
        self.hydamo = hydamo
        self.structures = []
        self.crossdefs = []
        self.crosslocs = []
        self.boundaries = []
        self.boundaries_ext = []
        self.boundaries_bc = []
        self.laterals_ext = []
        self.laterals_bc = []
        self.friction_defs = []
        self.obspoints = []
        self.storagenodes = []
        self.inifields = []
        self.onedfields = []

        self.assign_default_profiles = assign_default_profiles
        self.forcingmodel = ForcingModel()
        self.forcingmodel.filepath = "boundaryconditions.bc"
        self.forcingmodel.forcing = []

        self.onedfieldmodels = []

        self.write_all()

    def write_all(self):
        """Wrapper function to convert all seperate objects"""
        self.regular_weirs_to_dhydro()
        self.orifices_to_dhydro()
        self.universal_weirs_to_dhydro()
        self.bridges_to_dhydro()
        self.culverts_to_dhydro()
        self.pumps_to_dhydro()
        self.compounds_to_dhydro()
        self.crosssection_locations_to_dhydro()
        self.crosssection_definitions_to_dhydro()
        self.friction_definitions_to_dhydro()
        self.boundaries_to_dhydro()
        self.laterals_to_dhydro()
        self.observation_points_to_dhydro()
        self.storagenodes_to_dhydro()
        self.inifields_to_dhydro()

    @staticmethod
    def _clear_comments(lst):
        """Convenience function to remove comment statements in INI files"""
        if isinstance(lst, list):
            for item in lst:
                [setattr(item.comments, field[0], "") for field in item.comments]
        else:
            [setattr(lst.comments, field[0], "") for field in lst.comments]

    def compounds_to_dhydro(self):
        """Convert compound structures to Compound-model"""
        structs = [
            Compound(**struc)
            for struc in self.hydamo.structures.compounds_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def regular_weirs_to_dhydro(self):
        """Convert regular weirs to Weir-model"""
        structs = [
            Weir(**struc)
            for struc in self.hydamo.structures.rweirs_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def orifices_to_dhydro(self):
        """Convert orifices to Orfice-models"""
        structs = [
            Orifice(**struc)
            for struc in self.hydamo.structures.orifices_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def universal_weirs_to_dhydro(self):
        """Convert universal weirs to UniversalWeir-models"""
        structs = [
            UniversalWeir(**struc)
            for struc in self.hydamo.structures.uweirs_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def bridges_to_dhydro(self):
        """Convert bridges to Bridge-models"""
        structs = [
            Bridge(**struc)
            for struc in self.hydamo.structures.bridges_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def culverts_to_dhydro(self):
        """Convert culverts to Culvert-models"""
        structs = [
            Culvert(**struc)
            for struc in self.hydamo.structures.culverts_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def pumps_to_dhydro(self):
        """Convert pumps to Pump-models"""
        structs = [
            Pump(**struc)
            for struc in self.hydamo.structures.pumps_df.to_dict("records")
        ]
        self._clear_comments(structs)
        self.structures += structs

    def crosssection_locations_to_dhydro(self):
        """Convert crosssection locations to CrossLoc models"""
        # Check which of the branches do not have a cross section. Add the default one to those
        branchids = set(
            [
                dct["branchid"]
                for dct in self.hydamo.crosssections.crosssection_loc.values()
            ]
        )

        # check if there locations specified where a default profile should be added
        if hasattr(self.hydamo.crosssections, "default_locations"):
            default_locs = self.hydamo.crosssections.default_locations
        else:
            default_locs = self.hydamo.branches.index[
                ~self.hydamo.branches["code"].isin(branchids)
            ]

        # If any are missing, check if a default cross section definition has been defined
        if (len(default_locs) > 0) & (self.assign_default_profiles):
            if self.hydamo.crosssections.default_definition is None:
                 print(
                    "Not all branches have a cross section appointed to them. We assume D-Hydro can interpolate over these branches, otherwise errors will occur."
                )
            logger.info(
                f"Adding default cross section definition to branches with ids: {', '.join(list(default_locs))}"
            )
            for branchid in default_locs:
                self.hydamo.crosssections.add_crosssection_location(
                    branchid=branchid,
                    chainage=self.hydamo.branches.at[branchid, "geometry"].length / 2,
                    definition=self.hydamo.crosssections.default_definition,
                    minz=np.nan,
                    shift=self.hydamo.crosssections.default_definition_shift,
                )

        # Add cross sections to dhydro format
        css = [
            CrossSection(**cloc)
            for cloc in self.hydamo.crosssections.crosssection_loc.values()
        ]

        self._clear_comments(css)
        self.crosslocs += css

    def crosssection_definitions_to_dhydro(self) -> None:
        """Convert crosssection definitions to Crossdef models"""

        def _get_cstype_part_of_dict(cstype: str) -> dict:
            """Reorganize the csdef dict"""
            return {
                key: dct
                for key, dct in self.hydamo.crosssections.crosssection_def.items()
                if dct["type"] == cstype
            }

        # Circles
        cs_circle = _get_cstype_part_of_dict("circle")
        cs = [CircleCrsDef(**cs) for cs in cs_circle.values()]
        self._clear_comments(cs)
        self.crossdefs += cs

        # YZ
        cs_yz = _get_cstype_part_of_dict("yz")
        cs = [YZCrsDef(**cs) for cs in cs_yz.values()]
        self._clear_comments(cs)
        self.crossdefs += cs

        # Rectangle
        cs_rect = _get_cstype_part_of_dict("rectangle")
        cs = [RectangleCrsDef(**cs) for cs in cs_rect.values()]
        self._clear_comments(cs)
        self.crossdefs += cs

        # Trapezium
        cs_zw = _get_cstype_part_of_dict("zw")
        cs = [ZWCrsDef(**cs) for cs in cs_zw.values()]
        self._clear_comments(cs)
        self.crossdefs += cs

    def boundaries_to_dhydro(self) -> None:
        """Convert dataframe of boundaries to ext and bc models"""
        for bound in self.hydamo.external_forcings.boundary_nodes.values():
            if bound["time"] is None:
                bnd_bc = Constant(
                    name=bound["nodeid"],
                    function="constant",
                    quantity=bound["quantity"],
                    unit=bound["value_unit"],
                    datablock=[[bound["value"]]],
                )
            else:
                bnd_bc = TimeSeries(
                    name=bound["nodeid"],
                    function="timeseries",
                    timeinterpolation="linear",
                    quantityunitpair=[
                        QuantityUnitPair(quantity="time", unit=bound["time_unit"]),
                        QuantityUnitPair(
                            quantity=bound["quantity"], unit=bound["value_unit"]
                        ),
                    ],
                    datablock=list(map(list, zip(bound["time"], bound["value"]))),
                )
            self.forcingmodel.forcing.append(bnd_bc)
        for bound in self.hydamo.external_forcings.boundary_nodes.values():
            bnd_ext = Boundary(
                nodeid=bound["nodeid"],
                quantity=bound["quantity"],
                forcingfile=self.forcingmodel,
            )
            bnd_ext.forcingfile.filepath = Path("boundaryconditions.bc")
            self.boundaries_ext.append(bnd_ext)

    def laterals_to_dhydro(self) -> None:
        """Convert dataframe of laterals to ext and bc models"""
                
        for key, lateral in self.hydamo.external_forcings.lateral_nodes.items():                    
            if isinstance(lateral["discharge"], str):
                # realtime boundary                
                lat_ext = Lateral(
                    id=key,
                    name=key,
                    type="discharge",
                    locationType="1d",
                    branchId=lateral["branchid"],
                    chainage=lateral["chainage"],
                    discharge=lateral["discharge"],
                )
            else:
                # time series or constant value
                if isinstance(lateral["discharge"], pd.Series):                    
                    lat_bc = TimeSeries(
                        name=key,
                        function="timeseries",
                        timeinterpolation="linear",
                        quantityunitpair=[
                            QuantityUnitPair(quantity="time", unit=lateral["time_unit"]),
                            QuantityUnitPair(
                                quantity="lateral_discharge", unit="m3/s"
                            ),
                        ],
                        datablock=list(map(list, zip(lateral["time"], lateral["value"]))),
                    )
                    self.laterals_bc.append(lat_bc)
                elif isinstance(lateral["discharge"], float):                    
                    lat_bc = Constant(
                        name=key,
                        function="constant",
                        quantity="lateral_discharge",
                        unit="m3/s",
                        datablock=[[lateral["discharge"]]],
                    )
                self.forcingmodel.forcing.append(lat_bc)                
                lat_ext = Lateral(
                    id=key,
                    name=key,
                    type=lateral["type"],
                    locationtype=lateral["locationtype"],
                    branchId=lateral["branchid"],
                    chainage=lateral["chainage"],
                    discharge=self.forcingmodel,
                )            
            self.laterals_ext.append(lat_ext)

    def friction_definitions_to_dhydro(self):
        """Convert friction definitions to FrictGlobal-objects"""
        frictdefs = [
            FrictGlobal(**frictdef)
            for frictdef in self.hydamo.roughness_definitions.values()
        ]
        self._clear_comments(frictdefs)
        self.friction_defs += frictdefs

    def storagenodes_to_dhydro(self):
        """Convert dataframe of storagenodes to StorageNode-objects"""
        stornodes = [
            StorageNode(**stornode)
            for stornode in self.hydamo.storagenodes.storagenodes.values()
        ]
        self._clear_comments(stornodes)
        self.storagenodes += stornodes

    def observation_points_to_dhydro(self):
        """Convert dataframe of observationpoints to ObserationPoint-objects"""
        if hasattr(self.hydamo.observationpoints, 'observation_points'):
            obspoints = [
                ObservationPoint(**obs)
                for obs in self.hydamo.observationpoints.observation_points.drop(columns="geometry").to_dict(
                    "records"
                )
            ]
            self._clear_comments(obspoints)
            self.obspoints += obspoints

    def inifields_to_dhydro(self):
        """Convert initial conditions to InitialField objects"""
        for (
            level
        ) in self.hydamo.external_forcings.initial_waterlevel_polygons.itertuples():
            inifield = InitialField(
                quantity="waterlevel",
                datafiletype="1dField",
                unit="m",
                datafile="initialwaterdepth.ini",
            )
            if level.geometry is None:
                onedfield = OneDFieldGlobal(
                    quantity="waterlevel",
                    locationtype=level.locationtype,
                    unit="m",
                    value=str(level.value),
                )
            self._clear_comments(inifield)
            self._clear_comments(onedfield)
            self.inifields.append(inifield)
            self.onedfields.append(onedfield)

        for (
            depth
        ) in self.hydamo.external_forcings.initial_waterdepth_polygons.itertuples():
            inifield = InitialField(
                quantity="waterdepth",
                datafiletype="1dField",
                # unit="m",
                datafile="initialwaterdepth.ini",
            )
            if depth.geometry is None:
                onedfield = OneDFieldGlobal(
                    quantity="waterdepth",
                    # locationtype=depth.locationtype,
                    unit="m",
                    value=str(depth.waterdepth),
                )
            self._clear_comments(inifield)
            self._clear_comments(onedfield)
            self.inifields.append(inifield)
            self.onedfieldmodels.append(onedfield)

            #          onedfieldmodel = OneDFieldModel(global_=sekf.onedfields[0])
            # onedfieldmodel.filepath = Path("initialwaterdepth.ini")
            # fm.geometry.inifieldfile = onedfieldmodel
            # self.onedfields.append(onedfield)
