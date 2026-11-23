"""Convert the HyDAMO data to HYDROLIB-core models (structures, cross sections, forcings, fields)."""

import itertools
import logging
from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pydantic import Field
from pyproj import CRS
from scipy.interpolate import LinearNDInterpolator

from hydrolib.core.dflowfm.bc.models import (
    Constant,
    ForcingBase,
    ForcingModel,
    QHTable,
    QuantityUnitPair,
    TimeInterpolation,
    TimeSeries,
)
from hydrolib.core.dflowfm.crosssection.models import (
    CircleCrsDef,
    CrossSection,
    CrossSectionDefinition,
    RectangleCrsDef,
    YZCrsDef,
    ZWCrsDef,
)
from hydrolib.core.dflowfm.ext.models import Boundary, ExtModel, Lateral
from hydrolib.core.dflowfm.friction.models import FrictGlobal
from hydrolib.core.dflowfm.ini.models import INIBasedModel
from hydrolib.core.dflowfm.inifield.models import (
    IniFieldModel,
    InitialField,
    ParameterField,
)
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.net.models import Network
from hydrolib.core.dflowfm.obs.models import ObservationPoint
from hydrolib.core.dflowfm.onedfield.models import (
    OneDFieldBranch,
    OneDFieldGlobal,
    OneDFieldModel,
)
from hydrolib.core.dflowfm.polyfile.models import Metadata, PolyFile, PolyObject
from hydrolib.core.dflowfm.polyfile.models import Point as PolyPoint
from hydrolib.core.dflowfm.storagenode.models import StorageNode
from hydrolib.core.dflowfm.structure.models import (
    Bridge,
    Compound,
    Culvert,
    Orifice,
    Pump,
    Structure,
    UniversalWeir,
    Weir,
)
from hydrolib.core.dflowfm.xyz.models import XYZModel, XYZPoint
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import common
from hydrolib.dhydamo.geometry.spatial import (
    line_parts,
    points_to_branch_values,
    polygons_to_branch_values,
    rasterize_polygons,
)
from hydrolib.dhydamo.io.rasterwriter import write_raster

logger = logging.getLogger(__name__)

# Version of the external forcings file (bnd.ext). hydrolib-core 1.6 writes 3.00 by default, which D-Flow FM in
# D-HYDRO 2026.02 refuses ("Unsupported format of new external forcing file ... v3.00. Current format: v2.02").
# See https://github.com/Deltares/HYDROLIB/issues/248.
EXT_FILE_VERSION = "2.02"

# The D-HYDRO GUI holds one 1D initial condition per quantity and always writes it to these files. All 1D fields
# per branch of a quantity are therefore combined into one file with this name.
ONEDFIELD_FILES = {
    "waterlevel": "InitialWaterLevel.ini",
    "waterdepth": "InitialWaterDepth.ini",
}


class ConstantWithInterpolation(Constant):
    """Constant [Forcing] block that also writes timeInterpolation.

    Workaround for the D-HYDRO GUI (2026.02), which raises an error when timeInterpolation is missing,
    although it has no meaning for a constant. hydrolib-core's Constant has no such field (a value passed
    to it is silently dropped). D-Flow FM (2026.02) accepts it and ignores it for constant values.
    """

    timeinterpolation: TimeInterpolation = Field(
        TimeInterpolation.linear, alias="timeInterpolation"
    )


class QHTableWithInterpolation(QHTable):
    """QH-table [Forcing] block that also writes timeInterpolation (same GUI workaround as ConstantWithInterpolation)."""

    timeinterpolation: TimeInterpolation = Field(
        TimeInterpolation.linear, alias="timeInterpolation"
    )


@dataclass
class _Mesh1dNodes:
    """Location of the 1D computational nodes.

    Attributes
    ----------
    x, y : np.ndarray
        Coordinates of the nodes.
    branchid : np.ndarray
        Branch id per node.
    chainage : np.ndarray
        Chainage of the node on its branch.
    points : gpd.array.GeometryArray
        Point geometry per node.
    """

    x: np.ndarray
    y: np.ndarray
    branchid: np.ndarray
    chainage: np.ndarray
    points: gpd.array.GeometryArray


class _NodeValues:
    """Initial water level or depth at the 1D nodes, built up by applying the initial fields in order.

    Reproduces how D-Flow FM applied the fields in tests with D-HYDRO 2026.02: where fields overlap the
    last one applies, also between water level and water depth; a water depth from polygons or samples
    removes earlier water levels from polygons or samples; a uniform water depth overrides spatial water
    levels.

    Parameters
    ----------
    n : int
        Number of nodes.
    """

    def __init__(self, n: int) -> None:
        self.quantity = np.full(n, None, dtype=object)
        self.value = np.full(n, np.nan)
        self.source = np.full(n, None, dtype=object)
        self.source_type = np.full(n, None, dtype=object)

    def apply(
        self, kind: str, values: np.ndarray, filename: str | None, filetype: str
    ) -> None:
        """Apply a field: nodes where it has a value get that value.

        Parameters
        ----------
        kind : str
            'waterlevel' or 'waterdepth'.
        values : np.ndarray
            Value per node, NaN where the field gives no value.
        filename : str | None
            Data file of the field.
        filetype : str
            'uniform', 'branch', 'sample' or 'polygon'.
        """
        if kind == "waterdepth" and filetype in ("polygon", "sample"):
            removed = (self.quantity == "waterlevel") & np.isin(
                self.source_type, ["polygon", "sample"]
            )
            self.quantity[removed], self.value[removed] = None, np.nan
            self.source[removed], self.source_type[removed] = None, None
        has_value = ~np.isnan(values)
        self.quantity[has_value], self.value[has_value] = kind, values[has_value]
        self.source[has_value], self.source_type[has_value] = filename, filetype

    def apply_uniform_depth(self, values: np.ndarray, filename: str | None) -> None:
        """Override all water levels with a uniform water depth, whatever the order of the fields.

        Parameters
        ----------
        values : np.ndarray
            The uniform water depth at every node.
        filename : str | None
            Data file of the uniform water depth.
        """
        spatial_level = self.quantity == "waterlevel"
        self.quantity[spatial_level], self.value[spatial_level] = (
            "waterdepth",
            values[spatial_level],
        )
        self.source[spatial_level] = filename


class _CombinedOneDFields:
    """Initial water level and water depth per branch, combined into one 1dField file per quantity.

    A field per branch fills whole branches and the last field wins, so per branch the values of the last field
    are kept, and a branch that a later field gives the other quantity is moved to that quantity.

    Attributes
    ----------
    branches : dict[str, dict[Hashable, tuple[list[float], list[float]]]]
        Chainages and values per branch id, for 'waterlevel' and 'waterdepth'.
    quantity : dict[str, str]
        Quantity name of the last field per kind, as used in the [Initial] block.
    """

    def __init__(self) -> None:
        self.branches: dict[str, dict[Hashable, tuple[list[float], list[float]]]] = {
            "waterlevel": {},
            "waterdepth": {},
        }
        self.quantity: dict[str, str] = {}

    def add(
        self,
        kind: str,
        quantity: str,
        branch_values: dict[Hashable, tuple[list[float], list[float]]],
    ) -> None:
        """Add the values of a field per branch; they replace earlier values on the same branches.

        Parameters
        ----------
        kind : str
            'waterlevel' or 'waterdepth'.
        quantity : str
            Quantity name of the field.
        branch_values : dict[Hashable, tuple[list[float], list[float]]]
            Chainages and values per branch id.
        """
        other = "waterdepth" if kind == "waterlevel" else "waterlevel"
        for branch_id in branch_values:
            self.branches[other].pop(branch_id, None)
        self.branches[kind].update(branch_values)
        self.quantity[kind] = quantity


class Df2HydrolibModel:
    """HYDROLIB-core objects for everything in a HyDAMO object.

    All conversions are done when the object is created; the results are in its
    attributes, e.g. ``structures``, ``crossdefs``, ``crosslocs`` and ``inifields``.

    Parameters
    ----------
    hydamo : HyDAMO
        The HyDAMO data to convert.
    assign_default_profiles : bool, optional
        Give the branches without a cross section the default cross section definition
        of ``hydamo.crosssections``, halfway along the branch. Defaults to False.
    """

    def __init__(self, hydamo: HyDAMO, assign_default_profiles: bool = False) -> None:
        self.hydamo = hydamo
        self.structures: list[Structure] = []
        self.crossdefs: list[CrossSectionDefinition] = []
        self.crosslocs: list[CrossSection] = []
        self.boundaries: list[Boundary] = []
        self.boundaries_ext: list[Boundary] = []
        self.boundaries_bc: list[ForcingBase] = []
        self.laterals_ext: list[Lateral] = []
        self.laterals_bc: list[ForcingBase] = []
        self.friction_defs: list[FrictGlobal] = []
        self.obspoints: list[ObservationPoint] = []
        self.storagenodes: list[StorageNode] = []
        self.inifields: list[InitialField] = []
        self.parameterfields: list[ParameterField] = []
        # data for the spatial fields, written by write_spatial_fields()
        self.spatial_data: dict[str, dict] = {}

        self.assign_default_profiles = assign_default_profiles
        self.forcingmodel = ForcingModel()
        self.forcingmodel.filepath = Path("boundaryconditions.bc")
        self.forcingmodel.forcing = []

        self.onedfieldmodels: list[OneDFieldGlobal] = []

        self.write_all()

    def write_all(self) -> None:
        """Convert all objects; called when the object is created."""
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
        self.spatial_fields_to_dhydro()

    def ext_model(self) -> ExtModel:
        """Return the external forcings model (bnd.ext) with the boundaries and laterals.

        Use as ``fm.external_forcing.extforcefilenew = models.ext_model()``. The file version
        is set to ``EXT_FILE_VERSION``, as D-HYDRO 2026.02 does not read the default version of
        hydrolib-core 1.6.

        Returns
        -------
        ExtModel
            The external forcings model, readable by D-HYDRO.
        """
        extmodel = ExtModel(boundary=self.boundaries_ext, lateral=self.laterals_ext)
        extmodel.general.fileversion = EXT_FILE_VERSION
        return extmodel

    @staticmethod
    def _clear_comments(lst: INIBasedModel | list) -> None:
        """Remove the comments of INI objects, in place."""
        if not isinstance(lst, list):
            lst = [lst]
        for item in lst:
            if item.comments is None:
                continue
            for field, _ in item.comments:
                setattr(item.comments, field, "")

    def compounds_to_dhydro(self) -> None:
        """Convert the compound structures to Compound objects."""
        structs = [
            Compound(**struc)
            # into=dict only affects the type mypy sees, not the result:
            # - without it, pandas-stubs gives dict[Hashable, Any], which mypy rejects for **struc;
            # - into=dict[str, Any] would be exact, but pandas raises a TypeError on it;
            # - the bare dict gives dict[Any, Any], which is safe: our column names are always strings.
            for struc in self.hydamo.structures.compounds_df.to_dict(
                "records", into=dict
            )
        ]
        self._clear_comments(structs)
        self.structures += structs

    def regular_weirs_to_dhydro(self) -> None:
        """Convert the regular weirs to Weir objects."""
        structs = [
            Weir(**struc)
            # into=dict: see compounds_to_dhydro
            for struc in self.hydamo.structures.rweirs_df.to_dict("records", into=dict)
        ]
        self._clear_comments(structs)
        self.structures += structs

    def orifices_to_dhydro(self) -> None:
        """Convert the orifices to Orifice objects."""
        structs = [
            Orifice(**struc)
            # into=dict: see compounds_to_dhydro
            for struc in self.hydamo.structures.orifices_df.to_dict(
                "records", into=dict
            )
        ]
        self._clear_comments(structs)
        self.structures += structs

    def universal_weirs_to_dhydro(self) -> None:
        """Convert the universal weirs to UniversalWeir objects."""
        structs = [
            UniversalWeir(**struc)
            # into=dict: see compounds_to_dhydro
            for struc in self.hydamo.structures.uweirs_df.to_dict("records", into=dict)
        ]
        self._clear_comments(structs)
        self.structures += structs

    def bridges_to_dhydro(self) -> None:
        """Convert the bridges to Bridge objects."""
        structs = [
            Bridge(**struc)
            # into=dict: see compounds_to_dhydro
            for struc in self.hydamo.structures.bridges_df.to_dict("records", into=dict)
        ]
        self._clear_comments(structs)
        self.structures += structs

    def culverts_to_dhydro(self) -> None:
        """Convert the culverts to Culvert objects."""
        structs = [
            Culvert(**struc)
            # into=dict: see compounds_to_dhydro
            for struc in self.hydamo.structures.culverts_df.to_dict(
                "records", into=dict
            )
        ]
        self._clear_comments(structs)
        self.structures += structs

    def pumps_to_dhydro(self) -> None:
        """Convert the pumps to Pump objects."""
        structs = [
            Pump(**struc)
            # into=dict: see compounds_to_dhydro
            for struc in self.hydamo.structures.pumps_df.to_dict("records", into=dict)
        ]
        self._clear_comments(structs)
        self.structures += structs

    def crosssection_locations_to_dhydro(self) -> None:
        """Convert the cross section locations to CrossSection objects.

        With ``assign_default_profiles``, the default definition is first added to the
        default locations (or else to the branches without a cross section).
        """
        # Check which of the branches do not have a cross section. Add the default one to those
        branchids = {
            dct["branchid"]
            for dct in self.hydamo.crosssections.crosssection_loc.values()
        }

        # check if there locations specified where a default profile should be added
        default_locs: list[str] | pd.Index
        if hasattr(self.hydamo.crosssections, "default_locations"):
            default_locs = self.hydamo.crosssections.default_locations
        else:
            default_locs = self.hydamo.branches.index[
                ~self.hydamo.branches["code"].isin(branchids)
            ]

        # If any are missing, check if a default cross section definition has been defined
        if (len(default_locs) > 0) & (self.assign_default_profiles):
            default_definition = self.hydamo.crosssections.default_definition
            if default_definition is None:
                logger.warning(
                    "Not all branches have a cross section appointed to them. We assume D-Hydro can interpolate over these branches, otherwise errors will occur."
                )
            else:
                logger.info(
                    f"Adding default cross section definition to branches with ids: {', '.join(list(default_locs))}"
                )
                half_lengths = self.hydamo.branches.geometry.length / 2
                for branchid in default_locs:
                    self.hydamo.crosssections.add_crosssection_location(
                        branchid=branchid,
                        chainage=float(half_lengths.loc[branchid]),
                        definition=default_definition,
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
        """Convert the cross section definitions to CrossSectionDefinition objects."""

        def _get_cstype_part_of_dict(cstype: str) -> dict:
            """Return the cross section definitions of one type."""
            return {
                key: dct
                for key, dct in self.hydamo.crosssections.crosssection_def.items()
                if dct["type"] == cstype
            }

        # Circles
        cs_circle = _get_cstype_part_of_dict("circle")
        circles = [CircleCrsDef(**cs) for cs in cs_circle.values()]
        self._clear_comments(circles)
        self.crossdefs += circles

        # YZ
        cs_yz = _get_cstype_part_of_dict("yz")
        yz_profiles = [YZCrsDef(**cs) for cs in cs_yz.values()]
        self._clear_comments(yz_profiles)
        self.crossdefs += yz_profiles

        # Rectangle
        cs_rect = _get_cstype_part_of_dict("rectangle")
        rectangles = [RectangleCrsDef(**cs) for cs in cs_rect.values()]
        self._clear_comments(rectangles)
        self.crossdefs += rectangles

        # Trapezium
        cs_zw = _get_cstype_part_of_dict("zw")
        zw_profiles = [ZWCrsDef(**cs) for cs in cs_zw.values()]
        self._clear_comments(zw_profiles)
        self.crossdefs += zw_profiles

    def boundaries_to_dhydro(self) -> None:
        """Convert the boundaries to [Boundary] blocks and forcings."""
        for bound in self.hydamo.external_forcings.boundary_nodes.values():
            bnd_bc: ConstantWithInterpolation | TimeSeries | QHTableWithInterpolation
            if bound["vec1"] is None:
                bnd_bc = ConstantWithInterpolation(
                    name=bound["nodeid"],
                    function="constant",
                    quantity=bound["quantity"],
                    unit=bound["unit2"],
                    datablock=[[bound["vec2"]]],
                )
            elif bound["unit1"].startswith("minutes"):
                bnd_bc = TimeSeries(
                    name=bound["nodeid"],
                    function="timeseries",
                    timeinterpolation="linear",
                    quantityunitpair=[
                        QuantityUnitPair(quantity="time", unit=bound["unit1"]),
                        QuantityUnitPair(
                            quantity=bound["quantity"], unit=bound["unit2"]
                        ),
                    ],
                    datablock=list(map(list, zip(bound["vec1"], bound["vec2"]))),
                )
            else:
                bnd_bc = QHTableWithInterpolation(
                    name=bound["nodeid"],
                    function="qhtable",
                    quantityunitpair=[
                        QuantityUnitPair(
                            quantity="qhbnd waterlevel", unit=bound["unit2"]
                        ),
                        QuantityUnitPair(
                            quantity="qhbnd discharge", unit=bound["unit1"]
                        ),
                    ],
                    datablock=list(map(list, zip(bound["vec2"], bound["vec1"]))),
                )

            self.forcingmodel.forcing.append(bnd_bc)
        for bound in self.hydamo.external_forcings.boundary_nodes.values():
            bnd_ext = Boundary(
                nodeid=bound["nodeid"],
                quantity=bound["quantity"],
                forcingfile=self.forcingmodel,
            )
            self.forcingmodel.filepath = Path("boundaryconditions.bc")
            self.boundaries_ext.append(bnd_ext)

    def laterals_to_dhydro(self) -> None:
        """Convert the laterals to [Lateral] blocks and forcings."""
        for key, lateral in self.hydamo.external_forcings.lateral_nodes.items():
            if isinstance(lateral["discharge"], str):
                # realtime boundary
                lat_ext = Lateral(
                    id=key,
                    locationtype="1d",
                    branchid=lateral["branchid"],
                    chainage=lateral["chainage"],
                    discharge=lateral["discharge"],
                )
            else:
                # time series or constant value
                lat_bc: TimeSeries | ConstantWithInterpolation
                if isinstance(lateral["discharge"], pd.Series):
                    lat_bc = TimeSeries(
                        name=key,
                        function="timeseries",
                        timeinterpolation="linear",
                        quantityunitpair=[
                            QuantityUnitPair(
                                quantity="time", unit=lateral["time_unit"]
                            ),
                            QuantityUnitPair(quantity="lateral_discharge", unit="m3/s"),
                        ],
                        datablock=list(
                            map(list, zip(lateral["time"], lateral["value"]))
                        ),
                    )
                    self.laterals_bc.append(lat_bc)
                else:
                    # add_lateral validates the discharge as a pd.Series, float or str
                    lat_bc = ConstantWithInterpolation(
                        name=key,
                        function="constant",
                        quantity="lateral_discharge",
                        unit="m3/s",
                        datablock=[[lateral["discharge"]]],
                    )
                self.forcingmodel.forcing.append(lat_bc)
                lat_ext = Lateral(
                    id=key,
                    locationtype=lateral["locationtype"],
                    branchid=lateral["branchid"],
                    chainage=float(lateral["chainage"]),
                    discharge=self.forcingmodel,
                )
            self.laterals_ext.append(lat_ext)

    def friction_definitions_to_dhydro(self) -> None:
        """Convert the friction definitions to FrictGlobal objects."""
        frictdefs = [
            FrictGlobal(**frictdef)
            for frictdef in self.hydamo.roughness_definitions.values()
        ]
        self._clear_comments(frictdefs)
        self.friction_defs += frictdefs

    def storagenodes_to_dhydro(self) -> None:
        """Convert the storage nodes to StorageNode objects."""
        stornodes = [
            StorageNode(**stornode)
            for stornode in self.hydamo.storagenodes.storagenodes.values()
        ]
        self._clear_comments(stornodes)
        self.storagenodes += stornodes

    def observation_points_to_dhydro(self) -> None:
        """Convert the observation points to ObservationPoint objects."""
        if hasattr(self.hydamo.observationpoints, "observation_points"):
            obspoints = [
                ObservationPoint(**obs)
                # into=dict: see compounds_to_dhydro
                for obs in self.hydamo.observationpoints.observation_points.drop(
                    columns="geometry"
                ).to_dict("records", into=dict)
            ]
            self._clear_comments(obspoints)
            self.obspoints += obspoints

    def inifields_to_dhydro(self) -> None:
        """Convert uniform initial water levels and depths to InitialField objects with a 1dField data file.

        The corresponding OneDFieldGlobal objects are stored in self.onedfieldmodels, in the same order as the
        1dField entries in self.inifields. Use write_initial_fields() to write them.
        """
        initial = [
            ("waterlevel", self.hydamo.external_forcings.initial_waterlevel_polygons),
            ("waterdepth", self.hydamo.external_forcings.initial_waterdepth_polygons),
        ]
        for quantity, gdf in initial:
            # uniform 1D values are written per branch by spatial_fields_to_dhydro
            if gdf.empty or (gdf["locationtype"] == "1d").all():
                continue
            if gdf.geometry.notna().any() or len(gdf) > 1:
                raise ValueError(
                    f"Initial {quantity} with polygons is not supported by set_initial_{quantity}. "
                    "Use hydamo.external_forcings.set_spatial_field() for spatially varying values."
                )
            row = gdf.iloc[0]
            # e.g. initialwaterlevel_2d.ini; the 1D files use the names of the GUI (ONEDFIELD_FILES)
            filename = f"initial{quantity}_{row['locationtype']}.ini"
            inifield = InitialField(
                quantity=quantity,
                datafiletype="1dField",
                datafile=filename,
                locationtype=row["locationtype"],
            )
            onedfield = OneDFieldGlobal(
                quantity=quantity,
                unit="m",
                value=str(row[quantity]),
            )
            self._clear_comments(inifield)
            self._clear_comments(onedfield)
            self.inifields.append(inifield)
            self.onedfieldmodels.append(onedfield)

    def spatial_fields_to_dhydro(self) -> None:
        """Convert spatial fields (branch values, polygons, rasters or samples) to InitialField/ParameterField objects.

        Branch values are derived from the HyDAMO branches. All initial water level and water depth fields per
        branch (including uniform 1D values) are combined into one 1dField file per quantity (ONEDFIELD_FILES), as
        the D-HYDRO GUI expects. This gives the same result in D-Flow FM: a field per branch fills whole branches
        and the last field wins, so per branch the values of the last field are kept, and a branch that a later
        field gives the other quantity is moved to that quantity. The combined block is placed at the position of
        the last field of that quantity. Polygon files get one .pol file and one block per distinct value. Only the
        references are created here; the data files themselves are written by write_spatial_fields().
        """
        # footprints of the initial water level and depth fields, to check where both apply
        level_depth_fields = []
        # initial water level and depth per branch, combined into one 1dField file per quantity
        combined = _CombinedOneDFields()
        # order of the blocks: ("combined", kind) or ("blocks", spatial field, [(filename, datafiletype, value)])
        order: list[tuple] = []
        for sf in self.hydamo.external_forcings.spatial_fields:
            kind = (
                self._level_or_depth(sf["quantity"])
                if sf["kind"] == "initial"
                else None
            )
            if sf["filetype"] == "branch" and kind is not None:
                # combined with the other fields of this quantity, at the position of the last one; see the docstring
                branch_values = self._branch_values(sf)
                combined.add(kind, sf["quantity"], branch_values)
                order = [entry for entry in order if entry != ("combined", kind)] + [
                    ("combined", kind)
                ]
                data: dict | None = {"branches": branch_values}
            else:
                order.append(("blocks", sf, self._field_blocks(sf)))
                data = self.spatial_data.get(sf["filename"])
            footprint = self._level_depth_footprint(sf, data)
            if footprint is not None:
                level_depth_fields.append(footprint)

        self._add_fields_in_order(order, combined)
        self._check_level_and_depth(level_depth_fields, self.hydamo.branches)
        self._check_uniform_depth(level_depth_fields)

    def _add_fields_in_order(
        self, order: list[tuple], combined: _CombinedOneDFields
    ) -> None:
        """Add the blocks of the spatial fields to the initial and parameter fields, in their order.

        Parameters
        ----------
        order : list[tuple]
            ("combined", kind) for a combined 1dField file, or ("blocks", spatial field,
            [(filename, datafiletype, value)]) for the blocks of another field.
        combined : _CombinedOneDFields
            The combined initial water level and depth per branch.
        """
        for entry in order:
            if entry[0] == "combined":
                self._add_combined_field(entry[1], combined)
                continue
            _, sf, blocks = entry
            for filename, datafiletype, value in blocks:
                self._add_block_field(sf, filename, datafiletype, value)

    def _field_blocks(self, sf: dict) -> list[tuple[str, str, float | None]]:
        """Return the data and blocks of a spatial field that is not combined with other fields.

        The data is stored in self.spatial_data, and written by write_spatial_fields().

        Parameters
        ----------
        sf : dict
            The spatial field, see ExternalForcings.set_spatial_field.

        Returns
        -------
        list[tuple[str, str, float | None]]
            (filename, datafiletype, value) per block; value is only set for polygon files.
        """
        builders = {
            "polygon": self._polygon_blocks,
            "branch": self._branch_blocks,
            "raster": self._raster_blocks,
        }
        return builders.get(sf["filetype"], self._sample_blocks)(sf)

    def _polygon_blocks(self, sf: dict) -> list[tuple[str, str, float | None]]:
        """One .pol file and one block per distinct value of a polygon field.

        Parameters
        ----------
        sf : dict
            The spatial field.

        Returns
        -------
        list[tuple[str, str, float | None]]
            (filename, 'polygon', value) per distinct value.
        """
        gdf, column = sf["gdf"], sf["column"]
        values = sorted(gdf[column].astype(float).unique())
        stem, suffix = Path(sf["filename"]).stem, Path(sf["filename"]).suffix or ".pol"
        blocks: list[tuple[str, str, float | None]] = []
        for i, value in enumerate(values):
            filename = sf["filename"] if len(values) == 1 else f"{stem}_{i:02d}{suffix}"
            geometries = gdf.loc[gdf[column].astype(float) == value].geometry
            self.spatial_data[filename] = {
                "filetype": "polygon",
                "polygons": [
                    poly for geom in geometries for poly in common.as_polygon_list(geom)
                ],
            }
            blocks.append((filename, "polygon", value))
        return blocks

    def _branch_values(
        self, sf: dict
    ) -> dict[Hashable, tuple[list[float], list[float]]]:
        """Values per branch of a field: a uniform value, or derived from polygons or points.

        Parameters
        ----------
        sf : dict
            The spatial field, with filetype 'branch'.

        Returns
        -------
        dict[Hashable, tuple[list[float], list[float]]]
            Chainages and values per branch id.

        Raises
        ------
        ValueError
            For a uniform value without branches.
        """
        gdf, column = sf["gdf"], sf["column"]
        if sf.get("uniform") is not None:
            branches = self.hydamo.branches
            if branches is None or branches.empty:
                raise ValueError(
                    f"A uniform initial {sf['quantity']} for 1D requires the branches."
                )
            branch_values = {
                branch_id: ([0.0, line.length], [sf["uniform"]] * 2)
                for branch_id, line in zip(branches.index, branches.geometry)
            }
        elif gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            branch_values = polygons_to_branch_values(self.hydamo.branches, gdf, column)
        else:
            branch_values = points_to_branch_values(
                self.hydamo.branches, gdf, column, max_distance=sf["max_snap_distance"]
            )
        if not branch_values:
            logger.warning(f"No values on any branch for {sf['filename']}.")
        return branch_values

    def _branch_blocks(self, sf: dict) -> list[tuple[str, str, float | None]]:
        """Return a 1dField file and block for a field per branch that is not combined.

        E.g. a parameter field.

        Parameters
        ----------
        sf : dict
            The spatial field, with filetype 'branch'.

        Returns
        -------
        list[tuple[str, str, float | None]]
            One block (filename, '1dField', None).
        """
        self.spatial_data[sf["filename"]] = {
            "filetype": "branch",
            "branches": self._branch_values(sf),
            "quantity": sf["quantity"],
            "nodata": sf["nodata"],
        }
        return [(sf["filename"], "1dField", None)]

    def _raster_blocks(self, sf: dict) -> list[tuple[str, str, float | None]]:
        """Return a raster (GeoTIFF or ArcInfo grid) and block for a polygon field.

        Parameters
        ----------
        sf : dict
            The spatial field, with filetype 'raster'.

        Returns
        -------
        list[tuple[str, str, float | None]]
            One block (filename, 'arcInfo' or 'GeoTIFF', None).
        """
        gdf = sf["gdf"]
        array, transform = rasterize_polygons(
            gdf, sf["column"], sf["cellsize"], fill=sf["nodata"]
        )
        self.spatial_data[sf["filename"]] = {
            "filetype": "raster",
            "array": array,
            "transform": transform,
            "crs": gdf.crs,
            "nodata": sf["nodata"],
        }
        datafiletype = (
            "arcInfo" if Path(sf["filename"]).suffix.lower() == ".asc" else "GeoTIFF"
        )
        return [(sf["filename"], datafiletype, None)]

    def _sample_blocks(self, sf: dict) -> list[tuple[str, str, float | None]]:
        """Return a sample file (.xyz) and block for a point field.

        Parameters
        ----------
        sf : dict
            The spatial field, with filetype 'sample'.

        Returns
        -------
        list[tuple[str, str, float | None]]
            One block (filename, 'sample', None).
        """
        gdf = sf["gdf"]
        self.spatial_data[sf["filename"]] = {
            "filetype": "sample",
            "x": gdf.geometry.x.to_numpy(),
            "y": gdf.geometry.y.to_numpy(),
            "z": gdf[sf["column"]].to_numpy(dtype=float),
        }
        return [(sf["filename"], "sample", None)]

    def _add_combined_field(self, kind: str, combined: _CombinedOneDFields) -> None:
        """Add the combined 1dField file and [Initial] block of a quantity, if it has values.

        Parameters
        ----------
        kind : str
            'waterlevel' or 'waterdepth'.
        combined : _CombinedOneDFields
            The combined values per branch.
        """
        if not combined.branches[kind]:
            return
        filename = ONEDFIELD_FILES[kind]
        self.spatial_data[filename] = {
            "filetype": "branch",
            "branches": combined.branches[kind],
            "quantity": combined.quantity[kind],
            "nodata": -999.0,
        }
        field = InitialField(
            quantity=combined.quantity[kind], datafile=filename, datafiletype="1dField"
        )
        self._clear_comments(field)
        self.inifields.append(field)

    def _add_block_field(
        self, sf: dict, filename: str, datafiletype: str, value: float | None
    ) -> None:
        """Add an [Initial] or [Parameter] block for a data file of a spatial field.

        Parameters
        ----------
        sf : dict
            The spatial field.
        filename : str
            The data file.
        datafiletype : str
            Data file type, e.g. '1dField', 'polygon', 'GeoTIFF' or 'sample'.
        value : float | None
            Value inside the polygons, for polygon files.
        """
        kwargs = {
            "quantity": sf["quantity"],
            "datafile": filename,
            "datafiletype": datafiletype,
            "interpolationmethod": sf["interpolationmethod"],
            "locationtype": sf["locationtype"],
            "operand": sf["operand"],
        }
        if value is not None:
            kwargs["value"] = value
        if datafiletype == "1dField":
            # interpolation and location type are not used for 1dField files
            del kwargs["interpolationmethod"], kwargs["locationtype"]
        if sf["interpolationmethod"] == "averaging":
            kwargs["averagingtype"] = sf["averagingtype"]

        if sf["kind"] == "initial":
            initial_field = InitialField(**kwargs)
            self._clear_comments(initial_field)
            self.inifields.append(initial_field)
        else:
            parameter_field = ParameterField(**kwargs)
            self._clear_comments(parameter_field)
            self.parameterfields.append(parameter_field)

    @staticmethod
    def _level_or_depth(quantity: str) -> str | None:
        """'waterlevel' or 'waterdepth' for the (old and new) names of these quantities, otherwise None."""
        kind = quantity.lower().replace("initial", "")
        return kind if kind in ("waterlevel", "waterdepth") else None

    def _level_depth_footprint(self, sf: dict, data: dict | None) -> dict | None:
        """Where an initial water level or depth field gives values on the 1D network.

        - 'branch': the whole branches in the 1dField file (values are kept constant beyond the first and
          last chainage);
        - 'polygon' and 'raster': the polygons;
        - 'sample': the convex hull of the points, which triangulation fills (tested with D-HYDRO 2026.02).

        Returns None for other quantities, parameter fields and 2D-only fields.
        """
        kind = self._level_or_depth(sf["quantity"])
        if sf["kind"] != "initial" or kind is None:
            return None
        if sf["filetype"] != "branch" and sf["locationtype"] not in ("1d", "all"):
            return None
        footprint = {
            "kind": kind,
            "quantity": sf["quantity"],
            "filename": sf["filename"],
            "filetype": sf["filetype"],
            # a uniform value (set_initial_waterlevel/-depth) is a background that later fields override on purpose
            "background": sf.get("uniform") is not None,
        }
        if sf["filetype"] == "branch":
            footprint["branches"] = set(data["branches"]) if data else set()
            footprint["area"] = None
        elif sf["filetype"] == "sample":
            if data is None:
                raise ValueError(f"No sample data for {sf['filename']}.")
            footprint["branches"] = None
            footprint["area"] = shapely.MultiPoint(
                list(zip(data["x"], data["y"]))
            ).convex_hull
        else:
            footprint["branches"] = None
            footprint["area"] = shapely.union_all(sf["gdf"].geometry.to_numpy())
        return footprint

    @staticmethod
    def _covered_parts(
        footprint: dict, branches: gpd.GeoDataFrame
    ) -> dict[str, list[tuple[float, float]]]:
        """Parts of the branches covered by a footprint, as {branch id: [(start chainage, end chainage), ...]}.

        Chainage intervals are compared instead of line geometries, which is robust to rounding in intersections.
        """
        if footprint["branches"] is not None:
            selected = branches.loc[branches.index.isin(list(footprint["branches"]))]
            return {
                branch_id: [(0.0, line.length)]
                for branch_id, line in zip(selected.index, selected.geometry)
            }
        candidates = branches.iloc[
            branches.sindex.query(footprint["area"], predicate="intersects")
        ]
        covered = {}
        for branch_id, line in zip(candidates.index, candidates.geometry):
            intervals = []
            for part in line_parts(line.intersection(footprint["area"])):
                chainages = [line.project(shapely.Point(c)) for c in part.coords]
                intervals.append((min(chainages), max(chainages)))
            if intervals:
                covered[branch_id] = intervals
        return covered

    @staticmethod
    def _overlap_length(
        intervals_i: list[tuple[float, float]], intervals_j: list[tuple[float, float]]
    ) -> float:
        """Total length of the overlap between two lists of chainage intervals."""
        return sum(
            max(0.0, min(b_i, b_j) - max(a_i, a_j))
            for a_i, b_i in intervals_i
            for a_j, b_j in intervals_j
        )

    @classmethod
    def _check_level_and_depth(
        cls, fields: list[dict], branches: gpd.GeoDataFrame, tolerance: float = 0.01
    ) -> None:
        """Warn where a water level field and a water depth field both give values on the same part of a branch.

        D-Flow FM applies the fields in order, and the last field wins, also between water level and water
        depth (tested with D-HYDRO 2026.02). The footprints are larger than the input geometries suggest:
        triangulation fills the convex hull of the samples, and a 1dField file fills whole branches. Overlap
        only at junctions (shorter than `tolerance`) is ignored.

        Also warns for a water depth from polygons, samples or rasters after a water level from polygons, samples
        or rasters. In DIMR tests (2026.02) such a water depth field removed those water levels everywhere, also
        outside its own area. Water levels per branch (1dField) were kept, and a water depth per branch removed
        nothing. Rasters were not tested and are treated like samples.
        """
        has_branches = (
            branches is not None and not branches.empty and "geometry" in branches
        )
        parts = (
            [cls._covered_parts(f, branches) for f in fields] if has_branches else []
        )

        for (i, field_i), (j, field_j) in itertools.combinations(enumerate(fields), 2):
            if (
                field_i["kind"] == field_j["kind"]
                or field_i["background"]
                or field_j["background"]
            ):
                continue
            message = cls._overlap_message(field_i, field_j)
            if has_branches:
                shared = cls._shared_branches(parts[i], parts[j], tolerance)
                if shared:
                    logger.warning(
                        f"{len(shared)} branch(es) {message} Branches: {shared[:10]}"
                    )
            elif cls._areas_overlap(field_i, field_j):
                # no network to check against: the areas overlap
                logger.warning(f"The areas of the 1D fields {message}")

        cls._warn_depth_after_level(fields)

    @staticmethod
    def _overlap_message(field_i: dict, field_j: dict) -> str:
        """Message for a water level and a water depth field that give values at the same place.

        Parameters
        ----------
        field_i, field_j : dict
            Footprints of the earlier and the later field.

        Returns
        -------
        str
            The message, with a note on why the footprints may be larger than expected.
        """
        notes = []
        if "sample" in (field_i["filetype"], field_j["filetype"]):
            notes.append(
                "samples with triangulation fill the convex hull of the points"
            )
        if "branch" in (field_i["filetype"], field_j["filetype"]):
            notes.append("a 1dField file fills the whole branch")
        note = f" Note: {'; '.join(notes)}." if notes else ""
        header = (
            f"get both a {field_i['quantity']} ({field_i['filename']}) and a {field_j['quantity']} "
            f"({field_j['filename']}). There the {field_j['quantity']} applies, as it is added last."
        )
        return f"{header}{note}"

    @classmethod
    def _shared_branches(
        cls,
        parts_i: dict[str, list[tuple[float, float]]],
        parts_j: dict[str, list[tuple[float, float]]],
        tolerance: float,
    ) -> list[str]:
        """Branches where two footprints overlap over more than `tolerance`.

        Parameters
        ----------
        parts_i, parts_j : dict[str, list[tuple[float, float]]]
            Covered chainage intervals per branch id, see _covered_parts.
        tolerance : float
            Overlap up to this length (e.g. only at junctions) is ignored.

        Returns
        -------
        list[str]
            Sorted branch ids.
        """
        return sorted(
            branch_id
            for branch_id in parts_i.keys() & parts_j.keys()
            if cls._overlap_length(parts_i[branch_id], parts_j[branch_id]) > tolerance
        )

    @staticmethod
    def _areas_overlap(field_i: dict, field_j: dict) -> bool:
        """Whether the areas of two footprints overlap; False if one of them has no area (fields per branch).

        Parameters
        ----------
        field_i, field_j : dict
            Footprints of two fields.

        Returns
        -------
        bool
            True if the areas overlap.
        """
        if field_i["area"] is None or field_j["area"] is None:
            return False
        return field_i["area"].intersection(field_j["area"]).area > 0.0

    @staticmethod
    def _warn_depth_after_level(fields: list[dict]) -> None:
        """Warn for a water depth from polygons, samples or rasters after such a water level.

        In tests with D-HYDRO 2026.02, such a water depth field removed those water levels everywhere, also
        outside its own area. Fields per branch (1dField) are not affected.

        Parameters
        ----------
        fields : list[dict]
            Footprints of the initial water level and depth fields, in their order.
        """
        for j, field in enumerate(fields):
            if field["kind"] != "waterdepth" or field["filetype"] == "branch":
                continue
            earlier = [
                f["filename"]
                for f in fields[:j]
                if f["kind"] == "waterlevel" and f["filetype"] != "branch"
            ]
            if earlier:
                logger.warning(
                    f"The {field['quantity']} field {field['filename']} ({field['filetype']}) is added after water level "
                    f"field(s) {earlier}. In tests with D-HYDRO 2026.02, a water depth from polygons or samples removed "
                    "earlier water levels from polygons or samples everywhere, also outside its own area. Add water "
                    "depth fields before water level fields, or write the water levels per branch."
                )

    def _check_uniform_depth(self, fields: list[dict]) -> None:
        """Warn when a uniform water depth is combined with spatial water levels: the depth overrides them.

        Only for a uniform depth for 2D or 'all' ([Global] value); a uniform 1D depth is written per branch and
        behaves like any other field.
        """
        uniform_depth = self.hydamo.external_forcings.initial_waterdepth_polygons
        levels = [
            f["filename"]
            for f in fields
            if f["kind"] == "waterlevel" and not f["background"]
        ]
        if (
            not uniform_depth.empty
            and (uniform_depth["locationtype"] != "1d").any()
            and levels
        ):
            logger.warning(
                f"A uniform initial water depth is combined with spatial water level field(s) {levels}. "
                "In D-HYDRO 2026.02 the uniform water depth overrides these water levels, whatever the order."
            )

    def write_spatial_fields(self, output_dir: Path | str) -> None:
        """Write the data files of the spatial fields (1dField files, polygon files, rasters and samples).

        The files are referenced relative to initialFields.ini, so output_dir should be the
        folder where the FMModel (and its initialFields.ini) is saved.

        Parameters
        ----------
        output_dir : Path or str
            The fm folder of the model.
        """
        output_dir = Path(output_dir)
        for filename, data in self.spatial_data.items():
            path = output_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if data["filetype"] == "branch":
                # The D-HYDRO GUI requires a [Global] block. D-Flow FM (2026.02) treats the nodata value as no
                # value, so branches without values keep the value of earlier fields or WaterLevIni.
                quantity = data["quantity"]
                unit = (
                    "m"
                    if quantity.lower().replace("initial", "")
                    in ("waterlevel", "waterdepth")
                    else "-"
                )
                OneDFieldModel(
                    global_=OneDFieldGlobal(
                        quantity=quantity, unit=unit, value=data["nodata"]
                    ),
                    branch=[
                        OneDFieldBranch(
                            branchid=str(branch_id),
                            numlocations=len(chainage),
                            chainage=chainage,
                            values=values,
                        )
                        for branch_id, (chainage, values) in data["branches"].items()
                    ],
                ).save(filepath=path)
            elif data["filetype"] == "polygon":
                objects = []
                for i, polygon in enumerate(data["polygons"]):
                    coords = list(polygon.exterior.coords)
                    objects.append(
                        PolyObject(
                            metadata=Metadata(
                                name=f"{path.stem}_{i:04d}",
                                n_rows=len(coords),
                                n_columns=2,
                            ),
                            points=[
                                PolyPoint(x=x, y=y, data=[]) for x, y, *_ in coords
                            ],
                        )
                    )
                PolyFile(objects=objects).save(filepath=path)
            elif data["filetype"] == "raster":
                write_raster(
                    path,
                    data["array"],
                    data["transform"],
                    crs=data["crs"],
                    nodata=data["nodata"],
                )
            else:
                xyz = XYZModel(
                    points=[
                        XYZPoint(x=x, y=y, z=z)
                        for x, y, z in zip(data["x"], data["y"], data["z"])
                    ]
                )
                xyz.save(filepath=path)

    def write_initial_fields(self, fm: FMModel, fm_dir: Path | str) -> IniFieldModel:
        """Add all initial fields and parameter fields to the FM model and write their data files.

        This writes the 1dField files (uniform values), rasters and sample files to fm_dir, and links them
        in initialFields.ini, using paths relative to the fm folder. Call this before saving the FM model.

        Parameters
        ----------
        fm : FMModel
            The FM model.
        fm_dir : Path or str
            Folder where the FM model (mdu file) is written.

        Returns
        -------
        IniFieldModel
            The initial fields model, also set as ``fm.geometry.inifieldfile``.
        """
        fm_dir = Path(fm_dir)
        fm_dir.mkdir(parents=True, exist_ok=True)

        onedfield_inifields = [f for f in self.inifields if f.datafiletype == "1dField"]
        for inifield, onedfield in zip(onedfield_inifields, self.onedfieldmodels):
            if inifield.datafile is None or inifield.datafile.filepath is None:
                raise ValueError(
                    f"The 1dField initial field for {inifield.quantity} has no data file."
                )
            onedfieldmodel = OneDFieldModel(global_=onedfield)
            onedfieldmodel.save(filepath=fm_dir / inifield.datafile.filepath.name)

        self.write_spatial_fields(fm_dir)

        inifieldmodel = IniFieldModel(
            initial=self.inifields, parameter=self.parameterfields
        )
        fm.geometry.inifieldfile = inifieldmodel
        return inifieldmodel

    def initial_conditions_at_nodes(
        self,
        network: Network,
        output_file: Path | str | None = None,
        crs: CRS | str | None = None,
    ) -> gpd.GeoDataFrame:
        """Approximate the initial water level or depth at the 1D computational nodes.

        A diagnostic to check the initial conditions, e.g. in a GIS, since the D-HYDRO GUI hardly shows them.
        It is not computed by D-Flow FM: it reproduces how D-Flow FM applied the initial fields in tests with
        D-HYDRO 2026.02 (1404 to 1421 of the 1421 nodes of the test model matched the water level at t=0;
        differences at junctions and structures). Other D-Flow FM versions may behave differently. The fields
        differences at junctions and structures). Other D-Flow FM versions may behave differently. The fields
        are applied in their order:

        - values per branch (1dField) are interpolated along the branch and kept constant beyond the first and
          last chainage; uniform values apply everywhere;
        - samples are interpolated linearly in triangles (inside their convex hull), polygon files apply inside
          the polygons; rasters are not evaluated;
        - where fields overlap the last one applies, also between water level and water depth;
        - a water depth from polygons or samples removes earlier water levels from polygons or samples;
        - a uniform water depth overrides spatial water levels.
        Nodes without a value keep the default water level of the MDU (WaterLevIni), shown as missing here.

        Parameters
        ----------
        network : Network
            The network with the 1D mesh, e.g. ``fm.geometry.netfile.network``.
        output_file : Path or str, optional
            Shapefile (.shp) or GeoPackage (.gpkg) to write. Defaults to None.
        crs : optional
            Coordinate reference system. Defaults to the crs of the HyDAMO branches, or
            EPSG:28992 (as written to the model by ``DIMRWriter.add_crs``) if they have none.

        Returns
        -------
        gpd.GeoDataFrame
            One point per node with branchid, chainage, quantity ('waterlevel' or
            'waterdepth'), value and source (data file of the field that sets the value).
        """
        mesh1d = network._mesh1d
        x, y = (
            np.asarray(mesh1d.mesh1d_node_x, dtype=float),
            np.asarray(mesh1d.mesh1d_node_y, dtype=float),
        )
        branch_names = np.asarray(mesh1d.network1d_branch_id, dtype=object)
        mesh_nodes = _Mesh1dNodes(
            x=x,
            y=y,
            branchid=branch_names[np.asarray(mesh1d.mesh1d_node_branch_id, dtype=int)],
            chainage=np.asarray(mesh1d.mesh1d_node_branch_offset, dtype=float),
            points=gpd.points_from_xy(x, y),
        )
        state = _NodeValues(len(x))
        uniform_depth = None

        uniform_fields = iter(self.onedfieldmodels)
        for inifield in self.inifields:
            field = self._inifield_at_nodes(inifield, uniform_fields, mesh_nodes)
            if field is None:
                continue
            kind, values, filename, filetype = field
            state.apply(kind, values, filename, filetype)
            if filetype == "uniform" and kind == "waterdepth":
                uniform_depth = (values, filename)

        if uniform_depth is not None:
            state.apply_uniform_depth(*uniform_depth)

        nodes = gpd.GeoDataFrame(
            {
                "node_id": np.asarray(mesh1d.mesh1d_node_id, dtype=object),
                "branchid": mesh_nodes.branchid,
                "chainage": mesh_nodes.chainage,
                "quantity": state.quantity,
                "value": state.value,
                "source": state.source,
            },
            geometry=mesh_nodes.points,
            crs=self._nodes_crs(crs),
        )
        if (nodes["quantity"].isna()).any():
            logger.info(
                f"{nodes['quantity'].isna().sum()} 1D nodes get no initial value and keep WaterLevIni."
            )
        if output_file is not None:
            nodes.to_file(output_file)
        return nodes

    def _inifield_at_nodes(
        self, inifield: InitialField, uniform_fields: Iterator, nodes: _Mesh1dNodes
    ) -> tuple[str, np.ndarray, str | None, str] | None:
        """Values of an initial field at the 1D nodes, if it sets a water level or depth there.

        Parameters
        ----------
        inifield : InitialField
            The initial field.
        uniform_fields : Iterator
            The [Global] blocks of the uniform values (self.onedfieldmodels); the next one is consumed for
            every uniform value, also when it is skipped.
        nodes : _Mesh1dNodes
            The 1D nodes.

        Returns
        -------
        tuple[str, np.ndarray, str | None, str] | None
            (kind, value per node, data file, filetype), with kind 'waterlevel' or 'waterdepth' and filetype
            'uniform' or that of the spatial data; None if the field gives no water level or depth at the nodes.
        """
        filename = (
            inifield.datafile.filepath.name
            if inifield.datafile and inifield.datafile.filepath
            else None
        )
        kind = self._level_or_depth(inifield.quantity)
        data = None
        if filename is not None:
            data = self.spatial_data.get(filename)
        if data is None and inifield.datafiletype == "1dField":
            onedfield = next(uniform_fields)
            values = self._uniform_values_at_nodes(
                inifield, kind, float(onedfield.value), len(nodes.x)
            )
            filetype = "uniform"
        elif kind is None or data is None:
            return None
        else:
            values = self._spatial_values_at_nodes(inifield, filename, data, nodes)
            filetype = data["filetype"]
        if values is None or kind is None:
            return None
        return kind, values, filename, filetype

    @staticmethod
    def _uniform_values_at_nodes(
        inifield: InitialField, kind: str | None, value: float, n: int
    ) -> np.ndarray | None:
        """Values of a uniform 1D water level or depth at the nodes.

        Parameters
        ----------
        inifield : InitialField
            The initial field of the uniform value.
        kind : str | None
            'waterlevel', 'waterdepth', or None for other quantities.
        value : float
            The uniform value.
        n : int
            Number of nodes.

        Returns
        -------
        np.ndarray | None
            The value at every node, or None for other quantities and 2D-only fields.
        """
        if kind is None or str(inifield.locationtype).lower() == "2d":
            return None
        return np.full(n, value)

    @staticmethod
    def _spatial_values_at_nodes(
        inifield: InitialField, filename: str | None, data: dict, nodes: _Mesh1dNodes
    ) -> np.ndarray | None:
        """Values of a spatial field at the nodes, as D-Flow FM applied them (D-HYDRO 2026.02).

        - values per branch (1dField) are interpolated along the branch and kept constant beyond the first
          and last chainage;
        - samples are interpolated linearly in triangles, inside their convex hull (at least 3 samples);
        - polygon files give the value of the field inside the polygons.

        Parameters
        ----------
        inifield : InitialField
            The initial field.
        filename : str | None
            Data file of the field, for messages.
        data : dict
            The data of the field in spatial_data.
        nodes : _Mesh1dNodes
            The 1D nodes.

        Returns
        -------
        np.ndarray | None
            Value per node, NaN where the field gives no value; None for rasters, which are not evaluated.
        """
        values = np.full(len(nodes.x), np.nan)
        if data["filetype"] == "branch":
            for bid, (branch_chainage, branch_values) in data["branches"].items():
                on_branch = nodes.branchid == bid
                values[on_branch] = np.interp(
                    nodes.chainage[on_branch], branch_chainage, branch_values
                )
        elif data["filetype"] == "sample":
            if len(data["x"]) >= 3:
                interpolator = LinearNDInterpolator(
                    np.c_[data["x"], data["y"]], data["z"]
                )
                values = interpolator(np.c_[nodes.x, nodes.y])
        elif data["filetype"] == "polygon":
            if inifield.value is None:
                raise ValueError(f"The polygon field {filename} has no value.")
            inside = gpd.GeoSeries(nodes.points).intersects(
                shapely.union_all(data["polygons"])
            )
            values[inside.to_numpy()] = float(inifield.value)
        else:
            logger.warning(f"Raster field {filename} is not evaluated at the nodes.")
            return None
        return values

    def _nodes_crs(self, crs: CRS | str | None) -> CRS | str:
        """Coordinate reference system for the 1D nodes.

        Parameters
        ----------
        crs : optional
            Requested coordinate reference system, or None.

        Returns
        -------
        crs
            The requested crs, otherwise the crs of the HyDAMO branches, otherwise EPSG:28992 (as written to
            the model by DIMRWriter.add_crs).
        """
        if crs is None and getattr(self.hydamo, "branches", None) is not None:
            crs = self.hydamo.branches.crs
        if crs is None:
            # the projection that DIMRWriter.add_crs writes to the model
            crs = "EPSG:28992"
            logger.info(
                "No coordinate system known for the branches; EPSG:28992 is used for the 1D nodes."
            )
        return crs
