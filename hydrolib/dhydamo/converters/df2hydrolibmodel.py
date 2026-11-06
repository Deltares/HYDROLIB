import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from pydantic import Field

from hydrolib.core.dflowfm.bc.models import (
    Constant,
    ForcingModel,
    QHTable,
    QuantityUnitPair,
    TimeInterpolation,
    TimeSeries,
)
from hydrolib.core.dflowfm.crosssection.models import (
    CircleCrsDef,
    CrossSection,
    RectangleCrsDef,
    YZCrsDef,
    ZWCrsDef,
)
from hydrolib.core.dflowfm.ext.models import Boundary, Lateral
from hydrolib.core.dflowfm.friction.models import FrictGlobal
from hydrolib.core.dflowfm.inifield.models import IniFieldModel, InitialField, ParameterField
from hydrolib.core.dflowfm.obs.models import ObservationPoint
from hydrolib.core.dflowfm.onedfield.models import OneDFieldBranch, OneDFieldGlobal, OneDFieldModel
from hydrolib.core.dflowfm.storagenode.models import StorageNode
from hydrolib.core.dflowfm.structure.models import (
    Bridge,
    Compound,
    Culvert,
    Orifice,
    Pump,
    UniversalWeir,
    Weir,
)
from hydrolib.core.dflowfm.polyfile.models import Metadata, PolyFile, PolyObject
from hydrolib.core.dflowfm.polyfile.models import Point as PolyPoint
from hydrolib.core.dflowfm.xyz.models import XYZModel, XYZPoint

from hydrolib.dhydamo.geometry import common
from hydrolib.dhydamo.geometry.spatial import (
    _line_parts,
    points_to_branch_values,
    polygons_to_branch_values,
    rasterize_polygons,
)
from hydrolib.dhydamo.io.rasterwriter import write_raster

logger = logging.getLogger(__name__)


class ConstantWithInterpolation(Constant):
    """Constant [Forcing] block that also writes timeInterpolation.

    hydrolib-core's Constant has no timeInterpolation field, but the D-HYDRO GUI raises an error when it is
    missing. D-Flow FM (2026.02) accepts it and ignores it for constant values.
    """

    timeinterpolation: TimeInterpolation = Field(TimeInterpolation.linear, alias="timeInterpolation")


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
        self.parameterfields = []
        # data for the spatial fields, written by write_spatial_fields()
        self.spatial_data = {}

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
        self.spatial_fields_to_dhydro()

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
                 logger.warning(
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
            if bound["vec1"] is None:
                bnd_bc = ConstantWithInterpolation(
                    name=bound["nodeid"],
                    function="constant",
                    quantity=bound["quantity"],
                    unit=bound["unit2"],
                    datablock=[[bound["vec2"]]],
                )
            elif bound['unit1'].startswith('minutes'):
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
                bnd_bc = QHTable(                 
                        name=bound["nodeid"],
                        function="qhtable",
                        timeinterpolation="linear",
                        quantityunitpair=[
                            QuantityUnitPair(quantity="qhbnd waterlevel", unit=bound["unit2"]),
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
        """Convert dataframe of laterals to ext and bc models"""

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
        """Convert uniform initial water levels and depths to InitialField objects with a 1dField data file.

        The corresponding OneDFieldGlobal objects are stored in self.onedfieldmodels, in the same order as the
        1dField entries in self.inifields. Use write_initial_fields() to write them.
        """
        initial = [
            ("waterlevel", self.hydamo.external_forcings.initial_waterlevel_polygons, "initialwaterlevel.ini"),
            ("waterdepth", self.hydamo.external_forcings.initial_waterdepth_polygons, "initialwaterdepth.ini"),
        ]
        for quantity, gdf, filename in initial:
            # uniform 1D values are written per branch by spatial_fields_to_dhydro
            if gdf.empty or (gdf["locationtype"] == "1d").all():
                continue
            if gdf.geometry.notnull().any() or len(gdf) > 1:
                raise ValueError(
                    f"Initial {quantity} with polygons is not supported by set_initial_{quantity}. "
                    "Use hydamo.external_forcings.set_spatial_field() for spatially varying values."
                )
            row = gdf.iloc[0]
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

    def spatial_fields_to_dhydro(self):
        """Convert spatial fields (branch values, polygons, rasters or samples) to InitialField/ParameterField objects.

        Branch values are derived from the HyDAMO branches and written as one 1dField file. Polygon files
        get one .pol file and one block per distinct value. Only the references are created here;
        the data files themselves are written by write_spatial_fields().
        """
        # footprints of the initial water level and depth fields, to check where both apply
        level_depth_fields = []
        for sf in self.hydamo.external_forcings.spatial_fields:
            gdf, column = sf["gdf"], sf["column"]

            # list of (filename, datafiletype, value) per block
            blocks = []
            if sf["filetype"] == "polygon":
                values = sorted(gdf[column].astype(float).unique())
                stem, suffix = Path(sf["filename"]).stem, Path(sf["filename"]).suffix or ".pol"
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

            elif sf["filetype"] == "branch":
                if sf.get("uniform") is not None:
                    branches = self.hydamo.branches
                    if branches is None or branches.empty:
                        raise ValueError(f"A uniform initial {sf['quantity']} for 1D requires the branches.")
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
                    logger.warning(f"No values on any branch for {sf['filename']}; the file is empty.")
                self.spatial_data[sf["filename"]] = {
                    "filetype": "branch",
                    "branches": branch_values,
                    "quantity": sf["quantity"],
                    "nodata": sf["nodata"],
                }
                blocks.append((sf["filename"], "1dField", None))

            elif sf["filetype"] == "raster":
                array, transform = rasterize_polygons(
                    gdf, column, sf["cellsize"], fill=sf["nodata"]
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
                blocks.append((sf["filename"], datafiletype, None))

            else:
                self.spatial_data[sf["filename"]] = {
                    "filetype": "sample",
                    "x": gdf.geometry.x.to_numpy(),
                    "y": gdf.geometry.y.to_numpy(),
                    "z": gdf[column].to_numpy(dtype=float),
                }
                blocks.append((sf["filename"], "sample", None))

            for filename, datafiletype, value in blocks:
                kwargs = dict(
                    quantity=sf["quantity"],
                    datafile=filename,
                    datafiletype=datafiletype,
                    interpolationmethod=sf["interpolationmethod"],
                    locationtype=sf["locationtype"],
                    operand=sf["operand"],
                )
                if value is not None:
                    kwargs["value"] = value
                if datafiletype == "1dField":
                    # interpolation and location type are not used for 1dField files
                    del kwargs["interpolationmethod"], kwargs["locationtype"]
                if sf["interpolationmethod"] == "averaging":
                    kwargs["averagingtype"] = sf["averagingtype"]

                if sf["kind"] == "initial":
                    field = InitialField(**kwargs)
                    self._clear_comments(field)
                    self.inifields.append(field)
                else:
                    field = ParameterField(**kwargs)
                    self._clear_comments(field)
                    self.parameterfields.append(field)

            footprint = self._level_depth_footprint(sf, self.spatial_data.get(sf["filename"]))
            if footprint is not None:
                level_depth_fields.append(footprint)

        self._check_level_and_depth(level_depth_fields, self.hydamo.branches)
        self._check_uniform_depth(level_depth_fields)

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
            footprint["branches"] = None
            footprint["area"] = shapely.MultiPoint(list(zip(data["x"], data["y"]))).convex_hull
        else:
            footprint["branches"] = None
            footprint["area"] = shapely.union_all(sf["gdf"].geometry.to_numpy())
        return footprint

    @staticmethod
    def _covered_parts(footprint: dict, branches: gpd.GeoDataFrame) -> dict[str, list[tuple[float, float]]]:
        """Parts of the branches covered by a footprint, as {branch id: [(start chainage, end chainage), ...]}.

        Chainage intervals are compared instead of line geometries, which is robust to rounding in intersections.
        """
        if footprint["branches"] is not None:
            selected = branches.loc[branches.index.isin(list(footprint["branches"]))]
            return {branch_id: [(0.0, line.length)] for branch_id, line in zip(selected.index, selected.geometry)}
        candidates = branches.iloc[branches.sindex.query(footprint["area"], predicate="intersects")]
        covered = {}
        for branch_id, line in zip(candidates.index, candidates.geometry):
            intervals = []
            for part in _line_parts(line.intersection(footprint["area"])):
                chainages = [line.project(shapely.Point(c)) for c in part.coords]
                intervals.append((min(chainages), max(chainages)))
            if intervals:
                covered[branch_id] = intervals
        return covered

    @staticmethod
    def _overlap_length(intervals_i: list[tuple[float, float]], intervals_j: list[tuple[float, float]]) -> float:
        """Total length of the overlap between two lists of chainage intervals."""
        return sum(
            max(0.0, min(b_i, b_j) - max(a_i, a_j)) for a_i, b_i in intervals_i for a_j, b_j in intervals_j
        )

    @classmethod
    def _check_level_and_depth(cls, fields: list[dict], branches: gpd.GeoDataFrame, tolerance: float = 0.01) -> None:
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
        has_branches = branches is not None and not branches.empty and "geometry" in branches
        parts = [cls._covered_parts(f, branches) if has_branches else None for f in fields]

        for i, field_i in enumerate(fields):
            for j in range(i + 1, len(fields)):
                field_j = fields[j]
                if field_i["kind"] == field_j["kind"] or field_i["background"] or field_j["background"]:
                    continue

                notes = []
                if "sample" in (field_i["filetype"], field_j["filetype"]):
                    notes.append("samples with triangulation fill the convex hull of the points")
                if "branch" in (field_i["filetype"], field_j["filetype"]):
                    notes.append("a 1dField file fills the whole branch")
                note = f" Note: {'; '.join(notes)}." if notes else ""
                header = (
                    f"get both a {field_i['quantity']} ({field_i['filename']}) and a {field_j['quantity']} "
                    f"({field_j['filename']}). There the {field_j['quantity']} applies, as it is added last."
                )

                if parts[i] is not None:
                    shared = sorted(
                        branch_id
                        for branch_id in parts[i].keys() & parts[j].keys()
                        if cls._overlap_length(parts[i][branch_id], parts[j][branch_id]) > tolerance
                    )
                    if shared:
                        logger.warning(f"{len(shared)} branch(es) {header}{note} Branches: {shared[:10]}")
                elif field_i["area"] is not None and field_j["area"] is not None:
                    # no network to check against: compare the areas
                    if field_i["area"].intersection(field_j["area"]).area > 0.0:
                        logger.warning(f"The areas of the 1D fields {header}{note}")

        for j, field in enumerate(fields):
            if field["kind"] != "waterdepth" or field["filetype"] == "branch":
                continue
            earlier = [f["filename"] for f in fields[:j] if f["kind"] == "waterlevel" and f["filetype"] != "branch"]
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
        levels = [f["filename"] for f in fields if f["kind"] == "waterlevel" and not f["background"]]
        if not uniform_depth.empty and (uniform_depth["locationtype"] != "1d").any() and levels:
            logger.warning(
                f"A uniform initial water depth is combined with spatial water level field(s) {levels}. "
                "In D-HYDRO 2026.02 the uniform water depth overrides these water levels, whatever the order."
            )

    def write_spatial_fields(self, output_dir: Path | str) -> None:
        """Write the data files of the spatial fields (1dField files, polygon files, rasters and samples).

        The files are referenced relative to initialFields.ini, so output_dir should be the
        folder where the FMModel (and its initialFields.ini) is saved.

        Args:
            output_dir (Path | str): fm-folder of the model
        """
        output_dir = Path(output_dir)
        for filename, data in self.spatial_data.items():
            path = output_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if data["filetype"] == "branch":
                # The D-HYDRO GUI requires a [Global] block. D-Flow FM (2026.02) treats the nodata value as no
                # value, so branches without values keep the value of earlier fields or WaterLevIni.
                quantity = data["quantity"]
                unit = "m" if quantity.lower().replace("initial", "") in ("waterlevel", "waterdepth") else "-"
                OneDFieldModel(
                    global_=OneDFieldGlobal(quantity=quantity, unit=unit, value=data["nodata"]),
                    branch=[
                        OneDFieldBranch(
                            branchid=str(branch_id), numlocations=len(chainage), chainage=chainage, values=values
                        )
                        for branch_id, (chainage, values) in data["branches"].items()
                    ]
                ).save(filepath=path)
            elif data["filetype"] == "polygon":
                objects = []
                for i, polygon in enumerate(data["polygons"]):
                    coords = list(polygon.exterior.coords)
                    objects.append(
                        PolyObject(
                            metadata=Metadata(name=f"{path.stem}_{i:04d}", n_rows=len(coords), n_columns=2),
                            points=[PolyPoint(x=x, y=y, data=[]) for x, y, *_ in coords],
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

    def write_initial_fields(self, fm, fm_dir: Path | str) -> IniFieldModel:
        """Add all initial fields and parameter fields to the FM model and write their data files.

        This writes the 1dField files (uniform values), rasters and sample files to fm_dir, and links them
        in initialFields.ini, using paths relative to the fm folder. Call this before saving the FM model.

        Args:
            fm (FMModel): the FM model
            fm_dir (Path | str): folder where the FM model (mdu-file) is written

        Returns:
            IniFieldModel: the initial fields model, also set as fm.geometry.inifieldfile
        """
        fm_dir = Path(fm_dir)
        fm_dir.mkdir(parents=True, exist_ok=True)

        onedfield_inifields = [f for f in self.inifields if f.datafiletype == "1dField"]
        for inifield, onedfield in zip(onedfield_inifields, self.onedfieldmodels):
            onedfieldmodel = OneDFieldModel(global_=onedfield)
            onedfieldmodel.save(filepath=fm_dir / inifield.datafile.filepath.name)

        self.write_spatial_fields(fm_dir)

        inifieldmodel = IniFieldModel(initial=self.inifields, parameter=self.parameterfields)
        fm.geometry.inifieldfile = inifieldmodel
        return inifieldmodel

    def initial_conditions_at_nodes(self, network, output_file: Path | str = None, crs=None) -> gpd.GeoDataFrame:
        """Initial water level or depth at the 1D computational nodes, as D-Flow FM applies the initial fields.

        Useful to check the initial conditions, e.g. in a GIS, since the D-HYDRO GUI hardly shows them. The
        fields are applied in their order, following the behaviour found in tests with D-HYDRO 2026.02:
        - values per branch (1dField) are interpolated along the branch and kept constant beyond the first and
          last chainage; uniform values apply everywhere;
        - samples are interpolated linearly in triangles (inside their convex hull), polygon files apply inside
          the polygons; rasters are not evaluated;
        - where fields overlap the last one applies, also between water level and water depth;
        - a water depth from polygons or samples removes earlier water levels from polygons or samples;
        - a uniform water depth overrides spatial water levels.
        Nodes without a value keep the default water level of the MDU (WaterLevIni), shown as missing here.

        Args:
            network (Network): the network with the 1D mesh, e.g. fm.geometry.netfile.network
            output_file (Path | str, optional): shapefile (.shp) or GeoPackage (.gpkg) to write. Defaults to None.
            crs (optional): coordinate reference system. Defaults to the crs of the HyDAMO branches, or
                EPSG:28992 (as written to the model by DIMRWriter.add_crs) if they have none.

        Returns:
            gpd.GeoDataFrame: one point per node with branchid, chainage, quantity ('waterlevel' or 'waterdepth'),
                value and source (data file of the field that sets the value)
        """
        from scipy.interpolate import LinearNDInterpolator

        mesh1d = network._mesh1d
        x, y = np.asarray(mesh1d.mesh1d_node_x, dtype=float), np.asarray(mesh1d.mesh1d_node_y, dtype=float)
        branch_names = np.asarray(mesh1d.network1d_branch_id, dtype=object)
        branchid = branch_names[np.asarray(mesh1d.mesh1d_node_branch_id, dtype=int)]
        chainage = np.asarray(mesh1d.mesh1d_node_branch_offset, dtype=float)
        n = len(x)
        points = gpd.points_from_xy(x, y)

        quantity = np.full(n, None, dtype=object)
        value = np.full(n, np.nan)
        source = np.full(n, None, dtype=object)
        source_type = np.full(n, None, dtype=object)
        uniform_depth = None

        uniform_fields = iter(self.onedfieldmodels)
        for inifield in self.inifields:
            filename = inifield.datafile.filepath.name if inifield.datafile and inifield.datafile.filepath else None
            kind = self._level_or_depth(inifield.quantity)
            data = self.spatial_data.get(filename)
            if data is None and inifield.datafiletype == "1dField":
                onedfield = next(uniform_fields)
                if kind is None or str(inifield.locationtype).lower() == "2d":
                    continue
                values, filetype = np.full(n, float(onedfield.value)), "uniform"
                if kind == "waterdepth":
                    uniform_depth = (float(onedfield.value), filename)
            elif kind is None or data is None:
                continue
            elif data["filetype"] == "branch":
                values, filetype = np.full(n, np.nan), "branch"
                for bid, (branch_chainage, branch_values) in data["branches"].items():
                    on_branch = branchid == bid
                    values[on_branch] = np.interp(chainage[on_branch], branch_chainage, branch_values)
            elif data["filetype"] == "sample":
                values, filetype = np.full(n, np.nan), "sample"
                if len(data["x"]) >= 3:
                    interpolator = LinearNDInterpolator(np.c_[data["x"], data["y"]], data["z"])
                    values = interpolator(np.c_[x, y])
            elif data["filetype"] == "polygon":
                values, filetype = np.full(n, np.nan), "polygon"
                inside = gpd.GeoSeries(points).intersects(shapely.union_all(data["polygons"]))
                values[inside.to_numpy()] = float(inifield.value)
            else:
                logger.warning(f"Raster field {filename} is not evaluated at the nodes.")
                continue

            if kind == "waterdepth" and filetype in ("polygon", "sample"):
                removed = (quantity == "waterlevel") & np.isin(source_type, ["polygon", "sample"])
                quantity[removed], value[removed], source[removed], source_type[removed] = None, np.nan, None, None
            has_value = ~np.isnan(values)
            quantity[has_value], value[has_value] = kind, values[has_value]
            source[has_value], source_type[has_value] = filename, filetype

        if uniform_depth is not None:
            spatial_level = quantity == "waterlevel"
            quantity[spatial_level], value[spatial_level], source[spatial_level] = "waterdepth", uniform_depth[0], uniform_depth[1]

        if crs is None and getattr(self.hydamo, "branches", None) is not None:
            crs = self.hydamo.branches.crs
        if crs is None:
            # the projection that DIMRWriter.add_crs writes to the model
            crs = "EPSG:28992"
        nodes = gpd.GeoDataFrame(
            {
                "node_id": np.asarray(mesh1d.mesh1d_node_id, dtype=object),
                "branchid": branchid,
                "chainage": chainage,
                "quantity": quantity,
                "value": value,
                "source": source,
            },
            geometry=points,
            crs=crs,
        )
        if (nodes["quantity"].isna()).any():
            logger.info(f"{nodes['quantity'].isna().sum()} 1D nodes get no initial value and keep WaterLevIni.")
        if output_file is not None:
            nodes.to_file(output_file)
        return nodes
