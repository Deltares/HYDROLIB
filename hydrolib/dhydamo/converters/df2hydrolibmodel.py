import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hydrolib.core.dflowfm.bc.models import (
    Constant,
    ForcingModel,
    QHTable,
    QuantityUnitPair,
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
from hydrolib.core.dflowfm.onedfield.models import OneDFieldGlobal, OneDFieldModel
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
from hydrolib.dhydamo.geometry.spatial import rasterize_polygons
from hydrolib.dhydamo.io.rasterwriter import write_raster

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
                bnd_bc = Constant(
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
            if gdf.empty:
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
        """Convert spatial fields (polygons, rasters or samples) to InitialField/ParameterField objects.

        Polygons get one .pol file and one block per distinct value. Only the references are created here;
        the data files themselves are written by write_spatial_fields().
        """
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

    def write_spatial_fields(self, output_dir: Path | str) -> None:
        """Write the data files of the spatial fields (polygon files, rasters and samples).

        The files are referenced relative to initialFields.ini, so output_dir should be the
        folder where the FMModel (and its initialFields.ini) is saved.

        Args:
            output_dir (Path | str): fm-folder of the model
        """
        output_dir = Path(output_dir)
        for filename, data in self.spatial_data.items():
            path = output_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if data["filetype"] == "polygon":
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
