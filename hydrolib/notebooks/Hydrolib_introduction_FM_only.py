# Basis
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from hydrolib.core.io.bc.models import ForcingModel
from hydrolib.core.io.crosssection.models import *
from hydrolib.core.io.dimr.models import DIMR, FMComponent, RRComponent
from hydrolib.core.io.ext.models import ExtModel
from hydrolib.core.io.friction.models import FrictionModel
from hydrolib.core.io.inifield.models import IniFieldModel
from hydrolib.core.io.mdu.models import FMModel  # , RainfallRunoffModel
from hydrolib.core.io.obs.models import ObservationPointModel
from hydrolib.core.io.onedfield.models import OneDFieldModel
from hydrolib.core.io.structure.models import *

sys.path.append('../')
# from dhydamo.core.drr import DRRModel

# Importing relevant classes from delft3dfmpy
from hydrolib.dhydamo.core.hydamo import HyDAMO

# from hydrolib.dhydamo.io.fmconverter import RoughnessVariant#
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.geometry.viz import plot_network
from hydrolib.dhydamo.io.dfmwriter import DFLowFMModelWriter
from hydrolib.dhydamo.io.drrwriter import DRRWriter

# path to the package containing the dummy-data
data_path = os.path.abspath("hydrolib/tests/data")

# path to write the models
output_path = os.path.abspath("hydrolib/model")

gpkg_file = os.path.join(data_path, "Example_model.gpkg")

hydamo = HyDAMO(extent_file=os.path.join(data_path, "OLO_stroomgebied_incl.maas.shp"))

# show content
hydamo.branches.show_gpkg(gpkg_file)
hydamo.branches.read_gpkg_layer(gpkg_file, layer_name="HydroObject", index_col="code")

hydamo.profile.read_gpkg_layer(
    gpkg_file,
    layer_name="ProfielPunt",
    groupby_column="profiellijnid",
    order_column="codevolgnummer",
    id_col="code",
)
hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")
hydamo.profile.snap_to_branch(hydamo.branches, snap_method="intersecting")
hydamo.profile.dropna(axis=0, inplace=True, subset=["branch_offset"])
hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")

hydamo.profile.drop("code", axis=1, inplace=True)
hydamo.profile["code"] = hydamo.profile["profiellijnid"]

hydamo.culverts.read_gpkg_layer(
    gpkg_file, layer_name="DuikerSifonHevel", index_col="code"
)
hydamo.culverts.snap_to_branch(hydamo.branches, snap_method="ends", maxdist=5)
hydamo.culverts.dropna(axis=0, inplace=True, subset=["branch_offset"])

hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw")
hydamo.weirs.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
hydamo.weirs.dropna(axis=0, inplace=True, subset=["branch_offset"])
hydamo.opening.read_gpkg_layer(gpkg_file, layer_name="Kunstwerkopening")
hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")

idx = hydamo.management_device[
    hydamo.management_device["duikersifonhevelid"].notnull()
].index
for i in idx:
    globid = hydamo.culverts[
        hydamo.culverts.code == hydamo.management_device.duikersifonhevelid.loc[i]
    ].globalid.values[0]
    hydamo.management_device.at[i, "duikersifonhevelid"] = globid

hydamo.pumpstations.read_gpkg_layer(gpkg_file, layer_name="Gemaal", index_col="code")
hydamo.pumpstations.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")

hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
hydamo.bridges.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
hydamo.bridges.dropna(axis=0, inplace=True, subset=["branch_offset"])

hydamo.boundary_conditions.read_gpkg_layer(
    gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code"
)
hydamo.boundary_conditions.snap_to_branch(
    hydamo.branches, snap_method="overal", maxdist=10
)

hydamo.catchments.read_gpkg_layer(
    gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code"
)

hydamo.laterals.read_gpkg_layer(gpkg_file, layer_name="lateraleknoop")
for ind, cat in hydamo.catchments.iterrows():
    hydamo.catchments.loc[ind, "lateraleknoopcode"] = hydamo.laterals[
        hydamo.laterals.globalid == cat.lateraleknoopid
    ].code.values[0]
hydamo.laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5000)

hydamo.structures.convert.weirs(
    hydamo.weirs,
    hydamo.profile_group,
    hydamo.profile_line,
    hydamo.profile,
    hydamo.opening,
    hydamo.management_device,
)
hydamo.structures.convert.culverts(
    hydamo.culverts, management_device=hydamo.management_device
)
hydamo.structures.convert.bridges(
    hydamo.bridges,
    profile_groups=hydamo.profile_group,
    profile_lines=hydamo.profile_line,
    profiles=hydamo.profile,
)
hydamo.structures.convert.pumps(
    hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management
)

hydamo.structures.add_rweir(
    id="rwtest",
    name="rwtest",
    branchid="W_1386_0",
    chainage=600.0,
    crestlevel=18.0,
    crestwidth=3.0,
    corrcoeff=1.0,
)
hydamo.structures.add_orifice(
    id="otest",
    name="otest",
    branchid="W_1386_0",
    chainage=500.0,
    crestlevel=18.0,
    crestwidth=3.0,
    corrcoeff=1.0,
    gateloweredgelevel=17.0,
    uselimitflowpos=False,
    limitflowpos=0.0,
    uselimitflowneg=False,
    limitflowneg=0.0,
)
hydamo.structures.add_uweir(
    id="uwtest",
    name="uwtest",
    branchid="W_1386_0",
    chainage=400.0,
    dischargecoeff=0.9,
    crestlevel=18.0,
    numlevels=3,
    yvalues="0 0.5 1.0",
    zvalues="19.0 18.0 19.0",
)
hydamo.structures.add_bridge(
    id="btest",
    name="btest",
    branchid="W_1386_0",
    chainage=300.0,
    length=8.0,
    csdefid="rect_3.60",
    inletlosscoeff=0.9,
    outletlosscoeff=0.9,
    shift=0.0,
    frictiontype="Manning",
    friction=0.06,
)
hydamo.structures.add_culvert(
    id="ctest",
    name="ctest",
    branchid="W_1386_0",
    chainage=200.0,
    leftlevel=18.0,
    rightlevel=17.0,
    length=30.0,
    crosssection={"shape": "circle", "diameter": 0.40},
    inletlosscoeff=0.9,
    outletlosscoeff=0.9,
    frictiontype="Manning",
    frictionvalue=0.06,
)
hydamo.structures.add_pump(
    id="ptest",
    name="ptest",
    branchid="W_1386_0",
    chainage=100.0,
    capacity=1.0,
    startlevelsuctionside=[14.0],
    stoplevelsuctionside=[13.8],
)

fm = FMModel()

# TODO:  write a convenience function in mesh(?) and to separate structures
[
    mesh.mesh1d_add_branch(fm.geometry.netfile.network, branch.geometry, 20.0)
    for branch in hydamo.branches.itertuples()
]

# 2d mesh extent
extent = gpd.read_file(r"hydrolib\tests\data\2D_extent.shp").at[0, "geometry"]
# buffer for refinement
buffer = hydamo.branches.buffer(10.0).unary_union
# add triangular mesh, refene around channels and add 1d_to_2d links (TODO: check if all options work)
mesh.mesh2d_add_triangular(fm.geometry.netfile.network, extent, edge_length=20.0)
mesh.mesh2d_refine(fm.geometry.netfile.network, buffer, 1)
mesh.links1d2d_add_links_1d_to_2d(fm.geometry.netfile.network)

# Add cross sections from hydamo
hydamo.crosssections.convert.profiles(
    crosssections=hydamo.profile,
    crosssection_roughness=hydamo.profile_roughness,
    profile_groups=hydamo.profile_group,
    profile_lines=hydamo.profile_line,
    param_profile=hydamo.param_profile,
    param_profile_values=hydamo.param_profile_values,
    branches=hydamo.branches,
    roughness_variant="High",
)

hydamo.external_forcings.convert.boundaries(
    hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network
)
series = pd.Series(np.sin(np.linspace(0, 100, 100)) + 1.0)
series.index = [
    pd.Timestamp("2016-01-01 00:00:00") + pd.Timedelta(hours=i) for i in range(100)
]
hydamo.external_forcings.add_boundary_condition(
    "RVW_01", (197464.0, 392130.0), "dischargebnd", series, fm.geometry.netfile.network
)
hydamo.external_forcings.add_lateral("LAT_01", "W_242209_0", "5.0", series)

hydamo.crosssections.crosssection_loc = hydamo.dict_to_dataframe(
    hydamo.crosssections.crosssection_loc
)
hydamo.crosssections.crosssection_def = hydamo.dict_to_dataframe(
    hydamo.crosssections.crosssection_def
)

hydamo.observationpoints.add_points(
    [Point((200200, 395600)), (200200, 396200)],
    ["ObsPt1", "ObsPt2"],
    locationTypes=["1d", "1d"],
    snap_distance=10.0,
)
hydamo.storagenodes.add_storagenode(
    "test",
    "123_123",
    usestreetstorage="true",
    nodetype="unspecified",
    name=np.nan,
    usetable="false",
    bedlevel=12.0,
    area=100,
    streetlevel=14.0,
    streetstoragearea=10.0,
    storagetype="reservoir",
    levels=np.nan,
    storagearea=np.nan,
    interpolate="linear",
)
hydamo.storagenodes.storagenodes = hydamo.dict_to_dataframe(
    hydamo.storagenodes.storagenodes
)

hydamo.external_forcings.set_initial_waterdepth(0.5)


drrmodel = DRRModel()

# all data and settings to create the RR-model
lu_file = os.path.join(data_path, "rasters", "sobek_landuse.tif")
ahn_file = os.path.join(data_path, "rasters", "AHN_2m_clipped_filled.tif")
soil_file = os.path.join(data_path, "rasters", "sobek_soil.tif")
surface_storage = 10.0
infiltration_capacity = 100.0
initial_gwd = 1.2  # water level depth below surface

runoff_resistance = 1.0
infil_resistance = 300.0
layer_depths = [0.0, 1.0, 2.0]
layer_resistances = [30, 200, 10000]
street_storage = 10.0
sewer_storage = 10.0
pumpcapacity = 10.0
roof_storage = 10.0

meteo_areas = hydamo.catchments

drrmodel.unpaved.io.unpaved_from_input(
    hydamo.catchments,
    lu_file,
    ahn_file,
    soil_file,
    surface_storage,
    infiltration_capacity,
    initial_gwd,
    meteo_areas,
)
drrmodel.unpaved.io.ernst_from_input(
    hydamo.catchments,
    depths=layer_depths,
    resistance=layer_resistances,
    infiltration_resistance=infil_resistance,
    runoff_resistance=runoff_resistance,
)
drrmodel.paved.io.paved_from_input(
    catchments=hydamo.catchments,
    landuse=lu_file,
    surface_level=ahn_file,
    street_storage=street_storage,
    sewer_storage=sewer_storage,
    pump_capacity=pumpcapacity,
    meteo_areas=meteo_areas,
    zonalstats_alltouched=True,
)
drrmodel.greenhouse.io.greenhouse_from_input(
    hydamo.catchments,
    lu_file,
    ahn_file,
    roof_storage,
    meteo_areas,
    zonalstats_alltouched=True,
)
drrmodel.openwater.io.openwater_from_input(
    hydamo.catchments, lu_file, meteo_areas, zonalstats_alltouched=True
)
drrmodel.external_forcings.io.boundary_from_input(
    hydamo.laterals, hydamo.catchments, drrmodel
)

hydamo.external_forcings.convert.laterals(
    hydamo.laterals,
    lateral_discharges=None,
    rr_boundaries=drrmodel.external_forcings.boundary_nodes,
)
hydamo.external_forcings.lateral_nodes = hydamo.dict_to_dataframe(
    hydamo.external_forcings.lateral_nodes
)
hydamo.external_forcings.boundary_nodes = hydamo.dict_to_dataframe(
    hydamo.external_forcings.boundary_nodes
)

fm.filepath = Path(output_path) / "fm" / "test.mdu"

forcingmodel = ForcingModel()
forcingmodel.filepath = Path(output_path) / "fm" / "boundaryconditions.bc"

seepage_folder = os.path.join(data_path, "rasters", "seepage")
precip_folder = os.path.join(data_path, "rasters", "precipitation")
evap_folder = os.path.join(data_path, "rasters", "evaporation")
drrmodel.external_forcings.io.seepage_from_input(hydamo.catchments, seepage_folder)
drrmodel.external_forcings.io.precip_from_input(
    meteo_areas, precip_folder=None, precip_file=str(Path(data_path) / "DEFAULT.BUI")
)
drrmodel.external_forcings.io.evap_from_input(
    meteo_areas, evap_folder=None, evap_file=str(Path(data_path) / "DEFAULT.EVP")
)

writer = DFLowFMModelWriter(hydamo, forcingmodel)
fm.geometry.structurefile = [StructureModel(structure=writer.structures)]
fm.geometry.crosslocfile = CrossLocModel(crosssection=writer.crosslocs)
fm.geometry.crossdeffile = CrossDefModel(definition=writer.crossdefs)
fm.geometry.frictfile = [FrictionModel(global_=writer.friction_defs)]
fm.output.obsfile = [ObservationPointModel(observationpoint=writer.obspoints)]
extmodel = ExtModel()
extmodel.boundary = writer.boundaries_ext
extmodel.lateral = writer.laterals_ext
forcingmodel.forcing = writer.laterals_bc + writer.boundaries_bc
fm.external_forcing.extforcefilenew = extmodel
fm.external_forcing.forcingfile = forcingmodel
fm.geometry.inifieldfile = IniFieldModel(initial=writer.inifields)
onedfieldmodel = OneDFieldModel(global_=writer.onedfields[0])
onedfieldmodel.filepath = Path("initialwaterdepth.ini")
fm.geometry.onedfieldfile = [onedfieldmodel]

drrmodel.d3b_parameters["Timestepsize"] = 300
drrmodel.d3b_parameters[
    "StartTime"
] = "'2016/06/01;00:00:00'"  # should be equal to refdate for D-HYDRO
drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;00:00:00'"
drrmodel.d3b_parameters["RestartIn"] = 0
drrmodel.d3b_parameters["RestartOut"] = 0
drrmodel.d3b_parameters["RestartFileNamePrefix"] = "Test"
drrmodel.d3b_parameters["UnsaturatedZone"] = 1
drrmodel.d3b_parameters["UnpavedPercolationLikeSobek213"] = -1
drrmodel.d3b_parameters["VolumeCheckFactorToCF"] = 100000

# rr_writer = DRRWriter(drrmodel, output_dir= output_path, name='test',wwtp=(199000.,396000.))
# rr_writer.write_all()


# wegschrijven
dimr = DIMR()
dimr.component.append(
    FMComponent(
        name="test",
        workingDir=Path(output_path) / "fm",
        model=fm,
        inputfile=fm.filepath,
    )
    # RRComponent(name="test", workingDir=r"D:\3640.20\HYDROLIB-dhydamo\hydrolib\model", inputfile=fm.filepath, model=rr)
)
dimr.save(recurse=True)


print("done.")
# drrmodel.dimr_path = dimr_path
# print('Writing model')
