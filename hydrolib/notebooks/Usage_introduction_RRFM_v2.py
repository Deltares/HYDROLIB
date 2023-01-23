# Basis
from pathlib import Path
import shutil
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# import contextily as cx
import os

sys.path.insert(0, ".")

# and from hydrolib-core
from hydrolib.core.dimr.models import DIMR, FMComponent
from hydrolib.core.dflowfm.inifield.models import IniFieldModel, DiskOnlyFileModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel
from hydrolib.core.dflowfm.structure.models import StructureModel
from hydrolib.core.dflowfm.crosssection.models import CrossLocModel, CrossDefModel
from hydrolib.core.dflowfm.ext.models import ExtModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.obs.models import ObservationPointModel

# Importing relevant classes from Hydrolib-dhydamo
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from hydrolib.dhydamo.geometry.viz import plot_network


data_path = Path("hydrolib/tests/data").resolve()
assert data_path.exists()
# path to write the models
output_path = Path("hydrolib/tests/model").resolve().mkdir(parentes=True, exist_ok=True)
assert output_path.exists()


gpkg_file = str(data_path / "Example_model.gpkg")
# initialize a hydamo object
hydamo = HyDAMO(extent_file=data_path / "OLO_stroomgebied_incl.maas.shp")

TwoD = True
RTC = True
RR = True

# show content
hydamo.branches.show_gpkg(gpkg_file)

# read branchs
hydamo.branches.read_gpkg_layer(gpkg_file, layer_name="HydroObject", index_col="code")
# read profiles
hydamo.profile.read_gpkg_layer(
    gpkg_file,
    layer_name="ProfielPunt",
    groupby_column="profiellijnid",
    order_column="codevolgnummer",
    id_col="code",
    index_col="code",
)
hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")
hydamo.profile.snap_to_branch(hydamo.branches, snap_method="intersecting")
hydamo.profile.dropna(axis=0, inplace=True, subset=["branch_offset"])
hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")
hydamo.profile.drop("code", axis=1, inplace=True)
hydamo.profile["code"] = hydamo.profile["profiellijnid"]
# structures
hydamo.culverts.read_gpkg_layer(
    gpkg_file, layer_name="DuikerSifonHevel", index_col="code"
)
hydamo.culverts.snap_to_branch(hydamo.branches, snap_method="ends", maxdist=5)
hydamo.culverts.dropna(axis=0, inplace=True, subset=["branch_offset"])

hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw", index_col="code")
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
hydamo.pumpstations.dropna(axis=0, inplace=True, subset=["branch_offset"])
hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")

hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
hydamo.bridges.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
hydamo.bridges.dropna(axis=0, inplace=True, subset=["branch_offset"])
# boundaries
hydamo.boundary_conditions.read_gpkg_layer(
    gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code"
)
hydamo.boundary_conditions.snap_to_branch(
    hydamo.branches, snap_method="overal", maxdist=10
)

# catchments
hydamo.catchments.read_gpkg_layer(
    gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code"
)
# laterals
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


hydamo.structures.add_orifice(
    id="otest",
    name="otest",
    branchid="W_1386_0",
    chainage=5.0,
    crestlevel=18.0,
    crestwidth=3.0,
    corrcoeff=1.0,
    gateloweredgelevel=18.5,
    uselimitflowpos=False,
    limitflowpos=0.0,
    uselimitflowneg=False,
    limitflowneg=0.0,
)

# hydamo.structures.add_uweir(
#     id="uwtest",
#     name="uwtest",
#     branchid="W_1386_0",
#     chainage=6.0,
#     dischargecoeff=0.9,
#     crestlevel=18.0,
#     numlevels=3,
#     yvalues="0 0.5 1.0",
#     zvalues="19.0 18.0 19.0",
# )
# hydamo.structures.add_bridge(
#     id="btest",
#     name="btest",
#     branchid="W_1386_0",
#     chainage=7.0,
#     length=8.0,
#     csdefid=hydamo.structures.bridges_df.iloc[0].csdefid,
#     inletlosscoeff=0.9,
#     outletlosscoeff=0.9,
#     shift=0.0,
#     frictiontype="Manning",
#     friction=0.06,
# )
# hydamo.structures.add_culvert(
#     id="ctest",
#     name="ctest",
#     branchid="W_1386_0",
#     chainage=8.0,
#     leftlevel=18.0,
#     rightlevel=17.0,
#     length=30.0,
#     crosssection={"shape": "circle", "diameter": 0.40},
#     inletlosscoeff=0.9,
#     outletlosscoeff=0.9,
#     bedfrictiontype="StricklerKs",
#     bedfriction=75,
# )
# hydamo.structures.add_pump(
#     id="ptest",
#     name="ptest",
#     branchid="W_1386_0",
#     chainage=9.0,
#     capacity=1.0,
#     startlevelsuctionside=[14.0],
#     stoplevelsuctionside=[13.8],
# )

# hydamo.structures.culverts_df.head()


structures = hydamo.structures.as_dataframe(
    rweirs=True,
    bridges=True,
    uweirs=True,
    culverts=True,
    orifices=True,
    pumps=True,
)

fm = FMModel()
# Set start and stop time
fm.time.refdate = 20160601
fm.time.tstop = 2 * 3600 * 24
fm.output.hisinterval = [3600.0]
fm.time.dtuser = 300.0

mesh.mesh1d_add_branches_from_gdf(
    fm.geometry.netfile.network,
    branches=hydamo.branches,
    branch_name_col="code",
    node_distance=20,
    max_dist_to_struc=None,
    structures=structures,
)

if TwoD:
    # 2d mesh extent
    extent = gpd.read_file(data_path / "2D_extent.shp").at[0, "geometry"]

    # add triangular mesh within the 2D extent
    network = fm.geometry.netfile.network
    # mesh.mesh2d_add_triangular(network, extent, edge_length=50.0)
    mesh.mesh2d_add_rectilinear(network, extent, dx=50, dy=50)
    print("Nodes before refinement:", network._mesh2d.mesh2d_node_x.size)

    # refine around the branches. This does only work for a polygon without holes, so use the exterior
    buffer = Polygon(hydamo.branches.buffer(50.0).unary_union.exterior)
    mesh.mesh2d_refine(network, buffer, 1)
    print("Nodes before refinement:", network._mesh2d.mesh2d_node_x.size)

    # add terrain level
    mesh.mesh2d_altitude_from_raster(
        network, data_path / "rasters/AHN_2m_clipped_filled.tif", "face", "mean"
    )

    # add 1d2d links
    mesh.links1d2d_add_links_1d_to_2d(fm.geometry.netfile.network)


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
missing = hydamo.crosssections.get_branches_without_crosssection()
print(f"{len(missing)} branches are still missing a cross section.")

print(
    f"{len(hydamo.crosssections.get_structures_without_crosssection())} structures are still missing a cross section."
)


j = 0
hydamo.branches["order"] = np.nan
for i in hydamo.branches.naam.unique():
    if i != None:
        if (
            all(
                x in missing
                for x in hydamo.branches.loc[
                    hydamo.branches.loc[:, "naam"] == i, "code"
                ]
            )
            == False
        ):
            hydamo.branches.loc[hydamo.branches.loc[:, "naam"] == i, "order"] = int(j)
            j = j + 1

interpolation = []
for i in hydamo.branches.order.unique():
    if i > 0:
        mesh.mesh1d_set_branch_order(
            fm.geometry.netfile.network,
            hydamo.branches.code[hydamo.branches.order == i].to_list(),
            idx=int(i),
        )
        interpolation = (
            interpolation + hydamo.branches.code[hydamo.branches.order == i].to_list()
        )

missing_after_interpolation = np.setdiff1d(missing, interpolation)
print(
    "After interpolation",
    len(missing_after_interpolation),
    "crosssections are missing.",
)

# Set a default cross section
default = hydamo.crosssections.add_rectangle_definition(
    height=5.0,
    width=5.0,
    closed=False,
    roughnesstype="StricklerKs",
    roughnessvalue=30,
    name="default",
)
hydamo.crosssections.set_default_definition(definition=default, shift=10.0)
hydamo.crosssections.set_default_locations(missing_after_interpolation)

hydamo.observationpoints.add_points(
    [
        Point(199617, 394885),
        Point(199421, 393769),
        Point(199398, 393770),
        Point(200198, 396489),
    ],
    ["Obs_BV152054", "ObsS_96684_1", "ObsS_96684_2", "ObsS_96544"],
    locationTypes=["1d", "1d", "1d", "1d"],
    snap_distance=10.0,
)


# hydamo.storagenodes.add_storagenode(
#     "test",
#     "123_123",
#     usestreetstorage="true",
#     nodetype="unspecified",
#     name=np.nan,
#     usetable="false",
#     bedlevel=12.0,
#     area=100,
#     streetlevel=14.0,
#     streetstoragearea=10.0,
#     storagetype="reservoir",
#     levels=np.nan,
#     storagearea=np.nan,
#     interpolate="linear",
# )


hydamo.external_forcings.convert.boundaries(
    hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network
)


# #### Add a fictional time series to use in the BC file


series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 1.0)
series.index = [
    pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)
]

hydamo.external_forcings.add_boundary_condition(
    "RVW_01", (197464.0, 392130.0), "dischargebnd", series, fm.geometry.netfile.network
)


hydamo.dict_to_dataframe(hydamo.external_forcings.boundary_nodes)


hydamo.external_forcings.set_initial_waterdepth(1.5)

if RTC:
    drtcmodel = DRTCModel(
        hydamo,
        fm,
        output_path=output_path,
        complex_controllers_folder=data_path / "complex_controllers",
        rtc_timestep=60.0,
    )

    pid_settings = {}
    pid_settings["global"] = {
        "ki": -0.05,
        "kp": -0.03,
        "kd": 0.0,
        "maxspeed": 0.00033,
    }
    if not hydamo.management.typecontroller.empty:
        timeseries = pd.read_csv(data_path / "timecontrollers.csv")
        timeseries.index = timeseries.Time

        pid_settings["kst_pid"] = {
            "ki": -0.03,
            "kp": -0.0,
            "kd": 0.0,
            "maxspeed": 0.00033,
        }

        drtcmodel.from_hydamo(pid_settings=pid_settings, timeseries=timeseries)

    drtcmodel.add_pid_controller(
        structure_id="S_96544",
        steering_variable="Crest level (s)",
        target_variable="Water level (op)",
        setpoint=18.2,
        observation_location="ObsS_96544",
        lower_bound=18.0,
        upper_bound=18.4,
        pid_settings=pid_settings["global"],
    )

    drtcmodel.add_time_controller(
        structure_id="S_96548",
        steering_variable="Crest level (s)",
        data=timeseries.iloc[:, 1],
    )

if RR:
    drrmodel = DRRModel()

    # all data and settings to create the RR-model
    lu_file = data_path / "rasters" / "sobek_landuse.tif"
    ahn_file = data_path / "rasters" / "AHN_2m_clipped_filled.tif"
    soil_file = data_path / "rasters" / "sobek_soil.tif"
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

    hydamo.dict_to_dataframe(drrmodel.paved.pav_nodes).tail()

    drrmodel.paved.add_paved(
        id="test_pav",
        area="100",
        surface_level="18.1",
        street_storage="10.0",
        sewer_storage="10.0",
        pump_capacity="1.",
        meteo_area=hydamo.catchments.iloc[0].code,
        px=str(hydamo.catchments.iloc[0].geometry.centroid.coords[0][0]),
        py=str(hydamo.catchments.iloc[0].geometry.centroid.coords[0][1]),
        boundary_node=list(drrmodel.external_forcings.boundary_nodes.keys())[0],
    )

    seepage_folder = data_path / "rasters" / "seepage"
    precip_folder = data_path / "rasters" / "precipitation"
    precip_file = str(data_path / "DEFAULT.BUI")
    evap_folder = data_path / "rasters" / "evaporation"
    drrmodel.external_forcings.io.seepage_from_input(hydamo.catchments, seepage_folder)
    drrmodel.external_forcings.io.precip_from_input(
        meteo_areas, precip_folder=None, precip_file=precip_file
    )
    drrmodel.external_forcings.io.evap_from_input(
        meteo_areas, evap_folder=evap_folder, evap_file=None
    )

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

    hydamo.external_forcings.convert.laterals(
        hydamo.laterals,
        lateral_discharges=None,
        rr_boundaries=drrmodel.external_forcings.boundary_nodes,
    )
else:
    lateral_discharges = hydamo.laterals["afvoer"]
    lateral_discharges.index = hydamo.laterals.code
    hydamo.external_forcings.convert.laterals(
        hydamo.laterals, lateral_discharges=lateral_discharges, rr_boundaries=None
    )

hydamo.dict_to_dataframe(hydamo.external_forcings.lateral_nodes).tail()


# main filepath
fm.filepath = Path(output_path) / "fm" / "demo.mdu"
# we first need to set the forcing model, because it is referred to in the ext model components

# forcingmodel.filepath = Path(output_path) / "fm" / "boundaryconditions.bc"


models = Df2HydrolibModel(hydamo)

fm.geometry.structurefile = [StructureModel(structure=models.structures)]
fm.geometry.crosslocfile = CrossLocModel(crosssection=models.crosslocs)
fm.geometry.crossdeffile = CrossDefModel(definition=models.crossdefs)
fm.geometry.bedlevtype = 1
fm.geometry.frictfile = []
for i, fric_def in enumerate(models.friction_defs):
    fric_model = FrictionModel(global_=fric_def)
    fric_model.filepath = f"roughness_{i}.ini"
    fm.geometry.frictfile.append(fric_model)

fm.output.obsfile = [ObservationPointModel(observationpoint=models.obspoints)]

extmodel = ExtModel()
extmodel.boundary = models.boundaries_ext
extmodel.lateral = models.laterals_ext
fm.external_forcing.extforcefilenew = extmodel

fm.geometry.inifieldfile = IniFieldModel(initial=models.inifields)

for ifield, onedfield in enumerate(models.onedfieldmodels):
    # eventually this is the way, but it has not been implemented yet in Hydrolib core
    # fm.geometry.inifieldfile.initial[ifield].datafile = OneDFieldModel(global_=onedfield)

    # this is a workaround to do the same
    onedfield_filepath = output_path / "fm" / "initialwaterdepth.ini"
    onedfieldmodel = OneDFieldModel(global_=onedfield)
    onedfieldmodel.save(filepath=onedfield_filepath)
    fm.geometry.inifieldfile.initial[ifield].datafile = DiskOnlyFileModel(
        filepath=onedfield_filepath
    )

dimr = DIMR()
dimr.component.append(
    FMComponent(
        name="DFM",
        workingDir=Path(output_path) / "fm",
        model=fm,
        inputfile=fm.filepath,
    )
    # RRComponent(name="test", workingDir="."", inputfile=fm.filepath, model=rr)
)
dimr.save(recurse=True)

# shutil.copy(data_path / "initialWaterDepth.ini", output_path / "fm")

if RR:
    rr_writer = DRRWriter(
        drrmodel, output_dir=output_path, name="test", wwtp=(199000.0, 396000.0)
    )
    rr_writer.write_all()

from hydrolib.dhydamo.io.dimrwriter import DIMRWriter

if RTC:
    drtcmodel.write_xml_v1()

if not RR:
    drrmodel = None
if not RTC:
    drtcmodel = None

dimr = DIMRWriter(output_path=output_path)
dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)
dimr.write_runbat()

print("Done.")

# %%
