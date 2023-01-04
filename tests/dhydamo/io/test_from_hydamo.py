from pathlib import Path

import numpy as np
import pandas as pd
from hydrolib.dhydamo.geometry import mesh
from shapely.geometry import Point
from hydrolib.core.dflowfm.bc.models import ForcingModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.core.dflowfm.mdu.models import FMModel

hydamo_data_path = (
    Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "tests" / "data"
)

print(hydamo_data_path)


def test_hydamo_object_from_gpkg():

    # initialize a hydamo object
    extent_file = hydamo_data_path / "OLO_stroomgebied_incl.maas.shp"
    assert extent_file.exists()
    hydamo = HyDAMO(extent_file=extent_file)

    assert hydamo.clipgeo.area == 139373665.08206934

    # all data is contained in one geopackage called 'Example model'
    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    # Read branches
    hydamo.branches.read_gpkg_layer(
        str(gpkg_file), layer_name="HydroObject", index_col="code"
    )
    assert len(hydamo.branches) == 61
    assert hydamo.branches.length.sum() == 28371.461117125935

    hydamo.profile.read_gpkg_layer(
        gpkg_file,
        layer_name="ProfielPunt",
        groupby_column="profiellijnid",
        order_column="codevolgnummer",
        id_col="code",
    )
    from shapely.geometry import LineString

    assert len(hydamo.profile) == 359
    assert hydamo.profile.geom_type.values[0] == "LineString"

    #     # read roughness
    hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")
    hydamo.profile.snap_to_branch(hydamo.branches, snap_method="intersecting")

    # seven profiles are too far from a branch and are dropped
    assert len(hydamo.profile.branch_offset[hydamo.profile.branch_offset.isnull()]) == 7

    hydamo.profile.dropna(axis=0, inplace=True, subset=["branch_offset"])
    hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
    hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")
    hydamo.profile.drop("code", axis=1, inplace=True)
    hydamo.profile["code"] = hydamo.profile["profiellijnid"]

    # the dataset contains two profile groups (a bridge and a uweir)
    assert hydamo.profile_group.shape == (2, 4)
    # a profile_lines' profielgroepid's should correspond to profile_group globalids
    assert hydamo.profile_line.profielgroepid.values[0] in list(
        hydamo.profile_group.globalid
    )

    #     # Read Weirs
    hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw")
    hydamo.weirs.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
    hydamo.weirs.dropna(axis=0, inplace=True, subset=["branch_offset"])
    hydamo.opening.read_gpkg_layer(gpkg_file, layer_name="Kunstwerkopening")
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")

    assert len(hydamo.weirs) == 25
    assert np.round(hydamo.weirs.doorstroombreedte.mean(), 2) == 2.17

    # Read culverts
    hydamo.culverts.read_gpkg_layer(
        gpkg_file, layer_name="DuikerSifonHevel", index_col="code"
    )
    hydamo.culverts.snap_to_branch(hydamo.branches, snap_method="ends", maxdist=5)
    hydamo.culverts.dropna(axis=0, inplace=True, subset=["branch_offset"])
    assert len(hydamo.culverts) == 90
    assert hydamo.culverts.length.sum() == 2497.687230867272

    # Read management device
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")
    idx = hydamo.management_device.loc[
        hydamo.management_device["duikersifonhevelid"].notnull()
    ].index
    for i in idx:
        globid = hydamo.culverts.loc[
            hydamo.culverts["code"].eq(
                hydamo.management_device.at[i, "duikersifonhevelid"]
            ),
            "globalid",
        ].values[0]
        hydamo.management_device.at[i, "duikersifonhevelid"] = globid
    assert len(hydamo.management_device) == 27

    # Read pumpstations
    hydamo.pumpstations.read_gpkg_layer(
        gpkg_file, layer_name="Gemaal", index_col="code"
    )
    hydamo.pumpstations.snap_to_branch(
        hydamo.branches, snap_method="overal", maxdist=10
    )
    hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
    hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")

    assert len(hydamo.pumps) == 1

    # Read bridges
    hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
    hydamo.bridges.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
    hydamo.bridges.dropna(axis=0, inplace=True, subset=["branch_offset"])

    assert len(hydamo.bridges) == 1
    assert hydamo.bridges.branch_offset.values[0] == 182.117

    # Read boundary conditions
    hydamo.boundary_conditions.read_gpkg_layer(
        gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code"
    )
    hydamo.boundary_conditions.snap_to_branch(
        hydamo.branches, snap_method="overal", maxdist=10
    )
    assert len(hydamo.boundary_conditions) == 1

    assert not hydamo.boundary_conditions.branch_offset.empty

    # Read catchments
    hydamo.catchments.read_gpkg_layer(
        gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code"
    )
    assert len(hydamo.catchments) == 129
    assert hydamo.catchments.oppervlakt.sum() == 2872.9925

    # Read laterals
    hydamo.laterals.read_gpkg_layer(gpkg_file, layer_name="lateraleknoop")
    for ind, cat in hydamo.catchments.iterrows():
        hydamo.catchments.loc[ind, "lateraleknoopcode"] = hydamo.laterals[
            hydamo.laterals.globalid == cat.lateraleknoopid
        ].code.values[0]
    hydamo.laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5000)

    assert len(hydamo.laterals) == 121
    assert np.round(hydamo.laterals.afvoer.mean(),4) == 0.0058

    return hydamo


def test_convert_structures():
    # iniate a hydamo object
    hydamo = test_hydamo_object_from_gpkg()

    # Convert
    hydamo.structures.convert.weirs(
        hydamo.weirs,
        hydamo.profile_group,
        hydamo.profile_line,
        hydamo.profile,
        hydamo.opening,
        hydamo.management_device,
    )
    # one weir is converted to an orifice
    assert hydamo.structures.rweirs_df.shape[0] == hydamo.weirs.shape[0] - 1

    hydamo.structures.convert.culverts(
        hydamo.culverts, management_device=hydamo.management_device
    )
    assert hydamo.structures.culverts_df.shape[0] == hydamo.culverts.shape[0]

    hydamo.structures.convert.bridges(
        hydamo.bridges,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        profiles=hydamo.profile,
    )
    assert hydamo.structures.bridges_df.shape[0] == hydamo.bridges.shape[0]

    hydamo.structures.convert.pumps(
        hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management
    )

    assert hydamo.structures.pumps_df.shape[0] == hydamo.pumps.shape[0]
    return hydamo


def test_convert_crosssections():

    # initiate a hydamo object
    hydamo = test_hydamo_object_from_gpkg()
    hydamo = test_convert_structures()

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

    assert len(hydamo.crosssections.crosssection_def) == 442
    assert len(hydamo.crosssections.crosssection_loc) == 349

    # check whether the bridge profile in in the crosssection definitions
    assert (
        hydamo.structures.bridges_df.csdefid.values[0]
        in hydamo.crosssections.crosssection_def.keys()
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

    assert "default" in hydamo.crosssections.crosssection_def.keys()


def test_add_structures_manually():

    # iniate a hydamo object
    hydamo = test_hydamo_object_from_gpkg()

    # Add structures manually
    hydamo.structures.add_rweir(
        id="rwtest",
        name="rwtest",
        branchid="W_1386_0",
        chainage=2.0,
        crestlevel=18.0,
        crestwidth=3.0,
        corrcoeff=1.0,
    )
    assert "rwtest" in hydamo.structures.rweirs_df.id.values[0]

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
    assert "otest" in hydamo.structures.orifices_df.id.values[0]

    hydamo.structures.add_uweir(
        id="uwtest",
        name="uwtest",
        branchid="W_1386_0",
        chainage=6.0,
        dischargecoeff=0.9,
        crestlevel=18.0,
        numlevels=3,
        yvalues="0 0.5 1.0",
        zvalues="19.0 18.0 19.0",
    )
    assert "uwtest" in hydamo.structures.uweirs_df.id.values[0]

    hydamo.structures.add_bridge(
        id="btest",
        name="btest",
        branchid="W_1386_0",
        chainage=7.0,
        length=8.0,
        csdefid="rect_3.60",
        inletlosscoeff=0.9,
        outletlosscoeff=0.9,
        shift=0.0,
        frictiontype="Manning",
        friction=0.06,
    )
    assert "btest" in hydamo.structures.bridges_df.id.values[0]

    hydamo.structures.add_culvert(
        id="ctest",
        name="ctest",
        branchid="W_1386_0",
        chainage=8.0,
        leftlevel=18.0,
        rightlevel=17.0,
        length=30.0,
        crosssection={"shape": "circle", "diameter": 0.40},
        inletlosscoeff=0.9,
        outletlosscoeff=0.9,
        bedfrictiontype="Manning",
        bedfriction=0.06,
    )
    assert "ctest" in hydamo.structures.culverts_df.id.values[0]

    hydamo.structures.add_pump(
        id="ptest",
        name="ptest",
        branchid="W_1386_0",
        chainage=9.0,
        capacity=1.0,
        startlevelsuctionside=[14.0],
        stoplevelsuctionside=[13.8],
    )
    assert "ptest" in hydamo.structures.pumps_df.id.values[0]
    return hydamo


def test_observationpoints():
    hydamo = test_hydamo_object_from_gpkg()

    hydamo.observationpoints.add_points(
        [Point((200200, 395600)), (200200, 396200)],
        ["ObsPt1", "ObsPt2"],
        locationTypes=["1d", "1d"],
        snap_distance=10.0,
    )
    assert len(hydamo.observationpoints.observation_points) == 2


def test_storagenodes():
    hydamo = test_hydamo_object_from_gpkg()
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

    assert len(hydamo.storagenodes.storagenodes.keys()) == 1


def test_convert_boundararies():
    hydamo = test_hydamo_object_from_gpkg()

    fm = FMModel()

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=None,
    )

    hydamo.external_forcings.convert.boundaries(
        hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network
    )
    assert [i == "RVM_02" for i in hydamo.external_forcings.boundary_nodes.keys()][0]
    assert len(hydamo.external_forcings.boundary_nodes.keys()) == 1


def test_add_boundaries():
    hydamo = test_hydamo_object_from_gpkg()

    fm = FMModel()

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=None,
    )

    series = pd.Series(np.sin(np.linspace(2, 8, 100) * -1) + 1.0)
    series.index = [
        pd.Timestamp("2016-01-01 00:00:00") + pd.Timedelta(hours=i) for i in range(100)
    ]

    hydamo.external_forcings.add_boundary_condition(
        "RVW_01",
        (197464.0, 392130.0),
        "dischargebnd",
        series,
        fm.geometry.netfile.network,
    )
    assert len(hydamo.external_forcings.boundary_nodes.keys()) == 1


def test_add_initialfields():
    hydamo = test_hydamo_object_from_gpkg()

    hydamo.external_forcings.set_initial_waterdepth(1.5)

    assert (
        hydamo.external_forcings.initial_waterdepth_polygons.waterdepth.values[0] == 1.5
    )


def test_write_laterals():

    # Get full hydamo object
    hydamo = test_hydamo_object_from_gpkg()

    series = pd.Series(np.sin(np.linspace(2, 8, 100) * -1) + 1.0)
    series.index = [
        pd.Timestamp("2016-01-01 00:00:00") + pd.Timedelta(hours=i) for i in range(100)
    ]
    series.plot()
    hydamo.external_forcings.add_lateral("LAT_01", "W_242209_0", "5.0", series)

    assert (
        np.mean(hydamo.external_forcings.lateral_nodes["LAT_01"]["value"])
        == 1.0351497742173033
    )
