from pathlib import Path

import numpy as np
import pandas as pd
from hydrolib.dhydamo.core import hydamo
from hydrolib.dhydamo.geometry import mesh
from shapely.geometry import Point
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from hydrolib.core.dflowfm.mdu.models import FMModel


hydamo_data_path = (
    Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "tests" / "data"
)

def _check_related(hydamo, hydamo_name):
    # assert type of object
    extendedgdf = getattr(hydamo, hydamo_name)
    assert isinstance(extendedgdf, ExtendedGeoDataFrame)

    # Check relations recursively
    if extendedgdf.related is not None:
        for target_str, relation in extendedgdf.related.items():
            print(f"{target_str}:")
            _recursive_check_related(hydamo, hydamo_name, target_str, **relation)

def _recursive_check_related(hydamo, source_str, target_str, via, on, coupled_to):
    source = getattr(hydamo, source_str)
    target = getattr(hydamo, target_str)

    assert not source.empty
    assert not target.empty
    assert via in source.columns
    assert on in target.columns

    if coupled_to is not None:
        for next_target_str, next_relation in coupled_to.items():
            return _recursive_check_related(hydamo, target_str, next_target_str, **next_relation)

def test_hydamo_related():
    # all data is contained in one geopackage called 'Example model'
    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    hydamo = HyDAMO()
    hydamo.branches.read_gpkg_layer(gpkg_file, layer_name="HydroObject", index_col="code")
    hydamo.profile.read_gpkg_layer(gpkg_file, layer_name="ProfielPunt", groupby_column="profiellijnid", order_column="codevolgnummer", id_col="code")
    hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")
    hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
    hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")
    hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw")
    hydamo.opening.read_gpkg_layer(gpkg_file, layer_name="Kunstwerkopening")
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")
    hydamo.culverts.read_gpkg_layer(gpkg_file, layer_name="DuikerSifonHevel", index_col="code")
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")
    hydamo.pumpstations.read_gpkg_layer(gpkg_file, layer_name="Gemaal", index_col="code")
    hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
    hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")
    hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
    hydamo.boundary_conditions.read_gpkg_layer(gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code")
    hydamo.catchments.read_gpkg_layer(gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code", check_geotype=False)
    hydamo.laterals.read_gpkg_layer(gpkg_file, layer_name="lateraleknoop")
    hydamo.sewer_areas.read_shp(hydamo_data_path / 'rioleringsgebieden.shp', index_col='code', column_mapping={'Code':'code', 'Berging_mm':'riool_berging_mm', 'POC_m3s':'riool_poc_m3s' })
    hydamo.overflows.read_shp(hydamo_data_path / 'overstorten.shp', column_mapping={'codegerela': 'codegerelateerdobject'})

    _check_related(hydamo, "branches")
    _check_related(hydamo, "profile")
    _check_related(hydamo, "profile_line")
    _check_related(hydamo, "weirs")
    _check_related(hydamo, "bridges")
    _check_related(hydamo, "culverts")
    _check_related(hydamo, "pumpstations")
    _check_related(hydamo, "boundary_conditions")
    _check_related(hydamo, "catchments")
    _check_related(hydamo, "laterals")
    _check_related(hydamo, "overflows")
    _check_related(hydamo, "sewer_areas")

def _hydamo_object_from_gpkg():
    # initialize a hydamo object
    extent_file = hydamo_data_path / "OLO_stroomgebied_incl.maas.shp"
    assert extent_file.exists()
    hydamo = HyDAMO(extent_file=extent_file)

    # all data is contained in one geopackage called 'Example model'
    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    # Read branches
    hydamo.branches.read_gpkg_layer(
        str(gpkg_file), layer_name="HydroObject", index_col="code"
    )

    hydamo.profile.read_gpkg_layer(
        gpkg_file,
        layer_name="ProfielPunt",
        groupby_column="profiellijnid",
        order_column="codevolgnummer",
        id_col="code",
    )

    # read roughness
    hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")

    hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
    hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")
    hydamo.profile.drop("code", axis=1, inplace=True)
    hydamo.profile["code"] = hydamo.profile["profiellijnid"]
    len_profile_before = len(hydamo.profile)
    hydamo.snap_to_branch_and_drop(hydamo.profile, hydamo.branches, snap_method="intersecting", drop_related=True)

    # Read Weirs
    hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw")
    hydamo.opening.read_gpkg_layer(gpkg_file, layer_name="Kunstwerkopening")
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")

    # Read culverts
    hydamo.culverts.read_gpkg_layer(
        gpkg_file, layer_name="DuikerSifonHevel", index_col="code"
    )

    # Read management device
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")

    hydamo.snap_to_branch_and_drop(hydamo.weirs, hydamo.branches, snap_method="overal", maxdist=10, drop_related=True)

    hydamo.snap_to_branch_and_drop(hydamo.culverts, hydamo.branches, snap_method="ends", maxdist=5, drop_related=True)

    # Read pumpstations
    hydamo.pumpstations.read_gpkg_layer(
        gpkg_file, layer_name="Gemaal", index_col="code"
    )
    hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
    hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")

    hydamo.snap_to_branch_and_drop(hydamo.pumpstations, hydamo.branches, snap_method="overal", maxdist=10, drop_related=True)

    # Read bridges
    hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
    hydamo.snap_to_branch_and_drop(hydamo.bridges, hydamo.branches, snap_method="overal", maxdist=1100, drop_related=True)

    # Read boundary conditions
    hydamo.boundary_conditions.read_gpkg_layer(
        gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code"
    )
    hydamo.boundary_conditions.snap_to_branch(
        hydamo.branches, snap_method="overal", maxdist=10
    )

    # Read catchments
    hydamo.catchments.read_gpkg_layer(
        gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code", check_geotype=False,
    )

    # Read laterals
    # read laterals
    hydamo.laterals.read_gpkg_layer(gpkg_file, layer_name="lateraleknoop")
    hydamo.laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5000)
    hydamo.catchments['boundary_node'] = [hydamo.laterals[hydamo.laterals.globalid==c['lateraleknoopid']].code.values[0] for _,c in hydamo.catchments.iterrows()]
        
    return hydamo, len_profile_before


def test_hydamo_object_from_gpkg():
    hydamo, len_profile_before = _hydamo_object_from_gpkg()

    assert np.round(hydamo.clipgeo.area) == 139373665
    assert len(hydamo.branches) == 61
    assert np.round(hydamo.branches.length.sum()) == 28371
    assert len_profile_before == 359
    # seven profiles are too far from a branch and are dropped
    assert len(hydamo.profile) == 352
    assert hydamo.profile.geom_type.values[0] == "LineString"
    # the dataset contains two profile groups (a bridge and a uweir)
    assert hydamo.profile_group.shape == (2, 4)
    # a profile_lines' profielgroepid's should correspond to profile_group globalids
    assert hydamo.profile_line.profielgroepid.values[0] in list(
        hydamo.profile_group.globalid
    )
    assert len(hydamo.management_device) == 32
    assert len(hydamo.weirs) == 25
    assert np.round(hydamo.weirs.doorstroombreedte.mean()) == 2
    assert len(hydamo.culverts) == 90
    assert np.round(hydamo.culverts.length.sum()) == 2498
    assert len(hydamo.pumps) == 1
    assert len(hydamo.bridges) == 1
    assert np.round(hydamo.bridges.branch_offset.values[0]) == 182
    assert len(hydamo.boundary_conditions) == 1
    assert not hydamo.boundary_conditions.branch_offset.empty
    assert len(hydamo.catchments) == 121
    assert np.round(hydamo.catchments.oppervlakt.sum()) == 2662
    assert len(hydamo.laterals) == 121
    assert np.round(hydamo.laterals.afvoer.mean()) == 0

def _convert_structures(hydamo=None):
    # iniate a hydamo object
    if hydamo is None:
        hydamo, _ = _hydamo_object_from_gpkg()

    # Convert
    hydamo.structures.convert.weirs(
        weirs=hydamo.weirs,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        profiles=hydamo.profile,
        opening=hydamo.opening,
        management_device=hydamo.management_device,
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

    return hydamo

def test_convert_structures(hydamo=None):
    hydamo = _convert_structures(hydamo=hydamo)
    # one weir is converted to an orifice, one tol a universal weir. w weirs have in total 4 extra openings, so fictional weirs are added.
    assert hydamo.structures.rweirs_df.shape[0] == hydamo.weirs.shape[0] - 2 + 4
    assert len(hydamo.structures.compounds_df)==2
    assert hydamo.structures.compounds_df[hydamo.structures.compounds_df.id=='cmp_S_98740'].structureids.squeeze() == 'S_98740_1;S_98740_2;S_98740_3;S_98740_4'
    assert hydamo.structures.culverts_df.shape[0] == hydamo.culverts.shape[0]
    assert hydamo.structures.bridges_df.shape[0] == hydamo.bridges.shape[0]
    assert hydamo.structures.pumps_df.shape[0] == hydamo.pumps.shape[0]

def _convert_crosssections():
    # initiate a hydamo object
    hydamo = _convert_structures()

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

    return hydamo

def test_convert_crosssections():
    hydamo = _convert_crosssections()

    assert len(hydamo.crosssections.crosssection_def) == 443
    assert len(hydamo.crosssections.crosssection_loc) == 349

    # check whether the bridge profile in in the crosssection definitions
    assert (
        hydamo.structures.bridges_df.csdefid.values[0]
        in hydamo.crosssections.crosssection_def.keys()
    )
    assert "default" in hydamo.crosssections.crosssection_def.keys()


def _add_structures_manually():
    # iniate a hydamo object
    hydamo, _ = _hydamo_object_from_gpkg()

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

    hydamo.structures.add_pump(
        id="ptest",
        name="ptest",
        branchid="W_1386_0",
        chainage=9.0,
        capacity=1.0,
        startlevelsuctionside=[14.0],
        stoplevelsuctionside=[13.8],
    )

    return hydamo

def test_add_structures_manually():
    hydamo = _add_structures_manually()
    assert "rwtest" in hydamo.structures.rweirs_df.id.values[0]
    assert "otest" in hydamo.structures.orifices_df.id.values[0]
    assert "uwtest" in hydamo.structures.uweirs_df.id.values[0]
    assert "btest" in hydamo.structures.bridges_df.id.values[0]
    assert "ctest" in hydamo.structures.culverts_df.id.values[0]
    assert "ptest" in hydamo.structures.pumps_df.id.values[0]

def test_observationpoints():
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.observationpoints.add_points(
        [Point((200200, 395600)), (200200, 396200)],
        ["ObsPt1", "ObsPt2"],
        locationTypes=["1d", "1d"],
        snap_distance=10.0,
    )
    assert len(hydamo.observationpoints.observation_points) == 2


def test_storagenodes():
    hydamo, _ = _hydamo_object_from_gpkg()

    fm = FMModel()

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=None,
    )

    hydamo.storagenodes.add_storagenode(
        id='sto_test',
        xy=(141001, 395030),
        name='sto_test',
        usetable="true",
        levels=' '.join(np.arange(17.1, 19.6, 0.1).astype(str)),
        storagearea=' '.join(np.arange(100, 1000, 900/25.).astype(str)),
        interpolate="linear",
        network=fm.geometry.netfile.network
    )

    hydamo.storagenodes.add_storagenode(
        id="test",
        nodeid='199501.863000_395084.466000',
        usestreetstorage="true",
        nodetype="unspecified",
        name='sto_test',
        usetable="false",
        bedlevel=18.0,
        area=100,
        streetlevel=14.0,
        streetstoragearea=10.0,
        storagetype="reservoir",
        levels=np.nan,
        storagearea=np.nan,
        interpolate="linear",
    )

    assert len(hydamo.storagenodes.storagenodes.keys()) == 2


def test_convert_boundaries():
    hydamo, _ = _hydamo_object_from_gpkg()

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
    hydamo, _ = _hydamo_object_from_gpkg()

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
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.external_forcings.set_initial_waterdepth(1.5)

    assert (
        np.round(hydamo.external_forcings.initial_waterdepth_polygons.waterdepth.values[0]) == 2
    )


def test_write_laterals():
    # Get full hydamo object
    hydamo, _ = _hydamo_object_from_gpkg()

    series = pd.Series(np.sin(np.linspace(2, 8, 100) * -1) + 1.0)
    series.index = [
        pd.Timestamp("2016-01-01 00:00:00") + pd.Timedelta(hours=i) for i in range(100)
    ]
    series.plot()
    hydamo.external_forcings.add_lateral("LAT_01", "W_242209_0", "5.0", series)

    assert (
        np.round(np.mean(hydamo.external_forcings.lateral_nodes["LAT_01"]["value"]))
        == 1
    )
