from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
import pytest
from netCDF4 import Dataset
from hydrolib.dhydamo.core import hydamo
from hydrolib.dhydamo.core.hydamo import HyDAMO

hydamo_data_path = (
    Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "sample_data" / "data"
)


def _mixed_extended_geometries():
    gdf = gpd.GeoDataFrame(
        {
            "code": [
                "polygon_result",
                "multipolygon_result",
                "point_result",
                "geometrycollection_result",
            ],
            "geometry": [
                Polygon([(-1.0, 0.5), (1.0, 0.5), (1.0, 1.5), (-1.0, 1.5)]),
                MultiPolygon(
                    [
                        Polygon(
                            [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
                        ),
                        Polygon(
                            [(1.2, 1.2), (1.8, 1.2), (1.8, 1.8), (1.2, 1.8)]
                        ),
                    ]
                ),
                Polygon([(2.0, 2.0), (3.0, 2.0), (2.0, 3.0)]),
                MultiPolygon(
                    [
                        Polygon(
                            [(-1.0, 0.2), (0.5, 0.2), (0.5, 1.2), (-1.0, 1.2)]
                        ),
                        Polygon(
                            [(1.2, 0.5), (2.5, 0.5), (2.5, 1.5), (1.2, 1.5)]
                        ),
                        Polygon(
                            [(2.0, 2.0), (3.0, 2.0), (2.0, 3.0)]
                        ),
                    ]
                ),
            ],
        },
        crs="EPSG:28992",
    )
    geoms = ExtendedGeoDataFrame(geotype=BaseGeometry, required_columns=["code"])
    geoms.set_data(gdf, index_col="code")
    return geoms


def _check_related(hydamo_obj, hydamo_name):
    # assert type of object
    extendedgdf = getattr(hydamo_obj, hydamo_name)
    assert isinstance(extendedgdf, ExtendedGeoDataFrame)

    # Check relations recursively
    if extendedgdf.related is not None:
        for target_str, relation in extendedgdf.related.items():
            print(f"{target_str}:")
            _recursive_check_related(hydamo_obj, hydamo_name, target_str, **relation)

def _recursive_check_related(hydamo_obj, source_str, target_str, via, on, coupled_to):
    source = getattr(hydamo_obj, source_str)
    target = getattr(hydamo_obj, target_str)

    assert not source.empty
    assert not target.empty
    assert via in source.columns
    assert on in target.columns

    if coupled_to is not None:
        for next_target_str, next_relation in coupled_to.items():
            return _recursive_check_related(
                hydamo_obj, target_str, next_target_str, **next_relation
            )

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

def test_clip_hydamo_object():
    extent_file = hydamo_data_path / "test_deelmodel.shp"
    assert extent_file.exists()
    extent = gpd.read_file(extent_file).union_all()

    hydamo = HyDAMO(extent_file=extent_file)

    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    # Read branches
    hydamo.branches.read_gpkg_layer(str(gpkg_file), layer_name="HydroObject", index_col="code", clip=extent, cliptype="clip")    

    assert hydamo.branches.shape[0] ==  18

    assert np.isclose(np.round(hydamo.branches.loc['W_242208_0'].geometry.length, 2), 426.4)


def test_clip_extended_geodataframe():
    geoms = _mixed_extended_geometries()
    extent = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    assert set(geoms.geometry.geom_type.tolist()) == {"Polygon", "MultiPolygon"}

    geoms.clip(extent, cliptype="clip", clip_and_drop=False)

    assert geoms.shape[0] == 4
    assert geoms.loc["polygon_result"].geometry.geom_type == "Polygon"
    assert geoms.loc["multipolygon_result"].geometry.geom_type == "MultiPolygon"
    assert geoms.loc["point_result"].geometry.geom_type == "Point"
    assert geoms.loc["geometrycollection_result"].geometry.geom_type == "GeometryCollection"

def test_clip_and_drop_extended_geodataframe():
    geoms = _mixed_extended_geometries()
    extent = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    assert set(geoms.geometry.geom_type.tolist()) == {"Polygon", "MultiPolygon"}

    geoms.clip(extent, cliptype="clip", clip_and_drop=True)

    assert geoms.shape[0] == 3
    assert not geoms.index.has_duplicates
    assert geoms.index.tolist().count("geometrycollection_result") == 1
    assert geoms.loc["polygon_result"].geometry.geom_type == "Polygon"
    assert geoms.loc["multipolygon_result"].geometry.geom_type == "MultiPolygon"
    assert "point_result" not in geoms.index
    assert geoms.loc["geometrycollection_result"].geometry.geom_type == "MultiPolygon"


def test_clip_and_drop_linestring_uses_first_with_warning(caplog):
    gdf = gpd.GeoDataFrame(
        {
            "code": ["line_split"],
            "geometry": [LineString([(-1.0, 1.0), (5.0, 1.0)])],
        },
        crs="EPSG:28992",
    )
    lines = ExtendedGeoDataFrame(geotype=LineString, required_columns=["code"])
    lines.set_data(gdf, index_col="code")

    extent = Polygon(
        [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)],
        holes=[[(1.0, 0.5), (3.0, 0.5), (3.0, 1.5), (1.0, 1.5)]],
    )

    with caplog.at_level("WARNING"):
        lines.clip(extent, cliptype="clip", clip_and_drop=True)

    assert "Cannot deduplicate index" in caplog.text
    assert lines.shape[0] == 1
    assert lines.index.tolist() == ["line_split"]
    assert lines.geometry.iloc[0].geom_type == "LineString"

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
    hydamo.catchments['boundary_node'] = [hydamo.laterals[hydamo.laterals.globalid==c['lateraleknoopid']].code.to_numpy()[0] for _,c in hydamo.catchments.iterrows()]
        
    return hydamo, len_profile_before


def test_hydamo_object_from_gpkg():
    hydamo, len_profile_before = _hydamo_object_from_gpkg()

    assert np.isclose(np.round(hydamo.clipgeo.area), 139373665)
    assert len(hydamo.branches) == 61
    assert np.isclose(np.round(hydamo.branches.length.sum()), 28371)
    assert len_profile_before == 359
    # seven profiles are too far from a branch and are dropped
    assert len(hydamo.profile) == 352
    assert hydamo.profile.geom_type.to_numpy()[0] == "LineString"
    # the dataset contains two profile groups (a bridge and a uweir)
    assert hydamo.profile_group.shape == (2, 4)
    # a profile_lines' profielgroepid's should correspond to profile_group globalids
    assert hydamo.profile_line.profielgroepid.to_numpy()[0] in list(
        hydamo.profile_group.globalid
    )
    assert len(hydamo.management_device) == 32
    assert len(hydamo.weirs) == 25
    assert np.isclose(np.round(hydamo.weirs.doorstroombreedte.mean()), 2)
    assert len(hydamo.culverts) == 90
    assert np.isclose(np.round(hydamo.culverts.length.sum()), 2498)
    assert len(hydamo.pumps) == 1
    assert len(hydamo.bridges) == 1
    assert np.isclose(np.round(hydamo.bridges.branch_offset.to_numpy()[0]), 182)
    assert len(hydamo.boundary_conditions) == 1
    assert not hydamo.boundary_conditions.branch_offset.empty
    assert len(hydamo.catchments) == 121
    assert np.isclose(np.round(hydamo.catchments.oppervlakt.sum()), 2662)
    assert len(hydamo.laterals) == 121
    assert np.isclose(np.round(hydamo.laterals.afvoer.mean()), 0)

def _convert_structures(hydamo_obj=None):
    # iniate a hydamo object
    if hydamo_obj is None:
        hydamo_obj, _ = _hydamo_object_from_gpkg()

    # Convert
    hydamo_obj.structures.convert.weirs(
        weirs=hydamo_obj.weirs,
        profile_groups=hydamo_obj.profile_group,
        profile_lines=hydamo_obj.profile_line,
        profiles=hydamo_obj.profile,
        opening=hydamo_obj.opening,
        management_device=hydamo_obj.management_device,
    )

    hydamo_obj.structures.convert.culverts(
        hydamo_obj.culverts, management_device=hydamo_obj.management_device
    )

    hydamo_obj.structures.convert.bridges(
        hydamo_obj.bridges,
        profile_groups=hydamo_obj.profile_group,
        profile_lines=hydamo_obj.profile_line,
        profiles=hydamo_obj.profile,
    )

    hydamo_obj.structures.convert.pumps(
        hydamo_obj.pumpstations, pumps=hydamo_obj.pumps, management=hydamo_obj.management
    )

    return hydamo_obj

def test_convert_structures(hydamo_obj=None):
    hydamo = _convert_structures(hydamo_obj=hydamo_obj)
    # one weir is converted to an orifice, one tol a universal weir. w weirs have in total 4 extra openings, so fictional weirs are added.
    assert hydamo.structures.rweirs_df.shape[0] == hydamo.weirs.shape[0] - 2 + 4
    assert len(hydamo.structures.compounds_df)==2
    assert hydamo.structures.compounds_df[hydamo.structures.compounds_df.id=='cmp_S_98740'].structureids.squeeze() == 'S_98740_1;S_98740_2;S_98740_3;S_98740_4'
    assert hydamo.structures.culverts_df.shape[0] == hydamo.culverts.shape[0]
    
    # number culverts with flowdirection positive should be 1
    assert hydamo.structures.culverts_df[hydamo.structures.culverts_df.allowedflowdir=='positive'].shape[0] == 1
    # number of culverts with a gate shouuld be 1
    assert hydamo.structures.culverts_df[hydamo.structures.culverts_df.valveonoff==1].shape[0] == 1
    
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
        hydamo.structures.bridges_df.csdefid.to_numpy()[0]
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
        branchid="W_1486_0",
        chainage=12.0,
        crestlevel=18.0,
        crestwidth=3.0,
        corrcoeff=1.0,
    )

    hydamo.structures.add_orifice(
        id="otest",
        name="otest",
        branchid="W_1486_0",
        chainage=62.0,
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
        branchid="W_1486_0",
        chainage=27.0,
        dischargecoeff=1.0,
        crestlevel=18.0,
        numlevels=4,
        yvalues="0.0 1.0 2.0 3.0",
        zvalues="19.0 18.0 18.2 19.0",
    )

    hydamo.structures.add_bridge(
        id="btest",
        name="btest",
        branchid="W_1486_0",
        chainage=112.0,
        length=8.0,
        csdefid="B_11547",
        inletlosscoeff=0.9,
        outletlosscoeff=0.9,
        shift=0.0,
        frictiontype="Manning",
        friction=0.06,
    )

    hydamo.structures.add_culvert(
        id="ctest",
        name="ctest",
        branchid="W_1486_0",
        chainage=212.0,
        leftlevel=18.0,
        rightlevel=17.0,
        length=30.0,
        crosssection={"shape": "circle", "diameter": 0.40},
        inletlosscoeff=0.9,
        outletlosscoeff=0.9,
        valveonoff = 1,
        valveopeningheight=0.5,    
        numlosscoeff=2,
        relopening=[0.,1.],
        losscoeff=[0.7, 0.3],
        bedfrictiontype="Manning",
        bedfriction=0.06,
    )

    hydamo.structures.add_pump(
        id="ptest",
        name="ptest",
        branchid="W_1486_0",
        chainage=412.0,
        capacity=1.0,
        startlevelsuctionside=[17.5],
        stoplevelsuctionside=[17.3],
    )

    return hydamo

def test_add_structures_manually():
    hydamo = _add_structures_manually()
    # hydamo.structures.orifices_df.loc[hydamo.structures.orifices_df.id == 'orifice_test', 'gateloweredgelevel'] == 18.
    
    assert "rwtest" in hydamo.structures.rweirs_df.id.to_numpy()[0]
    assert "otest" in hydamo.structures.orifices_df.id.to_numpy()[0]
    assert "uwtest" in hydamo.structures.uweirs_df.id.to_numpy()[0]
    assert "btest" in hydamo.structures.bridges_df.id.to_numpy()[0]
    assert "ctest" in hydamo.structures.culverts_df.id.to_numpy()[0]
    assert "ptest" in hydamo.structures.pumps_df.id.to_numpy()[0]    

def test_observationpoints():
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.observationpoints.add_points(
        [Point((200200, 395600)), (200200, 396200)],
        ["ObsPt1", "ObsPt2"],
        locationTypes=["1d", "1d"],
        snap_distance=10.0,
    )
    assert len(hydamo.observationpoints.observation_points) == 2


@pytest.mark.parametrize(
    ("snapping_distance", "shape", "outcome"),
    [
        (10, 1, 'W_242224_0'),
        (0.1, 1, '')   ],
)
def test_observationpoint_snapping(snapping_distance, shape, outcome):
    hydamo, _ = _hydamo_object_from_gpkg()
    hydamo.observationpoints.add_points(
        [(200200, 396200)],
        ["TestObs"],
        locationTypes=["1d"],
        snap_distance=snapping_distance,
    )

    assert hydamo.observationpoints.observation_points.shape[0] == shape
    assert hydamo.observationpoints.observation_points.branchid.values[0] == outcome

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


def test_add_boundary_condition_accepts_integer_value():
    class _DummyMesh1DState:
        def __init__(self):
            self.mesh1d_node_id = np.array([1])
            self.mesh1d_node_x = np.array([197464.0])
            self.mesh1d_node_y = np.array([392130.0])

    class _DummyMesh1D:
        def __init__(self):
            self._mesh1d = _DummyMesh1DState()

    hydamo = HyDAMO()
    hydamo.external_forcings.add_boundary_condition(
        "RVW_INT",
        (197464.0, 392130.0),
        "dischargebnd",
        1,
        _DummyMesh1D(),
    )

    boundary = hydamo.external_forcings.boundary_nodes["RVW_INT"]
    assert boundary["vec1"] is None
    assert isinstance(boundary["vec2"], float)
    assert int(boundary["vec2"]) == 1
    assert boundary["unit2"] == "m3/s"


    # pseude-code:
    # invoeropties:
    # - linestrings with waterlevels
    # - linestrings with waterdepths
    # - polygons with streefpeilen. Of een combinatie van die twee?
    # - his-bestand met waterstanden op rekenpunten 

    # polygonen --> pol-bestanden met waterdiepte/waterstand
    # his-bestand --> linestrings --> xyz-file
    # linestrings --> xyz-file

    # end pseudocode

    # tijdelijke code voor maken test data
    # data_path = Path(r'hydrolib\tests\data')
    # branches = gpd.read_file(data_path / 'Example_model.gpkg', layer='hydroobject')
    # linestrings_wl = branches.copy(deep=True)
    # linestrings_wl['init_wlev'] = 17.5
    # linestrings_wl['init_wlev'].iloc[10] = 18.0
    # linestrings_wl['init_wlev'].iloc[20] = 18.0
    # linestrings_wl.to_file(data_path / 'Example_model.gpkg', layer='beginwaterstanden')
    # linestrings_wd = branches.copy(deep=True)
    # linestrings_wd['init_wdep'] = 1.5
    # linestrings_wd['init_wdep'].iloc[10] = 2.0
    # linestrings_wd['init_wdep'].iloc[20] = 2.0
    # linestrings_wd.to_file(data_path / 'Example_model.gpkg', layer='beginwaterdieptes')

    # catchments = gpd.read_file(data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied') 
    # polys_wl = branches.copy(deep=True)
    # polys_wl['streefpeil'] = 17.5
    # polys_wl['streefpeil'].iloc[10] = 16.5
    # polys_wl['streefpeil'].iloc[20] = 18.5
    # polys_wl.to_file(data_path / 'Example_model.gpkg', layer='streefpeilen')
    # # eind tijdelijke code       
    

def test_add_initialfields_uniform_depth():

    uniform_depth = 1.5
     
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.external_forcings.set_initial_waterdepth(uniform_depth)

    assert len(hydamo.external_forcings.initial_waterdepth_polygons) == 1
    assert hydamo.external_forcings.initial_waterdepth_polygons.waterdepth.to_numpy()[0] == uniform_depth
     
    
def test_add_initialfields_uniform_level():

    uniform_level = 17.5
     
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.external_forcings.set_initial_waterlevel(uniform_level)

    assert len(hydamo.external_forcings.initial_waterlevel_polygons) == 1
    assert hydamo.external_forcings.initial_waterlevel_polygons.waterlevel.to_numpy()[0] == uniform_level
    
def test_add_initialfields_level_from_polygon():

    poly_values = gpd.read_file(hydamo_data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied')
    poly_values['waterstand'] = 17.5
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.external_forcings.set_initial_waterlevel(poly_values, name_column='code', value_column='waterstand')

    assert len(hydamo.external_forcings.initial_waterlevel_polygons) == 121
    assert np.min(hydamo.external_forcings.initial_waterlevel_polygons.waterlevel.to_numpy()[0]) == 17.5
        

def test_add_initialfields_depth_from_polygon():

    poly_values = gpd.read_file(hydamo_data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied')
    poly_values['waterdiepte'] = 1.5

    hydamo, _ = _hydamo_object_from_gpkg()
    hydamo.external_forcings.set_initial_waterdepth(poly_values, name_column='code', value_column='waterdiepte')

    assert len(hydamo.external_forcings.initial_waterdepth_polygons) == 121
    assert np.min(hydamo.external_forcings.initial_waterdepth_polygons.waterdepth.to_numpy()[0]) == 1.5

def test_read_values_from_modelrun_fou():

    hydamo, _ = _hydamo_object_from_gpkg()

    nc_path = hydamo_data_path / 'model_result' / 'DFM_fou.nc'
    nc_field = 'mesh1d_fourier001_max' 
    netcdf_index = None
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, nc_field, netcdf_index=netcdf_index)
    assert not df.empty
 
def test_read_values_from_modelrun_map():

    hydamo, _ = _hydamo_object_from_gpkg()

    # reduced map file with only the last time step, see model_result/reduce_map.py
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    nc_field = 'mesh1d_s1' 
    netcdf_index = -1
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, nc_field, netcdf_index=netcdf_index)
    assert not df.empty

def test_correct_values_from_modelrun():

    hydamo, _ = _hydamo_object_from_gpkg()
    
    nc_path = hydamo_data_path / 'model_result' / 'DFM_fou.nc'
    nc_field = 'mesh1d_fourier001_max_depth' 

    poly_values = gpd.read_file(hydamo_data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied')
    selection = poly_values[poly_values['code'].isin([246.,285.,293.])]
    selection['waterdiepte'] = 0
                            
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, nc_field)
    df_cor = hydamo.external_forcings.correct_values_from_modelrun(df, selection, value_column='waterdiepte')
    
    assert not df_cor.empty
    assert len(df_cor) == len(df)
    assert len(df_cor[df_cor['value']==0]) == 69

def test_write_laterals():
    # Get full hydamo object
    hydamo, _ = _hydamo_object_from_gpkg()

    series = pd.Series(np.sin(np.linspace(2, 8, 100) * -1) + 1.0)
    series.index = [
        pd.Timestamp("2016-01-01 00:00:00") + pd.Timedelta(hours=i) for i in range(100)
    ]
    series.plot()
    hydamo.external_forcings.add_lateral("LAT_01", "W_242209_0", "5.0", series)

    assert np.isclose(
        np.round(np.mean(hydamo.external_forcings.lateral_nodes["LAT_01"]["value"])), 1
    )


def test_read_values_from_modelrun_edges():
    # velocities are on the 1D flow links (edges), not on the nodes
    hydamo, _ = _hydamo_object_from_gpkg()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_u1')

    assert len(df) == 1423
    assert {'branchid', 'chainage'} <= set(df.columns)
    assert df['value'].notna().all()


def test_read_values_from_modelrun_time():
    hydamo, _ = _hydamo_object_from_gpkg()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map.nc'
    by_time = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', time='2016-06-02 12:00')
    with Dataset(nc_path) as nc:
        index = int(np.argmin(np.abs(nc['time'][:] - 36 * 3600)))
    by_index = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', netcdf_index=index)
    assert np.allclose(by_time['value'], by_index['value'])

    with pytest.raises(ValueError):
        hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', time='2030-01-01')


def test_read_values_from_modelrun_level_from_depth(tmp_path):
    # a map-file without water level: the level is computed as bed level + water depth
    src = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    dst = tmp_path / 'map_without_s1.nc'
    with Dataset(src) as s, Dataset(dst, 'w') as d:
        for name, dim in s.dimensions.items():
            d.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, var in s.variables.items():
            if name == 'mesh1d_s1':
                continue
            fill = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
            v = d.createVariable(name, var.datatype, var.dimensions, fill_value=fill)
            v.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'})
            v[:] = var[:]

    hydamo, _ = _hydamo_object_from_gpkg()
    level = hydamo.external_forcings.read_values_from_modelrun(src, 'mesh1d_s1')
    derived = hydamo.external_forcings.read_values_from_modelrun(dst, 'mesh1d_s1')
    assert len(derived) == len(level)
    # compare wet nodes; at dry nodes the map-file can still contain a small water depth
    with Dataset(src) as nc:
        wet = ~np.ma.getmaskarray(nc['mesh1d_s1'][-1, :])
    assert np.allclose(derived['value'][wet], level['value'][wet], atol=1e-3)
    assert (derived['value'] >= level['value'] - 1e-3).all()


def test_read_values_from_modelrun_default_last_time_step():
    hydamo, _ = _hydamo_object_from_gpkg()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map.nc'
    default = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1')
    last = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', netcdf_index=-1)
    assert np.allclose(default['value'], last['value'])


def test_set_initial_velocity():
    hydamo, _ = _hydamo_object_from_gpkg()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    u = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_u1')
    assert 'direction' in u.columns

    hydamo.external_forcings.set_initial_velocity(u)
    fields = hydamo.external_forcings.spatial_fields
    assert [f['quantity'] for f in fields] == ['initialvelocityx', 'initialvelocityy']
    vx, vy = (f['gdf']['_component'].to_numpy() for f in fields)
    assert np.allclose(np.hypot(vx, vy), np.abs(u['value']))

    # D-Flow FM does not accept 'initialvelocity' itself
    with pytest.raises(ValueError):
        hydamo.external_forcings.set_spatial_field(u, 'value', quantity='initialvelocity')
