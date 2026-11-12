import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from netCDF4 import Dataset
from rasterio.transform import from_origin
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry

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


def test_extended_geodataframe_copy_preserves_metadata():
    geoms = _mixed_extended_geometries()

    copied = geoms.copy()

    assert copied.required_columns == geoms.required_columns
    assert copied.geotype is geoms.geotype


def test_clip_hydamo_object():
    extent_file = hydamo_data_path / "test_deelmodel.shp"
    assert extent_file.exists()
    extent = gpd.read_file(extent_file).union_all()

    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    hydamo = HyDAMO()
    hydamo.load_from_gpkg(
        gpkg_file,
        hydamo_version="2.2",
        clip_layers={"branches": (extent, "clip")},
    )

    assert hydamo.branches.shape[0] == 18
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

    hydamo.load_from_gpkg(gpkg_file, hydamo_version="2.2")

    len_profile_before = len(hydamo.profile)
    hydamo.snap_to_branch_and_drop(hydamo.profile, hydamo.branches, snap_method="intersecting", drop_related=True)
    hydamo.snap_to_branch_and_drop(hydamo.weirs, hydamo.branches, snap_method="overal", maxdist=10, drop_related=True)
    hydamo.snap_to_branch_and_drop(hydamo.culverts, hydamo.branches, snap_method="ends", maxdist=5, drop_related=True)
    hydamo.snap_to_branch_and_drop(hydamo.pumpstations, hydamo.branches, snap_method="overal", maxdist=10, drop_related=True)
    hydamo.snap_to_branch_and_drop(hydamo.bridges, hydamo.branches, snap_method="overal", maxdist=1100, drop_related=True)
    hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
    hydamo.laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5000)

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

def test_convert_structures():
    hydamo = _convert_structures()
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

def _pump_hydamo(management_counts):
    """Build one station; each tuple entry gives a pump's management row count."""
    hydamo = HyDAMO()
    hydamo.branches.set_data(
        gpd.GeoDataFrame({"code": ["B1"], "geometry": [LineString([(0, 0), (100, 0)])]}),
        index_col="code",
    )
    hydamo.pumpstations.set_data(
        gpd.GeoDataFrame(
            {
                "code": ["S1"],
                "globalid": ["station-1"],
                "branch_id": ["B1"],
                "branch_offset": [50.0],
                "geometry": [Point(50, 0)],
            }
        ),
        index_col="code",
    )
    # Distinct capacities (m³/min) and levels reveal incorrect pump/control links.
    pump_rows = []
    for pump_index in range(len(management_counts)):
        pump_rows.append(
            {
                "code": f"P{pump_index}",
                "globalid": f"pump-{pump_index}",
                "gemaalid": "station-1",
                "maximalecapaciteit": 60.0 * (pump_index + 1),
            }
        )
    pumps = pd.DataFrame(
        pump_rows,
        columns=["code", "globalid", "gemaalid", "maximalecapaciteit"],
    )
    hydamo.pumps.set_data(pumps, index_col="code")
    management_rows = []
    for pump_index, management_count in enumerate(management_counts):
        for management_index in range(management_count):
            management_rows.append(
                {
                    "code": f"M{pump_index}-{management_index}",
                    "globalid": f"management-{pump_index}-{management_index}",
                    "pompid": f"pump-{pump_index}",
                    "bovengrens": 17.5 + pump_index,
                    "ondergrens": 17.3 + pump_index,
                }
            )
    management = pd.DataFrame(
        management_rows,
        columns=["code", "globalid", "pompid", "bovengrens", "ondergrens"],
    )
    hydamo.management.set_data(management, index_col="code")
    return hydamo


@pytest.mark.parametrize(
    ("management_counts", "expected"),
    [
        pytest.param((1,), [("S1", 1.0, [17.5], [17.3])], id="single-pump"),
        pytest.param((0,), [], id="single-pump-without-management"),
        pytest.param(
            (1, 1),
            [("S1_1", 1.0, [17.5], [17.3]), ("S1_2", 2.0, [18.5], [18.3])],
            id="compound-station",
        ),
        pytest.param((0, 1), [("S1_2", 2.0, [18.5], [18.3])], id="skip-first-pump"),
        pytest.param((0, 0), [], id="all-pumps-without-management"),
    ],
)
def test_convert_pumps(management_counts, expected):
    """Check IDs, capacity conversion to m³/s, control levels, and compound members.

    Expected rows contain (id, capacity, start levels, stop levels).
    Missing management skips a pump without renumbering the remaining pumps.
    """
    hydamo = _pump_hydamo(management_counts)
    hydamo.structures.convert.pumps(
        hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management
    )
    pumps = list(hydamo.structures.pumps_df.itertuples())
    assert [
        (pump.id, pump.capacity, pump.startlevelsuctionside, pump.stoplevelsuctionside)
        for pump in pumps
    ] == expected
    for pump in pumps:
        assert pump.startleveldeliveryside == pump.startlevelsuctionside
        assert pump.stopleveldeliveryside == pump.stoplevelsuctionside
        # Catch the original list-of-Series regression, even for a single value.
        assert np.isscalar(pump.startlevelsuctionside[0])
        assert np.isscalar(pump.stoplevelsuctionside[0])

    compounds = hydamo.structures.compounds_df
    if len(management_counts) > 1:
        # Current behavior retains an empty compound when every pump is skipped.
        assert compounds.id.tolist() == ["cmp_S1"]
        assert compounds.iloc[0].numstructures == len(expected)
        assert compounds.iloc[0].structureids == ";".join(row[0] for row in expected)
    else:
        assert compounds.empty


@pytest.mark.parametrize(
    "management_counts",
    [
        pytest.param((2,), id="single-pump"),
        pytest.param((1, 2), id="second-pump-in-compound"),
    ],
)
def test_convert_pumps_rejects_duplicate_management(management_counts):
    """Reject multiple control rows per pump instead of choosing the first."""
    hydamo = _pump_hydamo(management_counts)
    with pytest.raises(
        ValueError, match="Pump station S1: expected one management row per pump, found 2"
    ):
        hydamo.structures.convert.pumps(
            hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management
        )


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
    bridge_profile = hydamo.structures.bridges_df.csdefid.to_numpy()[0]
    assert bridge_profile  in hydamo.crosssections.crosssection_def
    assert "default" in hydamo.crosssections.crosssection_def


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
        branchid="W_242224_0",
        chainage=300.0,
        dischargecoeff=0.9,
        # in the main watercourse (about 2 m3/s), crest below the water level (bed 13.7 m, water level 14.3-15.2 m),
        # so that the weir flows in test_run
        crestlevel=14.0,
        numlevels=3,
        yvalues="0 2.5 5.0",
        zvalues="15.0 14.0 15.0",
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
    ("snapping_distance", "outcome"),
    [
        (10, 'W_242224_0'),
        (0.1, ''),
    ],
)
def test_observationpoint_snapping(snapping_distance, outcome):
    hydamo, _ = _hydamo_object_from_gpkg()
    hydamo.observationpoints.add_points(
        [(200200, 396200)],
        ["TestObs"],
        locationTypes=["1d"],
        snap_distance=snapping_distance,
    )

    assert len(hydamo.observationpoints.observation_points) == 1
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
    assert next(i == "RVM_02" for i in hydamo.external_forcings.boundary_nodes)
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


def test_add_initialfields_uniform_depth():
    uniform_depth = 1.5
    hydamo = HyDAMO()

    hydamo.external_forcings.set_initial_waterdepth(uniform_depth)

    # a uniform 1D value is a field per branch over all branches
    fields = [f for f in hydamo.external_forcings.spatial_fields if f.get('uniform') is not None]
    assert [(f['quantity'], f['uniform']) for f in fields] == [('waterdepth', uniform_depth)]
    assert hydamo.external_forcings.initial_waterdepth_polygons.empty


def test_add_initialfields_uniform_level():
    uniform_level = 17.5
    hydamo = HyDAMO()

    hydamo.external_forcings.set_initial_waterlevel(uniform_level)

    # a uniform 1D value is a field per branch over all branches
    fields = [f for f in hydamo.external_forcings.spatial_fields if f.get('uniform') is not None]
    assert [(f['quantity'], f['uniform']) for f in fields] == [('waterlevel', uniform_level)]
    assert hydamo.external_forcings.initial_waterlevel_polygons.empty

def test_add_initialfields_level_from_polygon():
    # polygons are passed on to set_spatial_field, written per branch for 1D
    poly_values = gpd.read_file(hydamo_data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied')
    poly_values['waterstand'] = 17.0 + 0.01 * np.arange(len(poly_values))
    hydamo, _ = _hydamo_object_from_gpkg()

    hydamo.external_forcings.set_initial_waterlevel(poly_values, value_column='waterstand')

    field = hydamo.external_forcings.spatial_fields[-1]
    assert (field['quantity'], field['filetype'], field['locationtype']) == ('waterlevel', 'branch', '1d')
    np.testing.assert_allclose(field['gdf']['waterstand'].to_numpy(), poly_values['waterstand'].to_numpy())
    models = Df2HydrolibModel(hydamo)
    assert [(f.quantity, f.datafiletype) for f in models.inifields] == [('waterlevel', '1dField')]
    values = {v for _, branch_values in models.spatial_data['InitialWaterLevel.ini']['branches'].values() for v in branch_values}
    assert len(values) > 1 and values <= set(poly_values['waterstand'])


def test_add_initialfields_polygon_requires_value_column():
    poly_values = gpd.read_file(hydamo_data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied')
    hydamo = HyDAMO()
    with pytest.raises(ValueError, match="value_column"):
        hydamo.external_forcings.set_initial_waterlevel(poly_values)

@pytest.mark.slow
@pytest.mark.skipif(not (hydamo_data_path / 'model_result' / 'DFM_fou.nc').exists(), reason='DFM_fou.nc is not in the repository')
def test_read_values_from_modelrun_fou():
    hydamo = HyDAMO()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_fou.nc'
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_fourier001_max')
    assert len(df) == 1421
    assert np.isfinite(df['value']).all()


def test_read_values_from_modelrun_map():
    hydamo = HyDAMO()
    # reduced map file with only the last time step, see tests/tools/reduce_map.py
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', netcdf_index=-1)

    assert len(df) == 1421
    assert {'x', 'y', 'value', 'branchid', 'chainage'} <= set(df.columns)
    first = df.iloc[0]
    assert (first['branchid'], first['chainage']) == ('W_2645_0', 0.0)
    assert first['value'] == pytest.approx(22.246769, abs=1e-6)


def test_read_values_from_modelrun_fills_dry_nodes():
    # dry nodes have no water level in the map-file: they get the bed level (water level) or 0 (water depth)
    hydamo = HyDAMO()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    with Dataset(nc_path) as nc:
        dry = np.ma.getmaskarray(nc['mesh1d_s1'][-1, :])
        bed_level = np.asarray(nc['mesh1d_flowelem_bl'][:])
    assert dry.any()

    level = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1')
    np.testing.assert_allclose(level['value'].to_numpy()[dry], bed_level[dry])
    depth = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_waterdepth')
    assert (depth['value'].to_numpy()[dry] >= 0).all()
    assert len(level) == len(depth) == len(bed_level)


def test_read_values_from_modelrun_time_options():
    hydamo = HyDAMO()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    # the reduced file only contains 3 June 2016 00:00
    by_time = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', time='2016-06-03 00:00')
    default = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1')
    np.testing.assert_allclose(by_time['value'], default['value'])

    with pytest.raises(ValueError, match="outside the period"):
        hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', time='2030-01-01')
    with pytest.raises(ValueError, match="not both"):
        hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', time='2016-06-03', netcdf_index=-1)
    # a variable without time dimension
    with pytest.raises(ValueError, match="no time dimension"):
        hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_flowelem_bl', netcdf_index=0)


def test_correct_values_from_modelrun():
    hydamo, _ = _hydamo_object_from_gpkg()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    poly_values = gpd.read_file(hydamo_data_path / 'Example_model.gpkg', layer='afvoergebiedaanvoergebied')
    selection = poly_values[poly_values['code'].isin([246.0, 285.0, 293.0])].copy()
    selection['streefpeil'] = 15.0

    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', crs=poly_values.crs)
    df_cor = hydamo.external_forcings.correct_values_from_modelrun(df, selection, value_column='streefpeil')

    # the model points come first: those inside the polygons get the target level, the others keep their value
    inside = df.intersects(selection.union_all()).to_numpy()
    original = df_cor.iloc[:len(df)]
    assert inside.sum() > 0
    np.testing.assert_allclose(original['value'].to_numpy()[inside], 15.0)
    np.testing.assert_allclose(original['value'].to_numpy()[~inside], df['value'].to_numpy()[~inside])
    # then two points per crossing of a polygon boundary with a branch, at least one of them inside a polygon
    added = df_cor.iloc[len(df):]
    assert len(added) > 0
    assert len(added) % 2 == 0
    assert (added['value'].to_numpy().reshape(-1, 2) == 15.0).any(axis=1).all()


def test_correct_values_from_modelrun_without_branches():
    # without branches no points are added at the polygon boundaries
    hydamo = HyDAMO()
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
    df = hydamo.external_forcings.read_values_from_modelrun(nc_path, 'mesh1d_s1', crs=28992)
    polygon = gpd.GeoDataFrame(
        {'streefpeil': [15.0]}, geometry=[df.geometry.iloc[:5].union_all().buffer(1.0)], crs=28992
    )

    df_cor = hydamo.external_forcings.correct_values_from_modelrun(df, polygon, value_column='streefpeil')
    assert len(df_cor) == len(df)
    assert (df_cor['value'].iloc[:5] == 15.0).all()


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

@pytest.mark.slow
@pytest.mark.skipif(not (hydamo_data_path / 'model_result' / 'DFM_map.nc').exists(), reason='DFM_map.nc is not in the repository')
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
    nc_path = hydamo_data_path / 'model_result' / 'DFM_map_last.nc'
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


@pytest.mark.parametrize("qspec", [2.0, -1.0])
def test_create_laterals_excludes_nodata_and_keeps_negative_discharge(tmp_path, qspec):
    hydamo, _ = _hydamo_object_from_gpkg()
    xmin, ymin, xmax, ymax = hydamo.catchments.total_bounds
    cell = 25.0
    left, top = xmin - cell, ymax + cell
    columns = int((xmax - xmin) // cell) + 2
    rows = int((ymax - ymin) // cell) + 2
    values = np.full((rows, columns), qspec, dtype=np.float32)
    # A nodata strip through the catchments.
    strip_column = columns // 2
    values[:, strip_column] = -9999.0
    path = tmp_path / "qspec.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=columns,
        height=rows,
        count=1,
        dtype=values.dtype,
        crs="EPSG:28992",
        transform=from_origin(left, top, cell, cell),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)
    strip_x = left + strip_column * cell

    hydamo.create_laterals(qspec_file=path)

    # Uniform discharge, so every catchment gets it, whether it overlaps the
    # nodata strip or not; negative discharge stays negative.
    assert hydamo.catchments.intersects(box(strip_x, ymin, strip_x + cell, ymax)).any()
    np.testing.assert_allclose(
        hydamo.laterals["afvoer"].to_numpy(dtype=float),
        qspec * hydamo.catchments.area.to_numpy() / (1000 * 86400),
        rtol=1e-6,
    )


def test_make_index_unique_suffixes_duplicates_without_code_column():
    from hydrolib.dhydamo.io.common import _make_index_unique

    gdf = gpd.GeoDataFrame({"globalid": ["a", "a", "b", "a"]}, geometry=[Point(0, 0)] * 4)
    _make_index_unique(gdf, "globalid")

    assert gdf["globalid"].tolist() == ["a", "a_1", "b", "a_2"]


@pytest.mark.parametrize(
    ("maximumflowwidth", "levels", "flowwidths"),
    [
        (8.0, "10.0 11.000", "4.00 8.00"),  # up to the bank top, 1 m above the bed
        (float("nan"), "10.0 100", "4.00 364.00"),  # bank height missing: up to a level of 100 m
        (2.0, "10.0 100", "4.00 364.00"),  # bank below the bed: up to a level of 100 m
    ],
)
def test_open_trapezium_profile(maximumflowwidth, levels, flowwidths, caplog):
    crosssections = HyDAMO().crosssections
    with caplog.at_level(logging.WARNING):
        name = crosssections.add_trapezium_definition(
            slope=2.0,
            maximumflowwidth=maximumflowwidth,
            bottomwidth=4.0,
            closed=0,
            roughnesstype="StricklerKs",
            roughnessvalue=30.0,
            bottomlevel=10.0,
        )
    definition = crosssections.crosssection_def[name]
    assert (definition["levels"], definition["flowwidths"]) == (levels, flowwidths)
    valid = maximumflowwidth > 4.0
    assert any("no valid height up to the bank top" in message for message in caplog.messages) != valid
