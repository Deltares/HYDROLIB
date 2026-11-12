import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from hydrolib.core.dflowfm.inifield.models import IniFieldModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel
from hydrolib.core.dflowfm.polyfile.models import PolyFile
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.geometry.spatial import (
    points_to_branch_values,
    polygons_to_branch_values,
)
from shapely.geometry import LineString, Point, Polygon, box


def _polygons():
    return gpd.GeoDataFrame(
        {"peil": [1.0, 2.5, 1.0]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)],
        crs=28992,
    )


def _points():
    return gpd.GeoDataFrame(
        geometry=[Point(1, 1, 0.3), Point(5, 5, 0.4), Point(9, 2, 0.5)], crs=28992
    )


def _write(hydamo, path):
    models = Df2HydrolibModel(hydamo)
    ini = IniFieldModel(initial=models.inifields, parameter=models.parameterfields)
    ini.filepath = path / "initialFields.ini"
    ini.save()
    models.write_spatial_fields(path)
    return models


def test_polygon_field_written(tmp_path):
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", filetype="polygon")
    models = _write(hydamo, tmp_path)

    # one block and one polygon file per distinct value
    assert [f.datafiletype for f in models.inifields] == ["polygon", "polygon"]
    assert [f.interpolationmethod for f in models.inifields] == ["constant", "constant"]
    assert [f.value for f in models.inifields] == [1.0, 2.5]
    assert [f.datafile.filepath.name for f in models.inifields] == ["initial_waterlevel_00_00.pol", "initial_waterlevel_00_01.pol"]

    assert len(PolyFile(filepath=tmp_path / "initial_waterlevel_00_00.pol").objects) == 2
    assert len(PolyFile(filepath=tmp_path / "initial_waterlevel_00_01.pol").objects) == 1


def test_raster_field_written(tmp_path):
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(
        _polygons(),
        "peil",
        quantity="frictioncoefficient",
        kind="parameter",
        filetype="raster",
        cellsize=5.0,
    )
    models = _write(hydamo, tmp_path)

    field = models.parameterfields[0]
    assert (field.datafiletype, field.datafile.filepath.name) == ("GeoTIFF", "parameter_frictioncoefficient_00.tif")
    with rasterio.open(tmp_path / "parameter_frictioncoefficient_00.tif") as src:
        assert src.shape == (2, 6)
        np.testing.assert_array_equal(np.unique(src.read(1)), [1.0, 2.5])


def test_sample_field_written(tmp_path):
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(_points(), filetype="sample")
    models = _write(hydamo, tmp_path)

    field = models.inifields[0]
    assert (field.datafiletype, field.interpolationmethod) == ("sample", "triangulation")
    xyz = np.loadtxt(tmp_path / "initial_waterlevel_00.xyz")
    np.testing.assert_allclose(xyz[:, 2], [0.3, 0.4, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"column": "peil", "filetype": "raster"},
            "cellsize is required",
            id="raster_without_cellsize",
        ),
        pytest.param(
            {"column": "peil", "kind": "boundary"},
            "kind should be 'initial' or 'parameter'",
            id="invalid_kind",
        ),
        pytest.param(
            {"column": "peil", "filetype": "sample"},
            "For polygons, filetype should be",
            id="polygons_as_samples",
        ),
        pytest.param(
            {"column": "peil", "filetype": "polygon", "interpolationmethod": "triangulation"},
            "only support interpolationmethod 'constant'",
            id="polygon_file_not_constant",
        ),
        pytest.param(
            {"column": "peil", "locationtype": "2d"},
            "only available for locationtype '1d'",
            id="branch_values_for_2d",
        ),
        pytest.param(
            {"column": "peil", "filetype": "raster", "cellsize": 1.0, "interpolationmethod": "averaging"},
            "assigns no values to 1D nodes",
            id="averaging_on_1d",
        ),
        pytest.param({"column": None}, "Provide a value column", id="no_value_column"),
    ],
)
def test_spatial_field_invalid_polygon_input(kwargs, message):
    hydamo = HyDAMO()
    polygons = _polygons()
    with pytest.raises(ValueError, match=message):
        hydamo.external_forcings.set_spatial_field(polygons, **kwargs)


def test_spatial_field_points_without_z():
    hydamo = HyDAMO()
    points = gpd.GeoDataFrame(geometry=[Point(1, 1), Point(2, 2)], crs=28992)
    with pytest.raises(ValueError, match="Provide a value column"):
        hydamo.external_forcings.set_spatial_field(points)


def _branches():
    # two branches meeting at (30, 5)
    # like hydamo.branches: the code both as index and as column
    return gpd.GeoDataFrame(
        {"code": ["br1", "br2"]},
        geometry=[LineString([(0, 5), (30, 5)]), LineString([(30, 5), (30, 25)])],
        index=pd.Index(["br1", "br2"], name="code"),
        crs=28992,
    )


def test_polygons_to_branch_values():
    values = polygons_to_branch_values(_branches(), _polygons(), "peil", step=0.01)

    # br1 crosses the three polygons (1.0, 2.5, 1.0) with steps at chainage 10 and 20
    chainage, branch_values = values["br1"]
    np.testing.assert_allclose(chainage, [0, 9.995, 10.005, 19.995, 20.005, 30])
    np.testing.assert_allclose(branch_values, [1.0, 1.0, 2.5, 2.5, 1.0, 1.0])
    # br2 is inside the third polygon up to y = 10
    chainage, branch_values = values["br2"]
    np.testing.assert_allclose(chainage, [0, 4.995])
    np.testing.assert_allclose(branch_values, [1.0, 1.0])


def test_points_to_branch_values():
    points = gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        geometry=[Point(5, 6), Point(30, 5), Point(31, 15), Point(100, 100)],
        crs=28992,
    )
    values = points_to_branch_values(_branches(), points, "value", max_distance=5.0)

    # the point at the junction (30, 5) is snapped to one branch, but used at the end of both
    np.testing.assert_allclose(values["br1"][0], [5, 30])
    np.testing.assert_allclose(values["br1"][1], [1.0, 2.0])
    np.testing.assert_allclose(values["br2"][0], [0, 10])
    np.testing.assert_allclose(values["br2"][1], [2.0, 3.0])


def test_branch_field_written(tmp_path):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil")
    models = _write(hydamo, tmp_path)

    field = models.inifields[0]
    assert (field.datafiletype, field.datafile.filepath.name) == ("1dField", "InitialWaterLevel.ini")
    onedfield = OneDFieldModel(filepath=tmp_path / "InitialWaterLevel.ini")
    assert [b.branchid for b in onedfield.branch] == ["br1", "br2"]
    # a [Global] block with the nodata value, required by the D-HYDRO GUI
    assert (onedfield.global_.quantity, onedfield.global_.value) == ("waterlevel", -999.0)
    np.testing.assert_allclose(onedfield.branch[0].values, [1.0, 1.0, 2.5, 2.5, 1.0, 1.0])


def test_level_and_depth_on_same_branch_warns(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 15, 10)], crs=28992)
    depth = gpd.GeoDataFrame({"diepte": [0.5]}, geometry=[box(15, 0, 35, 30)], crs=28992)
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth")
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel")

    with caplog.at_level(logging.WARNING):
        Df2HydrolibModel(hydamo)

    # br1 crosses both areas, br2 only the depth area
    messages = [msg for msg in caplog.messages if "get both a waterdepth" in msg]
    assert len(messages) == 1
    assert "['br1']" in messages[0]


def _level_depth_warnings(hydamo, caplog):
    """Warnings about water level and water depth fields: (overlap messages, order messages, other messages)."""
    with caplog.at_level(logging.WARNING):
        Df2HydrolibModel(hydamo)
    overlap = [m for m in caplog.messages if "get both a" in m]
    order = [m for m in caplog.messages if "is added after water level" in m]
    uniform = [m for m in caplog.messages if "uniform initial water depth" in m]
    return overlap, order, uniform


def _depth_points(*xy):
    return gpd.GeoDataFrame({"diepte": [0.5] * len(xy)}, geometry=[Point(x, y) for x, y in xy], crs=28992)


def test_level_polygons_and_depth_samples_overlap_through_convex_hull(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    # none of the points is inside the level polygon, but their convex hull covers br1 from x=0 to 11
    depth = _depth_points((-5, -5), (15, -5), (5, 20))
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel", filetype="polygon")
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth", filetype="sample")

    overlap, order, _ = _level_depth_warnings(hydamo, caplog)
    assert len(overlap) == 1
    assert "['br1']" in overlap[0]
    assert "convex hull" in overlap[0]
    # depth samples after a water level field
    assert len(order) == 1
    assert "initial_waterdepth_01.xyz" in order[0]


def test_disjoint_level_and_depth_polygons_depth_first_no_warning(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    depth = gpd.GeoDataFrame({"diepte": [0.5]}, geometry=[box(15, 0, 25, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth", filetype="polygon")
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel", filetype="polygon")

    assert _level_depth_warnings(hydamo, caplog) == ([], [], [])


def test_level_branches_and_depth_samples(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    # the level is written per branch, so it fills the whole of br1; the samples cover br2 from y=15 to 24
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel")
    hydamo.external_forcings.set_spatial_field(
        _depth_points((29, 15), (31, 15), (30, 24)), "diepte", quantity="waterdepth", filetype="sample"
    )

    overlap, order, _ = _level_depth_warnings(hydamo, caplog)
    assert overlap == []
    # water levels per branch are not removed by a later water depth (DIMR test m6)
    assert order == []


def test_depth_samples_and_level_branches_overlap_on_whole_branch(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    # the samples are beyond the level polygon, but the 1dField file gives br1 a water level over its whole length
    depth = _depth_points((20, 0), (25, 0), (22, 9))
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth", filetype="sample")
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel")

    overlap, order, _ = _level_depth_warnings(hydamo, caplog)
    assert len(overlap) == 1
    assert "['br1']" in overlap[0]
    assert "whole branch" in overlap[0]
    assert order == []


def test_overlap_only_at_junction_is_ignored(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    # br1 (y=5, up to x=30) gets the level, br2 (x=30, from y=5) the depth; both polygons have a corner at the
    # junction (30, 5), so the covered parts only share that point
    level = gpd.GeoDataFrame(
        {"peil": [1.0]}, geometry=[Polygon([(-1, 0), (31, 0), (30, 5), (-1, 10)])], crs=28992
    )
    depth = gpd.GeoDataFrame(
        {"diepte": [0.5]}, geometry=[Polygon([(30, 5), (35, 0), (35, 30), (25, 30)])], crs=28992
    )
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth", filetype="polygon")
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel", filetype="polygon")

    overlap, _, _ = _level_depth_warnings(hydamo, caplog)
    assert overlap == []


def test_level_and_depth_areas_overlap_without_network(caplog):
    hydamo = HyDAMO()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    depth = gpd.GeoDataFrame({"diepte": [0.5]}, geometry=[box(5, 0, 15, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth", filetype="polygon")
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel", filetype="polygon")

    overlap, _, _ = _level_depth_warnings(hydamo, caplog)
    assert len(overlap) == 1
    assert overlap[0].startswith("The areas of the 1D fields")


def test_uniform_depth_for_all_with_spatial_level_warns(caplog):
    # a uniform depth for 1D and 2D is a [Global] value, which overrides spatial water levels
    hydamo = HyDAMO()
    hydamo.external_forcings.set_initial_waterdepth(1.5, locationtype="all")
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel", filetype="polygon")

    _, _, uniform = _level_depth_warnings(hydamo, caplog)
    assert len(uniform) == 1
    assert "initial_waterlevel_00.pol" in uniform[0]


def test_uniform_1d_value_is_a_background_per_branch(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    levels = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(levels, "peil", quantity="waterlevel")
    hydamo.external_forcings.set_initial_waterdepth(1.0)
    hydamo.external_forcings.set_initial_waterdepth(1.5)  # replaces the earlier uniform depth

    # the uniform depth is written per branch for all branches, after the level field (the order of the calls)
    overlap, order, uniform = _level_depth_warnings(hydamo, caplog)
    assert (overlap, order, uniform) == ([], [], [])
    models = Df2HydrolibModel(hydamo)
    assert [(f.quantity, f.datafile.filepath.name) for f in models.inifields] == [
        ("waterdepth", "InitialWaterDepth.ini"),
    ]
    # the depth is added last and covers all branches, so no branch keeps the water level
    assert models.spatial_data["InitialWaterDepth.ini"]["branches"] == {
        "br1": ([0.0, 30.0], [1.5, 1.5]),
        "br2": ([0.0, 20.0], [1.5, 1.5]),
    }


def test_generated_file_names():
    # file names are generated: unique per call, with the extension of the file type
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", quantity="waterdepth")
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", filetype="polygon")
    hydamo.external_forcings.set_spatial_field(_points(), filetype="sample")
    names = [f["filename"] for f in hydamo.external_forcings.spatial_fields]
    assert names == ["initial_waterdepth_00.ini", "initial_waterlevel_01.pol", "initial_waterlevel_02.xyz"]


def test_correction_adds_hard_boundary():
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    # model values every 5 m along br1 (y=5, x from 0 to 30), rising from 1.0 to 4.0
    chainage = np.arange(0.0, 31.0, 5.0)
    model_values = gpd.GeoDataFrame(
        {"value": 1.0 + chainage / 10.0, "branchid": "br1", "chainage": chainage},
        geometry=[Point(c, 5) for c in chainage],
        crs=28992,
    )
    correction = gpd.GeoDataFrame({"peil": [9.0]}, geometry=[box(12, 0, 18, 10)], crs=28992)
    corrected = hydamo.external_forcings.correct_values_from_modelrun(model_values, correction, "peil", boundary_step=0.01)

    values = points_to_branch_values(hydamo.branches, corrected, "value")
    chainages, branch_values = values["br1"]
    lookup = dict(zip(np.round(chainages, 3), branch_values))
    # the point at 15 m is inside the polygon; two points at each boundary (12 and 18 m)
    assert lookup[15.0] == 9.0
    assert lookup[11.995] == pytest.approx(2.1995)
    assert lookup[12.005] == 9.0
    assert lookup[17.995] == 9.0
    assert lookup[18.005] == pytest.approx(2.8005)

    # without boundary points
    plain = hydamo.external_forcings.correct_values_from_modelrun(model_values, correction, "peil", boundary_step=None)
    assert len(plain) == len(model_values)


def test_initial_conditions_at_nodes(tmp_path):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    fm = FMModel()
    mesh.mesh1d_add_branches_from_gdf(fm.geometry.netfile.network, branches=hydamo.branches, branch_name_col="code", node_distance=5)

    # water depth on br2 first, then the target levels per polygon on br1 only (a polygon touching br2 would
    # give the whole of br2 a water level, as the last field)
    depth = gpd.GeoDataFrame({"diepte": [0.5]}, geometry=[box(29, 6, 31, 30)], crs=28992)
    levels = gpd.GeoDataFrame({"peil": [1.0, 2.5]}, geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth")
    hydamo.external_forcings.set_spatial_field(levels, "peil", quantity="waterlevel")
    models = Df2HydrolibModel(hydamo)

    nodes = models.initial_conditions_at_nodes(fm.geometry.netfile.network, output_file=tmp_path / "nodes.shp")
    on_br1 = nodes[nodes["branchid"] == "br1"].set_index("chainage")
    assert (on_br1["quantity"] == "waterlevel").all()
    assert on_br1.loc[5.0, "value"] == 1.0
    assert on_br1.loc[15.0, "value"] == 2.5
    on_br2 = nodes[(nodes["branchid"] == "br2") & (nodes["chainage"] > 0)]
    assert (on_br2["quantity"] == "waterdepth").all()
    assert (on_br2["value"] == 0.5).all()
    assert len(gpd.read_file(tmp_path / "nodes.shp")) == len(nodes)


@pytest.mark.parametrize("two_d_first", [False, True])
def test_uniform_1d_and_2d_values_use_different_files(tmp_path, two_d_first):
    # a uniform 1D value (field per branch) must not affect a uniform 2D value ([Global] file), in either order
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    if two_d_first:
        hydamo.external_forcings.set_initial_waterlevel(2.0, locationtype="2d")
        hydamo.external_forcings.set_initial_waterlevel(1.0)
    else:
        hydamo.external_forcings.set_initial_waterlevel(1.0)
        hydamo.external_forcings.set_initial_waterlevel(2.0, locationtype="2d")
    models = Df2HydrolibModel(hydamo)
    models.write_initial_fields(FMModel(), tmp_path)

    files = sorted(f.datafile.filepath.name for f in models.inifields)
    assert files == ["InitialWaterLevel.ini", "initialwaterlevel_2d.ini"]
    assert OneDFieldModel(filepath=tmp_path / "initialwaterlevel_2d.ini").global_.value == 2.0
    assert OneDFieldModel(filepath=tmp_path / "InitialWaterLevel.ini").global_.value == -999.0


@pytest.mark.parametrize("quantity", ["waterlevel", "waterdepth"])
def test_uniform_2d_value_unaffected_by_1d_fields(tmp_path, quantity):
    # existing 2D behaviour: a uniform 2D value is a [Global] block with locationType 2d in its own file,
    # also when 1D fields (uniform and per branch) are added before and after it
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    set_initial = getattr(hydamo.external_forcings, f"set_initial_{quantity}")
    set_initial(1.0)
    set_initial(2.0, locationtype="2d")
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", quantity=quantity)
    models = Df2HydrolibModel(hydamo)
    models.write_initial_fields(FMModel(), tmp_path)

    two_d = [f for f in models.inifields if str(f.locationtype).lower() == "2d"]
    assert len(two_d) == 1
    assert (two_d[0].quantity, two_d[0].datafiletype, two_d[0].datafile.filepath.name) == (
        quantity,
        "1dField",
        f"initial{quantity}_2d.ini",
    )
    onedfield = OneDFieldModel(filepath=tmp_path / f"initial{quantity}_2d.ini")
    assert (onedfield.global_.quantity, onedfield.global_.value, onedfield.branch) == (quantity, 2.0, [])

    # the 2D value is not applied to the 1D nodes
    fm = FMModel()
    mesh.mesh1d_add_branches_from_gdf(fm.geometry.netfile.network, branches=hydamo.branches, branch_name_col="code", node_distance=5)
    nodes = models.initial_conditions_at_nodes(fm.geometry.netfile.network)
    assert f"initial{quantity}_2d.ini" not in set(nodes["source"].dropna())


def test_initial_polygons_for_2d_raise():
    # spatially varying 2D values are not added via set_initial_*; this keeps 2D behaviour unchanged
    hydamo = HyDAMO()
    polygons = _polygons()
    with pytest.raises(ValueError, match="only supported for 1D"):
        hydamo.external_forcings.set_initial_waterlevel(polygons, value_column="peil", locationtype="2d")


def test_1d_fields_are_combined_per_quantity():
    # all 1D fields per branch are combined into one file per quantity, with the names the D-HYDRO GUI uses;
    # per branch the last field applies, as in D-Flow FM
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    first = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 10, 10)], crs=28992)  # br1
    second = gpd.GeoDataFrame({"peil": [3.0]}, geometry=[box(20, 0, 25, 10)], crs=28992)  # br1, other part
    depth = gpd.GeoDataFrame({"diepte": [0.5]}, geometry=[box(29, 6, 31, 30)], crs=28992)  # br2
    hydamo.external_forcings.set_initial_waterlevel(2.0)  # background on all branches
    hydamo.external_forcings.set_spatial_field(first, "peil", quantity="waterlevel")
    hydamo.external_forcings.set_spatial_field(second, "peil", quantity="waterlevel")
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth")
    models = Df2HydrolibModel(hydamo)

    assert [(f.quantity, f.datafile.filepath.name) for f in models.inifields] == [
        ("waterlevel", "InitialWaterLevel.ini"),
        ("waterdepth", "InitialWaterDepth.ini"),
    ]
    levels = models.spatial_data["InitialWaterLevel.ini"]["branches"]
    depths = models.spatial_data["InitialWaterDepth.ini"]["branches"]
    # br1: the last level field (3.0) fills the whole branch; br2: the depth replaces the background level
    assert set(levels) == {"br1"}
    assert set(levels["br1"][1]) == {3.0}
    assert set(depths) == {"br2"}
    assert set(depths["br2"][1]) == {0.5}


def test_uniform_1d_value_requires_branches():
    hydamo = HyDAMO()
    hydamo.external_forcings.set_initial_waterlevel(1.0)
    with pytest.raises(ValueError, match="requires the branches"):
        Df2HydrolibModel(hydamo)


def test_field_without_values_on_branches_warns(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    far_away = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(1000, 1000, 1010, 1010)], crs=28992)
    hydamo.external_forcings.set_spatial_field(far_away, "peil")
    with caplog.at_level(logging.WARNING):
        models = Df2HydrolibModel(hydamo)
    assert any("No values on any branch" in message for message in caplog.messages)
    assert models.inifields == []


def test_branch_field_of_other_quantity_is_written_separately():
    # only initial water levels and depths are combined; other quantities per branch get their own file
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", quantity="frictioncoefficient", kind="parameter")
    models = Df2HydrolibModel(hydamo)

    field = models.parameterfields[0]
    assert (field.quantity, field.datafiletype) == ("frictioncoefficient", "1dField")
    assert field.datafile.filepath.name == "parameter_frictioncoefficient_00.ini"


def test_initial_conditions_at_nodes_from_samples_and_polygons():
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    fm = FMModel()
    mesh.mesh1d_add_branches_from_gdf(fm.geometry.netfile.network, branches=hydamo.branches, branch_name_col="code", node_distance=5)
    # samples with a constant water level around br1, then a polygon file with a water level for part of br1
    points = gpd.GeoDataFrame({"value": [2.0] * 4}, geometry=[Point(-1, 0), Point(31, 0), Point(31, 10), Point(-1, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(points, "value", filetype="sample")
    polygon = gpd.GeoDataFrame({"peil": [3.0]}, geometry=[box(12, 0, 18, 10)], crs=28992)
    hydamo.external_forcings.set_spatial_field(polygon, "peil", filetype="polygon")
    nodes = Df2HydrolibModel(hydamo).initial_conditions_at_nodes(fm.geometry.netfile.network)

    on_br1 = nodes[nodes["branchid"] == "br1"].set_index("chainage")
    assert on_br1.loc[5.0, "value"] == pytest.approx(2.0)  # samples (inside their convex hull)
    assert on_br1.loc[15.0, "value"] == 3.0  # polygon file, added last
    assert set(on_br1["quantity"]) == {"waterlevel"}


def test_points_on_unknown_branches_are_left_out(caplog):
    points = gpd.GeoDataFrame(
        {"value": [1.0, 2.0], "branchid": ["br1", "unknown"], "chainage": [5.0, 5.0]},
        geometry=[Point(5, 5), Point(50, 50)],
        crs=28992,
    )
    with caplog.at_level(logging.WARNING):
        values = points_to_branch_values(_branches(), points, "value")
    assert set(values) == {"br1"}
    assert any("not in the model" in message for message in caplog.messages)


def test_write_raster_unsupported_extension(tmp_path):
    from affine import Affine
    from hydrolib.dhydamo.io.rasterwriter import write_raster

    with pytest.raises(ValueError, match="Unsupported raster extension"):
        write_raster(tmp_path / "raster.png", np.zeros((2, 2)), Affine.identity())
