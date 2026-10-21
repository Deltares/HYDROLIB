import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from hydrolib.core.dflowfm.inifield.models import IniFieldModel
from hydrolib.core.dflowfm.polyfile.models import PolyFile
from shapely.geometry import LineString, Point, box

from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel

from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry.spatial import points_to_branch_values, polygons_to_branch_values


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
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", filetype="polygon", filename="streefpeil.pol")
    models = _write(hydamo, tmp_path)

    # one block and one polygon file per distinct value
    assert [f.datafiletype for f in models.inifields] == ["polygon", "polygon"]
    assert [f.interpolationmethod for f in models.inifields] == ["constant", "constant"]
    assert [f.value for f in models.inifields] == [1.0, 2.5]
    assert [f.datafile.filepath.name for f in models.inifields] == ["streefpeil_00.pol", "streefpeil_01.pol"]

    assert len(PolyFile(filepath=tmp_path / "streefpeil_00.pol").objects) == 2
    assert len(PolyFile(filepath=tmp_path / "streefpeil_01.pol").objects) == 1


def test_raster_field_written(tmp_path):
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(
        _polygons(),
        "peil",
        quantity="frictioncoefficient",
        kind="parameter",
        filetype="raster",
        cellsize=5.0,
        filename="roughness.asc",
        locationtype="2d",
        averagingtype="mean",
    )
    models = _write(hydamo, tmp_path)

    field = models.parameterfields[0]
    assert (field.datafiletype, field.interpolationmethod, field.averagingtype) == ("arcInfo", "averaging", "mean")
    with rasterio.open(tmp_path / "roughness.asc") as src:
        assert src.shape == (2, 6)
        np.testing.assert_array_equal(np.unique(src.read(1)), [1.0, 2.5])


def test_sample_field_written(tmp_path):
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(_points(), filename="initial_waterlevel.xyz")
    models = _write(hydamo, tmp_path)

    field = models.inifields[0]
    assert (field.datafiletype, field.interpolationmethod) == ("sample", "triangulation")
    xyz = np.loadtxt(tmp_path / "initial_waterlevel.xyz")
    np.testing.assert_allclose(xyz[:, 2], [0.3, 0.4, 0.5])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(column="peil", filetype="raster"),  # raster without cellsize
        dict(column="peil", kind="boundary"),  # invalid kind
        dict(column="peil", filetype="sample"),  # polygons as samples
        dict(column="peil", filetype="polygon", interpolationmethod="triangulation"),  # polygon files only support constant
        dict(column="peil", locationtype="2d"),  # branch values only for 1D
        dict(column="peil", filetype="raster", cellsize=1.0, interpolationmethod="averaging"),  # averaging on 1D
        dict(column=None),  # polygons without a value column
    ],
)
def test_spatial_field_invalid_polygon_input(kwargs):
    hydamo = HyDAMO()
    with pytest.raises(ValueError):
        hydamo.external_forcings.set_spatial_field(_polygons(), **kwargs)


def test_spatial_field_points_without_z():
    hydamo = HyDAMO()
    points = gpd.GeoDataFrame(geometry=[Point(1, 1), Point(2, 2)], crs=28992)
    with pytest.raises(ValueError):
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
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", filename="streefpeil.ini")
    models = _write(hydamo, tmp_path)

    field = models.inifields[0]
    assert (field.datafiletype, field.datafile.filepath.name) == ("1dField", "streefpeil.ini")
    onedfield = OneDFieldModel(filepath=tmp_path / "streefpeil.ini")
    assert [b.branchid for b in onedfield.branch] == ["br1", "br2"]
    # a [Global] block with the nodata value, required by the D-HYDRO GUI
    assert (onedfield.global_.quantity, onedfield.global_.value) == ("waterlevel", -999.0)
    np.testing.assert_allclose(onedfield.branch[0].values, [1.0, 1.0, 2.5, 2.5, 1.0, 1.0])


def test_level_and_depth_on_same_branch_warns(caplog):
    hydamo = HyDAMO()
    hydamo.branches = _branches()
    level = gpd.GeoDataFrame({"peil": [1.0]}, geometry=[box(0, 0, 15, 10)], crs=28992)
    depth = gpd.GeoDataFrame({"diepte": [0.5]}, geometry=[box(15, 0, 35, 30)], crs=28992)
    hydamo.external_forcings.set_spatial_field(depth, "diepte", quantity="waterdepth", filename="depth.ini")
    hydamo.external_forcings.set_spatial_field(level, "peil", quantity="waterlevel", filename="level.ini")

    with caplog.at_level(logging.WARNING):
        Df2HydrolibModel(hydamo)

    # br1 crosses both areas, br2 only the depth area
    messages = [msg for msg in caplog.messages if "get both a waterdepth" in msg]
    assert len(messages) == 1
    assert "['br1']" in messages[0]


@pytest.mark.parametrize(("filetype", "extension"), [("branch", ".ini"), ("polygon", ".pol")])
def test_filename_extension_follows_filetype(filetype, extension):
    hydamo = HyDAMO()
    hydamo.external_forcings.set_spatial_field(_polygons(), "peil", filetype=filetype, filename="initial_waterlevel")
    assert hydamo.external_forcings.spatial_fields[0]["filename"] == f"initial_waterlevel{extension}"
