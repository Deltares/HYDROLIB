from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from hydrolib.dhydamo.geometry.zonal import zonal_category_counts, zonal_stats
from osgeo import gdal, osr
from shapely.geometry import box

VALUES = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int16)
TRANSFORM = Affine.translation(0, 3) @ Affine.scale(1, -1)
CRS = "EPSG:3857"


@pytest.fixture
def raster_factory():
    paths = []

    def create(values, *, nodata=None, transform=None, crs=CRS):
        path = f"/vsimem/zonal-{uuid4().hex}.tif"
        dataset = gdal.GetDriverByName("GTiff").Create(
            path, values.shape[1], values.shape[0], 1, gdal.GDT_Int16
        )
        transform = transform or Affine.translation(0, values.shape[0]) @ Affine.scale(
            1, -1
        )
        dataset.SetGeoTransform(transform.to_gdal())
        if crs is not None:
            spatial_ref = osr.SpatialReference()
            spatial_ref.SetFromUserInput(crs)
            dataset.SetSpatialRef(spatial_ref)
        band = dataset.GetRasterBand(1)
        band.WriteArray(values)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        dataset = None
        paths.append(path)
        return path

    yield create

    for path in paths:
        gdal.Unlink(path)


@pytest.fixture
def full_zone():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=CRS)


def test_zonal_stats_calculates_mean_median_mode_and_excludes_nodata(full_zone):
    values = VALUES.copy()
    values[1, 1] = -9999

    scalar = zonal_stats(
        full_zone,
        values,
        statistics=("mean", "median"),
        affine=TRANSFORM,
        raster_crs=CRS,
        nodata=-9999,
    )
    mode = zonal_stats(
        full_zone,
        np.array([[1, 1, 2], [2, 2, 3], [3, 3, 3]], dtype=np.int16),
        statistics=("mode",),
        affine=TRANSFORM,
        raster_crs=CRS,
    )

    assert scalar.to_dict("list") == {"mean": [5.0], "median": [5.0]}
    assert mode.to_dict("list") == {"mode": [3.0]}


@pytest.mark.parametrize(
    ("all_touched", "expected_mean"),
    [(False, 2.5), (True, 3.0)],
)
def test_zonal_stats_pixel_selection_behavior(all_touched, expected_mean):
    zones = gpd.GeoDataFrame(geometry=[box(0.2, 1.1, 1.2, 2.8)], crs=CRS)

    result = zonal_stats(
        zones,
        VALUES,
        statistics=("mean",),
        all_touched=all_touched,
        affine=TRANSFORM,
        raster_crs=CRS,
    )

    assert result["mean"].tolist() == [expected_mean]


def test_zonal_category_counts_returns_integer_category_keys(raster_factory):
    values = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int16)
    zones = gpd.GeoDataFrame(geometry=[box(2, 0, 3, 2), box(0, 0, 2, 2)], crs=CRS)

    result = zonal_category_counts(zones, raster_factory(values))

    assert all(isinstance(category, int) for category in result.columns)


def test_zonal_stats_returns_nan_when_no_valid_pixels():
    no_overlap = zonal_stats(
        gpd.GeoDataFrame(geometry=[box(4, 4, 5, 5)], crs=CRS),
        VALUES,
        statistics=("mean",),
        affine=TRANSFORM,
        raster_crs=CRS,
    )
    all_nodata = zonal_stats(
        gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=CRS),
        np.full((3, 3), -9999, dtype=np.int16),
        statistics=("mean",),
        affine=TRANSFORM,
        raster_crs=CRS,
        nodata=-9999,
    )

    assert pd.isna(no_overlap.loc[0, "mean"])
    assert pd.isna(all_nodata.loc[0, "mean"])


def test_zonal_category_counts_returns_empty_for_zone_without_overlap(raster_factory):
    zones = gpd.GeoDataFrame(geometry=[box(4, 4, 5, 5)], crs=CRS)

    result = zonal_category_counts(zones, raster_factory(VALUES))

    assert result.index.tolist() == [0]
    assert result.empty


def test_zonal_stats_uses_raster_crs_for_crsless_mesh_zones(raster_factory):
    zones = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)])

    result = zonal_stats(
        zones, raster_factory(VALUES), statistics=("mean",), strategy="raster"
    )

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_applies_an_explicit_file_raster_crs_override(raster_factory):
    zones = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=CRS)
    raster = raster_factory(VALUES, crs=None)

    result = zonal_stats(zones, raster, statistics=("mean",), raster_crs=zones.crs)

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_preserves_shuffled_zone_order_and_index(raster_factory):
    zones = gpd.GeoDataFrame(
        {"name": ["right", "left"]},
        geometry=[box(2, 0, 3, 3), box(0, 0, 1, 3)],
        index=[42, 7],
        crs=CRS,
    )

    result = zonal_stats(
        zones, VALUES, statistics=("mean",), affine=TRANSFORM, raster_crs=CRS
    )
    categorical = zonal_category_counts(zones, raster_factory(VALUES))

    assert result.index.tolist() == [42, 7]
    assert result["mean"].tolist() == [6.0, 4.0]
    assert categorical.index.tolist() == [42, 7]
    assert categorical.loc[42].to_dict() == {1: 0, 3: 1, 4: 0, 6: 1, 7: 0, 9: 1}
    assert categorical.loc[7].to_dict() == {1: 1, 3: 0, 4: 1, 6: 0, 7: 1, 9: 0}


def test_zonal_stats_rejects_mismatched_crs(full_zone):
    mismatched_zone_crs = full_zone.to_crs("EPSG:4326")

    with pytest.raises(ValueError, match="Zones and raster must use the same CRS"):
        zonal_stats(
            mismatched_zone_crs,
            VALUES,
            statistics=("mean",),
            affine=TRANSFORM,
            raster_crs=CRS,
        )


def test_zonal_stats_skips_crs_check_when_raster_crs_is_unset(full_zone):
    result = zonal_stats(full_zone, VALUES, statistics=("mean",), affine=TRANSFORM)

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_applies_arbitrary_array_affine_transform():
    transform = Affine.translation(100, 203) @ Affine.scale(2, -1)
    zones = gpd.GeoDataFrame(geometry=[box(100, 200, 104, 203)], crs=CRS)

    result = zonal_stats(
        zones, VALUES, statistics=("mean",), affine=transform, raster_crs=CRS
    )

    assert result["mean"].tolist() == [4.5]


def test_zonal_stats_rejects_invalid_strategy(full_zone):
    with pytest.raises(ValueError, match="Unsupported GDAL zonal-statistics strategy"):
        zonal_stats(
            full_zone,
            VALUES,
            statistics=("mean",),
            strategy="invalid",
            affine=TRANSFORM,
            raster_crs=CRS,
        )


def test_zonal_stats_rejects_empty_statistics(full_zone):
    with pytest.raises(
        ValueError, match="statistics must contain at least one statistic"
    ):
        zonal_stats(full_zone, VALUES, statistics=(), affine=TRANSFORM, raster_crs=CRS)


def test_zonal_stats_rejects_unsupported_statistic(full_zone):
    with pytest.raises(ValueError, match="Unsupported scalar statistics"):
        zonal_stats(
            full_zone, VALUES, statistics=("sum",), affine=TRANSFORM, raster_crs=CRS
        )


def test_zonal_stats_rejects_empty_zones():
    empty_zones = gpd.GeoDataFrame(geometry=[], crs=CRS)

    with pytest.raises(ValueError, match="zones must contain at least one geometry"):
        zonal_stats(
            empty_zones, VALUES, statistics=("mean",), affine=TRANSFORM, raster_crs=CRS
        )
