import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from hydrolib.dhydamo.geometry import zonal
from hydrolib.dhydamo.geometry.zonal import (
    zonal_category_counts,
    zonal_stats,
    zonal_stats_per_raster,
)
from shapely.geometry import box

VALUES = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int16)
TRANSFORM = Affine.translation(0, 3) @ Affine.scale(1, -1)
CRS = "EPSG:3857"
FS = "feature-sequential"
RS = "raster-sequential"
# Covers 0.64 of pixel 1, 0.16 of pixel 2, 0.72 of pixel 4, and 0.18 of pixel 5.
PARTIAL_ZONE = box(0.2, 1.1, 1.2, 2.8)


@pytest.fixture
def raster_factory(tmp_path):
    def create(values, *, nodata=None, crs=CRS, valid=None):
        path = tmp_path / f"raster{len(list(tmp_path.iterdir()))}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=values.dtype,
            transform=Affine.translation(0, values.shape[0]) @ Affine.scale(1, -1),
            crs=crs,
            nodata=nodata,
        ) as dataset:
            dataset.write(values, 1)
            if valid is not None:
                dataset.write_mask(np.where(valid, 255, 0).astype(np.uint8))
        return path

    return create


@pytest.fixture
def full_zone():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=CRS)


def test_zonal_stats_calculates_mean_median_mode_and_excludes_nodata(full_zone):
    values = VALUES.copy()
    values[1, 1] = -9999

    scalar = zonal_stats(
        full_zone,
        values,
        ("mean", "median"),
        strategy=FS,
        affine=TRANSFORM,
        nodata=-9999,
    )
    mode = zonal_stats(
        full_zone,
        np.array([[1, 1, 2], [2, 2, 3], [3, 3, 3]], dtype=np.int16),
        "mode",
        strategy=FS,
        affine=TRANSFORM,
    )

    assert scalar.to_dict("list") == {"mean": [5.0], "median": [5.0]}
    assert mode.to_dict("list") == {"mode": [3.0]}


@pytest.mark.parametrize(
    ("min_coverage_frac", "coverage_weight", "expected_mean"),
    [
        (0, "fraction", (0.64 * 1 + 0.16 * 2 + 0.72 * 4 + 0.18 * 5) / 1.7),
        (0, "none", (1 + 2 + 4 + 5) / 4),
        (0.5, "fraction", (0.64 * 1 + 0.72 * 4) / 1.36),
        (0.5, "none", (1 + 4) / 2),
    ],
)
def test_zonal_stats_coverage(
    raster_factory, min_coverage_frac, coverage_weight, expected_mean
):
    zones = gpd.GeoDataFrame(geometry=[PARTIAL_ZONE], crs=CRS)

    result = zonal_stats(
        zones,
        raster_factory(VALUES),
        "mean",
        strategy=FS,
        min_coverage_frac=min_coverage_frac,
        coverage_weight=coverage_weight,
    )

    assert result["mean"].tolist() == pytest.approx([expected_mean])


def test_zonal_stats_defaults_to_exact_coverage_weighting():
    zones = gpd.GeoDataFrame(geometry=[PARTIAL_ZONE], crs=CRS)

    default = zonal_stats(zones, VALUES, "mean", strategy=FS, affine=TRANSFORM)
    exact = zonal_stats(
        zones,
        VALUES,
        "mean",
        strategy=FS,
        min_coverage_frac=0,
        coverage_weight="fraction",
        affine=TRANSFORM,
    )

    pd.testing.assert_frame_equal(default, exact)


def test_zonal_stats_coverage_applies_to_statistics_with_arguments():
    zones = gpd.GeoDataFrame(geometry=[PARTIAL_ZONE], crs=CRS)

    result = zonal_stats(
        zones,
        VALUES,
        "quantile(q=0.5)",
        strategy=FS,
        coverage_weight="none",
        affine=TRANSFORM,
    )

    # Unweighted median of 1, 2, 4, 5; the coverage-weighted median is 2.
    assert result.to_dict("list") == {"quantile_50": [3.0]}


@pytest.mark.parametrize(
    ("statistic", "kwargs", "message"),
    [
        ("mean", {"min_coverage_frac": 1.5}, "min_coverage_frac must be from 0 to 1"),
        ("mean", {"coverage_weight": "area_cartesian"}, "coverage_weight must be"),
        ("mean(coverage_weight=none)", {}, "as keywords"),
        ("quantile(q=0.5,min_coverage_frac=0.5)", {}, "as keywords"),
    ],
)
def test_zonal_stats_rejects_invalid_coverage(full_zone, statistic, kwargs, message):
    with pytest.raises(ValueError, match=message):
        zonal_stats(
            full_zone, VALUES, statistic, strategy=FS, affine=TRANSFORM, **kwargs
        )


def test_zonal_stats_uses_exactextract_column_names(full_zone):
    result = zonal_stats(
        full_zone,
        VALUES,
        ("sum", "min", "max", "quantile(q=0.9)"),
        strategy=FS,
        affine=TRANSFORM,
    )

    assert list(result.columns) == ["sum", "min", "max", "quantile_90"]
    assert result.loc[0, ["sum", "min", "max"]].tolist() == [45.0, 1.0, 9.0]


def test_zonal_stats_rejects_statistics_with_clashing_names(full_zone):
    with pytest.raises(ValueError, match="clashing column names"):
        zonal_stats(
            full_zone,
            VALUES,
            ("quantile(q=0.5)", "quantile(q=0.50)"),
            strategy=FS,
            affine=TRANSFORM,
        )


def test_zonal_stats_returns_nan_when_no_valid_pixels():
    no_overlap = zonal_stats(
        gpd.GeoDataFrame(geometry=[box(4, 4, 5, 5)], crs=CRS),
        VALUES,
        ("mean", "median"),
        strategy=FS,
        affine=TRANSFORM,
    )
    all_nodata = zonal_stats(
        gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=CRS),
        np.full((3, 3), -9999, dtype=np.int16),
        "mean",
        strategy=FS,
        affine=TRANSFORM,
        nodata=-9999,
    )

    assert no_overlap.isna().all(axis=None)
    assert pd.isna(all_nodata.loc[0, "mean"])


def test_zonal_stats_accepts_crsless_mesh_zones(raster_factory):
    zones = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)])

    result = zonal_stats(
        zones,
        raster_factory(VALUES),
        "mean",
        strategy=RS,
    )

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_reads_file_raster_nodata(raster_factory, full_zone):
    values = VALUES.copy()
    values[1, 1] = -9999

    result = zonal_stats(
        full_zone,
        raster_factory(values, nodata=-9999),
        "mean",
        strategy=FS,
    )

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_reads_file_raster_mask_band(raster_factory, full_zone):
    values = VALUES.copy()
    values[1, 1] = 1000
    valid = np.ones(values.shape, dtype=bool)
    valid[1, 1] = False

    result = zonal_stats(
        full_zone,
        raster_factory(values, valid=valid),
        "mean",
        strategy=FS,
    )

    assert result.to_dict("list") == {"mean": [5.0]}


@pytest.mark.parametrize("strategy", [FS, RS])
def test_zonal_stats_preserves_shuffled_zone_order_and_index(strategy):
    zones = gpd.GeoDataFrame(
        {"name": ["right", "left"]},
        geometry=[box(2, 0, 3, 3), box(0, 0, 1, 3)],
        index=[42, 7],
        crs=CRS,
    )

    result = zonal_stats(zones, VALUES, "mean", strategy=strategy, affine=TRANSFORM)

    assert result.index.tolist() == [42, 7]
    assert result["mean"].tolist() == [6.0, 4.0]


def test_zonal_stats_rejects_mismatched_crs(raster_factory, full_zone):
    with pytest.raises(ValueError, match="differs from the raster CRS"):
        zonal_stats(
            full_zone.to_crs("EPSG:4326"),
            raster_factory(VALUES),
            "mean",
            strategy=FS,
        )


def test_zonal_stats_checks_raster_crs_of_a_raster_without_crs(
    raster_factory, full_zone
):
    raster = raster_factory(VALUES, crs=None)

    result = zonal_stats(
        full_zone,
        raster,
        "mean",
        strategy=FS,
        raster_crs=CRS,
    )
    with pytest.raises(ValueError, match="differs from the raster CRS"):
        zonal_stats(
            full_zone,
            raster,
            "mean",
            strategy=FS,
            raster_crs="EPSG:4326",
        )

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_rejects_mismatched_raster_crs_of_an_array(full_zone):
    with pytest.raises(ValueError, match="differs from the raster CRS"):
        zonal_stats(
            full_zone,
            VALUES,
            "mean",
            strategy=FS,
            affine=TRANSFORM,
            raster_crs=rasterio.crs.CRS.from_epsg(4326),
        )


def test_zonal_stats_accepts_an_open_rasterio_dataset(raster_factory, full_zone):
    with rasterio.open(raster_factory(VALUES)) as dataset:
        result = zonal_stats(full_zone, dataset, "mean", strategy=FS)

    assert result.to_dict("list") == {"mean": [5.0]}


def test_zonal_stats_applies_array_affine_transform():
    transform = Affine.translation(100, 203) @ Affine.scale(2, -1)
    zones = gpd.GeoDataFrame(geometry=[box(100, 200, 104, 203)], crs=CRS)

    result = zonal_stats(zones, VALUES, "mean", strategy=FS, affine=transform)

    assert result["mean"].tolist() == [4.5]


@pytest.mark.parametrize("affine", [None, Affine.identity(), Affine.rotation(10)])
def test_zonal_stats_rejects_array_without_north_up_transform(full_zone, affine):
    with pytest.raises(ValueError, match="north-up affine transform"):
        zonal_stats(full_zone, VALUES, "mean", strategy=FS, affine=affine)


def test_zonal_stats_rejects_empty_zones():
    with pytest.raises(ValueError, match="zones must contain at least one geometry"):
        zonal_stats(
            gpd.GeoDataFrame(geometry=[], crs=CRS),
            VALUES,
            "mean",
            strategy=FS,
            affine=TRANSFORM,
        )


def test_zonal_stats_per_raster_matches_zonal_stats_in_raster_order(
    raster_factory, monkeypatch
):
    # Three rasters over two calls.
    monkeypatch.setattr(zonal, "_RASTERS_PER_CALL", 2)
    rasters = [raster_factory(VALUES * factor) for factor in (1, 10, 100)]
    zones = gpd.GeoDataFrame(
        geometry=[PARTIAL_ZONE, box(0, 0, 3, 3)], index=[42, 7], crs=CRS
    )

    result = zonal_stats_per_raster(zones, rasters, "mean", strategy=FS)

    expected = pd.concat(
        [zonal_stats(zones, raster, "mean", strategy=FS)["mean"] for raster in rasters],
        axis=1,
        keys=[0, 1, 2],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_zonal_stats_per_raster_reads_numpy_and_lazy_arrays(full_zone):
    dask_array = pytest.importorskip("dask.array")
    rasters = [VALUES, dask_array.from_array(VALUES * 10)]

    result = zonal_stats_per_raster(
        full_zone, rasters, "mean", strategy=FS, affine=TRANSFORM
    )

    assert result.loc[0].tolist() == [5.0, 50.0]


def test_zonal_category_counts_per_integer_class(raster_factory):
    values = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int16)
    zones = gpd.GeoDataFrame(
        geometry=[box(2, 0, 3, 2), box(0, 0, 2, 2)], index=[42, 7], crs=CRS
    )

    result = zonal_category_counts(
        zones,
        raster_factory(values),
        strategy=FS,
    )

    assert all(isinstance(category, int) for category in result.columns)
    assert result.to_dict("index") == {42: {1: 0.0, 2: 2.0}, 7: {1: 3.0, 2: 1.0}}


def test_zonal_category_counts_uses_int_keys_for_integer_float_classes():
    zones = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs=CRS)

    result = zonal_category_counts(
        zones, VALUES.astype(np.float32), strategy=FS, affine=TRANSFORM
    )

    assert result.columns.tolist() == list(range(1, 10))
    assert result.dtypes.eq(np.float64).all()


@pytest.mark.parametrize(
    ("min_coverage_frac", "coverage_weight", "expected"),
    [
        (0, "fraction", {1: 0.64, 2: 0.16, 4: 0.72, 5: 0.18}),
        (0, "none", {1: 1.0, 2: 1.0, 4: 1.0, 5: 1.0}),
        (0.5, "fraction", {1: 0.64, 4: 0.72}),
        (0.5, "none", {1: 1.0, 4: 1.0}),
    ],
)
def test_zonal_category_counts_coverage(min_coverage_frac, coverage_weight, expected):
    zones = gpd.GeoDataFrame(geometry=[PARTIAL_ZONE], crs=CRS)

    result = zonal_category_counts(
        zones,
        VALUES,
        strategy=FS,
        min_coverage_frac=min_coverage_frac,
        coverage_weight=coverage_weight,
        affine=TRANSFORM,
    )

    assert result.loc[0].to_dict() == pytest.approx(expected)


def test_zonal_category_counts_returns_empty_for_zone_without_overlap(raster_factory):
    zones = gpd.GeoDataFrame(geometry=[box(4, 4, 5, 5)], crs=CRS)

    result = zonal_category_counts(
        zones,
        raster_factory(VALUES),
        strategy=FS,
    )

    assert result.index.tolist() == [0]
    assert result.empty


def test_zonal_functions_accept_a_geoseries():
    zones = gpd.GeoSeries([box(0, 0, 3, 3)], index=[5], crs=CRS)

    stats = zonal_stats(zones, VALUES, "mean", strategy=FS, affine=TRANSFORM)
    counts = zonal_category_counts(zones, VALUES, strategy=FS, affine=TRANSFORM)

    assert stats.to_dict("index") == {5: {"mean": 5.0}}
    assert counts.loc[5].sum() == 9.0
