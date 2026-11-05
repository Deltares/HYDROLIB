"""Zonal statistics computed with exactextract.

exactextract computes the exact fraction of every pixel covered by a zone. Two of
its arguments, applied to every statistic, use that fraction:

- ``min_coverage_frac`` selects the pixels: those covered less are ignored.
- ``coverage_weight`` sets how they count: ``"none"`` counts every selected pixel
  fully, ``"fraction"`` by the fraction of it the zone covers.

The defaults, ``min_coverage_frac=0`` and ``coverage_weight="fraction"``, give
exact coverage weighting: every touched pixel counts by the fraction of it the
zone covers. ``coverage_weight="none"`` counts every touched pixel fully, like
``all_touched=True`` in GDAL and rasterstats; adding ``min_coverage_frac=0.5``
counts only pixels that are at least half covered, close to their default
pixel-centre rule.

Raster files are read through rasterio. exactextract reads windows of at most
``max_cells_in_memory`` cells, so rasters may be larger than memory.
"""

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from exactextract import exact_extract
from exactextract.feature import Feature, FeatureSource
from exactextract.raster import NumPyRasterSource, RasterioRasterSource
from rasterio.enums import MaskFlags
from rasterio.io import DatasetReader
from rasterio.windows import Window
from tqdm.auto import tqdm

Strategy = Literal["feature-sequential", "raster-sequential"]
CoverageWeight = Literal["fraction", "none"]

_ZONE_ID = "_zonal_id"


class _Zone(Feature):
    """A zone as exactextract reads it: its position and its WKB geometry."""

    def __init__(self, position: int, geometry):
        super().__init__()
        self.position = position
        self.wkb = shapely.to_wkb(geometry)

    def get(self, name):
        return self.position

    def fields(self):
        return [_ZONE_ID]

    def geometry(self):
        return self.wkb

    def set_geometry_format(self):
        return "wkb"


class _ZoneSource(FeatureSource):
    """Zones for exactextract, used instead of its GeoDataFrame source.

    That source fails on zones without a CRS, such as mesh polygons, and passes
    every zone as a GeoJSON dict with all its columns. This one passes WKB and
    only a positional ID, accepts a GeoSeries, and reports no CRS, because
    :func:`_check_crs` already compared the CRSs.
    """

    def __init__(self, geometries: np.ndarray):
        super().__init__()
        self.geometries = geometries

    def count(self):
        return len(self.geometries)

    def __iter__(self):
        for position, geometry in enumerate(self.geometries):
            yield _Zone(position, geometry)

    def srs_wkt(self):
        return None


class _PlainRasterioSource(RasterioRasterSource):
    """Rasterio source that reads plain arrays and lets exactextract apply the
    nodata value, instead of the masked arrays of exactextract's own reader.
    Rasters masked otherwise than by a nodata value, or that are scaled, still go
    through the masked reader.
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        flags = dataset.mask_flag_enums[self.band_idx - 1]
        self.plain = not self.scaled and set(flags) <= {
            MaskFlags.nodata,
            MaskFlags.all_valid,
        }

    def read_window(self, x0, y0, nx, ny):
        if not self.plain:
            return super().read_window(x0, y0, nx, ny)
        return self.ds.read(self.band_idx, window=Window(x0, y0, nx, ny))


# On AHN 0.5 m benchmarks 27-35% faster than exactextract's own rasterio reader,
# and 7-10% faster than its osgeo reader. Benchmarks swap it out to compare.
_RASTERIO_SOURCE = _PlainRasterioSource


def zonal_stats(
    zones: gpd.GeoDataFrame | gpd.GeoSeries,
    raster: Path | str | np.ndarray,
    statistics: str | list[str] | tuple[str, ...],
    *,
    strategy: Strategy,
    min_coverage_frac: float = 0.0,
    coverage_weight: CoverageWeight = "fraction",
    affine=None,
    raster_crs=None,
    nodata: float | None = None,
    max_cells_in_memory: int = 30_000_000,
    progress=False,
) -> pd.DataFrame:
    """Calculate scalar raster statistics for every polygonal zone.

    Parameters
    ----------
    zones : geopandas.GeoDataFrame or geopandas.GeoSeries
        Polygon zones. Zones without a CRS, such as mesh polygons, are taken to
        be in the raster CRS.
    raster : pathlib.Path, str, numpy.ndarray, or rasterio dataset
        Raster file, an open rasterio dataset, or a two-dimensional array
        together with ``affine``. Only band 1 is used.
    statistics : str or sequence of str
        Scalar exactextract statistics, such as ``"mean"``, ``"median"``,
        ``"mode"``, ``"sum"``, or ``"quantile(q=0.9)"``. exactextract names the
        result columns, e.g. ``"mean"`` and ``"quantile_90"``. Set the coverage
        arguments below, not in these strings.
    strategy : {'feature-sequential', 'raster-sequential'}
        exactextract processing strategy. ``'feature-sequential'`` reads the
        raster under each zone in turn, which suits few, large zones.
        ``'raster-sequential'`` reads the raster once, which suits many small
        zones such as mesh cells.
    min_coverage_frac : float, default 0
        Smallest fraction of a pixel the zone must cover for the pixel to
        count, from 0 (every touched pixel) to 1.
    coverage_weight : {'fraction', 'none'}, default 'fraction'
        How the selected pixels count: ``'fraction'`` by the fraction of them
        the zone covers, ``'none'`` fully.
    affine : affine.Affine, optional
        North-up transform of an array raster.
    raster_crs : pyproj.CRS or str, optional
        CRS of a raster that has none of its own, such as an array.
    nodata : float, optional
        Value to exclude from an array raster. Files use their own nodata.
    max_cells_in_memory : int, default 30_000_000
        Largest number of raster cells exactextract reads at once.
    progress : bool or callable, default False
        Show a progress bar, or call exactextract's ``progress(fraction,
        message)`` as the calculation proceeds.

    Returns
    -------
    pandas.DataFrame
        One float column per statistic, aligned to ``zones.index``. Zones
        without valid pixels get ``NaN``.

    Notes
    -----
    Reads use ``GDAL_NUM_THREADS=ALL_CPUS`` to decompress the raster, unless
    ``GDAL_NUM_THREADS`` is set in the environment.
    """
    statistics = [statistics] if isinstance(statistics, str) else list(statistics)
    result = _extract(
        zones,
        raster,
        statistics,
        strategy=strategy,
        min_coverage_frac=min_coverage_frac,
        coverage_weight=coverage_weight,
        affine=affine,
        raster_crs=raster_crs,
        nodata=nodata,
        max_cells_in_memory=max_cells_in_memory,
        progress=progress,
    )
    if result.shape[1] != len(statistics):
        raise ValueError(
            f"Statistics {statistics} give clashing column names "
            f"{list(result.columns)}; request them in separate calls."
        )
    return result.apply(pd.to_numeric, errors="coerce").astype(np.float64)


def zonal_category_counts(
    zones: gpd.GeoDataFrame | gpd.GeoSeries,
    raster: Path | str | np.ndarray,
    *,
    strategy: Strategy,
    min_coverage_frac: float = 0.0,
    coverage_weight: CoverageWeight = "fraction",
    affine=None,
    raster_crs=None,
    nodata: float | None = None,
    max_cells_in_memory: int = 30_000_000,
    progress=False,
) -> pd.DataFrame:
    """Count the cells of each raster class within every polygonal zone.

    Parameters are those of :func:`zonal_stats`. With
    ``coverage_weight="fraction"``, partly covered cells count for their covered
    fraction, so counts are not whole numbers.

    Returns
    -------
    pandas.DataFrame
        One float column per class code, aligned to ``zones.index``. Integer
        class codes are ints. Classes absent from a zone count zero.
    """
    result = _extract(
        zones,
        raster,
        ["unique", "frac", "count"],
        strategy=strategy,
        min_coverage_frac=min_coverage_frac,
        coverage_weight=coverage_weight,
        affine=affine,
        raster_crs=raster_crs,
        nodata=nodata,
        max_cells_in_memory=max_cells_in_memory,
        progress=progress,
    )
    rows = [
        {
            int(code) if float(code).is_integer() else code: share * count
            for code, share in zip(classes.tolist(), np.asarray(frac).tolist())
        }
        if isinstance(classes, np.ndarray)  # None for zones outside the raster
        else {}
        for classes, frac, count in zip(
            result["unique"], result["frac"], result["count"]
        )
    ]
    counts = pd.DataFrame(rows, index=zones.index, dtype=np.float64)
    counts = counts.sort_index(axis=1).fillna(0.0)
    # Unweighted counts are whole cells; drop floating-point noise.
    return counts.round() if coverage_weight == "none" else counts


def _extract(
    zones,
    raster,
    statistics,
    *,
    strategy,
    min_coverage_frac,
    coverage_weight,
    affine,
    raster_crs,
    nodata,
    max_cells_in_memory,
    progress,
):
    """Run exactextract and return its results indexed like ``zones``."""
    if zones.empty:
        raise ValueError("zones must contain at least one geometry.")
    if not 0 <= min_coverage_frac <= 1:
        raise ValueError(f"min_coverage_frac must be from 0 to 1, not {min_coverage_frac}.")
    if coverage_weight not in ("fraction", "none"):
        raise ValueError(
            f"coverage_weight must be 'fraction' or 'none', not {coverage_weight!r}."
        )
    coverage = f"min_coverage_frac={min_coverage_frac},coverage_weight={coverage_weight}"
    ops = []
    for stat in statistics:
        if "min_coverage_frac" in stat or "coverage_weight" in stat:
            raise ValueError(
                f"Pass min_coverage_frac and coverage_weight as keywords, not in {stat!r}."
            )
        ops.append(f"{stat[:-1]},{coverage})" if stat.endswith(")") else f"{stat}({coverage})")

    with ExitStack() as stack:
        source, own_crs = _raster_source(raster, affine, nodata, stack)
        _check_crs(zones.crs, own_crs or raster_crs)
        if progress is True:
            progress = _progress_bar(stack)
        result = exact_extract(
            source,
            _ZoneSource(zones.geometry.to_numpy()),
            ops,
            include_cols=[_ZONE_ID],
            strategy=strategy,
            max_cells_in_memory=max_cells_in_memory,
            output="pandas",
            progress=progress,
        )
    return result.set_index(_ZONE_ID).reindex(range(len(zones))).set_axis(zones.index)


def _progress_bar(stack):
    """Return an exactextract progress callback that shows a percentage bar."""
    bar = stack.enter_context(
        tqdm(total=100, bar_format="{percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]")
    )

    def update(fraction, message):
        bar.update(100 * fraction - bar.n)

    return update


def _raster_source(raster, affine, nodata, stack):
    """Return an exactextract source for ``raster`` and the raster's own CRS.

    Files are opened in ``stack``, which also scopes ``GDAL_NUM_THREADS`` for
    every read.
    """
    if isinstance(raster, np.ndarray):
        if affine is None or affine.b or affine.d or affine.e >= 0:
            raise ValueError("An array raster needs a north-up affine transform.")
        rows, columns = raster.shape
        source = NumPyRasterSource(
            raster,
            affine.c,
            affine.f + affine.e * rows,
            affine.c + affine.a * columns,
            affine.f,
            nodata=nodata,
        )
        return source, None

    threads = os.environ.get("GDAL_NUM_THREADS", "ALL_CPUS")
    stack.enter_context(rasterio.Env(GDAL_NUM_THREADS=threads))
    if not isinstance(raster, DatasetReader):
        raster = stack.enter_context(rasterio.open(raster))
    return _RASTERIO_SOURCE(raster), raster.crs.to_wkt() if raster.crs else None


def _check_crs(zones_crs, raster_crs):
    """Raise if the zones and raster both have a CRS and they differ."""
    if zones_crs is None or raster_crs is None:
        return
    if not zones_crs.equals(raster_crs, ignore_axis_order=True):
        raise ValueError(
            f"Zones CRS {zones_crs.name!r} differs from the raster CRS; "
            "reproject the zones first."
        )
