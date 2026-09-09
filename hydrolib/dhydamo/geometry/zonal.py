from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from osgeo import gdal, gdal_array, ogr, osr

gdal.UseExceptions()

_ZONE_ID_FIELD = "_zonal_id"
_ZONE_LAYER = "zones"
_ZONE_GEOMETRY_FIELD = "geometry"


def zonal_stats(
    zones: gpd.GeoDataFrame,
    raster: Path | str | np.ndarray,
    *,
    statistics: tuple[Literal["mean", "median", "mode"], ...],
    all_touched: bool = False,
    affine=None,
    raster_crs=None,
    nodata: float | None = None,
    strategy: Literal["feature", "raster"] = "feature",
) -> pd.DataFrame:
    """Calculate scalar raster statistics for every polygonal zone.

    Parameters
    ----------
    zones : geopandas.GeoDataFrame
        Polygon zones. Its CRS must match the raster CRS. File-backed rasters
        assign their CRS when zones have none; this supports mesh polygons,
        whose coordinates are already in the raster CRS.
    raster : pathlib.Path, str, or numpy.ndarray
        Raster path or a two-dimensional array. Arrays require ``affine`` and
        ``raster_crs``.
    statistics : tuple of {'mean', 'median', 'mode'}
        Scalar statistics to calculate for each zone.
    all_touched : bool, default False
        Include every raster pixel touched by a zone instead of GDAL's default
        pixel-centre rule.
    affine : affine.Affine or sequence of float, optional
        Six-value geotransform for an array raster.
    raster_crs : pyproj.CRS or str, optional
        CRS for an array raster, or an explicit CRS override when a file-backed
        raster lacks spatial-reference metadata.
    nodata : float, optional
        Array value to exclude from the calculation.
    strategy : {'feature', 'raster'}, default 'feature'
        GDAL processing strategy for ``mean``/``mode``, run as a single
        ``gdal.Run`` call across every zone regardless of this value. The
        ``median`` reconstruction always uses ``strategy="feature"``
        internally, independent of this argument.

    Returns
    -------
    pandas.DataFrame
        Requested statistic columns aligned to ``zones.index``. Missing values
        are represented by ``NaN``.
    """
    if not statistics:
        raise ValueError("statistics must contain at least one statistic.")
    unsupported = set(statistics) - {"mean", "median", "mode"}
    if unsupported:
        raise ValueError(f"Unsupported scalar statistics: {sorted(unsupported)}.")

    if isinstance(raster, np.ndarray):
        raster_dataset = _array_to_mem_raster(
            raster, affine, raster_crs or zones.crs, nodata
        )
    else:
        raster_dataset = _open_file_raster(raster, raster_crs or zones.crs)

    with raster_dataset:
        if zones.crs is None and raster_dataset.GetSpatialRef() is not None:
            zones = zones.set_crs(raster_dataset.GetSpatialRef().ExportToWkt())

        direct_statistics = tuple(stat for stat in statistics if stat != "median")
        if direct_statistics:
            records = _run_gdal_zonal_stats(
                zones=zones,
                raster_dataset=raster_dataset,
                statistics=direct_statistics,
                pixel_mode=_pixel_mode(all_touched),
                strategy=strategy,
            )
        else:
            records = [{} for _ in range(len(zones))]

        if "median" in statistics:
            # GDAL does not implement median. Reconstruct it from the exact
            # value frequencies reported by count, unique, and frac. This
            # runs one zone at a time, so it can never benefit from
            # strategy="raster" (whose entire benefit is sharing a single
            # raster scan across many zones in one call); forcing
            # strategy="feature" here avoids paying a full-raster-scan cost
            # per zone, regardless of what strategy the caller requested for
            # the batched statistics above.
            for position, record in enumerate(records):
                record["median"] = _median_from_unique_frac(
                    _run_gdal_zonal_stats(
                        zones=zones.iloc[[position]],
                        raster_dataset=raster_dataset,
                        statistics=("count", "unique", "frac"),
                        pixel_mode=_pixel_mode(all_touched),
                        strategy="feature",
                    )[0]
                )

    return pd.DataFrame(records, index=zones.index).reindex(columns=statistics).apply(
        pd.to_numeric, errors="coerce"
    )


def zonal_category_counts(
    zones: gpd.GeoDataFrame,
    raster: Path | str | np.ndarray,
    *,
    all_touched: bool = False,
    affine=None,
    raster_crs=None,
    nodata: float | None = None,
    strategy: Literal["feature", "raster"] = "feature",
) -> pd.DataFrame:
    """Count each raster class within every polygonal zone.

    Parameters
    ----------
    zones : geopandas.GeoDataFrame
        Polygon zones. Its CRS must match the raster CRS. File-backed rasters
        assign their CRS when zones have none; this supports mesh polygons,
        whose coordinates are already in the raster CRS.
    raster : pathlib.Path, str, or numpy.ndarray
        Categorical raster path or a two-dimensional array. Arrays require
        ``affine`` and ``raster_crs``.
    all_touched : bool, default False
        Include every raster pixel touched by a zone instead of GDAL's default
        pixel-centre rule.
    affine : affine.Affine or sequence of float, optional
        Six-value geotransform for an array raster.
    raster_crs : pyproj.CRS or str, optional
        CRS for an array raster, or an explicit CRS override when a file-backed
        raster lacks spatial-reference metadata.
    nodata : float, optional
        Array value to exclude from the calculation.
    strategy : {'feature', 'raster'}, default 'feature'
        GDAL processing strategy.

    Returns
    -------
    pandas.DataFrame
        Integer class-code columns aligned to ``zones.index``. Missing classes
        have a count of zero. Zones without valid pixels have no counts.
    """
    if isinstance(raster, np.ndarray):
        raster_dataset = _array_to_mem_raster(
            raster, affine, raster_crs or zones.crs, nodata
        )
    else:
        raster_dataset = _open_file_raster(raster, raster_crs or zones.crs)

    with raster_dataset:
        if zones.crs is None and raster_dataset.GetSpatialRef() is not None:
            zones = zones.set_crs(raster_dataset.GetSpatialRef().ExportToWkt())
        records = _run_gdal_zonal_stats(
            zones=zones,
            raster_dataset=raster_dataset,
            statistics=("count", "unique", "frac"),
            pixel_mode=_pixel_mode(all_touched),
            strategy=strategy,
        )

    counts_by_zone = []
    for zone_id, record in enumerate(records):
        frequency_data = _unique_frac_counts(
            record, context=f" for zone ID {zone_id}"
        )
        if frequency_data is not None:
            _, unique, frequencies = frequency_data
            category_counts = {}
            for category, frequency in zip(unique, frequencies):
                category = float(category)
                if category.is_integer():
                    category = int(category)
                category_counts[category] = int(frequency)
            counts = category_counts
        else:
            counts = {}
        counts_by_zone.append(counts)

    columns = sorted({category for counts in counts_by_zone for category in counts})
    return (
        pd.DataFrame(counts_by_zone, index=zones.index)
        .reindex(columns=columns, fill_value=0)
        .fillna(0)
        .astype(np.int64)
    )


def _pixel_mode(all_touched: bool) -> Literal["default", "all-touched"]:
    """Map the public pixel-selection flag to GDAL's pixel mode.

    Parameters
    ----------
    all_touched : bool
        Whether all intersected pixels are included.

    Returns
    -------
    {'default', 'all-touched'}
        GDAL pixel mode corresponding to ``all_touched``.
    """
    return "all-touched" if all_touched else "default"


def _unique_frac_counts(
    record: dict, *, context: str = ""
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Reconstruct exact per-value pixel counts from count/unique/frac stats.

    Parameters
    ----------
    record : dict
        Decoded GDAL record containing ``count``, ``unique``, and ``frac``.
    context : str, default ''
        Text appended to errors to identify the source record.

    Returns
    -------
    tuple of (int, numpy.ndarray, numpy.ndarray) or None
        Valid-pixel count, unique values, and their integer frequencies; or
        ``None`` when GDAL reports no valid pixels.

    ``frac[i]`` is the fraction of valid zone pixels equal to ``unique[i]``.
    ``round(count * frac[i])`` recovers the exact per-value pixel count,
    which is enough information to reconstruct the fully expanded, sorted
    pixel list without ever materializing it. Returns ``None`` when the zone
    has no valid pixels.
    """
    count_value = record.get("count")
    unique_value = record.get("unique")
    frac_value = record.get("frac")
    if not count_value or unique_value is None or frac_value is None:
        return None

    count = round(float(count_value))
    unique = np.asarray(unique_value, dtype=np.float64)
    frequencies = np.rint(np.asarray(frac_value, dtype=np.float64) * count).astype(
        np.int64
    )

    if unique.size != frequencies.size:
        raise RuntimeError(
            f"GDAL returned unequal 'unique' and 'frac' lengths{context}."
        )
    if int(frequencies.sum()) != count:
        raise RuntimeError(
            "Could not recover exact per-value pixel counts from GDAL "
            f"fractions{context}: expected {count}, recovered "
            f"{int(frequencies.sum())}."
        )
    return count, unique, frequencies


def _median_from_unique_frac(record: dict) -> float | None:
    """Calculate a median from GDAL's unique values and frequencies.

    Parameters
    ----------
    record : dict
        Decoded GDAL record containing ``count``, ``unique``, and ``frac``.

    Returns
    -------
    float or None
        Median of the valid pixels, or ``None`` when there are no valid
        pixels.
    """
    counts = _unique_frac_counts(record)
    if counts is None:
        return None
    count, unique, frequencies = counts

    order = np.argsort(unique)
    sorted_values = unique[order]
    cumulative = np.cumsum(frequencies[order])
    lower_rank = (count - 1) // 2
    upper_rank = count // 2
    lower = sorted_values[np.searchsorted(cumulative, lower_rank, side="right")]
    upper = sorted_values[np.searchsorted(cumulative, upper_rank, side="right")]
    return float((lower + upper) / 2)


def _open_file_raster(path: Path | str, raster_crs=None):
    """Open a raster path and require a band and spatial reference.

    Parameters
    ----------
    path : pathlib.Path or str
        Raster file accepted by GDAL.
    raster_crs : pyproj.CRS or str, optional
        Explicit spatial reference used only when the raster file has no CRS.

    Returns
    -------
    osgeo.gdal.Dataset
        Read-only GDAL raster dataset.
    """
    dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
    if dataset is None:
        raise ValueError(f"Could not open raster '{path}'.")
    if dataset.RasterCount < 1:
        raise ValueError(f"Raster '{path}' must contain at least one band.")
    if dataset.GetSpatialRef() is None and raster_crs is not None:
        spatial_ref = _spatial_reference_from_crs(raster_crs, "Raster CRS override")
        dataset = gdal.Translate(
            "", str(path), format="VRT", outputSRS=spatial_ref.ExportToWkt()
        )
        if dataset is None:
            raise RuntimeError(f"Could not apply a CRS override to raster '{path}'.")
    return dataset


def _array_to_mem_raster(array: np.ndarray, affine, raster_crs, nodata):
    """Create a single-band GDAL memory raster from an array.

    Parameters
    ----------
    array : numpy.ndarray
        Two-dimensional source values.
    affine : affine.Affine or sequence of float
        Six-value geotransform for ``array``.
    raster_crs : pyproj.CRS or str
        Spatial reference for ``array``.
    nodata : float or None
        Value to mark as nodata on the output band.

    Returns
    -------
    osgeo.gdal.Dataset
        Single-band in-memory raster containing ``array``.
    """
    if array.ndim != 2:
        raise ValueError("An in-memory raster array must be two-dimensional.")
    if affine is None:
        raise ValueError("An in-memory raster requires an affine transform.")

    spatial_ref = (
        _spatial_reference_from_crs(raster_crs, "In-memory raster")
        if raster_crs is not None
        else None
    )
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    if gdal_type == gdal.GDT_Unknown:
        raise ValueError(f"Unsupported in-memory raster dtype '{array.dtype}'.")

    dataset = gdal.GetDriverByName("MEM").Create(
        "", array.shape[1], array.shape[0], 1, gdal_type
    )
    if dataset is None:
        raise RuntimeError("Could not create an in-memory GDAL raster.")

    geotransform = affine.to_gdal() if hasattr(affine, "to_gdal") else tuple(affine)
    if len(geotransform) != 6:
        raise ValueError("An affine transform must provide six GDAL values.")
    dataset.SetGeoTransform(geotransform)
    if spatial_ref is not None:
        dataset.SetSpatialRef(spatial_ref)

    band = dataset.GetRasterBand(1)
    band.WriteArray(np.ascontiguousarray(array))
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.FlushCache()
    return dataset


def _run_gdal_zonal_stats(
    *,
    zones: gpd.GeoDataFrame,
    raster_dataset,
    statistics: tuple[str, ...],
    pixel_mode: Literal["default", "all-touched", "fractional"],
    strategy: Literal["feature", "raster"],
) -> list[dict]:
    """Run GDAL zonal statistics and align decoded records to zone order.

    Parameters
    ----------
    zones : geopandas.GeoDataFrame
        Zones to copy into a GDAL in-memory vector dataset.
    raster_dataset : osgeo.gdal.Dataset
        Prepared source raster.
    statistics : tuple of str
        GDAL statistic names to request.
    pixel_mode : {'default', 'all-touched', 'fractional'}
        GDAL pixel-inclusion rule.
    strategy : {'feature', 'raster'}
        GDAL processing strategy.

    Returns
    -------
    list of dict
        Decoded statistic records in the input-zone order.
    """
    if strategy not in {"feature", "raster"}:
        raise ValueError(f"Unsupported GDAL zonal-statistics strategy '{strategy}'.")

    zone_srs = _validate_zones(zones)
    raster_srs = raster_dataset.GetSpatialRef()
    if zone_srs is not None and raster_srs is not None and not zone_srs.IsSame(raster_srs):
        raise ValueError("Zones and raster must use the same CRS.")

    with (
        _zones_to_mem_datasource(zones, zone_srs) as zones_dataset,
        gdal.Run(
            "raster zonal-stats",
            input=raster_dataset,
            zones=zones_dataset,
            zones_layer=_ZONE_LAYER,
            output_format="MEM",
            include_field=[_ZONE_ID_FIELD],
            band=[1],
            stat=list(statistics),
            pixels=pixel_mode,
            strategy=strategy,
            chunk_size="256 MB",
        ) as alg,
    ):
        output_dataset = alg.Output()
        return _read_output_records(output_dataset.GetLayer(0), len(zones), statistics)


def _validate_zones(zones: gpd.GeoDataFrame):
    """Validate zone container and CRS, then return its spatial reference.

    Parameters
    ----------
    zones : geopandas.GeoDataFrame
        Non-empty input zone table. CRS is optional.

    Returns
    -------
    osgeo.osr.SpatialReference or None
        GDAL spatial reference created from ``zones.crs``, when present.
    """
    if not isinstance(zones, gpd.GeoDataFrame):
        raise TypeError("zones must be a GeoDataFrame.")
    if zones.empty:
        raise ValueError("zones must contain at least one geometry.")
    return (
        _spatial_reference_from_crs(zones.crs, "Zones")
        if zones.crs is not None
        else None
    )


def _zones_to_mem_datasource(zones: gpd.GeoDataFrame, spatial_ref):
    """Copy zones and positional IDs into an in-memory OGR datasource.

    Parameters
    ----------
    zones : geopandas.GeoDataFrame
        Input zones to copy.
    spatial_ref : osgeo.osr.SpatialReference
        Spatial reference assigned to the output layer.

    Returns
    -------
    osgeo.ogr.DataSource
        In-memory datasource containing the ``zones`` layer and ``_zonal_id``.
    """
    driver = ogr.GetDriverByName("MEM")
    dataset = driver.CreateDataSource("")
    if dataset is None:
        raise RuntimeError("Could not create an in-memory GDAL zones datasource.")

    layer = dataset.CreateLayer(_ZONE_LAYER, spatial_ref, ogr.wkbUnknown)
    # WritePyArrow matches the geometry column by the geometry field's name;
    # a freshly created layer's default geometry field has no name.
    layer.AlterGeomFieldDefn(
        0,
        ogr.GeomFieldDefn(_ZONE_GEOMETRY_FIELD, ogr.wkbUnknown),
        ogr.ALTER_GEOM_FIELD_DEFN_NAME_FLAG,
    )
    layer.CreateField(ogr.FieldDefn(_ZONE_ID_FIELD, ogr.OFTInteger64))

    minimal_zones = gpd.GeoDataFrame(
        {_ZONE_ID_FIELD: np.arange(len(zones), dtype=np.int64)},
        geometry=zones.geometry.to_numpy(),
        crs=zones.crs,
    )
    table = pa.table(minimal_zones.to_arrow(geometry_encoding="WKB"))
    if layer.WritePyArrow(table) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not write zones to GDAL.")
    return dataset


def _read_output_records(
    layer, zone_count: int, statistics: tuple[str, ...]
) -> list[dict]:
    """Decode GDAL output and restore the original zone order.

    Parameters
    ----------
    layer : osgeo.ogr.Layer
        GDAL output layer containing ``_zonal_id`` and requested statistics.
    zone_count : int
        Number of zones supplied to GDAL.
    statistics : tuple of str
        Statistic field names to read.

    Returns
    -------
    list of dict
        One decoded statistics mapping per input zone.
    """
    stream = layer.GetArrowStreamAsPyArrow()
    batches = [pa.RecordBatch.from_struct_array(array) for array in stream]
    table = pa.Table.from_batches(batches) if batches else None

    records: dict[int, dict] = {}
    if table is not None and table.num_rows:
        zone_ids = table.column(_ZONE_ID_FIELD).to_pylist()
        columns = {stat: table.column(stat).to_pylist() for stat in statistics}
        for row, zone_id in enumerate(zone_ids):
            if zone_id is None:
                raise RuntimeError(f"GDAL zonal-statistics output lacks '{_ZONE_ID_FIELD}'.")
            zone_id = int(zone_id)
            if zone_id in records:
                raise RuntimeError(f"GDAL returned duplicate zone ID {zone_id}.")
            if not 0 <= zone_id < zone_count:
                raise RuntimeError(f"GDAL returned invalid zone ID {zone_id}.")
            records[zone_id] = {stat: columns[stat][row] for stat in statistics}

    missing = sorted(set(range(zone_count)) - records.keys())
    if missing:
        raise RuntimeError(f"GDAL did not return zone IDs {missing}.")
    return [records[position] for position in range(zone_count)]


def _spatial_reference_from_crs(crs, source_name: str):
    """Convert a CRS-like value into a GDAL spatial reference.

    Parameters
    ----------
    crs : pyproj.CRS or str
        CRS value accepted by GDAL's user-input parser.
    source_name : str
        Human-readable input name used in validation errors.

    Returns
    -------
    osgeo.osr.SpatialReference
        Parsed GDAL spatial reference.
    """
    if crs is None:
        raise ValueError(f"{source_name} must define a CRS.")
    value = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
    spatial_ref = osr.SpatialReference()
    if spatial_ref.SetFromUserInput(value) != ogr.OGRERR_NONE:
        raise ValueError(f"{source_name} has an invalid CRS '{crs}'.")
    return spatial_ref
