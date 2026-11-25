"""Derive the RR nodes, meteo and boundaries of a D-RR model from catchments and rasters."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pydantic import ConfigDict, StrictStr, validate_call
from shapely.geometry import Point

from hydrolib.dhydamo.core import drr
from hydrolib.dhydamo.core.relations import RelationsRR
from hydrolib.dhydamo.geometry import zonal
from hydrolib.dhydamo.io import idfreader
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame

logger = logging.getLogger(__name__)

NO_RASTERDATA_WARNING = "No rasterdata available for catchment %s."


def _raster_metadata(raster: Path | str) -> tuple[float, rasterio.crs.CRS]:
    """Return pixel area and CRS declared by a raster file.

    Parameters
    ----------
    raster : pathlib.Path or str
        Path to the raster whose transform and CRS are inspected.

    Returns
    -------
    tuple of (float, rasterio.crs.CRS)
        Absolute pixel area in the raster coordinate system and its CRS.

    Raises
    ------
    ValueError
        If the raster has no coordinate reference system.
    """
    with rasterio.open(raster) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Raster '{raster}' must define a CRS.")
        transform = dataset.transform
        metadata = (
            abs(transform.a * transform.e - transform.b * transform.d),
            dataset.crs,
        )
    return metadata


def _meteo_area(meteo_areas: gpd.GeoDataFrame, point: Point) -> str:
    """Return the code of the first meteo area that contains the point, otherwise of the first area."""
    containing = meteo_areas.loc[meteo_areas.geometry.contains(point)]
    if containing.empty:
        return str(meteo_areas.iloc[0, 0])
    return str(containing["code"].iloc[0])


class UnpavedIO:
    """Fill the unpaved nodes of an RR model from catchments and rasters.

    Parameters
    ----------
    unpaved : drr.Unpaved
        The unpaved nodes of the RR model, filled in place.
    """

    def __init__(self, unpaved: drr.Unpaved) -> None:
        self.unpaved = unpaved

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def unpaved_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: StrictStr | Path,
        surface_level: StrictStr | Path,
        soiltype: StrictStr | Path,
        surface_storage: StrictStr | Path | float,
        infiltration_capacity: StrictStr | Path | float,
        initial_gwd: StrictStr | Path | float,
        meteo_areas: ExtendedGeoDataFrame,
        greenhouse_areas: ExtendedGeoDataFrame | None = None,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add an unpaved node per catchment, with values from rasters.

        Parameters
        ----------
        catchments : ExtendedGeoDataFrame
            Catchment areas.
        landuse : str or Path
            Land use raster.
        surface_level : str or Path
            Surface level raster.
        soiltype : str or Path
            Soil type raster.
        surface_storage : str, Path or float
            Surface storage (mm): a number for a uniform value, or a raster.
        infiltration_capacity : str, Path or float
            Infiltration capacity (mm/d): a number for a uniform value, or a raster.
        initial_gwd : str, Path or float
            Initial groundwater depth (m): a number for a uniform value, or a raster.
        meteo_areas : ExtendedGeoDataFrame
            Meteo areas; each catchment gets the meteo station of the area that contains
            its centroid.
        greenhouse_areas : ExtendedGeoDataFrame, optional
            Greenhouse areas. Where they are not greenhouse in the land use raster, their
            area is subtracted from the most common land use of the catchment. Defaults
            to None.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.
        """
        px_area, reference_crs = _raster_metadata(landuse)

        # required rasters
        lu_counts = self.unpaved.drrmodel.zonal_cache.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )

        soiltypes = zonal.zonal_stats(
            gpd.GeoDataFrame(catchments),
            soiltype,
            statistics=("mode",),
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )

        stats_elev = self.unpaved.drrmodel.zonal_cache.zonal_stats(
            gpd.GeoDataFrame(catchments),
            surface_level,
            statistics=("median",),
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )

        # optional rasters
        if isinstance(surface_storage, (Path, str)):
            sstores = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                surface_storage,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )
        elif isinstance(surface_storage, int):
            surface_storage = float(surface_storage)
        if isinstance(infiltration_capacity, (Path, str)):
            infcaps = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                infiltration_capacity,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )
        elif isinstance(infiltration_capacity, int):
            infiltration_capacity = float(infiltration_capacity)
        if isinstance(initial_gwd, (Path, str)):
            ini_gwds = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                initial_gwd,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )
        elif isinstance(initial_gwd, int):
            initial_gwd = float(initial_gwd)

        # HyDAMO Crop code; hydamo name, sobek index, sobek name:
        # 1 aardappelen   3 potatoes
        # 2 graan         5 grain
        # 3 suikerbiet    4 sugarbeet
        # 4 mais          2 corn
        # 5 overige gew. 15 vegetables
        # 6 bloembollen  10 bulbous plants
        # 7 boomgaard     9 orchard
        # 8 gras          1 grass
        # 9 loofbos      11 dediduous
        # 10 naaldbos    12 conferous
        # 11 natuuur     13 nature
        # 12 braak       14 fallow
        sobek_indices = [3, 5, 4, 2, 15, 10, 9, 1, 11, 12, 13, 14]
        for num, (cat, cat_geometry) in enumerate(
            zip(catchments.itertuples(), catchments.geometry)
        ):
            # if no rasterdata could be obtained for this catchment, skip it.
            if pd.isna(stats_elev.iloc[num]["median"]):
                logger.warning(NO_RASTERDATA_WARNING, cat.code)
                self.unpaved.add_unpaved(
                    id="0.0",
                    total_area="0.0",
                    lu_areas="0.0",
                    surface_level="0.0",
                    soiltype="0.0",
                    surface_storage="0.0",
                    infiltration_capacity="0.0",
                    initial_gwd="0.0",
                    meteo_area="0.0",
                    px="0.0",
                    py="0.0",
                    boundary_node="0.0",
                )
                continue
            ms = _meteo_area(meteo_areas, cat_geometry.centroid)
            landuse_counts = lu_counts.iloc[num].copy()
            mapping = np.zeros(16, dtype=int)

            # subtract greenhouse area from most occurring land use if no greenhouse area is in the landuse map
            if (
                greenhouse_areas is not None
                and cat_geometry.intersects(greenhouse_areas.geometry).any()
            ):
                intersection_area = greenhouse_areas.geometry.intersection(
                    cat_geometry
                ).area
                intersection_area = intersection_area[
                    intersection_area > 0.0
                ].to_numpy()[0]
                if landuse_counts.get(15, 0) > 0:
                    # divide area to subtract between greenhouses and the most occurring area
                    remainder = np.max(
                        [0.0, intersection_area - float(landuse_counts[15] * px_area)]
                    )
                else:
                    remainder = intersection_area
                maxind = landuse_counts.idxmax()
                logger.info(
                    "Catchment %s: subtracting %s m2 from class %s for supplied greenhouse area.",
                    cat.code,
                    remainder,
                    maxind,
                )
                landuse_counts[maxind] = np.max(
                    [0.0, landuse_counts[maxind] - np.round(remainder / px_area)]
                )

            for i in range(1, 13):
                mapping[sobek_indices[i - 1] - 1] = landuse_counts.get(i, 0) * px_area
            lu_map = " ".join(map(str, mapping))
            elev = stats_elev.iloc[num]["median"]
            self.unpaved.add_unpaved(
                id=str(cat.code),
                total_area=f"{cat_geometry.area:.0f}",
                lu_areas=lu_map,
                surface_level=f"{elev:.2f}",
                soiltype=f"{soiltypes.iloc[num]['mode'] + 100.0:.0f}",
                surface_storage=(
                    f"{surface_storage:.3f}"
                    if isinstance(surface_storage, float)
                    else f"{sstores.iloc[num]['mean']:.3f}"
                ),
                infiltration_capacity=(
                    f"{infiltration_capacity:.3f}"
                    if isinstance(infiltration_capacity, float)
                    else f"{infcaps.iloc[num]['mean']:.3f}"
                ),
                initial_gwd=(
                    f"{initial_gwd:.2f}"
                    if isinstance(initial_gwd, float)
                    else f"{ini_gwds.iloc[num]['mean']:.2f}"
                ),
                meteo_area=str(ms),
                px=f"{cat_geometry.centroid.coords[0][0] - 10:.0f}",
                py=f"{cat_geometry.centroid.coords[0][1]:.0f}",
                boundary_node=str(cat.boundary_node),
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def ernst_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        depths: list,
        resistance: list,
        infiltration_resistance: float | None = None,
        runoff_resistance: float | None = None,
    ) -> None:
        """Add the same Ernst definition for the unpaved node of each catchment.

        Parameters
        ----------
        catchments : ExtendedGeoDataFrame
            Catchment areas.
        depths : list
            Layer depths (m).
        resistance : list
            Ernst resistance of each layer (d).
        infiltration_resistance : float, optional
            Resistance to infiltration (d). Defaults to None: 300.
        runoff_resistance : float, optional
            Resistance to surface runoff (d). Defaults to None: 1.
        """
        if infiltration_resistance is None:
            infiltration_resistance = 300.0
        if runoff_resistance is None:
            runoff_resistance = 1.0

        cvo = " ".join([str(res) for res in resistance])
        lv = " ".join([str(depth) for depth in depths])
        cvi = f"{infiltration_resistance:.2f}"
        cvs = f"{runoff_resistance:.2f}"
        for cat in catchments.itertuples():
            self.unpaved.add_ernst_def(
                id=str(cat.code), cvo=cvo, lv=lv, cvi=cvi, cvs=cvs
            )


class PavedIO:
    """Fill the paved nodes of an RR model from catchments, sewer areas and rasters.

    Parameters
    ----------
    paved : drr.Paved
        The paved nodes of the RR model, filled in place.
    """

    def __init__(self, paved: drr.Paved) -> None:
        self.paved = paved

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def paved_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: StrictStr | Path,
        surface_level: StrictStr | Path,
        street_storage: StrictStr | Path | float,
        sewer_storage: StrictStr | Path | float,
        pump_capacity: StrictStr | Path | float,
        meteo_areas: ExtendedGeoDataFrame,
        overflows: ExtendedGeoDataFrame | None = None,
        sewer_areas: ExtendedGeoDataFrame | None = None,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add the paved nodes: one per overflow of each sewer area, and one per catchment.

        The paved area of a catchment outside the sewer areas gets a node that drains to
        the catchment's boundary node.

        Parameters
        ----------
        catchments : ExtendedGeoDataFrame
            Catchment areas.
        landuse : str or Path
            Land use raster.
        surface_level : str or Path
            Surface level raster.
        street_storage : str, Path or float
            Street storage (mm): a number for a uniform value, or a raster.
        sewer_storage : str, Path or float
            Sewer storage (mm): a number for a uniform value, or a raster.
        pump_capacity : str, Path or float
            Pump capacity (mm/h): a number for a uniform value, or a raster.
        meteo_areas : ExtendedGeoDataFrame
            Meteo areas; each node gets the meteo station of the area that contains its
            centroid.
        overflows : ExtendedGeoDataFrame, optional
            Overflow locations, required with ``sewer_areas``. Defaults to None.
        sewer_areas : ExtendedGeoDataFrame, optional
            Sewer areas. Defaults to None.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.

        Raises
        ------
        ValueError
            If ``sewer_areas`` is given without ``overflows``.
        """
        px_area, reference_crs = _raster_metadata(landuse)

        lu_counts = self.paved.drrmodel.zonal_cache.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )
        stats_elev = self.paved.drrmodel.zonal_cache.zonal_stats(
            gpd.GeoDataFrame(catchments),
            surface_level,
            statistics=("median",),
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )

        if isinstance(street_storage, (Path, str)):
            str_stors = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                street_storage,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )

        if isinstance(sewer_storage, (Path, str)):
            sew_stors = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                sewer_storage,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )

        if isinstance(pump_capacity, (Path, str)):
            # raster of POC in mm/h
            pump_caps = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                pump_capacity,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )
        paved_columns = [
            "id",
            "area",
            "surface_level",
            "street_storage",
            "sewer_storage",
            "pump_capacity",
            "meteo_area",
            "px",
            "py",
            "boundary_node",
        ]
        if sewer_areas is not None:
            # if the parameters ara rasters, do the zonal statistics per sewage area as well.
            if isinstance(street_storage, (Path, str)):
                str_stors_sa = zonal.zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    street_storage,
                    statistics=("mean",),
                    min_coverage_frac=min_coverage_frac,
                    coverage_weight=coverage_weight,
                    strategy="feature-sequential",
                    raster_crs=reference_crs,
                )
            if isinstance(sewer_storage, (Path, str)):
                sew_stors_sa = zonal.zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    sewer_storage,
                    statistics=("mean",),
                    min_coverage_frac=min_coverage_frac,
                    coverage_weight=coverage_weight,
                    strategy="feature-sequential",
                    raster_crs=reference_crs,
                )
            if isinstance(pump_capacity, (Path, str)):
                pump_caps_sa = zonal.zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    pump_capacity,
                    statistics=("mean",),
                    min_coverage_frac=min_coverage_frac,
                    coverage_weight=coverage_weight,
                    strategy="feature-sequential",
                    raster_crs=reference_crs,
                )
            stats_sa_elev = zonal.zonal_stats(
                gpd.GeoDataFrame(sewer_areas),
                surface_level,
                statistics=("median",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
            )
            sewer_lu_counts = zonal.zonal_category_counts(
                gpd.GeoDataFrame(sewer_areas),
                landuse,
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
            )

            if overflows is None:
                raise ValueError("overflows is required with sewer_areas.")
            # find the paved area in the sewer areas
            for isew, (sew, sew_geometry) in enumerate(
                zip(sewer_areas.itertuples(), sewer_areas.geometry)
            ):
                sewer_pav_area = 0
                pixels = sewer_lu_counts.iloc[isew]
                pav_pixels = pixels.get(14, 0)
                if pav_pixels == 0:
                    logger.warning("No paved area in sewer area %s.", sew.code)
                    self.paved.add_paved(**dict.fromkeys(paved_columns, "0.0"))
                    continue
                sewer_pav_area += pav_pixels * px_area

                # subtract it fromthe total paved area in this catchment, make sure at least 0 remains
                # lu_counts[cat_ind][14.0] -=  pav_pixels
                # if lu_counts[cat_ind][14.0] < 0: lu_counts[cat_ind][14.0]  = 0

                elev = stats_sa_elev.iloc[isew]["median"]
                # find overflows related to this sewer area
                overflow_sa = RelationsRR.OVERFLOW_SEWER_AREA
                sewer_id = getattr(sew, overflow_sa.parent_key)
                ovf = overflows[overflows[overflow_sa.foreign_key] == sewer_id]
                for ov, ov_geometry in zip(ovf.itertuples(), ovf.geometry):
                    # find corresponding meteo-station
                    ms = _meteo_area(meteo_areas, sew_geometry.centroid)

                    # if a float is given, a standard value is passed. If a string is given, a rastername is assumed to zonal statistics are applied.
                    if isinstance(street_storage, float):
                        street_storage_val = f"{street_storage:.2f}"
                    elif isinstance(street_storage, (Path, str)):
                        street_storage_val = f"{str_stors_sa.iloc[isew]['mean']:.2f}"
                    else:
                        raise TypeError(
                            "Street_storage has the wrong datatype. It should be a filename (Path or string) or number (float or int)."
                        )

                    # three options: it can be an attribute of a sewer area, a uniform value or a raster
                    riool_berging_mm = getattr(sew, "riool_berging_mm", None)
                    if (
                        riool_berging_mm is None
                        or np.isnan(riool_berging_mm)
                        or not isinstance(riool_berging_mm, float)
                    ):
                        if isinstance(sewer_storage, float):
                            sewer_storage_val = f"{sewer_storage:.2f}"
                        elif isinstance(sewer_storage, (Path, str)):
                            sewer_storage_val = f"{sew_stors_sa.iloc[isew]['mean']:.2f}"
                        else:
                            raise TypeError(
                                "Sewer_storage has the wrong datatype. It should be a filename (Path or string) or number (float or int)."
                            )
                    else:
                        sewer_storage_val = f"{riool_berging_mm:.2f}"

                    # three options: it can be an attribute of a sewer area, a uniform value or a raster
                    riool_poc_m3s = getattr(sew, "riool_poc_m3s", None)
                    if (
                        riool_poc_m3s is None
                        or np.isnan(riool_poc_m3s)
                        or not isinstance(riool_poc_m3s, float)
                    ):
                        if isinstance(pump_capacity, float):
                            # convert the value from mm/h to m3/s
                            pump_capacity_val = f"{pump_capacity * (float(sewer_pav_area) * ov.fractie) / (1000.0 * 3600.0):.8f}"
                        elif isinstance(pump_capacity, (Path, str)):
                            # convert the value (extracted from the raster) from mm/h to m3/s
                            pump_capacity_val = f"{pump_caps_sa.iloc[isew]['mean'] * (float(sewer_pav_area) * ov.fractie) / (1000.0 * 3600.0):.8f}"
                        else:
                            raise TypeError(
                                "Pump_capacity has the wrong datatype. It should be a filename (Path or string) or number (float or int)."
                            )
                    else:
                        # use the attribute value
                        pump_capacity_val = f"{riool_poc_m3s * ov.fractie:.8f}"

                    # add prefix to the overflow id to create the paved-node id
                    self.paved.add_paved(
                        id=str(ov.code),
                        area=str(sewer_pav_area * ov.fractie),
                        surface_level=f"{elev:.2f}",
                        street_storage=street_storage_val,
                        sewer_storage=sewer_storage_val,
                        pump_capacity=pump_capacity_val,
                        meteo_area=str(ms),
                        px=f"{ov_geometry.coords[0][0] + 10:.0f}",
                        py=f"{ov_geometry.coords[0][1]:.0f}",
                        boundary_node=ov.code,
                    )

        if sewer_areas is not None:
            # Land use of the catchment parts outside the sewer areas, by
            # catchment position. Catchments that overlap no sewer area use
            # lu_counts.
            sewer_union = sewer_areas.union_all()
            # GeoSeries(...): the stubs type reset_index as a plain Series
            geometries = gpd.GeoSeries(catchments.geometry.reset_index(drop=True))
            overlaps = geometries.intersects(sewer_union)
            # the stubs type a masked GeoSeries as Series[BaseGeometry]; it is a GeoSeries
            outside = geometries[overlaps].difference(sewer_union)  # type: ignore[operator]
            outside = outside[outside.area > 0]
            outside_counts = pd.DataFrame()
            if not outside.empty:
                outside_counts = zonal.zonal_category_counts(
                    outside,
                    landuse,
                    min_coverage_frac=min_coverage_frac,
                    coverage_weight=coverage_weight,
                    strategy="feature-sequential",
                )

        for num, (cat, cat_geometry) in enumerate(
            zip(catchments.itertuples(), catchments.geometry)
        ):
            # if no rasterdata could be obtained for this catchment, skip it.
            if pd.isna(stats_elev.iloc[num]["median"]):
                logger.warning(NO_RASTERDATA_WARNING, cat.code)
                self.paved.add_paved(**dict.fromkeys(paved_columns, "0.0"))
                continue
            if sewer_areas is None or not overlaps[num]:
                pav_area = str(lu_counts.iloc[num].get(14, 0) * px_area)
            elif num in outside_counts.index:
                pav_area = str(outside_counts.loc[num].get(14, 0) * px_area)
            else:
                logger.info(
                    f"No paved area outside sewer area in catchments {cat.code}."
                )
                pav_area = "0.0"

            # find corresponding meteo-station
            ms = _meteo_area(meteo_areas, cat_geometry.centroid)

            elev = stats_elev.iloc[num]["median"]
            # if a float is given, a standard value is passed. If a string is given, a rastername is assumed to zonal statistics are applied.
            street_storage_val = (
                f"{street_storage:.2f}"
                if isinstance(street_storage, float)
                else f"{str_stors.iloc[num]['mean']:.2f}"
            )
            sewer_storage_val = (
                f"{sewer_storage:.2f}"
                if isinstance(sewer_storage, float)
                else f"{sew_stors.iloc[num]['mean']:.2f}"
            )
            pump_capacity_val = (
                f"{(pump_capacity * float(pav_area)) / (1000.0 * 3600.0):.8f}"
                if isinstance(pump_capacity, float)
                else f"{pump_caps.iloc[num]['mean'] * (float(pav_area)) / (1000.0 * 3600.0):.8f}"
            )
            self.paved.add_paved(
                id=str(cat.code),
                area=str(pav_area),
                surface_level=f"{elev:.2f}",
                street_storage=street_storage_val,
                sewer_storage=sewer_storage_val,
                pump_capacity=pump_capacity_val,
                meteo_area=str(ms),
                px=f"{cat_geometry.centroid.coords[0][0] + 10:.0f}",
                py=f"{cat_geometry.centroid.coords[0][1]:.0f}",
                boundary_node=str(cat.boundary_node),
            )


class GreenhouseIO:
    """Fill the greenhouse nodes of an RR model from catchments, greenhouse areas and rasters.

    Parameters
    ----------
    greenhouse : drr.Greenhouse
        The greenhouse nodes of the RR model, filled in place.
    """

    def __init__(self, greenhouse: drr.Greenhouse) -> None:
        self.greenhouse = greenhouse

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def greenhouse_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Path | str,
        surface_level: Path | str,
        roof_storage: StrictStr | float,
        meteo_areas: ExtendedGeoDataFrame,
        greenhouse_areas: ExtendedGeoDataFrame | None = None,
        greenhouse_laterals: ExtendedGeoDataFrame | None = None,
        basin_storage_class: int = 2,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add the greenhouse nodes: one per greenhouse area, and one per catchment.

        The catchment node gets the greenhouse area of the land use raster, minus what
        is already in the greenhouse areas.

        Parameters
        ----------
        catchments : ExtendedGeoDataFrame
            Catchment areas.
        landuse : str or Path
            Land use raster.
        surface_level : str or Path
            Surface level raster.
        roof_storage : str or float
            Roof storage (mm): a number for a uniform value, or a raster.
        meteo_areas : ExtendedGeoDataFrame
            Meteo areas; each node gets the meteo station of the area that contains its
            centroid.
        greenhouse_areas : ExtendedGeoDataFrame, optional
            Known greenhouse areas, optionally with a ``basin_storage_class`` column.
            Defaults to None.
        greenhouse_laterals : ExtendedGeoDataFrame, optional
            Outlet points of the greenhouse areas, required with ``greenhouse_areas``.
            Defaults to None.
        basin_storage_class : int, optional
            Basin storage class of greenhouses without their own. Defaults to 2.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.

        Raises
        ------
        ValueError
            If ``greenhouse_areas`` is given without ``greenhouse_laterals``.
        """
        px_area, reference_crs = _raster_metadata(landuse)

        lu_counts = self.greenhouse.drrmodel.zonal_cache.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )
        stats_elev = self.greenhouse.drrmodel.zonal_cache.zonal_stats(
            gpd.GeoDataFrame(catchments),
            surface_level,
            statistics=("median",),
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )
        if greenhouse_areas is not None:
            stats_elev_gh = zonal.zonal_stats(
                gpd.GeoDataFrame(greenhouse_areas),
                surface_level,
                statistics=("median",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
            )

        # optional rasters
        if isinstance(roof_storage, (Path, str)):
            roofstors = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                roof_storage,
                statistics=("mean",),
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                raster_crs=reference_crs,
            )
            if greenhouse_areas is not None:
                roofstors_gh = zonal.zonal_stats(
                    gpd.GeoDataFrame(greenhouse_areas),
                    roof_storage,
                    statistics=("mean",),
                    min_coverage_frac=min_coverage_frac,
                    coverage_weight=coverage_weight,
                    strategy="feature-sequential",
                    raster_crs=reference_crs,
                )

        gh_columns = [
            "id",
            "area",
            "surface_level",
            "roof_storage",
            "basin_storage_class",
            "meteo_area",
            "px",
            "py",
            "boundary_node",
        ]
        if greenhouse_areas is not None:
            if greenhouse_laterals is None:
                raise ValueError(
                    "greenhouse_laterals is required with greenhouse_areas."
                )
            for num, (gh, gh_geometry) in enumerate(
                zip(greenhouse_areas.itertuples(), greenhouse_areas.geometry)
            ):
                # find corresponding meteo-station
                if pd.isna(stats_elev_gh.iloc[num]["median"]):
                    logger.warning(NO_RASTERDATA_WARNING, gh.code)
                    self.greenhouse.add_greenhouse(**dict.fromkeys(gh_columns, "0.0"))
                    continue
                ms = _meteo_area(meteo_areas, gh_geometry.centroid)

                elev = stats_elev_gh.iloc[num]["median"]
                # optional columns: a missing column counts as no value
                roof_storage_mm = getattr(gh, "roof_storage_mm", np.nan)
                gh_basin_storage_class = getattr(gh, "basin_storage_class", np.nan)
                if not pd.isna(roof_storage_mm):
                    roof_storage_val = f"{roof_storage_mm:.2f}"
                elif isinstance(roof_storage, float):
                    roof_storage_val = f"{roof_storage:.2f}"
                else:
                    roof_storage_val = f"{roofstors_gh.iloc[num]['mean']:.2f}"
                if not pd.isna(gh_basin_storage_class):
                    basin_storage_class_val = f"{gh_basin_storage_class:g}"
                else:
                    basin_storage_class_val = f"{basin_storage_class:g}"
                gh_area = RelationsRR.GREENHOUSE_LATERAL_AREA
                area_id = getattr(gh, gh_area.parent_key)
                related_laterals = greenhouse_laterals[
                    greenhouse_laterals[gh_area.foreign_key] == area_id
                ]
                latcode = related_laterals.code.to_numpy()[0]
                self.greenhouse.add_greenhouse(
                    id=str(gh.code),
                    area=f"{gh_geometry.area:.2f}",
                    surface_level=f"{elev:.2f}",
                    roof_storage=roof_storage_val,
                    basin_storage_class=basin_storage_class_val,
                    meteo_area=str(ms),
                    px=f"{gh_geometry.centroid.coords[0][0]:.0f}",
                    py=f"{gh_geometry.centroid.coords[0][1]:.0f}",
                    boundary_node=str(latcode),
                )

        for num, (cat, cat_geometry) in enumerate(
            zip(catchments.itertuples(), catchments.geometry)
        ):
            # if no rasterdata could be obtained for this catchment, skip it.
            if pd.isna(stats_elev.iloc[num]["median"]):
                logger.warning(NO_RASTERDATA_WARNING, cat.code)
                self.greenhouse.add_greenhouse(**dict.fromkeys(gh_columns, "0.0"))
                continue

            # find corresponding meteo-station
            ms = _meteo_area(meteo_areas, cat_geometry.centroid)

            if (
                greenhouse_areas is not None
                and cat_geometry.intersects(greenhouse_areas.geometry).any()
            ):
                intersection_area = greenhouse_areas.geometry.intersection(
                    cat_geometry
                ).area
                intersection_area = intersection_area[
                    intersection_area > 0.0
                ].to_numpy()[0]
                landuse_counts = lu_counts.iloc[num]
                if landuse_counts.get(15, 0) > 0:
                    # divide area to subtract between greenhouses and the most occurring area
                    logger.info(
                        "Catchment: %s: subtracting %s m2 from greenhouse area in landuse map.",
                        cat.code,
                        np.min([(landuse_counts[15] * px_area, intersection_area)]),
                    )
                    lu_counts.loc[lu_counts.index[num], 15] = np.max(
                        [
                            0.0,
                            landuse_counts[15] - np.round(intersection_area / px_area),
                        ]
                    )

            elev = stats_elev.iloc[num]["median"]
            roof_storage_val = (
                f"{roof_storage:.2f}"
                if isinstance(roof_storage, float)
                else f"{roofstors.iloc[num]['mean']:.2f}"
            )
            self.greenhouse.add_greenhouse(
                id=str(cat.code),
                area=(str(lu_counts.iloc[num].get(15, 0) * px_area)),
                surface_level=f"{elev:.2f}",
                roof_storage=roof_storage_val,
                basin_storage_class=f"{basin_storage_class:g}",
                meteo_area=str(ms),
                px=f"{cat_geometry.centroid.coords[0][0] + 20:.0f}",
                py=f"{cat_geometry.centroid.coords[0][1]:.0f}",
                boundary_node=str(cat.boundary_node),
            )


class OpenwaterIO:
    """Fill the open water nodes of an RR model from catchments and a land use raster.

    Parameters
    ----------
    openwater : drr.Openwater
        The open water nodes of the RR model, filled in place.
    """

    def __init__(self, openwater: drr.Openwater) -> None:
        self.openwater = openwater

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def openwater_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Path | str,
        meteo_areas: ExtendedGeoDataFrame,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add an open water node per catchment, with the open water area of the land use raster.

        Parameters
        ----------
        catchments : ExtendedGeoDataFrame
            Catchment areas.
        landuse : str or Path
            Land use raster.
        meteo_areas : ExtendedGeoDataFrame
            Meteo areas; each node gets the meteo station of the area that contains its
            centroid.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.
        """
        px_area, _ = _raster_metadata(landuse)

        lu_counts = self.openwater.drrmodel.zonal_cache.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
        )

        for num, (cat, cat_geometry) in enumerate(
            zip(catchments.itertuples(), catchments.geometry)
        ):
            # find corresponding meteo-station
            ms = _meteo_area(meteo_areas, cat_geometry.centroid)

            self.openwater.add_openwater(
                id=str(cat.code),
                area=str(lu_counts.iloc[num].get(13, 0) * px_area),
                meteo_area=str(ms),
                px=f"{cat_geometry.centroid.coords[0][0] - 20:.0f}",
                py=f"{cat_geometry.centroid.coords[0][1]:.0f}",
                boundary_node=str(cat.boundary_node),
            )


class ExternalForcingsIO:
    """Fill the meteo, seepage and boundary nodes of an RR model.

    Parameters
    ----------
    external_forcings : drr.ExternalForcings
        The external forcings of the RR model, filled in place.
    """

    def __init__(self, external_forcings: drr.ExternalForcings) -> None:
        self.external_forcings = external_forcings

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def seepage_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        seepage_folder: Path | str,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add a seepage time series per catchment, the mean of each raster in a folder.

        The time of each step is read from the raster (IDF or netCDF).

        Parameters
        ----------
        catchments : ExtendedGeoDataFrame
            Catchment areas.
        seepage_folder : str or Path
            Folder with the seepage rasters, one per time step.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.
        """
        rasters: list[zonal.Raster] = []
        times: list[pd.Timestamp] = []
        affine = raster_crs = nodata = None
        # sorted: the time series follows the order of the files
        for path in sorted(Path(seepage_folder).iterdir()):
            if path.suffix == ".idf":
                # Lazy: zonal_stats_per_raster reads it when its turn comes.
                dataset = idfreader.open(path).squeeze()
                if affine is None:
                    affine = idfreader.affine_from_idf(dataset)
                    raster_crs = dataset.attrs.get("crs")
                    nodata = dataset.attrs.get("nodata", np.nan)
                elif idfreader.affine_from_idf(dataset) != affine:
                    raise ValueError(
                        f"IDF seepage file {path.name} differs in grid from the others."
                    )
                rasters.append(dataset)
                # the one time of the file
                times.append(pd.DatetimeIndex(dataset["time"].to_numpy().ravel())[0])
            else:
                rasters.append(path)
                times.append(self.external_forcings.drrmodel._time_from_filename(path))
        stats = zonal.zonal_stats_per_raster(
            gpd.GeoDataFrame(catchments),
            rasters,
            "mean",
            min_coverage_frac=min_coverage_frac,
            coverage_weight=coverage_weight,
            strategy="feature-sequential",
            affine=affine,
            raster_crs=raster_crs,
            nodata=nodata,
            progress=True,
        )
        result = pd.DataFrame(
            stats.to_numpy().T,
            index=times,
            columns=["sep_" + str(cat) for cat in catchments.code],
        )
        if affine is not None:
            # if an NHI model (IDF files) is used, convert units from m3 to mm/d
            result = (result / (1e-3 * (affine[0] * -affine[4]))) / (
                (times[2] - times[1]).total_seconds() / 86400.0
            )
        for column, series in result.items():
            self.external_forcings.add_seepage(str(column), series)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def precip_from_input(
        self,
        areas: ExtendedGeoDataFrame,
        precip_folder: Path | str | None = None,
        precip_file: Path | str | None = None,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add a precipitation time series per meteo area, or use an existing meteo file.

        Parameters
        ----------
        areas : ExtendedGeoDataFrame
            Meteo areas for which time series are created.
        precip_folder : str or Path, optional
            Folder with precipitation rasters, one per time step. Only used without
            ``precip_file``. Defaults to None.
        precip_file : str or Path, optional
            Existing meteo file (``.bui``), used instead of the rasters. Defaults to None.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.

        Raises
        ------
        ValueError
            If neither ``precip_folder`` nor ``precip_file`` is given.
        """
        if precip_file is None:
            if precip_folder is None:
                raise ValueError("Provide precip_folder or precip_file.")
            # sorted: the time series follows the order of the files
            paths = sorted(Path(precip_folder).iterdir())
            stats = zonal.zonal_stats_per_raster(
                gpd.GeoDataFrame(areas),
                paths,
                "mean",
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                progress=True,
            )
            result = pd.DataFrame(
                stats.to_numpy().T,
                index=[
                    self.external_forcings.drrmodel._time_from_filename(path)
                    for path in paths
                ],
                columns=["ms_" + str(area) for area in areas.code],
            )
            for column, series in result.items():
                self.external_forcings.add_precip(str(column), series)
        else:
            self.external_forcings.precip = str(precip_file)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def evap_from_input(
        self,
        areas: ExtendedGeoDataFrame,
        evap_folder: Path | str | None = None,
        evap_file: Path | str | None = None,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Add an evaporation time series per meteo area, or use an existing meteo file.

        Parameters
        ----------
        areas : ExtendedGeoDataFrame
            Meteo areas for which time series are created.
        evap_folder : str or Path, optional
            Folder with evaporation rasters, one per time step. Only used without
            ``evap_file``. Defaults to None.
        evap_file : str or Path, optional
            Existing meteo file (``.evp``), used instead of the rasters. Defaults to None.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a zone must cover for the pixel to count in the
            zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the zone covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.

        Raises
        ------
        ValueError
            If neither ``evap_folder`` nor ``evap_file`` is given.
        """
        if evap_file is None:
            if evap_folder is None:
                raise ValueError("Provide evap_folder or evap_file.")
            # aggregated evap
            # areas['dissolve'] = 1
            # agg_areas = areas.iloc[0:len(areas),:].dissolve(by='dissolve',aggfunc='mean')
            # sorted: the time series follows the order of the files
            paths = sorted(Path(evap_folder).iterdir())
            stats = zonal.zonal_stats_per_raster(
                gpd.GeoDataFrame(areas),
                paths,
                "mean",
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
                strategy="feature-sequential",
                progress=True,
            )
            result = pd.DataFrame(
                stats.to_numpy().T,
                index=[
                    self.external_forcings.drrmodel._time_from_filename(path)
                    for path in paths
                ],
                columns=["ms_" + str(area) for area in areas.code],
            )
            for column, series in result.items():
                self.external_forcings.add_evap(str(column), series)
        else:
            self.external_forcings.evap = str(evap_file)

    # defer_build: drr.DRRModel does not exist yet when drr imports this module
    @validate_call(
        config=ConfigDict(
            arbitrary_types_allowed=True, coerce_numbers_to_str=True, defer_build=True
        )
    )
    def boundary_from_input(
        self,
        boundary_nodes: ExtendedGeoDataFrame,
        catchments: ExtendedGeoDataFrame,
        drrmodel: drr.DRRModel,
        overflows: ExtendedGeoDataFrame | None = None,
        greenhouse_laterals: ExtendedGeoDataFrame | None = None,
    ) -> None:
        """Add the RR boundary nodes, which link the RR nodes to the FM laterals.

        Call this after the RR nodes are added. Catchments without any RR node with an
        area are removed from ``catchments``, in place.

        Parameters
        ----------
        boundary_nodes : ExtendedGeoDataFrame
            Laterals, linked to the catchments by their key.
        catchments : ExtendedGeoDataFrame
            Catchment areas, with a ``boundary_node`` column.
        drrmodel : drr.DRRModel
            The RR model, with its nodes added.
        overflows : ExtendedGeoDataFrame, optional
            Overflow locations, each added as a boundary node. Defaults to None.
        greenhouse_laterals : ExtendedGeoDataFrame, optional
            Outlet points of greenhouse areas, each added as a boundary node. Defaults to
            None.
        """
        # boundary nodes of the RR nodes with an area
        nodes_with_area = set()
        for node in drrmodel.unpaved.unp_nodes.values():
            # an unpaved node has an area per land use
            land_use_areas = [float(area) for area in node["ar"].split(" ")]
            if np.sum(land_use_areas) > 0.0:
                nodes_with_area.add(node["boundary_node"])
        for nodes in (
            drrmodel.paved.pav_nodes,
            drrmodel.greenhouse.gh_nodes,
            drrmodel.openwater.ow_nodes,
        ):
            for node in nodes.values():
                if float(node["ar"]) > 0.0:
                    nodes_with_area.add(node["boundary_node"])

        # the catchments that have no area attached and no nodes that will be attached to the boundary
        not_occurring = [
            node for node in catchments["boundary_node"] if node not in nodes_with_area
        ]

        drop_idx = catchments[
            catchments.boundary_node.isin(not_occurring)
        ].index.to_list()
        if drop_idx:
            logger.warning(
                "%d catchments removed because of an area of 0 m2.",
                len(drop_idx),
            )
            # inplace: returning a copy of an ExtendedGeoDataFrame gives a plain GeoDataFrame
            catchments.drop(drop_idx, inplace=True)  # noqa: PD002

        catchment_lat = RelationsRR.CATCHMENT_LATERAL
        for cat in catchments.itertuples():
            lateral_id = getattr(cat, catchment_lat.foreign_key)
            lat = boundary_nodes[boundary_nodes[catchment_lat.parent_key] == lateral_id]
            if lat.empty:
                logger.warning(
                    f"{cat.code} not connected to a boundary node. Skipping."
                )
                self.external_forcings.add_boundary_node(id="0.0", px="0.0", py="0.0")
                continue
            self.external_forcings.add_boundary_node(
                id=f"lat_{cat.code}",
                px=str(lat.geometry.x.iloc[0]).strip(),
                py=str(lat.geometry.y.iloc[0]).strip(),
            )
        if overflows is not None:
            logger.info("Adding overflows to the boundary nodes.")
            for ovf, ovf_geometry in zip(overflows.itertuples(), overflows.geometry):
                self.external_forcings.add_boundary_node(
                    id=str(ovf.code),
                    px=str(ovf_geometry.coords[0][0]),
                    py=str(ovf_geometry.coords[0][1]),
                )
        if greenhouse_laterals is not None:
            logger.info("Adding greenhouse_laterals to the boundary nodes.")
            for gh, gh_geometry in zip(
                greenhouse_laterals.itertuples(), greenhouse_laterals.geometry
            ):
                self.external_forcings.add_boundary_node(
                    id=str(gh.code),
                    px=str(gh_geometry.coords[0][0]),
                    py=str(gh_geometry.coords[0][1]),
                )
