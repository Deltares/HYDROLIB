import logging
import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pydantic.v1 import ConfigDict, StrictStr, validate_arguments
from tqdm.auto import tqdm

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


class UnpavedIO:
    def __init__(self, unpaved):
        self.unpaved = unpaved

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
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
        zonalstats_alltouched: bool = False,
        greenhouse_areas: ExtendedGeoDataFrame = None
    ):
        """Generate contents of an unpaved node from raster data

        Args:
            catchments (ExtendedGeoDataFrame): catchment areas
            landuse (str): filename of land use raster
            surface_level (str): file name of surface level raster
            soiltype (str): file name of soiltype raster
            surface_storage (Union): numeric for spatially uniform surface storage (mm), or raster for distributed values
            infiltration_capacity (Union): numeric for spatially uniform infiltration capacity (mm/d), or raster for distributed values
            initial_gwd (Union): numeric for spatially uniform initial groundwater levels (m), or raster for distributed values
            meteo_areas (ExtendedGeoDataFrame): meteo areas, for each station a meteo time series is assigned
            zonalstats_alltouched (bool, optional): method to carry out zonal statistics, see rasterstats docx. Defaults to False.
        """
        all_touched = zonalstats_alltouched
        px_area, reference_crs = _raster_metadata(landuse)

        # required rasters
        warnings.filterwarnings("ignore")
        lu_counts = zonal.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            all_touched=all_touched,
        )

        soiltypes = zonal.zonal_stats(
            gpd.GeoDataFrame(catchments),
            soiltype,
            statistics=("mode",),
            all_touched=all_touched,
        )

        # @TODO mean in naam maar median
        mean_elev = zonal.zonal_stats(
            gpd.GeoDataFrame(catchments),
            surface_level,
            statistics=("median",),
            all_touched=all_touched,
        )

        # optional rasters
        if isinstance(surface_storage, (Path, str)):
            sstores = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                surface_storage,
                statistics=("mean",),
                all_touched=True,
                raster_crs=reference_crs,
            )
        elif isinstance(surface_storage, int):
            surface_storage = float(surface_storage)
        if isinstance(infiltration_capacity, (Path, str)):
            infcaps = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                infiltration_capacity,
                statistics=("mean",),
                all_touched=True,
                raster_crs=reference_crs,
            )
        elif isinstance(infiltration_capacity, int):
            infiltration_capacity = float(infiltration_capacity)
        if isinstance(initial_gwd, (Path, str)):
            ini_gwds = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                initial_gwd,
                statistics=("mean",),
                all_touched=True,
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
        for num, cat in enumerate(catchments.itertuples()):
            # if no rasterdata could be obtained for this catchment, skip it.
            if pd.isna(mean_elev.iloc[num]["median"]): # @TODO mean in naam maar median
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
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, 0] if not tm else tm[0].code
            landuse_counts = lu_counts.iloc[num].copy()
            mapping = np.zeros(16, dtype=int)
            
            # subtract greenhouse area from most occurring land use if no greenhouse area is in the landuse map
            if greenhouse_areas is not None and cat.geometry.intersects(greenhouse_areas.geometry).any():
                intersection_area = cat.geometry.intersection(greenhouse_areas.geometry).area
                intersection_area = intersection_area[intersection_area > 0.].to_numpy()[0]                    
                if landuse_counts.get(15, 0) > 0:
                    # divide area to subtract between greenhouses and the most occurring area
                    remainder = np.max([0., intersection_area - float(landuse_counts[15] * px_area)])
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
            elev = mean_elev.iloc[num]["median"] # @TODO mean in naam maar median
            self.unpaved.add_unpaved(
                id=str(cat.code),
                total_area=f"{cat.geometry.area:.0f}",
                lu_areas=lu_map,
                surface_level=f"{elev:.2f}",
                soiltype=f'{soiltypes.iloc[num]["mode"] + 100.0:.0f}',
                surface_storage=(
                    f"{surface_storage:.3f}"
                    if isinstance(surface_storage, float)
                    else f'{sstores.iloc[num]["mean"]:.3f}'
                ),
                infiltration_capacity=(
                    f"{infiltration_capacity:.3f}"
                    if isinstance(infiltration_capacity, float)
                    else f'{infcaps.iloc[num]["mean"]:.3f}'
                ),
                initial_gwd=(
                    f"{initial_gwd:.2f}"
                    if isinstance(initial_gwd, float)
                    else f'{ini_gwds.iloc[num]["mean"]:.2f}'
                ),
                meteo_area=str(ms),
                px=f"{cat.geometry.centroid.coords[0][0]-10:.0f}",
                py=f"{cat.geometry.centroid.coords[0][1]:.0f}",
                boundary_node=cat.boundary_node,
            )

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def ernst_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        depths: list,
        resistance: list,
        infiltration_resistance: float | None = None,
        runoff_resistance: float | None = None,
    ) -> None:
        """Generate an Ernst definition for an unpaved node.

        Args:
            catchments (ExtendedGeoDataFrame): Cahchment areas
            depths (list): list of layer depths (m)
            resistance (list): list of layer Ernst resistances (d-1)
            infiltration_resistance (int or float, optional): resistance to infiltration. Defaults to 300 d-1.
            runoff_resistance (int or float, optional): resistance to suface runoff. Defaults to 1 d-1.
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
    def __init__(self, paved):
        self.paved = paved

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def paved_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: StrictStr | Path,
        surface_level: StrictStr | Path,
        street_storage:StrictStr | Path | float,
        sewer_storage:StrictStr | Path | float,
        pump_capacity:StrictStr | Path | float,
        meteo_areas: ExtendedGeoDataFrame,
        overflows: ExtendedGeoDataFrame = None,
        sewer_areas: ExtendedGeoDataFrame = None,
        zonalstats_alltouched: bool = False,
        
    ) -> None:
        """Generate contents of RR paved nodes

        Args:
            catchments (ExtendedGeoDataFrame): catchment areas
            landuse (str): filename of landuse raster
            surface_level (str): file name of suface level raster
            street_storage (Union): numeric for spatially uniform street storage (mm), or raster for distributed values
            sewer_storage (Union): numeric for spatially uniform sewer storage (mm), or raster for distributed values
            pump_capacity (Union): numeric for spatially uniform pump capaities (mm), or raster for distributed values
            meteo_areas (ExtendedGeoDataFrame): meteo areas, for each station a meteo time series is assigned
            overflows (ExtendedGeoDataFrame, optional): overflow locations. Defaults to None.
            sewer_areas (ExtendedGeoDataFrame, optional): sewer area locations. Defaults to None.
            zonalstats_alltouched (bool, optional): method to carry out zonal statistis, see rasterstats docx. Defaults to False.onalstats_alltouched (bool, optional): method to. Defaults to False.

        Returns:
            _type_: _description_
        """
        all_touched = zonalstats_alltouched
        px_area, reference_crs = _raster_metadata(landuse)

        lu_counts = zonal.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            all_touched=all_touched,
        )
        mean_elev = zonal.zonal_stats(
            gpd.GeoDataFrame(catchments),
            surface_level,
            statistics=("median",), # @TODO mean in naam maar median
            all_touched=all_touched,
        )

        if isinstance(street_storage, (Path, str)):
            str_stors = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                street_storage,
                statistics=("mean",),
                all_touched=True,
                raster_crs=reference_crs,
            )
        
        if isinstance(sewer_storage, (Path, str)):
            sew_stors = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                sewer_storage,
                statistics=("mean",),
                all_touched=True,
                raster_crs=reference_crs,
            )
        
        if isinstance(pump_capacity,  (Path, str)):
            # raster of POC in mm/h
            pump_caps = zonal.zonal_stats(
                gpd.GeoDataFrame(catchments),
                pump_capacity,
                statistics=("mean",),
                all_touched=True,
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
            if isinstance(street_storage,(Path, str)):
                str_stors_sa = zonal.zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    street_storage,
                    statistics=("mean",),
                    all_touched=True,
                    raster_crs=reference_crs,
                )
            if isinstance(sewer_storage, (Path, str)):
                sew_stors_sa = zonal.zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    sewer_storage,
                    statistics=("mean",),
                    all_touched=True,
                    raster_crs=reference_crs,
                )
            if isinstance(pump_capacity, (Path, str)):
                pump_caps_sa = zonal.zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    pump_capacity,
                    statistics=("mean",),
                    all_touched=True,
                    raster_crs=reference_crs,
                )
            mean_sa_elev = zonal.zonal_stats(
                gpd.GeoDataFrame(sewer_areas),
                surface_level,
                statistics=("median",), # @TODO mean in naam maar median
                all_touched=True,
            )
            sewer_lu_counts = zonal.zonal_category_counts(
                gpd.GeoDataFrame(sewer_areas), landuse, all_touched=all_touched
            )

            # find the paved area in the sewer areas
            for isew, sew in enumerate(sewer_areas.itertuples()):
                pav_area = 0
                pixels = sewer_lu_counts.iloc[isew]
                pav_pixels = pixels.get(14, 0)
                if pav_pixels == 0:
                    logger.warning("No paved area in sewer area %s.", sew.code)
                    self.paved.add_paved(**dict.fromkeys(paved_columns, "0.0"))
                    continue
                pav_area += pav_pixels * px_area

                # subtract it fromthe total paved area in this catchment, make sure at least 0 remains
                # lu_counts[cat_ind][14.0] -=  pav_pixels
                # if lu_counts[cat_ind][14.0] < 0: lu_counts[cat_ind][14.0]  = 0

                elev = mean_sa_elev.iloc[isew]["median"] # @TODO mean in naam maar median
                # find overflows related to this sewer area
                ovf = overflows[overflows.codegerelateerdobject == sew.code]
                for ov in ovf.itertuples():
                    # find corresponding meteo-station
                    tm = [
                        m
                        for m in meteo_areas.itertuples()
                        if m.geometry.contains(sew.geometry.centroid)
                    ]
                    ms = meteo_areas.iloc[0, 0] if not tm else tm[0].code

                    # if a float is given, a standard value is passed. If a string is given, a rastername is assumed to zonal statistics are applied.
                    if isinstance(street_storage, float):
                        street_storage_val = f"{street_storage:.2f}"
                    elif isinstance(street_storage, (Path, str)):
                        street_storage_val = f'{str_stors_sa.iloc[isew]["mean"]:.2f}'
                    else:
                        raise TypeError('Street_storage has the wrong datatype. It should be a filename (Path or string) or number (float or int).')

                    # three options: it can be an attribute of a sewer area, a uniform value or a raster
                    riool_berging_mm = getattr(sew, "riool_berging_mm", None)
                    if riool_berging_mm is None or np.isnan(riool_berging_mm) or not isinstance(riool_berging_mm, float):
                        if isinstance(sewer_storage, float):
                            sewer_storage_val = f"{sewer_storage:.2f}"
                        elif isinstance(sewer_storage, (Path, str)):
                            sewer_storage_val = f'{sew_stors_sa.iloc[isew]["mean"]:.2f}'
                        else:
                            raise TypeError('Sewer_storage has the wrong datatype. It should be a filename (Path or string) or number (float or int).')
                    else:
                        sewer_storage_val = f'{riool_berging_mm:.2f}'

                    # three options: it can be an attribute of a sewer area, a uniform value or a raster
                    riool_poc_m3s = getattr(sew, "riool_poc_m3s", None)
                    if riool_poc_m3s is None or np.isnan(riool_poc_m3s) or not isinstance(riool_poc_m3s, float):
                        if isinstance(pump_capacity, float):
                            # convert the value from mm/h to m3/s
                            pump_capacity_val = f"{pump_capacity * (float(pav_area) * ov.fractie) / (1000. * 3600.):.8f}"
                        elif isinstance(pump_capacity, (Path, str)):
                            # convert the value (extracted from the raster) from mm/h to m3/s
                            pump_capacity_val = f'{pump_caps_sa.iloc[isew]["mean"] * (float(pav_area) * ov.fractie) / (1000. * 3600.):.8f}'
                        else:
                            raise TypeError('Pump_capacity has the wrong datatype. It should be a filename (Path or string) or number (float or int).')
                    else:
                        # use the attribute value
                        pump_capacity_val = f'{riool_poc_m3s * ov.fractie:.8f}'

                    # add prefix to the overflow id to create the paved-node id
                    self.paved.add_paved(
                        id=str(ov.code),
                        area=str(pav_area * ov.fractie),
                        surface_level=f"{elev:.2f}",
                        street_storage=street_storage_val,
                        sewer_storage=sewer_storage_val,
                        pump_capacity=pump_capacity_val,
                        meteo_area=str(ms),
                        px=f"{ov.geometry.coords[0][0]+10:.0f}",
                        py=f"{ov.geometry.coords[0][1]:.0f}",
                        boundary_node=ov.code,
                    )

        for num, cat in enumerate(catchments.itertuples()):
            # if no rasterdata could be obtained for this catchment, skip it.
            if pd.isna(mean_elev.iloc[num]["median"]): # @TODO mean in naam maar median
                logger.warning(NO_RASTERDATA_WARNING, cat.code)
                self.paved.add_paved(**dict.fromkeys(paved_columns, "0.0"))
                continue
            if sewer_areas is not None:
                # part of the catchment that is also in a sewer area
                if cat.geometry.intersects(sewer_areas.union_all()):
                    # the part of the catchment OUTSIDE the sewer area
                    area_outside_sewer = cat.geometry.difference(
                        sewer_areas.union_all()
                    )
                    if area_outside_sewer.area == 0:
                        logger.info(
                            f"No paved area outside sewer area in catchments {cat.code}."
                        )
                        pav_area = 0.0
                    else:
                        # the paved ara in the catchment OUTSIDE the sewer area
                        pixels = zonal.zonal_category_counts(
                            gpd.GeoDataFrame(
                                geometry=[area_outside_sewer], crs=catchments.crs
                            ),
                            landuse,
                            all_touched=all_touched,
                        ).iloc[0]
                        pav_area = str(pixels.get(14, 0) * px_area)
                else:
                    # all of the catchment is outside the sewer area
                    pixels = zonal.zonal_category_counts(
                        gpd.GeoDataFrame(
                            geometry=[cat.geometry], crs=catchments.crs
                        ),
                        landuse,
                        all_touched=all_touched,
                    ).iloc[0]
                    pav_area = str(pixels.get(14, 0) * px_area)
            else:
                # there is no sewer area at all
                pav_area = str(lu_counts.iloc[num].get(14, 0) * px_area)

            # find corresponding meteo-station
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, 0] if not tm else tm[0].code

            elev = mean_elev.iloc[num]["median"] # @TODO mean in naam maar median
            # if a float is given, a standard value is passed. If a string is given, a rastername is assumed to zonal statistics are applied.
            street_storage_val = (
                f"{street_storage:.2f}"
                if isinstance(street_storage, float)
                else f'{str_stors.iloc[num]["mean"]:.2f}'
            )
            sewer_storage_val = (
                f"{sewer_storage:.2f}"
                if isinstance(sewer_storage, float)
                else f'{sew_stors.iloc[num]["mean"]:.2f}'
            )
            pump_capacity_val = (
                f'{(pump_capacity * float(pav_area)) / (1000. * 3600.):.8f}'
                if isinstance(pump_capacity, float)
                else f'{pump_caps.iloc[num]["mean"] * (float(pav_area)) / (1000. * 3600.):.8f}'
            )
            self.paved.add_paved(
                id=str(cat.code),
                area=str(pav_area),
                surface_level=f"{elev:.2f}",
                street_storage=street_storage_val,
                sewer_storage=sewer_storage_val,
                pump_capacity=pump_capacity_val,
                meteo_area=str(ms),
                px=f"{cat.geometry.centroid.coords[0][0]+10:.0f}",
                py=f"{cat.geometry.centroid.coords[0][1]:.0f}",
                boundary_node=cat.boundary_node,
            )


class GreenhouseIO:
    def __init__(self, greenhouse):
        self.greenhouse = greenhouse

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def greenhouse_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Path | str,
        surface_level: Path | str,
        roof_storage: StrictStr | float,
        meteo_areas: ExtendedGeoDataFrame,
        zonalstats_alltouched: bool = False,
        greenhouse_areas: ExtendedGeoDataFrame=None,
        greenhouse_laterals: ExtendedGeoDataFrame=None,
        basin_storage_class: int=2    
    ) -> None:
        """Generate contents of an RR greenhouse node

        Args:
            catchments (ExtendedGeoDataFrame): catchment areas
            greenhouuse_areas (ExtendedGeoDataFrame): known set of greenhouse areas with attiributes
            greenhouse_laterals (ExtendedGeoDataFrame) : known set of outlet points for greenhouses
            landuse (str): filename of land use raster
            surface_level (str): filename of surface level raster
            roofstorage (Union): float for spatially uniform sewer storage (mm), or raster for distributed values
            meteo_areas (ExtendedGeoDataFrame): meteo areas, for each station a meteo time series is assigned
            zonalstats_alltouched (bool, optional): method to carry out zonal statistis, see rasterstats docx. Defaults to False.onalstats_alltouched (bool, optional): method to. Defaults to False.
        """
        all_touched = zonalstats_alltouched
        px_area, reference_crs = _raster_metadata(landuse)

        lu_counts = zonal.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            all_touched=all_touched,
        )
        mean_elev = zonal.zonal_stats(
            gpd.GeoDataFrame(catchments),
            surface_level,
            statistics=("median",), # @TODO mean in naam maar median
            all_touched=all_touched,
        )
        if greenhouse_areas is not None:
            mean_elev_gh = zonal.zonal_stats(
                 gpd.GeoDataFrame(greenhouse_areas),
                 surface_level,
                 statistics=("median",), # @TODO mean in naam maar median
                 all_touched=all_touched,
            )
            
        # optional rasters
        if isinstance(roof_storage, (Path, str)):
            roofstors = zonal.zonal_stats(
                 gpd.GeoDataFrame(catchments),
                 roof_storage,
                 statistics=("mean",),
                 all_touched=True,
                 raster_crs=reference_crs,
            )
            if greenhouse_areas is not None:
                roofstors_gh = zonal.zonal_stats(
                     gpd.GeoDataFrame(greenhouse_areas),
                     roof_storage,
                     statistics=("mean",),
                     all_touched=True,
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
            for num, gh in enumerate(greenhouse_areas.itertuples()):
                # find corresponding meteo-station
                if pd.isna(mean_elev_gh.iloc[num]["median"]): # @TODO mean in naam maar median
                    logger.warning(NO_RASTERDATA_WARNING, gh.code)
                    self.greenhouse.add_greenhouse(**dict.fromkeys(gh_columns, "0.0"))
                    continue
                tm = [
                    m
                    for m in meteo_areas.itertuples()
                    if m.geometry.contains(gh.geometry.centroid)
                ]
                ms = meteo_areas.iloc[0, 0] if not tm else tm[0].code

                elev = mean_elev_gh.iloc[num]["median"] # @TODO mean in naam maar median
                if hasattr(gh, 'roof_storage_mm') and not np.isnan(gh.roof_storage_mm):
                    roof_storage_val = f"{gh.roof_storage_mm:.2f}"
                elif isinstance(roof_storage, float):
                    roof_storage_val = f"{roof_storage:.2f}"
                else:
                    roof_storage_val = f'{roofstors_gh.iloc[num]["mean"]:.2f}'
                if hasattr(gh, 'basin_storage_class') and not np.isnan(gh.basin_storage_class):
                    basin_storage_class_val = f"{gh.basin_storage_class:g}"
                else:
                    basin_storage_class_val = f'{basin_storage_class:g}'
                latcode = greenhouse_laterals[greenhouse_laterals.codegerelateerdobject == gh.code].code.to_numpy()[0]
                self.greenhouse.add_greenhouse(
                    id=str(gh.code),
                    area=gh.geometry.area,
                    surface_level=f"{elev:.2f}",
                    roof_storage=roof_storage_val,
                    basin_storage_class=basin_storage_class_val,
                    meteo_area=str(ms),
                    px=f"{gh.geometry.centroid.coords[0][0]:.0f}",
                    py=f"{gh.geometry.centroid.coords[0][1]:.0f}",
                    boundary_node=str(latcode),
                )

        for num, cat in enumerate(catchments.itertuples()):
            # if no rasterdata could be obtained for this catchment, skip it.
            if pd.isna(mean_elev.iloc[num]["median"]): # @TODO mean in naam maar median
                logger.warning(NO_RASTERDATA_WARNING, cat.code)
                self.greenhouse.add_greenhouse(**dict.fromkeys(gh_columns, "0.0"))
                continue

            # find corresponding meteo-station
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, 0] if not tm else tm[0].code

            if greenhouse_areas is not None and cat.geometry.intersects(greenhouse_areas.geometry).any():
                intersection_area = cat.geometry.intersection(greenhouse_areas.geometry).area
                intersection_area = intersection_area[intersection_area > 0.].to_numpy()[0]                    
                landuse_counts = lu_counts.iloc[num]
                if landuse_counts.get(15, 0) > 0:
                    # divide area to subtract between greenhouses and the most occurring area                                                
                    logger.info(
                        "Catchment: %s: subtracting %s m2 from greenhouse area in landuse map.",
                        cat.code,
                        np.min([(landuse_counts[15] * px_area, intersection_area)]),
                    )
                    lu_counts.iloc[num, lu_counts.columns.get_loc(15)] = np.max(
                        [0.0, landuse_counts[15] - np.round(intersection_area / px_area)]
                    )
            
            elev = mean_elev.iloc[num]["median"] # @TODO mean in naam maar median
            roof_storage_val = (
                f"{roof_storage:.2f}"
                if isinstance(roof_storage, float)
                else f'{roofstors.iloc[num]["mean"]:.2f}'
            )
            self.greenhouse.add_greenhouse(
                id=str(cat.code),
                area=(
                    str(lu_counts.iloc[num].get(15, 0) * px_area)
                ),
                surface_level=f"{elev:.2f}",
                roof_storage=roof_storage_val,
                basin_storage_class=f"{basin_storage_class:g}",
                meteo_area=str(ms),
                px=f"{cat.geometry.centroid.coords[0][0]+20:.0f}",
                py=f"{cat.geometry.centroid.coords[0][1]:.0f}",
                boundary_node=cat.boundary_node,
            )

class OpenwaterIO:
    def __init__(self, openwater):
        self.openwater = openwater

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def openwater_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Path | str,
        meteo_areas: ExtendedGeoDataFrame,
        zonalstats_alltouched: bool = False,
    ) -> None:
        """Generate contents of an RR open water node.

        Args:
            catchments (ExtendedGeoDataFrame): catchment areas
            landuse (str): filename of landuse raster
            meteo_areas (ExtendedGeoDataFrame): meteo areas, for each station a meteo time series is assigned
            zonalstats_alltouched (bool, optional): method to carry out zonal statistis, see rasterstats docx. Defaults to False.onalstats_alltouched (bool, optional): method to. Defaults to False.

        Returns:
            _type_: _description_
        """
        all_touched = zonalstats_alltouched
        px_area, _ = _raster_metadata(landuse)

        lu_counts = zonal.zonal_category_counts(
            gpd.GeoDataFrame(catchments),
            landuse,
            all_touched=all_touched,
        )

        for num, cat in enumerate(catchments.itertuples()):
            # find corresponding meteo-station
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, 0] if not tm else tm[0].code

            self.openwater.add_openwater(
                id=str(cat.code),
                area=str(lu_counts.iloc[num].get(13, 0) * px_area),
                meteo_area=str(ms),
                px=f"{cat.geometry.centroid.coords[0][0]-20:.0f}",
                py=f"{cat.geometry.centroid.coords[0][1]:.0f}",
                boundary_node=cat.boundary_node,
            )


class ExternalForcingsIO:
    def __init__(self, external_forcings):
        self.external_forcings = external_forcings

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def seepage_from_input(
        self, catchments: ExtendedGeoDataFrame, seepage_folder: Path | str
    ) -> None:
        """Perform zonal statistics to derive seepage time series per catchment. Time steps are derived from the data

        Args:
            catchments (ExtendedGeoDataFrame): catchment areas
            seepage_folder (str): folder where the seepage rasters are stored
        """
        warnings.filterwarnings("ignore")
        file_list = os.listdir(seepage_folder)
        file_list = [file for file in file_list if file.lower()]
        times = []
        convert_units=False
        arr = np.zeros((len(file_list), len(catchments.code)))
        zones = gpd.GeoDataFrame(catchments)
        for ifile, file in tqdm(
            enumerate(file_list), total=len(file_list), desc="Reading seepage files"
        ):
            path = os.path.join(seepage_folder, file)
            if file.endswith('.idf'):
                dataset = idfreader.open(path).squeeze()
                affine = idfreader.affine_from_idf(dataset)
                time = pd.Timestamp(dataset["time"].values)
                stats = zonal.zonal_stats(
                    zones,
                    dataset.to_numpy(),
                    statistics=("mean",),
                    all_touched=True,
                    affine=affine,
                    raster_crs=dataset.attrs.get("crs"),
                    nodata=dataset.attrs.get("nodata", np.nan),
                )
                convert_units=True
            else:
                time = self.external_forcings.drrmodel._time_from_filename(path)
                stats = zonal.zonal_stats(
                    zones, path, statistics=("mean",), all_touched=True, strategy="feature"
                )
            times.append(time)
            arr[ifile, :] = stats["mean"].to_numpy()
        result = pd.DataFrame(
            arr, columns=["sep_" + str(cat) for cat in catchments.code]
        )
        result.index = times
        if convert_units:
            # if an NHI model (IDF files) is used, convert units from m3 to mm/d
            result = (result / (1e-3 * (affine[0] * -affine[4]))) / (
                    (times[2] - times[1]).total_seconds() / 86400.0
            )
        [self.external_forcings.add_seepage(*sep) for sep in result.items()]


    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def precip_from_input(
        self,
        areas: ExtendedGeoDataFrame,
        precip_folder: Path | str | None = None,
        precip_file: Path | str | None = None,
    ) -> None:
        """Create time series of precipitation for every meteo_area, based on zonal statistics from rasters.

        Args:
            areas (ExtendedGeoDataFrame): meteo areas for which time series are created
            precip_folder (str, optional): folder where precipitation rasters are stored. Only used if no precip_file is given. Defaults to None.
            precip_file (str, optional): existing meteo-file, which is used if available.
        """
        if precip_file is None:
            warnings.filterwarnings("ignore")
            file_list = os.listdir(precip_folder)
            times = []
            zones = gpd.GeoDataFrame(areas)
            arr = np.zeros((len(file_list), len(areas.code)))
            for ifile, file in tqdm(
                enumerate(file_list),
                total=len(file_list),
                desc="Reading precipitation files",
            ):
                path = os.path.join(precip_folder, file)
                times.append(self.external_forcings.drrmodel._time_from_filename(path))
                stats = zonal.zonal_stats(
                    zones, path, statistics=("mean",), all_touched=True, strategy="feature"
                )
                arr[ifile, :] = stats["mean"].to_numpy()
            result = pd.DataFrame(
                arr, columns=["ms_" + str(area) for area in areas.code]
            )
            result.index = times
            [self.external_forcings.add_precip(*prec) for prec in result.items()]
        else:
            self.external_forcings.precip = str(precip_file)

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def evap_from_input(
        self,
        areas: ExtendedGeoDataFrame,
        evap_folder: Path | str | None = None,
        evap_file: Path | str | None = None,
    ) -> None:
        """Create time series of evaporation for every meteo_area, based on zonal statistics from rasters.

        Args:
            areas (ExtendedGeoDataFrame): meteo areas for which time series are created
            evap_folder (str, optional): folder where precipitation rasters are stored. Only used if no precip_file is given. Defaults to None.
            evap_file (str, optional): existing meteo-file, which is used if available.
        """
        if evap_file is None:
            warnings.filterwarnings("ignore")
            file_list = os.listdir(evap_folder)
            # aggregated evap
            # areas['dissolve'] = 1
            # agg_areas = areas.iloc[0:len(areas),:].dissolve(by='dissolve',aggfunc='mean')
            times = []
            zones = gpd.GeoDataFrame(areas)
            arr = np.zeros((len(file_list), len(areas)))
            for ifile, file in tqdm(
                enumerate(file_list),
                total=len(file_list),
                desc="Reading evaporation files",
            ):
                path = os.path.join(evap_folder, file)
                times.append(self.external_forcings.drrmodel._time_from_filename(path))
                stats = zonal.zonal_stats(
                    zones, path, statistics=("mean",), all_touched=True, strategy="feature"
                )
                arr[ifile, :] = stats["mean"].to_numpy()
            result = pd.DataFrame(
                arr, columns=["ms_" + str(area) for area in areas.code]
            )
            result.index = times
            [self.external_forcings.add_evap(*evap) for evap in result.items()]
        else:
            self.external_forcings.evap = str(evap_file)

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def boundary_from_input(
        self,
        boundary_nodes: ExtendedGeoDataFrame,
        catchments: ExtendedGeoDataFrame,
        drrmodel,
        overflows: ExtendedGeoDataFrame = None,
        greenhouse_laterals: ExtendedGeoDataFrame=None
    ) -> None:
        """Generate RR-boundary nodes to link to RR-nodes and to FM-laterals.

        Args:
            boundary_nodes (ExtendedGeoDataFrame): boundary nodes
            catchments (ExtendedGeoDataFrame): catchment areas associated with them
            drrmodel (_type_): drrmodel object
            overflows (ExtendedGeoDataFrame, optional): overflow locations, if applicable. Defaults to None.
            greenhouse_laterals (ExtendedGeoDataFrame, optional): overflow locations, if applicable. Defaults to None.

        """
        # find the catchments that have no area attached and no nodes that will be attached to the boundary
        not_occurring = []
        for cat in catchments.itertuples():
            occurs = False
            if cat.boundary_node in [
                val["boundary_node"]
                for val in drrmodel.unpaved.unp_nodes.values()
                if np.sum([float(d) for d in val["ar"].split(" ")]) > 0.0
            ]:
                occurs = True
            if cat.boundary_node in [
                val["boundary_node"]
                for val in drrmodel.paved.pav_nodes.values()
                if float(val["ar"]) > 0.0
            ]:
                occurs = True
            if cat.boundary_node in [
                val["boundary_node"]
                for val in drrmodel.greenhouse.gh_nodes.values()
                if float(val["ar"]) > 0.0
            ]:
                occurs = True
            if cat.boundary_node in [
                val["boundary_node"]
                for val in drrmodel.openwater.ow_nodes.values()
                if float(val["ar"]) > 0.0
            ]:
                occurs = True
            if not occurs:
                not_occurring.append(cat.boundary_node)

     
        drop_idx = catchments[catchments.boundary_node.isin(not_occurring)].index.to_list()
        if drop_idx:
            logger.warning(
                "%d catchments removed because of an area of 0 m2.",
                len(drop_idx),
            )
            catchments.drop(drop_idx, inplace=True)

        for cat in catchments.itertuples():
            if boundary_nodes[boundary_nodes["globalid"] == cat.lateraleknoopid].empty:
                logger.warning(
                    f"{cat.code} not connected to a boundary node. Skipping."
                )
                self.external_forcings.add_boundary_node(id="0.0", px="0.0", py="0.0")
                continue
            self.external_forcings.add_boundary_node(
                id=f"lat_{cat.code}",
                px=str(
                    boundary_nodes[
                        boundary_nodes["globalid"] == cat.lateraleknoopid
                    ]["geometry"].x.iloc[0]
                ).strip(),
                py=str(
                    boundary_nodes[
                        boundary_nodes["globalid"] == cat.lateraleknoopid
                    ]["geometry"].y.iloc[0]
                ).strip(),
            )
        if overflows is not None:
            logger.info("Adding overflows to the boundary nodes.")
            for ovf in overflows.itertuples():
                self.external_forcings.add_boundary_node(
                    id=str(ovf.code),
                    px=str(ovf.geometry.coords[0][0]),
                    py=str(ovf.geometry.coords[0][1]),
                )
        if greenhouse_laterals is not None:
            logger.info("Adding greenhouse_laterals to the boundary nodes.")
            for gh in greenhouse_laterals.itertuples():
                self.external_forcings.add_boundary_node(
                    id=str(gh.code),
                    px=str(gh.geometry.coords[0][0]),
                    py=str(gh.geometry.coords[0][1]),
                )
