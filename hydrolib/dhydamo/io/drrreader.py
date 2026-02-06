import logging
import os
import warnings
from pathlib import Path
from typing import Union
import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic.v1 import validate_arguments, StrictStr
from rasterstats import zonal_stats
from rasterio.transform import from_origin
from tqdm.auto import tqdm
from hydrolib.dhydamo.io import idfreader
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame

logger = logging.getLogger(__name__)


class UnpavedIO:
    def __init__(self, unpaved):
        self.unpaved = unpaved

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def unpaved_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Union[StrictStr, Path],
        surface_level: Union[StrictStr, Path],
        soiltype: Union[StrictStr, Path],
        surface_storage: Union[StrictStr, Path, float],
        infiltration_capacity: Union[StrictStr, Path, float],
        initial_gwd: Union[StrictStr, Path, float],
        meteo_areas: ExtendedGeoDataFrame,
        zonalstats_alltouched: bool = None,        
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
        all_touched = False if zonalstats_alltouched is None else zonalstats_alltouched

        # required rasters
        warnings.filterwarnings("ignore")
        lu_rast, lu_affine = self.unpaved.drrmodel.read_raster(landuse, static=True)
        lu_counts = zonal_stats(
            gpd.GeoDataFrame(catchments),
            lu_rast.astype(int),
            affine=lu_affine,
            categorical=True,
            all_touched=all_touched,
        )

        soil_rast, affine = self.unpaved.drrmodel.read_raster(soiltype, static=True)
        soiltypes = zonal_stats(
            gpd.GeoDataFrame(catchments),
            soil_rast.astype(int),
            affine=affine,
            stats="majority",
            all_touched=all_touched,
        )

        rast, affine = self.unpaved.drrmodel.read_raster(surface_level, static=True)
        mean_elev = zonal_stats(
            gpd.GeoDataFrame(catchments), rast.astype(float), affine=affine, stats="median", all_touched=all_touched
        )

        # optional rasters
        if isinstance(surface_storage, str):
            rast, affine = self.unpaved.drrmodel.read_raster(
                surface_storage, static=True
            )
            sstores = zonal_stats(
                gpd.GeoDataFrame(catchments), rast.astype(float), affine=affine, stats="mean", all_touched=True
            )
        elif isinstance(surface_storage, int):
            surface_storage = float(surface_storage)
        if isinstance(infiltration_capacity, str):
            rast, affine = self.unpaved.drrmodel.read_raster(
                infiltration_capacity, static=True
            )
            infcaps = zonal_stats(
                gpd.GeoDataFrame(catchments), rast.astype(float), affine=affine, stats="mean", all_touched=True
            )
        elif isinstance(infiltration_capacity, int):
            infiltration_capacity = float(infiltration_capacity)
        if isinstance(initial_gwd, str):
            rast, affine = self.unpaved.drrmodel.read_raster(initial_gwd, static=True)
            ini_gwds = zonal_stats(
                gpd.GeoDataFrame(catchments), rast.astype(float), affine=affine, stats="mean", all_touched=True
            )
        elif isinstance(initial_gwd, int):
            initial_gwd = float(initial_gwd)

        # get raster cellsize
        px_area = lu_affine[0] * -lu_affine[4]

        unpaved_drr = ExtendedDataFrame(required_columns=["id"])
        unpaved_drr.set_data(
            pd.DataFrame(
                np.zeros((len(catchments), 12)),
                columns=[
                    "id",
                    "total_area",
                    "lu_areas",
                    "surface_level",
                    "soiltype",
                    "surface_storage",
                    "infiltration_capacity",
                    "initial_gwd",
                    "meteo_area",
                    "px",
                    "py",
                    "boundary_node",
                ],
                dtype="str",
            ),
            index_col="id",
        )

        unpaved_drr.index = catchments.code
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
            if mean_elev[num]["median"] is None:
                logger.warning("No rasterdata available for catchment %s.", cat.code)
                continue
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, :][0] if tm == [] else tm[0].code
            mapping = np.zeros(16, dtype=int)
            
            # subtract greenhouse area from most occurring land use if no greenhouse area is in the landuse map
            if greenhouse_areas is not None:
                if cat.geometry.intersects(greenhouse_areas.geometry).any():
                    intersection_area = cat.geometry.intersection(greenhouse_areas.geometry).area
                    intersection_area = intersection_area[intersection_area > 0.].values[0]                    
                    if 15 in lu_counts[num]:
                        # divide area to subtract between greenhouses and the most occurring area
                        remainder = np.max([0., intersection_area - float(lu_counts[num][15]*px_area)])                                                
                    else:    
                        remainder = intersection_area
                    maxind = np.argmax(list(lu_counts[num].values()))              
                    logger.info(
                        "Catchment %s: subtracting %s m2 from class %s for supplied greenhouse area.",
                        cat.code,
                        remainder,
                        maxind,
                    )
                    lu_counts[num][list(lu_counts[num].keys())[maxind]] = np.max([0., (lu_counts[num][list(lu_counts[num].keys())[maxind]] - np.round(remainder/px_area))])
            
            for i in range(1, 13):
                if i in lu_counts[num]:
                    mapping[sobek_indices[i - 1] - 1] = lu_counts[num][i] * px_area
            lu_map = " ".join(map(str, mapping))
            elev = mean_elev[num]["median"]
            unpaved_drr.at[cat.code, "id"] = str(cat.code)
            unpaved_drr.at[cat.code, "total_area"] = f"{cat.geometry.area:.0f}"
            unpaved_drr.at[cat.code, "lu_areas"] = lu_map
            unpaved_drr.at[cat.code, "surface_level"] = f"{elev:.2f}"
            unpaved_drr.at[
                cat.code, "soiltype"
            ] = f'{soiltypes[num]["majority"]+100.:.0f}'
            if isinstance(surface_storage, float):
                unpaved_drr.at[cat.code, "surface_storage"] = f"{surface_storage:.3f}"
            else:
                unpaved_drr.at[
                    cat.code, "surface_storage"
                ] = f'{sstores[num]["mean"]:.3f}'
            if isinstance(infiltration_capacity, float):
                unpaved_drr.at[
                    cat.code, "infiltration_capacity"
                ] = f"{infiltration_capacity:.3f}"
            else:
                unpaved_drr.at[
                    cat.code, "infiltration_capacity"
                ] = f'{infcaps[num]["mean"]:.3f}'
            if isinstance(initial_gwd, float):
                unpaved_drr.at[cat.code, "initial_gwd"] = f"{initial_gwd:.2f}"
            else:
                unpaved_drr.at[cat.code, "initial_gwd"] = f'{ini_gwds[num]["mean"]:.2f}'
            unpaved_drr.at[cat.code, "meteo_area"] = str(ms)
            unpaved_drr.at[
                cat.code, "px"
            ] = f"{cat.geometry.centroid.coords[0][0]-10:.0f}"
            unpaved_drr.at[cat.code, "py"] = f"{cat.geometry.centroid.coords[0][1]:.0f}"
            unpaved_drr.at[cat.code, "boundary_node"] = cat.boundary_node

        [
            self.unpaved.add_unpaved(**unpaved)
            for unpaved in unpaved_drr.to_dict("records")
        ]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def ernst_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        depths: list,
        resistance: list,
        infiltration_resistance: float = None,
        runoff_resistance: float = None,
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
        
        ernst_drr = ExtendedDataFrame(required_columns=["id"])
        ernst_drr.set_data(
            pd.DataFrame(
                np.zeros((len(catchments), 5)),
                columns=["id", "cvo", "lv", "cvi", "cvs"],
                dtype="str",
            ),
            index_col="id",
        )
        ernst_drr.index = catchments.code
        for num, cat in enumerate(catchments.itertuples()):
            ernst_drr.at[cat.code, "id"] = str(cat.code)
            ernst_drr.at[cat.code, "cvo"] = " ".join([str(res) for res in resistance])
            ernst_drr.at[cat.code, "lv"] = " ".join([str(depth) for depth in depths])
            ernst_drr.at[cat.code, "cvi"] = f'{infiltration_resistance:.2f}'
            ernst_drr.at[cat.code, "cvs"] = f'{runoff_resistance:.2f}'

        [self.unpaved.add_ernst_def(**ernst) for ernst in ernst_drr.to_dict("records")]


class PavedIO:
    def __init__(self, paved):
        self.paved = paved

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def paved_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Union[StrictStr, Path],
        surface_level: Union[StrictStr, Path],
        street_storage:Union[StrictStr, Path, float],
        sewer_storage:Union[StrictStr, Path,  float],
        pump_capacity:Union[StrictStr, Path, float],
        meteo_areas: ExtendedGeoDataFrame,
        overflows: ExtendedGeoDataFrame = None,
        sewer_areas: ExtendedGeoDataFrame = None,
        zonalstats_alltouched: bool = None
        
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
        all_touched = False if zonalstats_alltouched is None else zonalstats_alltouched

        lu_rast, lu_affine = self.paved.drrmodel.read_raster(landuse, static=True)
        lu_counts = zonal_stats(
            gpd.GeoDataFrame(catchments),
            lu_rast.astype(int),
            affine=lu_affine,
            categorical=True,
            all_touched=all_touched,
        )
        sl_rast, sl_affine = self.paved.drrmodel.read_raster(surface_level, static=True)
        mean_elev = zonal_stats(
            gpd.GeoDataFrame(catchments),
            sl_rast.astype(float),
            affine=sl_affine,
            stats="median",
            all_touched=all_touched,
        )

        if isinstance(street_storage, (Path, str)):
            strs_rast, strs_affine = self.paved.drrmodel.read_raster(
                street_storage, static=True
            )
            str_stors = zonal_stats(
                gpd.GeoDataFrame(catchments),
                strs_rast.astype(float),
                affine=strs_affine,
                stats="mean",
                all_touched=True,
            )
        
        if isinstance(sewer_storage, (Path, str)):
            sews_rast, sews_affine = self.paved.drrmodel.read_raster(
                sewer_storage, static=True
            )
            sew_stors = zonal_stats(
                gpd.GeoDataFrame(catchments),
                sews_rast.astype(float),
                affine=sews_affine,
                stats="mean",
                all_touched=True,
            )
        
        if isinstance(pump_capacity,  (Path, str)):
            # raster of POC in mm/h
            pump_rast, pump_affine = self.paved.drrmodel.read_raster(
                pump_capacity, static=True
            )
            pump_caps = zonal_stats(
                gpd.GeoDataFrame(catchments),
                pump_rast.astype(float),
                affine=pump_affine,
                stats="mean",
                all_touched=True,
            )
        
        def update_dict(dict1, dict2):
            for i in dict2.keys():
                if i in dict1:
                    dict1[i] += dict2[i]
                else:
                    dict1[i] = dict2[i]
            return dict1

        # get raster cellsize
        px_area = lu_affine[0] * -lu_affine[4]
        paved_drr = ExtendedDataFrame(required_columns=["id"])
        if sewer_areas is not None:
            # if the parameters ara rasters, do the zonal statistics per sewage area as well.
            if isinstance(street_storage,(Path, str)):
                str_stors_sa = zonal_stats(
                    sewer_areas,
                    strs_rast.astype(float),
                    affine=strs_affine,
                    stats="mean",
                    all_touched=True,
                )
            if isinstance(sewer_storage, (Path, str)):
                sew_stors_sa = zonal_stats(
                    sewer_areas,
                    sews_rast.astype(float),
                    affine=sews_affine,
                    stats="mean",
                    all_touched=True,
                )
            if isinstance(pump_capacity, (Path, str)):
                pump_caps_sa = zonal_stats(
                    gpd.GeoDataFrame(sewer_areas),
                    pump_rast.astype(float),
                    affine=pump_affine,
                    stats="mean",
                    all_touched=True,
                )
            mean_sa_elev = zonal_stats(
                gpd.GeoDataFrame(sewer_areas), sl_rast, affine=sl_affine, stats="median", all_touched=True
            )

            # initialize the array of paved nodes, which should contain a node for all catchments and all overflows
            paved_drr.set_data(
                pd.DataFrame(
                    np.zeros((len(catchments) + len(overflows), 10)),
                    columns=[
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
                    ],
                    dtype="str",
                ),
                index_col="id",
            )
            paved_drr.index = pd.concat([catchments.code, overflows.code], ignore_index=True)


            # find the paved area in the sewer areas
            for isew, sew in enumerate(sewer_areas.itertuples()):
                pav_area = 0
                pixels = zonal_stats(
                    sew.geometry,
                    lu_rast.astype(int),
                    affine=lu_affine,
                    categorical=True,
                    all_touched=all_touched,
                )[0]
                if 14.0 not in pixels:
                    logger.warning("No paved area in sewer area %s.", sew.code)
                    continue
                pav_pixels = pixels[14.0]
                pav_area += pav_pixels * px_area

                # subtract it fromthe total paved area in this catchment, make sure at least 0 remains
                # lu_counts[cat_ind][14.0] -=  pav_pixels
                # if lu_counts[cat_ind][14.0] < 0: lu_counts[cat_ind][14.0]  = 0

                elev = mean_sa_elev[isew]["median"]
                # find overflows related to this sewer area
                ovf = overflows[overflows.codegerelateerdobject == sew.code]
                for ov in ovf.itertuples():
                    # find corresponding meteo-station
                    tm = [
                        m
                        for m in meteo_areas.itertuples()
                        if m.geometry.contains(sew.geometry.centroid)
                    ]
                    ms = meteo_areas.iloc[0, :][0] if tm == [] else tm[0].code

                    # add prefix to the overflow id to create the paved-node id
                    paved_drr.at[ov.code, "id"] = str(ov.code)
                    paved_drr.at[ov.code, "area"] = str(pav_area * ov.fractie)
                    paved_drr.at[ov.code, "surface_level"] = f"{elev:.2f}"
                    
                    # if a float is given, a standard value is passed. If a string is given, a rastername is assumed to zonal statistics are applied.
                    if isinstance(street_storage,  float):
                        paved_drr.at[
                            ov.code, "street_storage"
                        ] = f"{street_storage:.2f}"
                    elif isinstance(street_storage, (Path, str)):
                        paved_drr.at[
                            ov.code, "street_storage"
                        ] = f'{str_stors_sa[isew]["mean"]:.2f}'
                    else:
                        raise ValueError('Street_storage has the wrong datatype. It should be a filename (Path or string) or number (float or int).')
                    
                    # three options: it can be an attribute of a sewer area, a uniform value or a raster
                    if sew.riool_berging_mm is None or np.isnan(sew.riool_berging_mm) or not isinstance(sew.riool_berging_mm, float):  
                        if isinstance(sewer_storage, float):
                            paved_drr.at[ov.code, "sewer_storage"] = f"{sewer_storage:.2f}"   
                        elif isinstance(sewer_storage, (Path, str)):
                            paved_drr.at[
                            ov.code, "sewer_storage"
                            ] = f'{sew_stors_sa[isew]["mean"]:.2f}'
                        else:
                            raise ValueError('Sewer_storage has the wrong datatype. It should be a filename (Path or string) or number (float or int).')                            
                    else:                                     
                        paved_drr.at[ov.code, "sewer_storage"] = f'{sew.riool_berging_mm:.2f}'
                    
                    # three options: it can be an attribute of a sewer area, a uniform value or a raster
                    if sew.riool_poc_m3s is None or np.isnan(sew.riool_poc_m3s) or not isinstance(sew.riool_poc_m3s, float):                    
                         if isinstance(pump_capacity, float):
                            # convert the value from mm/h to m3/s
                            paved_drr.at[ov.code, "pump_capacity"] = f"{pump_capacity * (float(pav_area) * ov.fractie) / (1000. * 3600.):.8f}"   
                         elif isinstance(pump_capacity, (Path, str)):
                            # convert the value (extracted from the raster) from mm/h to m3/s
                            paved_drr.at[
                            ov.code, "pump_capacity"
                            ] = f'{pump_caps_sa[isew]["mean"] * (float(pav_area) * ov.fractie) / (1000. * 3600.):.8f}'
                         else:
                            raise ValueError('Pump_capacity has the wrong datatype. It should be a filename (Path or string) or number (float or int).')        
                    else:
                        # use the attribute value
                        paved_drr.at[ov.code, "pump_capacity"] = f'{sew.riool_poc_m3s * ov.fractie:.8f}'
                                              
                    paved_drr.at[ov.code, "meteo_area"] = str(ms)
                    paved_drr.at[ov.code, "px"] = f"{ov.geometry.coords[0][0]+10:.0f}"
                    paved_drr.at[ov.code, "py"] = f"{ov.geometry.coords[0][1]:.0f}"
                    paved_drr.at[ov.code, "boundary_node"] = ov.code
        else:
            # in this case only the catchments are taken into account. A node is created for every catchment nonetheless, but only nodes with a remaining area >0 are written.
            paved_drr.set_data(
                pd.DataFrame(
                    np.zeros((len(catchments), 10)),
                    columns=[
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
                    ],
                    dtype="str",
                ),
                index_col="id",
            )
            paved_drr.index = catchments.code

        for num, cat in enumerate(catchments.itertuples()):
            # if no rasterdata could be obtained for this catchment, skip it.
            if mean_elev[num]["median"] is None:
                logger.warning("No rasterdata available for catchment %s.", cat.code)
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
                        pixels = zonal_stats(
                            area_outside_sewer,
                            lu_rast.astype(int),
                            affine=lu_affine,
                            categorical=True,
                            all_touched=all_touched,
                        )[0]
                        if 14.0 in pixels:
                            pav_area = str(pixels[14.0] * px_area)
                        else:
                            pav_area = 0.0
                else:
                    # all of the catchment is outside the sewer area
                    pixels = zonal_stats(
                                cat.geometry,
                                lu_rast.astype(int),
                                affine=lu_affine,
                                categorical=True,
                                all_touched=all_touched,
                            )[0]
                    if 14.0 in pixels:
                        pav_area = str(pixels[14.0] * px_area)
                    else:
                        pav_area = 0.0
            else:
                # there is no sewer area at all
                pav_area = (
                    str(lu_counts[num][14.0] * px_area)
                    if 14.0 in lu_counts[num]
                    else "0"
                )

            # find corresponding meteo-station
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, :][0] if tm == [] else tm[0].code

            elev = mean_elev[num]["median"]
            paved_drr.at[cat.code, "id"] = str(cat.code)
            paved_drr.at[cat.code, "area"] = str(pav_area)  #
            paved_drr.at[cat.code, "surface_level"] = f"{elev:.2f}"
            # if a float is given, a standard value is passed. If a string is given, a rastername is assumed to zonal statistics are applied.
            if isinstance(street_storage, float):
                paved_drr.at[cat.code, "street_storage"] = f"{street_storage:.2f}"
            else:
                paved_drr.at[
                    cat.code, "street_storage"
                ] = f'{str_stors[num]["mean"]:.2f}'
            if isinstance(sewer_storage, float):
                paved_drr.at[cat.code, "sewer_storage"] = f"{sewer_storage:.2f}"
            else:
                paved_drr.at[
                    cat.code, "sewer_storage"
                ] = f'{sew_stors[num]["mean"]:.2f}'
            if isinstance(pump_capacity, float):
                paved_drr.at[cat.code, "pump_capacity"] = f'{(pump_capacity * float(pav_area)) / (1000. * 3600.):.8f}'
            else:
                paved_drr.at[
                    cat.code, "pump_capacity"
                ] = f'{pump_caps[num]["mean"] * (float(pav_area)) / (1000. * 3600.):.8f}'
            
            paved_drr.at[cat.code, "meteo_area"] = str(ms)
            paved_drr.at[
                cat.code, "px"
            ] = f"{cat.geometry.centroid.coords[0][0]+10:.0f}"
            paved_drr.at[cat.code, "py"] = f"{cat.geometry.centroid.coords[0][1]:.0f}"
            paved_drr.at[cat.code, "boundary_node"] = cat.boundary_node
        [self.paved.add_paved(**pav) for pav in paved_drr.to_dict("records")]


class GreenhouseIO:
    def __init__(self, greenhouse):
        self.greenhouse = greenhouse

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def greenhouse_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Union[Path, str],
        surface_level: Union[Path, str],
        roof_storage: Union[StrictStr,float],
        meteo_areas: ExtendedGeoDataFrame,
        zonalstats_alltouched: bool = None,        
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
        all_touched = False if zonalstats_alltouched is None else zonalstats_alltouched

        lu_rast, lu_affine = self.greenhouse.drrmodel.read_raster(landuse, static=True)
        lu_counts = zonal_stats(
            gpd.GeoDataFrame(catchments),
            lu_rast.astype(int),
            affine=lu_affine,
            categorical=True,
            all_touched=all_touched,
        )
        rast, affine = self.greenhouse.drrmodel.read_raster(surface_level, static=True)
        mean_elev = zonal_stats(
            gpd.GeoDataFrame(catchments), rast.astype(float), affine=affine, stats="median", all_touched=all_touched
        )
        if greenhouse_areas is not None:
            mean_elev_gh = zonal_stats(
                 gpd.GeoDataFrame(greenhouse_areas), rast.astype(float), affine=affine, stats="median", all_touched=all_touched
            )
            
        # optional rasters
        if isinstance(roof_storage, (Path, str)):
            rast, affine = self.greenhouse.drrmodel.read_raster(
                roof_storage, static=True
            )
            roofstors = zonal_stats(
                 gpd.GeoDataFrame(catchments), rast.astype(float), affine=affine, stats="mean", all_touched=True
            )
            if greenhouse_areas is not None:
                roofstors_gh = zonal_stats(
                     gpd.GeoDataFrame(greenhouse_areas), rast.astype(float), affine=affine, stats="mean", all_touched=True
                )

        # get raster cellsize
        px_area = lu_affine[0] * -lu_affine[4]


        numgh = catchments.shape[0]
        indexgh = catchments.code
        if greenhouse_areas is not None:
            numgh = numgh + greenhouse_areas.shape[0]
            indexgh = indexgh + greenhouse_areas.code

        gh_drr = ExtendedDataFrame(required_columns=["id"])
        gh_drr.set_data(
            pd.DataFrame(
                np.zeros((numgh, 8)),
                columns=[
                    "id",
                    "area",
                    "surface_level",
                    "roof_storage",
                    "meteo_area",
                    "px",
                    "py",
                    "boundary_node",
                ],
                dtype="str",
            ),
            index_col="id",
        )
        gh_drr.index = indexgh
        if greenhouse_areas is not None:
            for num, gh in enumerate(greenhouse_areas.itertuples()):
                # find corresponding meteo-station
                if mean_elev_gh[num]["median"] is None:
                    logger.warning("No rasterdata available for catchment %s.", gh.code)
                    continue
                tm = [
                    m
                    for m in meteo_areas.itertuples()
                    if m.geometry.contains(gh.geometry.centroid)
                ]
                ms = meteo_areas.iloc[0, :][0] if tm == [] else tm[0].code

                elev = mean_elev_gh[num]["median"]
                gh_drr.at[gh.code, "id"] = str(gh.code)
                gh_drr.at[gh.code, "area"] = gh.geometry.area
                gh_drr.at[gh.code, "surface_level"] = f"{elev:.2f}"
                if hasattr(gh, 'roof_storage_mm') & (~(np.isnan(gh.roof_storage_mm))):
                    gh_drr.at[gh.code, "roof_storage"] = f"{gh.roof_storage_mm:.2f}"                                                 
                if isinstance(roof_storage, float):
                    gh_drr.at[gh.code, "roof_storage"] = f"{roof_storage:.2f}"
                else:
                    gh_drr.at[gh.code, "roof_storage"] = f'{roofstors_gh[num]["mean"]:.2f}'
                if hasattr(gh, 'basin_storage_class') & (~(np.isnan(gh.basin_storage_class))):
                    gh_drr.at[gh.code, 'basin_storage_class'] = f"{gh.basin_storage_class:g}"
                else:
                    gh_drr.at[gh.code, "basin_storage_class"] = f'{basin_storage_class:g}'
                gh_drr.at[gh.code, "meteo_area"] = str(ms)
                gh_drr.at[gh.code, "px"] = f"{gh.geometry.centroid.coords[0][0]:.0f}"
                gh_drr.at[gh.code, "py"] = f"{gh.geometry.centroid.coords[0][1]:.0f}"
                latcode = greenhouse_laterals[greenhouse_laterals.codegerelateerdobject == gh.code].code.values[0]
                gh_drr.at[gh.code, "boundary_node"] = str(latcode)           
            [self.greenhouse.add_greenhouse(**gh) for gh in gh_drr.to_dict("records")]

        for num, cat in enumerate(catchments.itertuples()):
            # if no rasterdata could be obtained for this catchment, skip it.
            if mean_elev[num]["median"] is None:
                logger.warning("No rasterdata available for catchment %s.", cat.code)
                continue

            # find corresponding meteo-station
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, :][0] if tm == [] else tm[0].code

            if greenhouse_areas is not None:
                if cat.geometry.intersects(greenhouse_areas.geometry).any():
                    intersection_area = cat.geometry.intersection(greenhouse_areas.geometry).area
                    intersection_area = intersection_area[intersection_area > 0.].values[0]                    
                    if 15 in lu_counts[num]:
                        # divide area to subtract between greenhouses and the most occurring area                                                
                        logger.info(
                            "Catchment: %s: subtracting %s m2 from greenhouse area in landuse map.",
                            cat.code,
                            np.min([(lu_counts[num][15] * px_area, intersection_area)]),
                        )
                        lu_counts[num][15] = np.max([0., (lu_counts[num][15] - np.round(intersection_area/px_area))])                                                               
            
            elev = mean_elev[num]["median"]
            gh_drr.at[cat.code, "id"] = str(cat.code)
            gh_drr.at[cat.code, "area"] = (
                str(lu_counts[num][15] * px_area) if 15 in lu_counts[num] else "0"
            )
            gh_drr.at[cat.code, "surface_level"] = f"{elev:.2f}"
            if isinstance(roof_storage, float):
                gh_drr.at[cat.code, "roof_storage"] = f"{roof_storage:.2f}"
            else:
                gh_drr.at[cat.code, "roof_storage"] = f'{roofstors[num]["mean"]:.2f}'
            gh_drr.at[cat.code, "basin_storage_class"] = f'{basin_storage_class:g}'
            gh_drr.at[cat.code, "meteo_area"] = str(ms)
            gh_drr.at[cat.code, "px"] = f"{cat.geometry.centroid.coords[0][0]+20:.0f}"
            gh_drr.at[cat.code, "py"] = f"{cat.geometry.centroid.coords[0][1]:.0f}"
            gh_drr.at[cat.code, "boundary_node"] = cat.boundary_node
        [self.greenhouse.add_greenhouse(**gh) for gh in gh_drr.to_dict("records")]

class OpenwaterIO:
    def __init__(self, openwater):
        self.openwater = openwater

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def openwater_from_input(
        self,
        catchments: ExtendedGeoDataFrame,
        landuse: Union[Path, str],
        meteo_areas: ExtendedGeoDataFrame,
        zonalstats_alltouched: bool = None,
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
        all_touched = False if zonalstats_alltouched is None else zonalstats_alltouched

        lu_rast, lu_affine = self.openwater.drrmodel.read_raster(landuse, static=True)
        lu_counts = zonal_stats(
            gpd.GeoDataFrame(catchments),
            lu_rast.astype(int),
            affine=lu_affine,
            categorical=True,
            all_touched=all_touched,
        )

        # get raster cellsize
        px_area = lu_affine[0] * -lu_affine[4]

        ow_drr = ExtendedDataFrame(required_columns=["id"])
        ow_drr.set_data(
            pd.DataFrame(
                np.zeros((len(catchments), 6)),
                columns=["id", "area", "meteo_area", "px", "py", "boundary_node"],
                dtype="str",
            ),
            index_col="id",
        )
        ow_drr.index = catchments.code
        for num, cat in enumerate(catchments.itertuples()):
            # find corresponding meteo-station
            tm = [
                m
                for m in meteo_areas.itertuples()
                if m.geometry.contains(cat.geometry.centroid)
            ]
            ms = meteo_areas.iloc[0, :][0] if tm == [] else tm[0].code

            ow_drr.at[cat.code, "id"] = str(cat.code)
            ow_drr.at[cat.code, "area"] = (
                str(lu_counts[num][13] * px_area) if 13 in lu_counts[num] else "0"
            )
            ow_drr.at[cat.code, "meteo_area"] = str(ms)
            ow_drr.at[cat.code, "px"] = f"{cat.geometry.centroid.coords[0][0]-20:.0f}"
            ow_drr.at[cat.code, "py"] = f"{cat.geometry.centroid.coords[0][1]:.0f}"
            ow_drr.at[cat.code, "boundary_node"] = cat.boundary_node
        [self.openwater.add_openwater(**ow) for ow in ow_drr.to_dict("records")]


class ExternalForcingsIO:
    def __init__(self, external_forcings):
        self.external_forcings = external_forcings

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def seepage_from_input(
        self, catchments: ExtendedGeoDataFrame, seepage_folder: Union[Path, str]
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
        for ifile, file in tqdm(
            enumerate(file_list), total=len(file_list), desc="Reading seepage files"
        ):
            if file.endswith('.idf'):
                dataset = idfreader.open(os.path.join(seepage_folder, file))                                
                array = dataset.squeeze().values
                header = idfreader.header(os.path.join(seepage_folder, file), pattern=None)
                affine = from_origin(
                    header["xmin"], header["ymax"], header["dx"], header["dx"]
                )
                time = header['time']
                convert_units=True
            else:
                array, affine, time = self.external_forcings.drrmodel.read_raster(
                    os.path.join(seepage_folder, file)
                )
            times.append(time)
            stats = zonal_stats(
                 gpd.GeoDataFrame(catchments), array, affine=affine, stats="mean", all_touched=True
            )
            arr[ifile, :] = [s["mean"] for s in stats]
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


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def precip_from_input(
        self,
        areas: ExtendedGeoDataFrame,
        precip_folder: Union[Path, str] = None,
        precip_file: Union[Path, str] = None,
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
            arr = np.zeros((len(file_list), len(areas.code)))
            for ifile, file in tqdm(
                enumerate(file_list),
                total=len(file_list),
                desc="Reading precipitation files",
            ):
                array, affine, time = self.external_forcings.drrmodel.read_raster(
                    os.path.join(precip_folder, file)
                )
                times.append(time)
                stats = zonal_stats(
                     gpd.GeoDataFrame(areas), array, affine=affine, stats="mean", all_touched=True
                )
                arr[ifile, :] = [s["mean"] for s in stats]
            result = pd.DataFrame(
                arr, columns=["ms_" + str(area) for area in areas.code]
            )
            result.index = times
            [self.external_forcings.add_precip(*prec) for prec in result.items()]
        else:
            self.external_forcings.precip = str(precip_file)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evap_from_input(
        self,
        areas: ExtendedGeoDataFrame,
        evap_folder: Union[Path, str] = None,
        evap_file: Union[Path, str] = None,
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
            arr = np.zeros((len(file_list), len(areas)))
            for ifile, file in tqdm(
                enumerate(file_list),
                total=len(file_list),
                desc="Reading evaporation files",
            ):
                array, affine, time = self.external_forcings.drrmodel.read_raster(
                    os.path.join(evap_folder, file)
                )
                times.append(time)
                stats = zonal_stats(
                     gpd.GeoDataFrame(areas), array, affine=affine, stats="mean", all_touched=True
                )
                arr[ifile, :] = [s["mean"] for s in stats]
            result = pd.DataFrame(
                arr, columns=["ms_" + str(area) for area in areas.code]
            )
            result.index = times
            [self.external_forcings.add_evap(*evap) for evap in result.items()]
        else:
            self.external_forcings.evap = str(evap_file)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
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
        if any(drop_idx):
            logger.warning(
                "%d catchments removed because of an area of 0 m2.",
                len(drop_idx),
            )
            catchments.drop(drop_idx, inplace=True)

        for i in not_occurring:
            catchments.drop(
                catchments[catchments.boundary_node == i].code.iloc[0],
                axis=0,
                inplace=True,
            )

        numlats = len(catchments)
        if overflows is not None:
            numlats = numlats + len(overflows)
        if greenhouse_laterals is not None:
            numlats = numlats + len(greenhouse_laterals)
            
        bnd_drr = ExtendedDataFrame(required_columns=["id"])
        bnd_drr.set_data(
            pd.DataFrame(
                np.zeros((numlats, 3)), columns=["id", "px", "py"], dtype="str"
            ),
            index_col="id",
        )
        index = catchments.code
        if overflows is not None:
            index = pd.concat([index, overflows.code], ignore_index=True)
        if greenhouse_laterals is not None:
            index = pd.concat([index, greenhouse_laterals.code], ignore_index=True)
        
        bnd_drr.index = index
        for num, cat in enumerate(catchments.itertuples()):
            # logger.info(num, cat.code)
            if boundary_nodes[boundary_nodes["globalid"] == cat.lateraleknoopid].empty:
                # raise IndexError(f'{cat.code} not connected to a boundary node. Skipping.')
                logger.warning(
                    f"{cat.code} not connected to a boundary node. Skipping."
                )
                continue
            bnd_drr.at[cat.code, "id"] = f'lat_{cat.code}'
            bnd_drr.at[cat.code, "px"] = str(
                boundary_nodes[boundary_nodes["globalid"] == cat.lateraleknoopid][
                    "geometry"
                ].x.iloc[0]
            ).strip()
            bnd_drr.at[cat.code, "py"] = str(
                boundary_nodes[boundary_nodes["globalid"] == cat.lateraleknoopid][
                    "geometry"
                ].y.iloc[0]
            ).strip()
        if overflows is not None:
            logger.info("Adding overflows to the boundary nodes.")
            for num, ovf in enumerate(overflows.itertuples()):
                bnd_drr.at[ovf.code, "id"] = str(ovf.code)
                bnd_drr.at[ovf.code, "px"] = str(ovf.geometry.coords[0][0])
                bnd_drr.at[ovf.code, "py"] = str(ovf.geometry.coords[0][1])
        if greenhouse_laterals is not None:
            logger.info("Adding greenhouse_laterals to the boundary nodes.")
            for num, gh in enumerate(greenhouse_laterals.itertuples()):
                bnd_drr.at[gh.code, "id"] = str(gh.code)
                bnd_drr.at[gh.code, "px"] = str(gh.geometry.coords[0][0])
                bnd_drr.at[gh.code, "py"] = str(gh.geometry.coords[0][1])
        [
            self.external_forcings.add_boundary_node(**bnd)
            for bnd in bnd_drr.to_dict("records")
        ]
