# -*- coding: utf-8 -*-

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from scipy.spatial import distance
import logging

from hydromt import config
import hydromt.io

logger = logging.getLogger(__name__)


__all__ = ["slice_geodataframe", "retype_geodataframe"]


def slice_geodataframe(gdf, required_query: str = None, required_columns: list = None):
    """Function to read gpd.GeoDataFrame with preprocessing: rename, slice, convert type and set index"""

    # check data
    if gdf is None or len(gdf) == 0:
        logger.error(f"GeoDataFrame: no slicing is applied. data is None or empty.")
        return gdf
    else:
        _data = gdf

    # check arguments
    if isinstance(required_columns, str):
        required_columns = [required_columns]

    # column wise slicing
    if required_columns is None:
        logger.debug(
            f"GeoDataFrame: no column-wise slicing and retyping applied. required_columns is not specified"
        )
        data = _data

    else:

        if not set(required_columns).issubset(_data.columns):
            logger.error(
                f"GeoDataFrame: cannot perform slicing. required_columns must exist in data columns {_data.columns}."
            )

        else:
            try:
                logger.debug(
                    f"GeoDataFrame: column-wise slicing data with required_columns: {required_columns}"
                )
                data = gpd.GeoDataFrame(geometry=_data["geometry"])
                for c in required_columns:
                    data[c] = _data[c]
            except Exception as e:
                logger.error(e)

    # row-wise slicing
    if required_query is not None:
        try:
            logger.debug(
                f"GeoDataFrame: row-wise slicing data with required_dtypes: {required_columns}"
            )
            data = data.query(required_query)
        except Exception as e:
            logger.error(e)
    else:
        logger.debug(
            f"GeoDataFrame: no row-wise slicing applied. required_query is not specified."
        )

    if len(data) == 0:
        logger.error(f"GeoDataFrame: Zero items are left after slicing.")

    return data


def retype_geodataframe(gdf, retype=None):

    if retype is None or len(retype) == 0:
        logger.debug(f"GeoDataFrame: no retyping is applied. retype is not specified.")

    else:
        cols = gdf.columns
        if not isinstance(retype, dict):
            astype = {c: retype for c in cols}
        else:
            if not set(retype.keys()).issubset(gdf.columns):
                logger.warning(
                    f"GeoDataFrame: retyping is not fully applied. retype must exist in {gdf.columns}."
                )
            astype = {k: v for k, v in retype.items() if k in cols}

        for c in cols:
            mv = astype.get(c, None)
            if mv is not None:
                if mv == "bool":
                    gdf[c] = gdf[c].replace(
                        {
                            "TRUE": True,
                            "True": True,
                            "true": True,
                            "1": True,
                            "FALSE": False,
                            "False": False,
                            "false": False,
                            "0": False,
                        }
                    )

                try:
                    gdf[c] = gdf[c].replace({"NULL": None, "None": None, "none": None})
                    gdf[c] = gdf[c].astype(mv)
                except ValueError:
                    raise ValueError(
                        f"GeoDataFrame: retype ({mv}) could not be performed on column {c}"
                    )
    return gdf
