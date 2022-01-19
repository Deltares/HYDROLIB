# -*- coding: utf-8 -*-

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry import (
    Polygon,
    LineString,
    Point,
    MultiLineString,
    box,
    MultiPoint,
)
from shapely.ops import split, snap
from scipy.spatial import distance
import pathlib
import configparser
import logging

from hydromt import config
import hydromt.io

logger = logging.getLogger(__name__)


__all__ = [
    "slice_geodataframe",
    "retype_geodataframe",
    "parse_ini",
    "append_data_columns_based_on_ini_query",
    "check_geodataframe",
    "split_lines",
]


## IO


def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False


def isint(x):
    try:
        int(x)
        return True
    except:
        return False


def parse_arg(arg, dtype=None):
    if dtype is None:
        if arg in ["None", "none", "NONE", None, "", "NULL"]:
            arg_ = None
        elif arg in ["FALSE", "False", "false", False]:
            arg_ = False
        elif arg in ["TRUE", "True", "true", True]:
            arg_ = True
        elif isint(arg):
            arg_ = int(arg)
        elif isfloat(arg):
            arg_ = float(arg)
        elif isinstance(arg, pathlib.Path):
            if arg.is_file() == False:
                arg_ = None  # path
            else:
                arg_ = str(arg)  # filespath
        elif isinstance(arg, dict):
            arg_ = arg
        elif "{" in arg:
            arg_ = eval(arg)  # dict
        elif "," in arg:
            arg_ = arg.split(",")
            arg_ = [a.strip() for a in arg_]  # list
        elif arg.startswith("'") & arg.endswith("'"):
            arg_ = arg[1:-1]  # reserve string
        else:
            arg_ = str(arg)  # str

    else:
        if dtype in [bool, int, float, str]:
            arg_ = dtype(arg)
        else:
            raise Warning("Not supported dtype")

    return arg_


def parse_ini(ini_fn):
    logger.info(f"parsing settings from {ini_fn}")
    config = configparser.ConfigParser(inline_comment_prefixes=[";", "#"])
    config.optionxform = str  # case sensitive parsing
    _ = config.read(ini_fn)
    config_ = {}
    for section in config.sections():
        config_[section] = {}
        for k, v in config[section].items():
            config_[section][k] = parse_arg(v)
    return config_


## GeoDataFrame handling


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


def append_data_columns_based_on_ini_query(
    data: gpd.GeoDataFrame, ini: configparser.ConfigParser, keys: list = []
):
    """append key,val pair as data columns for the input GeiDataFrame based on ini [default] or [query] sections"""
    # TODO check this function
    _columns = list(data.columns)
    for section in ini.keys():
        if section == "global":  # do nothing for global settings
            pass
        elif section == "default":  # append default key,value pairs
            for key, val in ini[section].items():
                try:
                    val = float(val)
                except:
                    pass
                if key in _columns:
                    data.loc[data[key].isna(), key] = val
                else:
                    data.loc[:, key] = val
        else:  # overwrite default key,value pairs based on [query]
            for key, val in ini[section].items():
                try:

                    try:
                        val = float(val)
                    except:
                        pass
                    if key in _columns:
                        d = data.query(section)
                        idx = d.loc[d[key].isna(), :].index
                        data.loc[idx, key] = val
                    else:
                        idx = data.query(section).index
                        data.loc[idx, key] = val
                except:
                    logger.warning(
                        f"Unable to query: adding default values from section {section} failed"
                    )
                    pass  # do not query if error
    _columns_ = list(data.columns)
    if len(keys) > 0:
        columns = _columns + keys
    else:
        columns = _columns_
    return data.loc[:, columns]


def check_geodataframe(gdf):
    if gdf is None or len(gdf) == 0:
        check = False
        logger.warning("GeoDataFrame: do not have valid features. ")
    else:
        check = True
    return check


## geometry
def split_lines(line, num_new_lines):
    """function to get a list of lines splitted from line"""
    _line = [line]
    points = [
        line.interpolate((i / num_new_lines), normalized=True)
        for i in range(0, num_new_lines + 1)
    ]

    new_lines = []
    for n, p in enumerate(points):
        split_line = split(snap(line, p, 1e-8), p)
        segments = [feature for feature in split_line]
        if n == 0:
            line = segments[0]
        elif n == num_new_lines:
            new_lines.append(segments[0])
        else:
            new_lines.append(segments[0])
            line = segments[-1]

    assert (
        len(new_lines) == num_new_lines
    ), "number of lines after splitting does not match input"
    assert np.isclose(
        sum([l.length for l in new_lines]), _line[0].length
    ), "length after splitting does not match input"

    return new_lines
