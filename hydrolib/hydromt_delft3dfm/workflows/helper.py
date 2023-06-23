# -*- coding: utf-8 -*-

import configparser
import logging
import pathlib

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)
from shapely.ops import snap, split

logger = logging.getLogger(__name__)


__all__ = [
    "slice_geodataframe",
    "retype_geodataframe",
    "write_shp",
    "parse_ini",
    "append_data_columns_based_on_ini_query",
    "split_lines",
    "check_geodataframe",
    "check_gpd_attributes",
    "update_data_columns_attributes_based_on_filter",
    "get_gdf_from_branches",
]


## IO


def isfloat(x):
    """Determines whether `x` is a float by trying to cast it to a float.

    Parameters
    ----------
    x
        The instance to check.

    Returns
    -------
    bool
        True if `x` is a float, otherwise False.
    """

    try:
        float(x)
        return True
    except:
        return False


def isint(x):
    """Determines whether `x` is an integer by trying to cast it to an integer.

    Parameters
    ----------
    x
        The instance to check.

    Returns
    -------
    bool
        True if `x` is an integer, otherwise False.
    """
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


def parse_ini(ini_fn) -> dict:
    """Parses an ini file to a dictionary.

    Parameters
    ----------
    ini_fn
        The location of the ini file.

    Returns
    -------
    dict
        A dictionary containing the sections with as value a dictionary with the keys and values.
    """
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


## GeoDataFrame handeling


def slice_geodataframe(
    gdf: gpd.GeoDataFrame,
    required_query: str = None,
    required_columns: list = None,
    logger=logger,
):
    """Function to read gpd.GeoDataFrame with preprocessing: rename, slice, convert type and set index.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame.
        The geo data frame to slice.
    required_query : str, optional
        The required query. Defaults to None.
    required_columns : list, optional
        The required columns. Default to None.
    logger
        The logger to log messages with.

    Returns
    -------
    gpd.GeoDataFrame
        The sliced geo data frame.
    """

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


def retype_geodataframe(gdf: gpd.GeoDataFrame, retype=None, logger=logger):
    """Retype a GeoDataFrame."""

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


def eval_funcs(gdf: gpd.GeoDataFrame, funcs: dict, logger=logger):
    """Evaluate funcs on the geo data frame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The geo data frame to update.
    funcs : dict
        A dictionary containing key-value pair describing a column name with a string describing the operation to evaluate.
    logger
        The logger to log messages with.

    Returns
    -------
    gpd.GeoDataFrame
        The geo data frame with the updated columns.
    """
    if funcs is None or len(funcs) == 0:
        logger.debug(f"GeoDataFrame: no funcs is applied. funcs is not specified.")
        return gdf

    for k, v in funcs.items():
        try:
            # eval funcs for columns that exist
            if v in gdf.columns:
                if "geometry" in v:
                    # ensure type of geoseries - might be a bug in pandas / geopandas
                    _ = type(gdf["geometry"])
                gdf[k] = gdf.eval(v)
                logger.debug(f"GeoDataFrame:  update column {k} based on {v}")
            # assign new columns using given values
            else:
                gdf[k] = v
                logger.debug(f"GeoDataFrame: update column {k} based on {v}")
        except Exception as e:
            logger.debug(f"GeoDataFrame: can not update column {k} based on {v}: {e}")
    return gdf


def write_shp(data: gpd.GeoDataFrame, filename: str, columns: list = None):
    """Write a geo data frame to a shape file.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        The geo data frame to write to file.
    filename : str
        The shape file name to write to.
    columns : list, optional
        The list of columns to write. The geometry column will be added if it it missing.
        If not specified, all the columns in the dataset will be written.
        Default to None.
    """

    if data is not None:
        # convert to numerical
        data = data.apply(pd.to_numeric, errors="ignore")
        # convert list to strings
        for c in data.columns:
            if isinstance(data[c][0], list):
                data[c] = data[c].apply(";".join)
        if columns is not None:
            if "geometry" not in columns:
                columns = columns + ["geometry"]
            gpd.GeoDataFrame(data[columns]).to_file(filename, index=False)
        else:
            gpd.GeoDataFrame(data).to_file(filename, index=False)


# data handling


def append_data_columns_based_on_ini_query(
    data: gpd.GeoDataFrame,
    ini: configparser.ConfigParser,
    keys: list = [],
    logger=logger,
):
    """append key,val pair as data columns for the input GeoDataFrame based on ini [default] or [query] sections

    Parameters
    ----------
    data : gpd.GeoDataFrame
        The geo data frame to append to.
    ini : configparser.ConfigParser
        The ini config parser.
    keys : list, optional
        The keys to add to the geo data frame columns.
    logger
        The logger to log messages with.

    Returns
    -------
    gpd.GeoDataFrame
        The updated geo data frame.
    """
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


def check_geodataframe(gdf: gpd.GeoDataFrame):
    """Check the geo data frame for None and length.
    A warning will be logged, if the geo data frame is None or empty.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The geo data frame to check.

    Returns
    -------
    bool
        True if `gdf` is not None and has at least one entry; otherwise, False.
    """

    if gdf is None or len(gdf) == 0:
        check = False
        logger.warning("GeoDataFrame: do not have valid features. ")
    else:
        check = True
    return check


## geometry
def cut_pieces(line, distances):
    """cut a line into pieces based on distances"""
    if distances[0] != 0:
        distances.insert(0, 0)
    if distances[-1] == line.length:
        distances.pop(-1)
    pieces = [line]
    for d in np.diff(np.sort(distances)):
        line = pieces.pop(-1)
        pieces.extend(cut(line, d))
    return pieces


def cut(line, distance):
    """Cuts a line in two at a distance from its starting point
    ref: https://shapely.readthedocs.io/en/stable/manual.html"""
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:]),
            ]


def split_lines(line, num_new_lines):
    """Get a list of lines splitted from a line.

    Parameters
    ----------
    line
        The line to split
    num_new_lines : int
        The desired number of lines.

    Returns
    -------
    list
        The new lines.
    """
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


def check_gpd_attributes(
    gdf: gpd.GeoDataFrame, required_columns: list, raise_error: bool = False
):
    """check if the geodataframe contains all required columns

    Parameters
    ----------
    gdf : gpd.GeoDataFrame, required
        GeoDataFrame to be checked
    required_columns: list of strings, optional
        Check if the geodataframe contains all required columns
    raise_error: boolean, optional
        Raise error if the check failed
    """
    if not (set(required_columns).issubset(gdf.columns)):
        if raise_error:
            raise ValueError(
                f"GeoDataFrame do not contains all required attributes: {required_columns}."
            )
        else:
            logger.warning(
                f"GeoDataFrame do not contains all required attributes: {required_columns}."
            )
        return False
    return True

def update_data_columns_attributes_based_on_filter(
    gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
    filter_column: str,
    filter_value: str = None,
):
    """
    Add or update columns in the geodataframe based on column and values in attributes dataframe

    If filter_column and filter_value is set, only update the attributes of the filtered geodataframe.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geodataframe containing user input
    df : attribute DataFrame
        a pd.DataFrame with attribute columns and values (e.g. width =  1) per filter_value in the filter_column (e.g. branch_type = pipe)
    filter_column : str
        Name of the column linking df to gdf.
    filter_value: str
        Value of the filter in the filter column.
        Must be used with filter_column

    Raises
    ------
    ValueError:
        if Name of the column linking df to gdf (`filter_column`) is not specified

    Returns
    -------
    gdf : gpd.GeoDataFrame
        geodataframe containing user input filled with new attribute values.

    """
    if filter_column is None:
        raise ValueError("Name of the column linking df to gdf must be specified")
    
    if filter_value is not None:
        attributes = df[df[filter_column] == filter_value]
    else:
        attributes = df
    
    # Update attributes
    for i in range(len(attributes.index)):
        row = attributes.iloc[i, :]
        filter_value = row.loc[filter_column]
        for colname in row.index[1:]:
            # If attribute is not at all in branches, add a new column
            if colname not in gdf.columns:
                gdf[colname] = pd.Series(dtype=attributes[colname].dtype)
            # Then fill in empty or NaN values with defaults
            gdf.loc[
                np.logical_and(
                    gdf[colname].isna(), gdf[filter_column] == filter_value
                ),
                colname,
            ] = row.loc[colname]

    return gdf


def get_gdf_from_branches(branches:gpd.GeoDataFrame, df:pd.DataFrame) -> gpd.GeoDataFrame:
    """Get geodataframe from dataframe. 
    Based on interpolation of branches, using columns ["branchid", "chainage" in df]
    
    Parameters
    ----------
    branches:gpd.GeoDataFrame
        line geometries of the branches
        Required varaibles: ["branchId"/"branchid", "geometry" ]
    df:pd.DataFrame
        dataframe containing the features located on branches
        Required varaibles: ["branchId"/"branchid", "chainage" ]
    
    Return
    ------
    gdf:gpd.GeoDataFrame
        dataframe cotaining the features located on branches, with point geometry
    
    """
    branches.columns = branches.columns.str.lower()
    branches = branches.set_index("branchid")
    
    df["geometry"] = None
    
    # Iterate over each point and interpolate a point along the corresponding line feature
    for i, row in df.iterrows():
        line_geometry = branches.loc[row.branchid, "geometry"]
        new_point_geometry = line_geometry.interpolate(row.chainage)
        df.loc[i, "geometry"] = new_point_geometry
    
    return gpd.GeoDataFrame(df)
