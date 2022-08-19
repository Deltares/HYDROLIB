from pathlib import Path
from typing import Dict, Union

import geopandas as gpd
import numpy as np

from .workflows import helper

__all__ = ["get_process_geodataframe"]


def get_process_geodataframe(
    self,
    path_or_key: str,
    id_col: str,
    clip_buffer: float = 0,  # TODO: think about whether to keep/remove, maybe good to put into the ini file.
    clip_predicate: str = "contains",
    retype: dict = dict(),
    funcs: dict = dict(),
    variables: list = None,
    required_query: str = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Function to open and process geodataframe.

    This function combines a wrapper around :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataframe`

    Parameters
    ----------
    path_or_key : str
        Data catalog key. If a path to a vector file is provided it will be added
        to the data_catalog with its based on the file basename without extension.
    id_col : str
        The id column name.
    clip_buffer : float, optional
        Buffer around the `bbox` or `geom` area of interest in meters. Defaults to 0.
    clip_predicate : {'contains', 'intersects', 'within', 'overlaps', 'crosses', 'touches'}, optional
        If predicate is provided, the GeoDataFrame is filtered by testing
        the predicate function against each item. Requires bbox or mask.
        Defaults to 'contains'.
    retype : dict, optional
        Dictionary containing retypes. Defaults to an empty dictionary.
    funcs : dict, optional
        A dictionary containing key-value pair describing a column name with a string describing the operation to evaluate. Defaults to an empty dictionary.
    variables : list, optional
        Names of GeoDataFrame columns to return. By default all columns are returned. Defaults to None.
    required_query : str, optional
        The required query. Defaults to None. 

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        GeoDataFrame

    Raises
    ------
    ValueError
        If `id_col` does not exist in the opened data frame.
    """

    #    # pop kwargs from catalogue
    #    d = self.data_catalog.to_dict(path_or_key)[path_or_key].pop("kwargs")

    #    # set clipping
    #    clip_buffer = d.get("clip_buffer", clip_buffer)
    #    clip_predicate = d.get("clip_predicate", clip_predicate)  # TODO: in ini file

    # read data + clip data + preprocessing data
    df = self.data_catalog.get_geodataframe(
        path_or_key,
        geom=self.region,
        buffer=clip_buffer,
        clip_predicate=clip_predicate,
        variables=variables,
    )
    self.logger.debug(
        f"GeoDataFrame: {len(df)} feature are read after clipping region with clip_buffer = {clip_buffer}, clip_predicate = {clip_predicate}"
    )

    # retype data
    #    retype = d.get("retype", None)
    df = helper.retype_geodataframe(df, retype)

    # eval funcs on data
    #    funcs = d.get("funcs", None)
    df = helper.eval_funcs(df, funcs)

    # slice data # TODO: test what can be achived by the alias in yml file
    # required_columns = d.get("required_columns", None) can be done with variables arg from get_geodataframe
    # required_query = d.get("required_query", None)
    df = helper.slice_geodataframe(
        df, required_columns=None, required_query=required_query
    )
    self.logger.debug(
        f"GeoDataFrame: {len(df)} feature are sliced after applying required_query = '{required_query}'"
    )

    # index data
    if id_col is None:
        pass
    elif id_col not in df.columns:
        raise ValueError(
            f"GeoDataFrame: cannot index data using id_col = {id_col}. id_col must exist in data columns ({df.columns})"
        )
    else:
        self.logger.debug(f"GeoDataFrame: indexing with id_col: {id_col}")
        df.index = df[id_col]
        df.index.name = id_col

    # remove nan in id
    df_na = df.index.isna()
    if len(df_na) > 0:
        df = df[~df_na]
        self.logger.debug(f"GeoDataFrame: removing index with NaN")

    # remove duplicated
    df_dp = df.duplicated()
    if len(df_dp) > 0:
        df = df.drop_duplicates()
        self.logger.debug(f"GeoDataFrame: removing duplicates")

    # report
    df_num = len(df)
    if df_num == 0:
        self.logger.warning(f"Zero features are read from {path_or_key}")
    else:
        self.logger.info(f"{len(df)} features read from {path_or_key}")

    return df
