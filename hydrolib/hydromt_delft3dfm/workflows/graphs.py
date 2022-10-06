# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from hydromt import config

logger = logging.getLogger(__name__)


def gpd_to_digraph(data: gpd.GeoDataFrame) -> nx.DiGraph():
    """Convert a `gpd.GeoDataFrame` to a `nx.DiGraph` by taking the first and last coordinate in a row as source and target, respectively.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        The data to convert.

    Returns
    -------
    nx.DiGraph
        The converted directed graph.
    """
    _ = data.copy()

    _["from_node"] = [row.geometry.coords[0] for index, row in _.iterrows()]
    _["to_node"] = [row.geometry.coords[-1] for index, row in _.iterrows()]
    G = nx.from_pandas_edgelist(
        _,
        source="from_node",
        target="to_node",
        create_using=nx.DiGraph,
        edge_attr=True,
    )
    return G
