# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
from hydromt import config
import networkx as nx

logger = logging.getLogger(__name__)

def gpd_to_digraph(data: gpd.GeoDataFrame) -> nx.DiGraph():

    _ = data.copy()

    _["from_node"] = [
        row.geometry.coords[0] for index, row in _.iterrows()
    ]
    _["to_node"] = [
        row.geometry.coords[-1] for index, row in _.iterrows()
    ]
    G = nx.from_pandas_edgelist(
        _,
        source="from_node",
        target="to_node",
        create_using=nx.DiGraph,
        edge_attr=True,
    )
    return G
