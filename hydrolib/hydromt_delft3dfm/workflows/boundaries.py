# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, Point

from .graphs import gpd_to_digraph

logger = logging.getLogger(__name__)


__all__ = [
    "generate_boundaries_from_branches",
    "validate_boundaries",
]


def generate_boundaries_from_branches(branches:gpd.GeoDataFrame,
                                      where:str = 'both'):
    """function to get possible boundary locations from branches with id

    [ ] convert branches to graph
    [ ] get boundary locations at where

    parameters:
    where: str
        Options available aare: ['upstream', 'downstream', 'both']
        Where at the branches should the boundaries be derived.
    """

    # convert branches to graph
    G = gpd_to_digraph(branches)

    # get boundary locations at where
    if where == 'downstream':
        endnodes = {dn:{**d,**{"where":"downstream"}} for up, dn, d in G.edges(data = True) if G.out_degree[dn] == 0 and G.degree[dn] == 1}
    elif where == 'upstream':
        endnodes = {up:{**d,**{"where":"upstream"}} for up, dn, d in G.edges(data = True) if G.in_degree[up] == 0 and G.degree[up] == 1}
    elif where == 'both':
        endnodes = {dn:{**d,**{"where":"downstream"}} for up, dn, d in G.edges(data = True) if G.out_degree[dn] == 0 and G.degree[dn] == 1}
        endnodes.update({up:{**d,**{"where":"upstream"}}for up, dn, d in G.edges(data = True) if G.in_degree[up] == 0 and G.degree[up] == 1})
    else:
        pass

    if len(endnodes) == 0:
        logger.error(f"cannot generate boundaries for given condition {branch_query}")

    endnodes_pd = pd.DataFrame().from_dict(endnodes, orient = 'index').drop(columns=["geometry"])
    endnodes_gpd = gpd.GeoDataFrame(data=endnodes_pd, geometry=[Point(endnode) for endnode in endnodes], crs = branches.crs)
    endnodes_gpd.reset_index(inplace=True)
    return endnodes_gpd

def validate_boundaries(boundaries: gpd.GeoDataFrame, branch_type:str = 'river'):
    """Validate boundaries per branch type"""

    if branch_type == 'river': # TODO add other open system branch_type
        for _, bnd in boundaries.iterrows():
            # TODO extended
            if bnd['where'] == 'downstream' and bnd['boundary_type'] == 'discharge':
                logger.warning(f'Boundary type voilets modeller suggestions: using downstream discharge boundary at branch {bnd["branchId"]}')


    if branch_type == 'pipe': # TODO add other close system branch_type
        for _, bnd in boundaries.iterrows():
            # TODO extended
            if bnd['where'] == 'upstream':
                logger.warning(f'Boundary type voilets modeller suggestions: using upstream boundary at branch {bnd["branchId"]}')

    return None

