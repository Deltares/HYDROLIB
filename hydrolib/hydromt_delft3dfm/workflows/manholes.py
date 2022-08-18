# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import xarray as xr
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, Point

from .graphs import gpd_to_digraph

logger = logging.getLogger(__name__)


__all__ = [
    "generate_manholes_on_branches",
]

def generate_manholes_on_branches(branches: gpd.GeoDataFrame, bedlevel_shift: float = 0.0,
                                           id_col: str = "manholeId", id_prefix:str = 'manhole_', id_suffix:str = '',
                                           logger=logging):
    """generate manhole location and bedlevel from pipes"""

    # generate
    logger.info('Generating manholes on pipes and/or tunnels')
    manholes_generated, branches_generated = __generate_manholes(branches, id_col, id_prefix,
                                                                 id_suffix)
    # apply additional shifts for bedlevel and streetlevel
    logger.info(f'Shifting manholes bedlevels based on bedlevel_shift = {bedlevel_shift}')
    manholes_generated['bedLevel'] = manholes_generated['bedLevel'] + bedlevel_shift

    return manholes_generated, branches_generated


def __generate_manholes(branches: gpd.GeoDataFrame,
                        id_col: str = 'manholeId', id_prefix='manhole_', id_suffix=''):
    """generate manhole locations !Do not split anymore"""

    # prepare branches
    if branches.index.name is None: # add temp index
        branches.index.name = '_index'
    pipes = branches.query('branchType == "pipe" | branchType == "tunnel"')
    channels = branches.query(' branchType == "river" | branchType == "Channel"')

    # generate nodes upstream and downstream for every pipe
    _nodes_pipes_up = pd.DataFrame(
        [(Point(l.geometry.coords[0]), l.invlev_up, li) for li, l in pipes[['geometry', 'invlev_up']].iterrows()],
        columns=['geometry', 'bedLevel', pipes.index.name])
    _nodes_pipes_dn = pd.DataFrame(
        [(Point(l.geometry.coords[-1]), l.invlev_dn, li) for li, l in pipes[['geometry', 'invlev_dn']].iterrows()],
        columns=['geometry', 'bedLevel', pipes.index.name])
    _nodes_pipes = pd.concat([_nodes_pipes_up, _nodes_pipes_dn])

    # get connecting pipes characteristics based on statistical functions
    # left join based on pipe index to get pipe charactersitics
    nodes_pipes = _nodes_pipes.set_index(pipes.index.name
                                         ).merge(pipes[["branchId", "ORIG_branchId",  "branchType", "diameter", "width"]], on=pipes.index.name
                                                 ).reset_index()

    # apply statistical funcs
    nodes_pipes['where'] = nodes_pipes["geometry"].apply(lambda geom: geom.wkb)
    from scipy.stats import mode
    nodes_pipes = __get_pipe_stats_for_manholes(nodes_pipes, 'where', 'diameter', 'max')
    nodes_pipes = __get_pipe_stats_for_manholes(nodes_pipes, 'where', 'width', 'max')
    nodes_pipes = __get_pipe_stats_for_manholes(nodes_pipes, 'where', 'bedLevel', 'min')
    nodes_pipes = __get_pipe_stats_for_manholes(nodes_pipes, 'where', 'ORIG_branchId', ';'.join)

    # drop duplicated nodes
    nodes_pipes = nodes_pipes.loc[nodes_pipes['where'].drop_duplicates().index, :]
    nodes_pipes.drop(columns='where', inplace=True)

    # remove pipes nodes on channels
    # generate nodes on channels
    _nodes_channels = pd.DataFrame([(Point(l.coords[0]), li) for li, l in channels['geometry'].iteritems()] +
                                   [(Point(l.coords[-1]), li) for li, l in channels['geometry'].iteritems()],
                                   columns=['geometry', channels.index.name])
    nodes_channels = _nodes_channels.set_index(channels.index.name
                                               ).merge(channels[["branchId", "ORIG_branchId", "branchType"]], on=channels.index.name)

    # snap channel nodes to pipe nodes
    nodes_channels = hydromt.gis_utils.nearest_merge(
        gpd.GeoDataFrame(nodes_channels, crs = branches.crs),
        gpd.GeoDataFrame(nodes_pipes, crs = branches.crs),
        max_dist=0.1,
        overwrite=True,
    )

    # if snapped, meaning there should be a channel nodes, therefore remove pipe nodes
    mask = ~nodes_pipes.index.isin(nodes_channels.index_right)
    nodes_pipes = nodes_pipes.loc[mask].reset_index(drop=True)

    # manhole generated
    manholes_generated = gpd.GeoDataFrame(nodes_pipes)
    manholes_generated.loc[:, id_col] = ['%s%s%s' % (id_prefix, str(x), id_suffix) for x in
                                         range(len(manholes_generated))]
    manholes_generated.index = manholes_generated.loc[:, id_col]

    # update manholes generated to pipes
    pipes = __update_pipes_from_manholes(manholes_generated, pipes)

    # merge pipe and channels
    branches_generated = pd.concat([pipes, channels], join='outer')
    if branches_generated.index.name == '_index': # remove temp index
        branches_generated.index.name = None

    return manholes_generated, branches_generated


def __update_pipes_from_manholes(manholes: gpd.GeoDataFrame, pipes: gpd.GeoDataFrame):
    """assign manholes to pipes based on geometry"""
    manholes_dict = {(m.geometry.x, m.geometry.y): mi for mi, m in manholes.iterrows()}
    if not {"manhole_up", "manhole_dn"}.issubset(pipes.columns):
        pipes["manhole_up"] = None
        pipes["manhole_dn"] = None
    for pi, p in pipes.iterrows():
        cs = p.geometry.coords
        if cs[0] in manholes_dict:
            pipes.at[pi, "manhole_up"] = manholes_dict[
                cs[0]]
        else:
            pipes.at[pi, "manhole_up"] = '' # empty if no manholes
        if cs[-1] in manholes_dict:
            pipes.at[pi, "manhole_dn"] = manholes_dict[
                cs[-1]]
        else:
            pipes.at[pi, "manhole_dn"] = '' # empty if no manholes

    return pipes


def __get_pipe_stats_for_manholes(manholes: gpd.GeoDataFrame, pipes_col: str,
                                  stats_col: str, method: str):
    """get the stats from all pipes connecting a single manholes

        parameters
        --------------------
        pipes_col: used to identify pipes connected to the manhole (multiple rows of pipes for a single manhole), e.g. BRANCH_ID.
        stats_col: the column used to obtain the stats, e.g. DIAMETER
        method: method used to obtain the stats: e.g. max
        """
    manholes.loc[:, stats_col] = manholes.groupby(pipes_col)[stats_col].transform(method)
    return manholes


def __fill_manholes(manholes: gpd.GeoDataFrame, manholes_ini: configparser.ConfigParser, fill_method: str = 'default'):
    """fill manholes attributes using fill method"""
    if fill_method == 'default':
        manholes_filled = append_data_columns_based_on_ini_query(manholes, manholes_ini)
    else:
        raise NotImplementedError(
            'Fill method is not recognized Please use only default')  # TODO BMA: add other fill method, e.g. landuse (In HydroMT)
    return manholes_filled


# TODO BMA: __fill_manholes_with_DTM

def _merge_user_manholes_to_generated_manholes(manholes_generated: gpd.GeoDataFrame, manholes_user: gpd.GeoDataFrame,
                                               snap_method:str = 'overall', snap_offset: float = 0., logger=logging):
    """
    snap user manholes to generated manholes and overwrite generated manholes fields
    """
    manholes_merged = merge_nodes_with_nodes_prior_by_snapping(manholes_user, manholes_generated, snap_method, snap_offset)
    return manholes_merged