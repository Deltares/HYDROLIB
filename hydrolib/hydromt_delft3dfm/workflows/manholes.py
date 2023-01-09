# -*- coding: utf-8 -*-

import logging

import geopandas as gpd
import pandas as pd
from hydromt import gis_utils
from shapely.geometry import Point

logger = logging.getLogger(__name__)


__all__ = [
    "generate_manholes_on_branches",
]


def generate_manholes_on_branches(
    branches: gpd.GeoDataFrame,
    use_branch_variables: list = ["diameter", "width"],
    bedlevel_shift: float = 0.0,
    id_prefix: str = "",
    id_suffix: str = "",
    logger=logging,
):
    """Generate manhole location and bedlevel from branches.

    Manholes will be generated at locations upstream and downstream of each pipe. No manholes will be generated at pipe/tunnel outlets into the channels/rivers.

    A new geom of manholes will be created.
    manholeId will be generated following the convension of id_prefix, id, and id_suffix.
    bedlevel will be generated from the lowest invert levels of the connecting pipes.
    Otehr attributes can be summersized from exisiting branch variables defined in ''use_branch_variables''.

    Adds:
        * **manholes** geom: 1D manholes vector
    Updates:
        * **branches** geom: 1D branches vector

    Parameters
    ----------
    branches: gpd.GeoDataFrame
        branches where manholes need to be generated. Use the entire network branches.
        Required columns: ['branchId', 'branchType', 'invlev_up', 'invlev_dn']
        Optional columns: ones defined in use_branch_variables
    use_branch_variables: list of str, Optional
        list of branch variables to include as attributes for manholes.
        If multiple branches connects to a single manhole, then the maximum will be taken.
        By default ['diameter', 'width'].
    bedlevel_shift: float, optional
        Shift applied to lowest pipe invert levels to derive manhole bedlevels [m] (bedlevel = pipe invert + bedlevel shift).
        By default 0.0 m, no shift is applied.
    id_prefix: str, optional
        prefix to add to the id convention
        By default "", no prefix are used.
    id_suffix: str, optional
        suffix to add to the id convention
        By default "", no suffix are used.

    """
    # generate
    logger.info("Generating manholes on pipes and/or tunnels")

    # prepare branches
    if branches.index.name is None:  # add temp index
        branches.index.name = "_index"
    pipes = branches.query(
        'branchType == "pipe" | branchType == "tunnel"'
    )  # include both pipes and tunnels
    channels = branches.query(
        'branchType == "river" | branchType == "Channel"'
    )  # include both channels and rivers

    # generate nodes upstream and downstream for every pipe
    _nodes_pipes_up = pd.DataFrame(
        [
            (Point(l.geometry.coords[0]), l.invlev_up, li)
            for li, l in pipes[["geometry", "invlev_up"]].iterrows()
        ],
        columns=["geometry", "bedLevel", pipes.index.name],
    )
    _nodes_pipes_dn = pd.DataFrame(
        [
            (Point(l.geometry.coords[-1]), l.invlev_dn, li)
            for li, l in pipes[["geometry", "invlev_dn"]].iterrows()
        ],
        columns=["geometry", "bedLevel", pipes.index.name],
    )
    _nodes_pipes = pd.concat([_nodes_pipes_up, _nodes_pipes_dn])
    _nodes_pipes["where"] = _nodes_pipes["geometry"].apply(lambda geom: geom.wkb)

    # get branch variables
    nodes_pipes = (
        _nodes_pipes.set_index(pipes.index.name)
        .merge(
            pipes[["branchId", "branchType"] + use_branch_variables],
            on=pipes.index.name,
        )
        .reset_index()
    )

    # summarize branch variables
    nodes_pipes = _get_pipe_stats_for_manholes(nodes_pipes, "where", "diameter", "max")
    nodes_pipes = _get_pipe_stats_for_manholes(nodes_pipes, "where", "width", "max")
    nodes_pipes = _get_pipe_stats_for_manholes(
        nodes_pipes, "where", "branchId", ";".join
    )

    # get bed level, use the lowest of all pipe inverts
    nodes_pipes = _get_pipe_stats_for_manholes(nodes_pipes, "where", "bedLevel", "min")
    # apply additional shifts for bedlevel and streetlevel
    if bedlevel_shift != 0:
        logger.info(
            f"Shifting manholes bedlevels based on bedlevel_shift = {bedlevel_shift}"
        )
        nodes_pipes["bedLevel"] = nodes_pipes["bedLevel"] + bedlevel_shift

    # internal admin
    # drop duplicated nodes
    nodes_pipes = nodes_pipes.loc[nodes_pipes["where"].drop_duplicates().index, :]
    nodes_pipes = gpd.GeoDataFrame(nodes_pipes, crs=branches.crs)

    # drop pipe nodes that are outlets (located at channels)
    _nodes_channels = gpd.GeoDataFrame(
        [(Point(l.coords[0]), li) for li, l in channels["geometry"].iteritems()]
        + [(Point(l.coords[-1]), li) for li, l in channels["geometry"].iteritems()],
        columns=["geometry", channels.index.name],
        crs=branches.crs,
    )

    nodes_channels = gpd.GeoDataFrame(_nodes_channels, crs=branches.crs)
    nodes_to_remove = gis_utils.nearest_merge(
        nodes_pipes, nodes_channels, max_dist=0.001, overwrite=True
    )
    nodes_pipes = nodes_pipes.loc[nodes_to_remove.index_right == -1]

    # manhole generated
    manholes_generated = gpd.GeoDataFrame(
        nodes_pipes.drop(columns="where"), crs=branches.crs
    )

    # add manholeId
    manholes_generated.loc[:, "manholeId"] = [
        f"{id_prefix}{x}{id_suffix}" for x in range(len(manholes_generated))
    ]
    manholes_generated.set_index("manholeId")

    # update manholes generated to pipes
    pipes_updated = _update_pipes_from_manholes(manholes_generated, pipes)

    # merge updated pipe and channels into branches
    branches_updated = pd.concat([pipes_updated, channels], join="outer")
    if branches_updated.index.name == "_index":  # remove temp index
        branches_updated.index.name = None

    return manholes_generated, branches_updated


def _update_pipes_from_manholes(manholes: gpd.GeoDataFrame, pipes: gpd.GeoDataFrame):
    """assign manholes 'manholeId' to pipes ['manhole_up', 'manhole_dn'] based on geometry"""
    manholes_dict = {
        (m.geometry.x, m.geometry.y): manholes.loc[mi, "manholeId"]
        for mi, m in manholes.iterrows()
    }
    if not {"manhole_up", "manhole_dn"}.issubset(pipes.columns):
        pipes["manhole_up"] = None
        pipes["manhole_dn"] = None
    for pi, p in pipes.iterrows():
        cs = p.geometry.coords
        if cs[0] in manholes_dict:
            pipes.at[pi, "manhole_up"] = manholes_dict[cs[0]]
        else:
            pipes.at[pi, "manhole_up"] = ""  # empty if no manholes
        if cs[-1] in manholes_dict:
            pipes.at[pi, "manhole_dn"] = manholes_dict[cs[-1]]
        else:
            pipes.at[pi, "manhole_dn"] = ""  # empty if no manholes

    return pipes


def _get_pipe_stats_for_manholes(
    manholes: gpd.GeoDataFrame, pipes_col: str, stats_col: str, method: str
):
    """get the stats from all pipes connecting a single manholes

    parameters
    --------------------
    pipes_col: used to identify pipes connected to the manhole (multiple rows of pipes for a single manhole), e.g. BRANCH_ID.
    stats_col: the column used to obtain the stats, e.g. DIAMETER
    method: method used to obtain the stats: e.g. max
    """
    manholes.loc[:, stats_col] = manholes.groupby(pipes_col)[stats_col].transform(
        method
    )
    return manholes
