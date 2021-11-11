
import numpy as np
import scipy as sp
import scipy.spatial
import sys
import copy
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiLineString, box
from shapely.ops import polygonize
import matplotlib.pyplot as plt
from delft3dfmpy import DFlowFMModel
import logging
import networkx as nx
import contextily as ctx
import random
from scipy.spatial import KDTree
import configparser
from delft3dfmpy.core import geometry
from scipy.spatial import distance
import shapely
from networkx.drawing.nx_agraph import graphviz_layout
from shapely.ops import linemerge
# ===============================================================
#                  checkes
# ===============================================================
# FIXME BMA: new checks are in setup functions - remove this
def to_boolean(inarg):
    if inarg in ['TRUE', 'True', 'true', '1', True, 1]:
        outarg = True
    elif inarg in ['FALSE', 'False', 'false', '0', True, 0]:
        outarg = False
    else:
        outarg = inarg
    return outarg

# ===============================================================
#                  PREPROCESSING
# ===============================================================
#FIXME BMA: For HydroMT we will rewrite this snap_branches to snap_lines and take set_branches out of it

def snap_branch_ends(branches:gpd.GeoDataFrame, offset: float = 0.01, subsets = [],  max_points:int = np.inf, id_col = 'BRANCH_ID', logger = logging):
    """
    Helper to snap branch ends to other branch ends within a given offset.


    Parameters
    ----------
    branches : gpd.GeoDataFrame
    offset : float [m]
        Maximum distance between end points. If the distance is larger, they are not snapped.
    subset : list
        A list of branch id subset to perform snapping (forced snapping)
    max_points: int
        maximum points allowed in a group.
        if snapping branch ends only, use max_points = 2
        if not specified, branch intersections will also be snapped
    Returns
    branches : gpd.GeoDataFrame
        Branches updated with snapped geometry
    """
    # Collect endpoints
    _endpoints = []
    for branch in branches.itertuples():
        _endpoints.append((branch.geometry.coords[0], branch.Index, 0))
        _endpoints.append((branch.geometry.coords[-1], branch.Index, -1))

    # determine which branches should be included
    if len(subsets) > 0:
        _endpoints = [[i for i in _endpoints if i[1] in subsets]]
    else:
        _endpoints = _endpoints

    # # group branch ends based on off set
    groups = {}
    coords = [i[0] for i in _endpoints]
    dist = distance.squareform(distance.pdist(coords))
    bdist = (dist <= offset)
    for row_i, row in enumerate(bdist):
        groups[_endpoints[row_i]] = []
        for col_i, col in enumerate(row):
            if col:
                groups[_endpoints[row_i]].append(_endpoints[col_i])

    # remove duplicated group, group that does not satisfy max_points in groups. Assign endpoints
    endpoints = {k:list(set(v)) for k,v in groups.items() if (len(set(v)) >= 2) and (len(set(v)) <= max_points)}
    logger.debug('Limit snapping to allow a max number of {max_points} contact points. If max number == 2, it means 1 to 1 snapping.')

    # Create a counter
    snapped = 0

    # snap each group (list) in endpoints together, by using the coords from the first point
    for point_reference, points_to_snap in endpoints.items():
        # get the point_reference coords as reference point
        ref_crd = point_reference[0]
        # for each of the rest
        for j, (endpoint, branchid, side) in enumerate(points_to_snap):
            # Change coordinates of branch
            crds = branches.at[branchid, 'geometry'].coords[:]
            if crds[side] != ref_crd:
                crds[side] = ref_crd
                branches.at[branchid, 'geometry'] = LineString(crds)
                snapped += 1
    logger.debug(f'Snapped {snapped} points.')

    return branches


def find_critical_intersections(branches: gpd.GeoDataFrame, offset: float = 0.01, logger=logging):
    """
    Find critical intersections: intersection that points are not snapped together (precision issue or human error)

    Using GIS methods
    # TODO BMA: seek potential improvement, based on distance matrix --> not yet found a nice way to do it
    """

    points_geometry = [Point(l.coords[0]) for li, l in branches.geometry.iteritems()] + [Point(l.coords[-1]) for
                                                                                         li, l in
                                                                                         branches.geometry.iteritems()]
    points_index = [li for li, l in branches.geometry.iteritems()] + [li for li, l in branches.geometry.iteritems()]
    points = gpd.GeoDataFrame([], geometry=points_geometry, index=points_index)
    points = drop_duplicate_geometry(points)
    points['buffer'] = points.buffer(offset)
    overlapping_polygons = find_overlapping_polygons(points['buffer'].tolist())
    points['buffer_overlapping'] = points['buffer'].isin(overlapping_polygons)
    n = sum(points['buffer_overlapping'])
    logger.info(f'trying to group points that are closer than {offset} for {n} points.')
    points_to_snap = points.query('buffer_overlapping == True')
    return points_to_snap


def find_overlapping_polygons(polygons: list):
    overlapping = []
    # overlapping_ratio = []
    for n, p in enumerate(polygons[:-1], 1):
        p_vs_gs = [g for g in polygons[n:] if p.overlaps(g)]
        if len(p_vs_gs) > 0:
            p_vs_gs.append(p)
            overlapping.append(p_vs_gs)
            # overlapping_ratio.append(max(p.intersection(g).area/p.area for g in p_vs_gs))
    return sum(overlapping, [])

def drop_duplicate_geometry(gdf):
    # convert to wkb
    gdf["geometry_str"] = gdf["geometry"].apply(lambda geom: geom.wkb)
    gdf = gdf.drop_duplicates(subset=["geometry_str"])
    return gdf.drop(columns=["geometry_str"])

def intersect_branches(branches:gpd.GeoDataFrame, offset: float, id_col = 'BRANCH_ID', logger = logging):
    """function to make branch intersect within a given threshold (offset)"""
    endpoints = [{'idx':idx, 'geometry': Point(line.coords[0])} for idx,line in branches.geometry.iteritems()]
    endpoints = endpoints + [{'idx':idx, 'geometry': Point(line.coords[-1])} for idx,line in branches.geometry.iteritems()]
    endpoints = gpd.GeoDataFrame(endpoints)
    geometry.find_nearest_branch(branches, branches, method='ends', maxdist=offset)



def split_branches(branches:gpd.GeoDataFrame, spacing_const:float = float('inf'), spacing_col:str = None, logger = logging):
    """
    Helper function to split branches based on a given spacing.
    If spacing_col is used (default), apply spacing as a categorical variable - distance used to split branches. priority issue --> Question to Rinske
    If spacing_const is used (overwrite), apply spacing as a constant -  distance used to split branches.
    Raise Error if neither exist.

    NOTE! branch generated will be straight line

    Parameters
    ----------
    branches : gpd.GeoDataFrame
    spacing_const : float
        Constent spacing which will overwrite the spacing_col.
    spacing_col: str
        Name of the column in branchs that contains spacing information.

    Returns
    split_branches : gpd.GeoDataFrame
        Branches after split, new ids will be overwritten for the bracn index. Old ids are stored in "OLD_" + index.
    """

    id_col = branches.index.name

    if spacing_col is None:
        logger.info(f'Splitting branches with spacing of {spacing_const} [m]')
        split_branches = _split_branches_by_spacing_const(branches, spacing_const, id_col = id_col, logger = logging)

    elif branches[spacing_col].astype(float).notna().any():
        logger.info(f'Splitting branches with spacing specifed in datamodel branches[{spacing_col}]')
        split_branches= []
        for spacing_subset, branches_subset in branches.groupby(spacing_col):
            if spacing_subset:
                split_branches_subset = _split_branches_by_spacing_const(branches_subset, spacing_subset, id_col = id_col, logger = logging)
            else:
                branches_subset.loc[:, f'ORIG_{id_col}'] = branches_subset[id_col]
                split_branches_subset = branches_subset
            split_branches.append(split_branches_subset)
        split_branches = pd.concat(split_branches)


    else: # no spacing information specified anywhere, do not apply splitting
        branches.loc[:, f'ORIG_{id_col}'] = branches[id_col]
        split_branches = branches

    # reassign branch id for the generated
    split_branches.index = split_branches[id_col]
    return split_branches

# fixme BMA: not generic functions
def _split_branches_by_spacing_const(branches:gpd.GeoDataFrame, spacing_const:float, id_col = "BRANCH_ID", logger = logging):
    """
    Helper function to split branches based on a given spacing constant.

    Parameters
    ----------
    branches : gpd.GeoDataFrame
    spacing_const : float
        Constent spacing which will overwrite the spacing_col.
    id_col: str
        Name of the column in branches that contains the id of the branches.

    Returns
    split_branches : gpd.GeoDataFrame
        Branches after split, new ids will be stored in id_col. Original ids are stored in "ORIG_" + id_col.
    """

    if spacing_const == float('inf'):
        branches[f'ORIG_{id_col}'] = branches[id_col]
        branches.index = branches[id_col]
        return branches

    # only split if spacing_const != inf
    # make a dflowfm helper model
    _dfmmodel = DFlowFMModel()

    # set branches to helper model
    _dfmmodel.network.set_branches(branches, id_col=id_col)

    # get nodes from helper model mesh1d
    _dfmmodel.network.generate_1dnetwork(one_d_mesh_distance=spacing_const)

    s = _dfmmodel.network.schematised
    edge_geom = []
    edge_offset = []
    edge_invertup = []
    edge_invertdn = []
    edge_bedlevup = []
    edge_bedlevdn = []
    edge_index = []
    branch_index = []
    for bid, b in s.iterrows():
        new_edges = [LineString([Point(p1), Point(p2)]) for p1,p2 in zip(b.geometry.coords[:-1], b.geometry.coords[1:])]
        lengths = np.r_[0, np.cumsum(np.hypot(np.diff(b.geometry.coords, axis=0)[:, 0], np.diff(b.geometry.coords, axis=0)[:, 1]))]
        edge_geom.extend(new_edges)
        edge_offset.extend(lengths[1:])
        edge_invertup.extend(np.interp(lengths[:-1], [0, lengths[-1]], [b.INVLEV_UP, b.INVLEV_DN]))
        edge_invertdn.extend(np.interp(lengths[1:], [0, lengths[-1]], [b.INVLEV_UP, b.INVLEV_DN]))
        edge_bedlevup.extend(np.interp(lengths[:-1], [0, lengths[-1]], [b.BEDLEV_UP, b.BEDLEV_DN]))
        edge_bedlevdn.extend(np.interp(lengths[1:], [0, lengths[-1]], [b.BEDLEV_UP, b.BEDLEV_DN]))
        edge_index.extend([bid + '_E' + str(i) for i in range(len(new_edges))])
        branch_index.extend([bid] * len(new_edges))

    edges = gpd.GeoDataFrame({'EDGE_ID': edge_index, 'geometry': edge_geom, id_col: branch_index,
                              'INVLEV_UP': edge_invertup, 'INVLEV_DN': edge_invertdn,
                              'BEDLEV_UP': edge_bedlevup, 'BEDLEV_DN': edge_bedlevdn,
                              }, crs=branches.crs)
    edges_attr = pd.concat([branches.loc[idx, :] for idx in branch_index], axis=1).transpose()
    edges = pd.concat([edges, edges_attr.drop(columns= list(set(edges.columns) - set(['EDGE_ID']))).reset_index()], axis=1)

    edges = edges.rename(columns={id_col: f'ORIG_{id_col}'})
    edges = edges.rename(columns={'EDGE_ID': id_col})
    edges.index = edges[id_col]
    split_branches = edges

    return split_branches

# FIXME BMA: check if HydroMT has break line functions
# TODO BMA: break line at intersection function --> might be required
# def break_lines_at_intersection(lines:gpd.GeoDataFrame, logger = logging):
# _lines = lines.copy()
# from shapely.geometry import MultiLineString
# from itertools import combinations
# intersection_points = []
# for line1, line2 in combinations([line for line in _lines.geometry],2):
#     if line1.intersects(line2):
#         p = line1.intersection(line2)
#         # if isinstance(p, Point):
#         intersection_points.append(p)

# intersection_points = gpd.GeoDataFrame(geometry = intersection_points, index  =range(len(intersection_points)))

#     """break the line at intersection point"""
#     _lines = lines.copy()
#     lines = _lines.unary_union
#     a = gpd.overlay(_lines, _lines, how='intersection')
#     lines = []
#     for idx1,c1 in _lines.geometry.iteritems():
#         for idx2,c2 in _lines.geometry.iteritems():
#             if idx1 != idx2 and c2.intersects(c1):
#                 for i,l in enumerate(split(c1, c2.intersection(c1))):
#                     lines.append({**_lines.loc[idx1].to_dict(),
#                                     **{_lines.index.name: str(idx1) + '_' + str(i), 'geometry': l}})
                # else:
                #     lines.append(c1)
# The below should not be needed
# for i,l in enumerate(split(c2, c2.intersection(c1))):
#                     lines.append({**_lines.loc[idx2].to_dict(),
#                                     **{_lines.index.name: str(idx2) + '_' + str(i), 'geometry': l}})
#     lines = gpd.GeoDataFrame(lines)
#     lines.index = lines[_lines.index.name]
#     return lines

# FIXME BMA: check if HydroMT has point snapping function with offset
def snap_nodes_to_nodes(nodes:gpd.GeoDataFrame, nodes_prior:gpd.GeoDataFrame, offset:float = 1, logger = logging):

    """
    Method to snap nodes to nodes_prior.
    index of nodes_prior will be overwritten by index of nodes.
    index column will be retained as in nodes_prior.

    Parameters
    offset : float
        Maximum distance between end points. If the distance is larger, they are not snapped.
    """
    id_col = str(nodes_prior.index.name)

    # Collect points_prior in nodes_prior
    points_prior = [(node.geometry.x, node.geometry.y) for node in nodes_prior.itertuples()]
    points_prior_index = nodes_prior.index.to_list()

    # Collect points in nodes
    points = [(node.geometry.x, node.geometry.y) for node in nodes.itertuples()]
    points_index = nodes.index.to_list()

    # Create KDTree of points_prior
    snapped = 0

    # For every points determine the distance to the nearest points_prior
    for pointid, point in zip(points_index, points):
        mindist, minidx = KDTree(points_prior).query(point)

        # snapped if dist is smaller than offset (also include 0.0).
        if mindist <= offset:

            # Change index columns of nodes_prior
            points_prior_index[minidx] = pointid
            snapped += 1

    logging.info(f'Snapped {snapped} points.')
    # reindex
    nodes_prior_ = nodes_prior.copy()
    nodes_prior_['ORIG_' + id_col] = list(nodes_prior.index)
    nodes_prior_[id_col] = points_prior_index
    nodes_prior_.index = nodes_prior_[id_col]
    return nodes_prior_


def merge_nodes_with_nodes_prior_by_snapping(nodes:gpd.GeoDataFrame, nodes_prior:gpd.GeoDataFrame, offset:float = 1, logger = logging):

    """
    Method to snap nodes to nodes_prior. snapped node will be replaced, not snapped will be added
    use nodes_prior geometry in nodes
    Parameters
    offset : float
        Maximum distance between end points. If the distance is larger, they are not snapped.
    """
    id_col = str(nodes_prior.index.name)

    # Collect points in nodes
    points = [(node.geometry.x, node.geometry.y) for node in nodes.itertuples()]
    points_index = nodes.index.to_list()

    # Collect points_prior in nodes_prior
    points_prior = [(node.geometry.x, node.geometry.y) for node in nodes_prior.itertuples()]
    points_prior_index = nodes_prior.index.to_list()

    # relocate, snap nodes to nodes_prior
    to_remove= []
    # For every points determine the distance to the nearest points_prior
    for i, (pointid, point) in enumerate(zip(points_index, points)):

        # Create KDTree of points_prior
        mindist, minidx = KDTree(points_prior).query(point)

        # snapped if dist is smaller than offset (also include 0.0).
        if mindist <= offset:

            # use nodes_prior geometry in nodes
            points[i] = points_prior[minidx]

            # remove from nodes_prior
            to_remove.append(points_prior_index[minidx])

    logging.info(f'Snapped {len(to_remove)} points.')

    # update snapped nodes with nodes_prior geometry
    nodes.geometry = [Point(x, y) for x,y in points]

    # remove snapped from nodes_prior
    nodes_prior.drop(index = to_remove, inplace=True)

    # combine attributes, nodes_prior_ will be overwritten by nodes
    nodes_post = nodes.combine_first(nodes_prior)

    return nodes_post


def filter_geometry_and_assign(nodes, subset, **kwargs):
    """filter the nodes using subset geometry, and assign new attributes to nodes"""

    a = nodes["geometry"].apply(lambda geom: geom.wkb)
    b = subset["geometry"].apply(lambda geom: geom.wkb)
    mask = a.isin(b)
    for k,v in kwargs.items():
        nodes.loc[mask,k] = v
    return nodes

# FIXME BMA: check if HydroMT has point snapping function with offset
def snap_nodes_to_lines(lines:gpd.GeoDataFrame, nodes:gpd.GeoDataFrame, offset:float = 1, logger = logging):
    """
    Method to snap nodes to lines.
    """
    from geopandas.tools import sjoin
    nodes_ = nodes.copy()
    snapped = 0
    for line_id, line in lines.geometry.iteritems():
        # buffer the line
        _buffer = line.buffer(offset)
        buffer = gpd.GeoDataFrame(geometry = [_buffer], crs = lines.crs)
        _points = np.array([[point.x, point.y] for point in nodes.geometry])
        mask = np.array(in_box(_points, _buffer.bounds))
        pointInPolys = sjoin(nodes_[mask],buffer, how='left')
        points = pointInPolys.dropna()
        mask = np.array([line.touches(point) for point in points.geometry])
        points = points[~mask]
        if len(points) > 0:
            snapped += len(points)
            nodes_.at[points.index, 'geometry'] = points.apply(lambda row: line.interpolate(line.project(row.geometry)), axis=1)
    print(f"Snapped {snapped} branches ends to branches")
    return nodes_




# ===============================================================
#                  Helpers
# ===============================================================
# FIXME BMA: not needed once starts with hydromT
def rename_shp(fn:str, rename:dict = None):
    data = gpd.read_file(fn)
    if rename is not None and len(rename) > 0:
        data.rename(columns=rename, inplace=True)
    else:
        pass
    data.to_file(fn)

def append_data_columns_based_on_ini_query(data:gpd.GeoDataFrame, ini:configparser.ConfigParser, keys:list = [], logger = logging):
    """append key,val pair as data columns for the input GeiDataFrame based on ini [default] or [query] sections"""
    _columns = list(data.columns)
    for section in ini.keys():
        if section == 'global': # do nothing for global settings
            pass
        elif section == 'default': # append default key,value pairs
            for key, val in ini[section].items():
                try:
                    val = float(val)
                except:
                    pass
                if key in _columns:
                    data.loc[data[key].isna(), key] = val
                else:
                    data.loc[:, key] = val
        else: # overwrite default key,value pairs based on [query]
            for key, val in ini[section].items():
                try:

                    try:
                        val = float(val)
                    except:
                        pass
                    if key in _columns:
                        d = data.query(section)
                        idx = d.loc[d[key].isna(),:].index
                        data.loc[idx, key] = val
                    else:
                        idx = data.query(section).index
                        data.loc[idx, key] = val
                except:
                    logger.warning(f'Unable to query: adding default values from section {section} failed')
                    pass #do not query if error
    _columns_ = list(data.columns)
    if len(keys) > 0:
        columns = _columns + keys
    else:
        columns = _columns_
    return data.loc[:,columns]

# FIXME BMA: check if hydromt has general leftjoin that can be used
# FIXME BMA/Question to Rinske: Maybe we should inplement non case sensitive columns in datamodel
def leftjoin_gpd_by_index(data1:gpd.GeoDataFrame, data2:gpd.GeoDataFrame, case_sensitive = False):
    """perform a leftjoin (columns cases, names from the left, values from the right)
    NOTE: this means the left dataframe has to have all the columns required
    """
    _data1 = data1.copy()
    _data2 = data2.copy()
    if not case_sensitive:
        _data1.columns = [c.lower() for c in _data1.columns]
        _data2.columns = [c.lower() for c in _data2.columns]
    # perform a left join
    for idx in _data1.index:
        if idx in _data2.index:
            for k,v in _data2.loc[idx].iteritems():
                if (k in _data1.columns) and (v is not None):
                    if isinstance(v, float):
                        if ~np.isnan(v):
                            _data1.loc[idx,k] = v
                    else:
                        _data1.loc[idx,k] = v
    _data1.columns = list(data1.columns)
    return _data1

def reduce_gdf_precision(gdf:gpd.GeoDataFrame, rounding_precision = 8):

    if isinstance(gdf.geometry[0], LineString):
        branches = gdf.copy()
        for i_branch, branch in enumerate(branches.itertuples()):
            points = shapely.wkt.loads(shapely.wkt.dumps(branch.geometry, rounding_precision=rounding_precision)).coords[:]
            branches.at[i_branch, 'geometry'] = LineString(points)

    elif isinstance(gdf.geometry[0], Point):
        points = gdf.copy()
        for i_point, point in enumerate(points.itertuples()):
            new_point = shapely.wkt.loads(shapely.wkt.dumps(point.geometry, rounding_precision=rounding_precision)).coords[:]
            points.at[i_point, 'geometry'] = Point(new_point)

    else:
        raise NotImplementedError

    return gdf


# ===============================================================
#                  Validate
# ===============================================================


def random_color():
    return tuple(
        [
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
        ]
    )

def make_graphplot_for_targetnodes(G:nx.DiGraph, target_nodes:list, target_nodes_labeldict:dict = None, layout = 'xy', ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 25))

    # layout graphviz
    if layout == 'graphviz':
        # get position
        pos = graphviz_layout(G, prog='dot', args='')

        # draw network
        nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)

        if target_nodes_labeldict is not None:

            # draw labels
            nx.draw_networkx_labels(
                G,
                pos,
                target_nodes_labeldict,
                font_size=16,
                font_color='k')

    # layout xy
    elif layout == 'xy':
        # get position
        pos = {xy: xy for xy in G.nodes()}
        # make plot for each target node
        RG = G.reverse()

        for target in target_nodes:
            c = random_color()

            # make target upstream a graph
            target_G = G.subgraph(list(dict(nx.bfs_predecessors(RG, target)).keys()) + [target])

            # draw graph
            nx.draw_networkx(
                target_G,
                pos,
                node_size=10,
                node_color=[c],
                width=2,
                edge_color=[c],
                with_labels=False,
                ax=ax,
            )

            # draw outlets
            nx.draw_networkx_nodes(
                target_G,
                pos,
                nodelist=[target],
                node_size=100,
                node_color="k",
                edgecolors=c,
                ax=ax,
            )

            # draw labels
            if target_nodes_labeldict is not None:
                nx.draw_networkx_labels(
                    target_G,
                    pos,
                    target_nodes_labeldict,
                    font_size=16,
                    font_color='k')
    return ax

def gpd_to_digraph(branches: gpd.GeoDataFrame) -> nx.DiGraph():

    G = nx.DiGraph()

    for index, row in branches.iterrows():
        from_node= row.geometry.coords[0]
        to_node = row.geometry.coords[-1]

        G.add_edge(from_node, to_node, **row.to_dict())

    return G

def validate_1dnetwork_connectivity(branches: gpd.GeoDataFrame, plotit=False, ax=None, exportpath = os.getcwd(), logger = logging):
    """Function to validate the connectivity of provided branch"""

    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create digraph
    G = gpd_to_digraph(branches)
    pos = {xy: xy for xy in G.nodes()}

    # convert to undirected graph
    UG = G.to_undirected()

    # find connected components in undirected graph
    outlets = []
    for i,SG in enumerate(nx.connected_components(UG)):

        # make components a subgraph
        SG = G.subgraph(SG)

        # find outlets of the subgraph
        outlets.append([n for n in SG.nodes() if G.out_degree(n) == 0])


    outlets = sum(outlets,[])
    outlet_ids = {p: [li for li, l in branches.geometry.iteritems() if l.intersects(Point(p))] for p in
                        outlets}

    # report
    if i == 0:
        logger.info('Validation results: the 1D network are fully connected.  Supress plotit function.')
    else:
        logger.info(f'Validation results: the 1D network are disconnected have {i+1} connected components')

    if plotit:
        ax = make_graphplot_for_targetnodes(G, outlets, outlet_ids, layout = 'graphviz')
        ax.set_title('Connectivity of the 1d network, with outlets' +
                   '(connectivity outlets, not neccessarily network outlets due to bi-directional flow, please check these)', wrap=True)
        plt.savefig(exportpath.joinpath('validate_1dnetwork_connectivity'))

    return None

def validate_1dnetwork_flowpath(
    branches: gpd.GeoDataFrame, branchType_col= 'branchType', plotit = False, ax=None, exportpath = os.getcwd(), logger = logging
):
    """function to validate flowpath (flowpath to outlet) for provided branch"""

    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create digraph
    G = gpd_to_digraph(branches)
    pos = {xy: xy for xy in G.nodes()}

    # create separate graphs for pipes and branches
    pipes = branches.query(f"{branchType_col} == 'Pipe'")
    channels = branches.query(f"{branchType_col} == 'Channel'")

    # validate 1d network based on pipes -> channel logic
    if len(pipes) > 0:
        # create graph
        PG = gpd_to_digraph(pipes)
        # pipes outlets
        pipes_outlets = [n for n in PG.nodes() if G.out_degree(n) == 0]
        pipes_outlet_ids = {p: [li for li, l in pipes.geometry.iteritems() if l.intersects(Point(p))] for p in
                               pipes_outlets}
        logger.info(
            f"Validation result: the 1d network has {len(pipes_outlets)} pipe outlets."
        )

    if len(channels) > 0:
        # create graph
        CG = gpd_to_digraph(channels)
        # pipes outlets
        channels_outlets = [n for n in CG.nodes() if G.out_degree(n) == 0]
        channels_outlet_ids = {p:[li for li,l in channels.geometry.iteritems() if l.intersects(Point(p))] for p in channels_outlets}
        logger.info(
            f"Validation result: the 1d network has {len(channels_outlets)} channel outlets."
        )

    if (len(channels) > 0) and (len(pipes) > 0):
        isolated_outlets = [p for p in pipes_outlets if not any(Point(p).intersects(l) for _,l in channels.geometry.iteritems())]
        isolated_outlet_ids = {}
        for p in isolated_outlets:
            isolated_outlet_id = [li for li,l in pipes.geometry.iteritems() if l.intersects(Point(p))]
            isolated_outlet_ids[p] = isolated_outlet_id
            logger.warning(
                f"Validation result: downstream of {isolated_outlet_id} are not located on channels. Please double check. "
            )

    # plot
    if plotit:
        ax = make_graphplot_for_targetnodes(G,
                                            target_nodes={**isolated_outlet_ids, **channels_outlet_ids}.keys(),
                                            target_nodes_labeldict= {**isolated_outlet_ids, **channels_outlet_ids})
        ctx.add_basemap(ax=ax, url=ctx.providers.OpenStreetMap.Mapnik, crs=branches.crs.to_epsg())
        ax.set_title('Flow path of the 1d network, with outlets' +
                   '(flowpath outlets, not neccessarily network outlets due to bi-directional flow , please check these)', wrap=True)
        plt.savefig(exportpath.joinpath('validate_1dnetwork_flowpath'))

    return None





# ===============================================================
#                  Subcat
# ===============================================================

# FIXME BMA: check if HydroMT has thiessam polygon functions that can be used
# Returns a new np.array of towers that within the bounding_box
def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[2]),
                          np.logical_and(bounding_box[1] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))

# Generates a bounded vornoi diagram with finite regions
def bounded_voronoi(towers, bounding_box):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)

    # Mirror points left, right, above, and under to provide finite regions for the edge regions of the bounding box
    points_center = towers[i, :]

    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])

    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[2] + (bounding_box[2] - points_right[:, 0])

    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[1] - (points_down[:, 1] - bounding_box[1])

    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)

    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # from scipy.spatial import voronoi_plot_2d
    # voronoi_plot_2d(vor)
    vor.filtered_points = points_center # creates a new attibute for points that form the diagram within the region
    vor.filtered_regions = [vor.regions[i] for i in vor.point_region[:vor.npoints//5]] # grabs the first fifth of the regions, which are the original regions

    return vor


def get_thiessen_polygons(towers: gpd.GeoDataFrame, region: gpd.GeoDataFrame = None, barriers:gpd.GeoDataFrame = None, logger = logging):
    """Function to get thiessen polygon in a polygon"""
    # get voronoi in bounding box
    _towers = np.array([[p.x, p.y] for p in towers.geometry])
    logger.debug(f'Get thiessen polygon extent.')

    if towers is not None:
        x1, y1, x2, y2 = towers.buffer(100).unary_union.bounds
        tbox = box(x1, y1, x2, y2)

    if region is not None:
        x1, y1, x2, y2 = region.buffer(100).unary_union.bounds
        rbox = box(x1, y1, x2, y2)
    else:
        rbox = tbox

    bbox = gpd.GeoDataFrame({'geometry':[tbox, rbox]}, crs = towers.crs)
    _bounding_box = bbox.total_bounds

    _vor = bounded_voronoi(_towers,_bounding_box)

    vor_in_bounds = gpd.GeoDataFrame(
        {'geometry':
        [poly for poly in polygonize([
        LineString(_vor.vertices[line])
        for line in _vor.ridge_vertices
        if -1 not in line])]}, crs = towers.crs)

    # assign index
    vor_in_bounds = gpd.sjoin(vor_in_bounds, towers, how='left', op = 'contains')

    # get voronoi in region (clip)
    if region is None:
        vor_in_region = vor_in_bounds
    else:
        vor_in_region = gpd.clip(vor_in_bounds, region).explode() # explode multi polygons
        vor_in_region = vor_in_region[vor_in_region.geom_type.values == 'Polygon']  # get rid of points
    # split voronoi at barries (polygon/line)
    if barriers is None:
        vor_in_region_avoid_barriers = vor_in_region
    elif barriers['geometry'].geom_type[0]=='Polygon':
        vor_in_region_avoid_barriers = gpd.overlay(
            vor_in_region, barriers, how='symmetric_difference').explode()
    elif barriers['geometry'].geom_type[0]=='LineString':
        # apply a buffer of 0.01 # Question: should this be included in user input?
        barriers['geometry'] = barriers.buffer(0.01)
        vor_in_region_avoid_barriers = gpd.overlay(
            vor_in_region, barriers, how='symmetric_difference').explode()
    else:
        logging.warn(f'barriers type is not recognized. Please use Polygon or LineString.')
        vor_in_region_avoid_barriers = vor_in_region

    # set index
    vor_in_region_avoid_barriers.index = vor_in_region_avoid_barriers['index_right']
    vor_in_region_avoid_barriers.index.name = towers.index.name

    return vor_in_region_avoid_barriers['geometry']

def link_polygons_to_points(polygons: gpd.GeoDataFrame, points: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame = None, maxdist:float = float('inf'), logger = logging) -> gpd.GeoDataFrame:

        """
        Method to link polygons to points within maxdist and avoid barriers (lines).

        Parameters
        polygons : gpd.GeoDataFrame of polygons: ID is the index column
        points: gpd.GeoDataFrame of points: ID is the index column
        maxdist : float
            Maximum distance between polygons centroid to points. If the distance is larger, they are not linked.
        barriers: gpd.GeoSeries of lines or Polygon
            Shapes that are barries where links are not allowed.

        Returns
        polygons_to_points: gpd.GeoDataFrame of links
        """

        if barriers is not None and barriers['geometry'].geom_type[0] == 'Polygon':
            barriers['geometry'] = barriers.exterior # to line
        elif barriers is not None and barriers['geometry'].geom_type[0] != 'LineString':
            raise NotImplementedError

        # Collect subcats centroids
        polygons['centroid'] = polygons.centroid
        from_points = [((poly.centroid.x, poly.centroid.y), poly.Index) for poly in polygons.itertuples()]
        to_points = [((p.geometry.x, p.geometry.y), p.Index) for p in points.itertuples()]

        # # group branch ends based on off set
        dist = distance.cdist(np.array([p for p,pi in from_points]),
                                                  np.array([p for p,pi in to_points]))

        groups= {}
        for row_i, row in enumerate(dist):
            groups[from_points[row_i]] = []
            for col_i, col in enumerate(row):
                if col:
                    if dist[row_i, col_i] <= maxdist:
                        groups[from_points[row_i]].append(to_points[col_i])

        # remove duplicated group, group that does not satisfy max_points in groups. Assign endpoints
        endpoints = {k: list(set(v)) for k, v in groups.items()}

        # dict
        polygons_to_points = {'FROM_INDEX': list(), 'TO_INDEX': list(), 'dist': list(), 'geometry':list()}

        # loop through from_points
        for i, (c, cid) in enumerate(from_points):

            # For from_points determine the distance to to_points
            to_points = [p for p in endpoints[(c, cid)]]
            found = False

            while not found and len(to_points) > 0:

                # Find distance and nearest to_points
                mindist, minidx = KDTree([m for m, mid in to_points]).query(c)

                # check if the distance is smaller than maxdist.
                if mindist <= maxdist:

                    m, mid = to_points[minidx]
                    c_to_m = LineString([c, m])
                    found = True

                    # double check if intersect with barries
                    if barriers is not None:

                        intersect = np.any([b.intersects(c_to_m) for _,b in barriers.geometry.iteritems()])
                        if intersect:
                            found = False
                            to_points.pop(minidx)

                    if found:
                        logger.debug(f'Link {cid} to {mid}.')
                        polygons_to_points['FROM_INDEX'].append(cid)
                        polygons_to_points['TO_INDEX'].append(mid)
                        polygons_to_points['dist'].append(mindist)
                        polygons_to_points['geometry'].append(c_to_m)

                    if not found and len(to_points) == 0:
                        logger.debug(f'Could not link {cid}.')

        return gpd.GeoDataFrame(polygons_to_points, crs = polygons.crs)



def redistribute_vertices(geom, distance):
    """ static function to redistribute vertices based on a distance without altering important geometry vertices"""

    if geom.geom_type == 'LineString':
        parts = [LineString([p1, p2]) for p1, p2 in zip(geom.coords[:-1], geom.coords[1:])]
        parts_new = []
        for part in parts:
            num_vert = int(round(part.length / distance))
            if num_vert == 0:
                num_vert = 1
            parts_new.append(LineString(
                [part.interpolate(float(n) / num_vert, normalized=True)
                 for n in range(num_vert + 1)]))
        multi_line = geometry.MultiLineString(parts_new)
        return linemerge(multi_line)
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))

